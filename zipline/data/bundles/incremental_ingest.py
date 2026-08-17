"""
Incremental ingest for the ``alpaca_api`` bundle.

Replaces the ``__main__`` block of the sibling ``alpaca_api`` module, which
re-fetched the same ~1500 sessions and wrote them into a *fresh* timestamped
directory on every cron run.  zipline's ``bundles.load()`` only ever reads the
newest directory, so that left 511 near-identical copies (415MB) on disk.

This script maintains ONE directory instead, tagged with a ``.central_bundle``
marker file:

  daily_equities.bcolz    Rewritten in full each run.  This is cheap (34 symbols
                          x 1500 sessions is ~21s / 776KB) and it is *required*:
                          bars are requested with ``adjustment='all'``, so a new
                          split restates the whole history, and an append-only
                          daily store would silently keep serving pre-split
                          prices for everything before the split date.

  minute_equities.bcolz   Appended only for sessions newer than what is already
                          on disk.  ``BcolzMinuteBarWriter.open()`` supports
                          reopening an existing store; ``BcolzDailyBarWriter``
                          has no equivalent, which is why the two halves are
                          handled differently.

At the end the directory is renamed to a fresh ingest timestamp, so zipline's
"most recent directory wins" lookup keeps working without patching the engine.

Usage
-----
    python -m zipline.data.bundles.incremental_ingest                   # nightly
    python -m zipline.data.bundles.incremental_ingest --rebuild         # start over
    python -m zipline.data.bundles.incremental_ingest --minute-sessions 500
    python -m zipline.data.bundles.incremental_ingest --no-minute       # daily only

Running the file by path still works; imports are absolute, exactly as in
``alpaca_api.py``, so the module does not depend on being imported as part of
the package.  It is deliberately *not* imported from ``bundles/__init__.py`` --
it registers nothing and its import would pull the Alpaca client into every
``import zipline`` for no benefit.

Old pre-existing timestamped ingest directories are left untouched; prune them
with ``zipline clean -b alpaca_api --keep-last N`` when you are satisfied this
path works.
"""

import argparse
import collections
import json
import os
import shutil
import time
from datetime import timedelta

import numpy as np
import pandas as pd
import trading_calendars
from trading_calendars import TradingCalendar

from zipline.assets import AssetDBWriter
from zipline.data.adjustments import SQLiteAdjustmentWriter
from zipline.data.bcolz_daily_bars import BcolzDailyBarWriter, BcolzDailyBarReader
from zipline.data.minute_bars import BcolzMinuteBarWriter, BcolzMinuteBarMetadata
from zipline.data.bundles import core as bundles_core
from zipline.data.bundles.alpaca_api import (
    NY,
    df_generator,
    initialize_client,
    list_assets,
)
from alpaca_trade_api.rest import TimeFrame, TimeFrameUnit
import zipline.data.bundles.alpaca_api as alpaca_api

BUNDLE_NAME = "alpaca_api"
MARKER = ".central_bundle"
SID_MAP_FILE = "sid_map.json"

DAILY_LOOKBACK_DAYS = 1500
MINUTE_LOOKBACK_SESSIONS = 250
MINUTES_PER_DAY = 390

# Alpaca allows 200 symbols per request; minute history is pulled in date slices
# so a single response stays a manageable size even though get_bars() pages
# transparently under the hood.
SYMBOL_BATCH = 200
MINUTE_CHUNK_DAYS = 30


# ---------------------------------------------------------------------------
# paths / sessions
# ---------------------------------------------------------------------------

def bundle_root():
    return os.path.join(
        os.environ.get("ZIPLINE_ROOT", os.path.expanduser("~/.zipline")),
        "data",
        BUNDLE_NAME,
    )


def find_central_dir(root):
    """Return the one directory carrying our marker file, or None."""
    if not os.path.isdir(root):
        return None
    found = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if os.path.isdir(path) and os.path.exists(os.path.join(path, MARKER)):
            found.append(path)
    if len(found) > 1:
        raise RuntimeError(
            "more than one directory carries the %s marker: %s\n"
            "keep exactly one and delete the rest" % (MARKER, found)
        )
    return found[0] if found else None


def last_completed_session(cal):
    """
    The most recent session that has actually closed.

    Same rule as alpaca_api.py's __main__: "today" is evaluated in exchange-local
    time and the session must be CLOSED, otherwise the API is asked for a session
    with no data yet and _fillna() forward-fills the previous close into it,
    fabricating a duplicate bar that the daily report then reads.
    """
    now_ny = pd.Timestamp("now", tz=NY)
    end_date = now_ny.date()
    while not cal.is_session(str(end_date)):
        end_date -= timedelta(days=1)
    if now_ny < cal.session_close(pd.Timestamp(end_date, tz="utc")):
        end_date -= timedelta(days=1)
        while not cal.is_session(str(end_date)):
            end_date -= timedelta(days=1)
    return pd.Timestamp(end_date, tz="utc")


def daily_start_session(cal, end_session):
    start = end_session - timedelta(days=DAILY_LOOKBACK_DAYS)
    while not cal.is_session(start):
        start -= timedelta(days=1)
    return start


# ---------------------------------------------------------------------------
# stable sid assignment
# ---------------------------------------------------------------------------

def load_sid_map(path, symbols):
    """
    Map symbol -> sid, stable across runs.

    This has to be persisted rather than recomputed.  ``list_assets()`` returns
    ``list(set(custom_asset_list))`` and CPython randomises string hashes per
    process, so the set's iteration order -- and therefore any positional sid
    assignment -- differs on every invocation.  That is harmless when each run
    rebuilds the whole bundle from scratch, but for an append-only minute store
    it would write tonight's AAPL bars into last night's MSFT ctable.
    """
    existing = {}
    if path is not None and os.path.exists(path):
        with open(path, "r") as f:
            existing = json.load(f)

    sid_map = {s: int(sid) for s, sid in existing.items()}
    next_sid = (max(sid_map.values()) + 1) if sid_map else 0
    for symbol in sorted(symbols):
        if symbol not in sid_map:
            sid_map[symbol] = next_sid
            next_sid += 1
    return sid_map


def save_sid_map(path, sid_map):
    with open(path, "w") as f:
        json.dump(sid_map, f, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# daily half -- full rewrite
# ---------------------------------------------------------------------------

METADATA_DTYPE = [
    ("symbol", "object"),
    ("start_date", "datetime64[ns]"),
    ("end_date", "datetime64[ns]"),
    ("first_traded", "datetime64[ns]"),
    ("auto_close_date", "datetime64[ns]"),
    ("exchange", "object"),
]


def empty_metadata(n_rows):
    df = pd.DataFrame(np.empty(n_rows, dtype=METADATA_DTYPE))
    # np.empty leaves the datetime columns full of garbage, which would survive
    # the dropna() below and produce assets with nonsense date ranges.
    for col in ("start_date", "end_date", "first_traded", "auto_close_date"):
        df[col] = pd.NaT
    df["symbol"] = None
    df["exchange"] = None
    return df


def write_daily(bundle_dir, cal, start_session, end_session, sid_map):
    """Rewrite daily_equities.bcolz from scratch; return the asset metadata."""
    daily_path = os.path.join(bundle_dir, "daily_equities.bcolz")
    if os.path.exists(daily_path):
        shutil.rmtree(daily_path)
    os.makedirs(daily_path)

    metadata = empty_metadata(max(sid_map.values()) + 1)
    written = []

    def rows():
        for (sid_df, symbol, start, end, first_traded, auto_close, exchange) in \
                df_generator(interval="1d",
                             start=start_session,
                             end=end_session,
                             assets_to_sids=sid_map):
            sid = sid_df[0]
            metadata.iloc[sid] = (symbol, start, end, first_traded,
                                  auto_close, exchange)
            written.append(sid)
            yield sid_df

    writer = BcolzDailyBarWriter(daily_path, cal, start_session, end_session)
    writer.write(rows(), assets=list(sid_map.values()), show_progress=True)

    metadata.dropna(inplace=True)
    print("  daily: wrote %d symbols, %s -> %s"
          % (len(written), start_session.date(), end_session.date()))
    return metadata


# ---------------------------------------------------------------------------
# minute half -- append only
# ---------------------------------------------------------------------------

FIELDS = ["open", "high", "low", "close", "volume"]
AGG = collections.OrderedDict([
    ("open", "first"),
    ("high", "max"),
    ("low", "min"),
    ("close", "last"),
    ("volume", "sum"),
])


def fetch_minute_chunk(symbols, start, end, valid_minutes, session_closes):
    """
    Pull 1-minute bars and return {symbol: DataFrame} indexed by zipline market
    minutes (UTC, tz-aware).

    Three alignment fixes over alpaca_api.get_aggs_from_alpaca()'s minute path,
    which has never actually been exercised (the bundle only ever registered
    '1d'):

    * Alpaca returns a UTC index covering full extended hours (08:00-23:59 UTC).
      That helper calls ``between_time("09:30", "16:00")`` directly on it, which
      selects 09:30-16:00 *UTC* -- i.e. 05:30-12:00 New York, mostly pre-market.

    * Alpaca labels each bar at its START (the 09:30 bar covers [09:30, 09:31));
      zipline labels at the END, so its session runs 09:31-16:00.  Everything
      needs a +1 minute shift or each bar lands one slot early.

    * That shift alone silently discards the closing auction.  The auction
      prints at exactly 16:00:00, so it sits in Alpaca's [16:00, 16:01) bar,
      which a blanket +1 shift relabels 16:01 -- past the end of the session.
      For AAPL on 2026-08-14 that bar was 7.17M of the day's 18.9M
      regular-session shares, and dropping it left zipline's 16:00 close at
      305.98 instead of the true 305.75.  Bars starting exactly at a session
      close are therefore relabelled onto the close itself and merged with the
      preceding minute.  Keying off the calendar's own close (rather than a
      literal 16:00) keeps early-close days correct, where the auction is at
      13:00.
    """
    # The data API parses start/end as RFC3339 and rejects naive datetimes
    # ("extra text" 400), so the UTC offset has to survive into the query string.
    resp = alpaca_api.CLIENT.get_bars(
        symbols,
        TimeFrame(1, TimeFrameUnit.Minute),
        start=pd.Timestamp(start).tz_convert("UTC").isoformat(),
        end=pd.Timestamp(end).tz_convert("UTC").isoformat(),
        adjustment="all",
    )
    df = resp.df
    if df.empty:
        return {}

    df = df.sort_index(kind="mergesort")
    idx = df.index
    is_close_bar = idx.isin(session_closes)
    df = df.copy()
    df.index = pd.DatetimeIndex(
        np.where(is_close_bar, idx, idx + pd.Timedelta(minutes=1)), tz="UTC",
    )

    # Keep only true market minutes.  Filtering against the calendar (rather
    # than a fixed 09:31-16:00 window) is the other half of what makes half-days
    # correct: the writer's own minute grid is a flat 390 slots from each
    # session's open and does NOT account for early closes, so post-close
    # extended-hours bars would otherwise be written into perfectly
    # valid-looking in-session slots.
    df = df[df.index.isin(valid_minutes)]
    if df.empty:
        return {}

    out = {}
    for symbol, group in df.groupby("symbol"):
        g = group[FIELDS]
        # The final minute now has two contributing rows (the 15:59 bar and the
        # auction).  Collapse duplicate labels rather than dropping either; the
        # frame is still in bar-start order, so first/last pick up correctly.
        if g.index.has_duplicates:
            g = g.groupby(level=0, sort=True).agg(AGG)
        else:
            g = g.sort_index()
        if not g.empty:
            out[symbol] = g
    return out


def ensure_minute_metadata(bundle_dir, cal, end_session, lookback_sessions):
    """
    Lay down an empty minute store if there is none.

    ``bundles.load()`` builds the BcolzMinuteBarReader eagerly, so a bundle with
    no ``minute_equities.bcolz/metadata.json`` cannot be loaded at all -- even by
    a daily-only consumer like dailyReport.py.  The stock ingest gets this for
    free because BcolzMinuteBarWriter writes its metadata from __init__.
    """
    minute_path = os.path.join(bundle_dir, "minute_equities.bcolz")
    if os.path.exists(os.path.join(minute_path, "metadata.json")):
        return
    if not os.path.exists(minute_path):
        os.makedirs(minute_path)
    start_session = cal.sessions_window(end_session, -(lookback_sessions - 1))[0]
    BcolzMinuteBarWriter(minute_path, cal, start_session, end_session,
                         MINUTES_PER_DAY)
    print("  minute: skipped (wrote empty store metadata only)")


def update_minute(bundle_dir, cal, end_session, sid_map, lookback_sessions):
    """Create or extend minute_equities.bcolz, fetching only what is missing."""
    minute_path = os.path.join(bundle_dir, "minute_equities.bcolz")
    sessions = cal.sessions_in_range(
        cal.sessions_window(end_session, -(lookback_sessions - 1))[0],
        end_session,
    )

    have_store = os.path.exists(os.path.join(minute_path, "metadata.json"))
    if have_store:
        writer = BcolzMinuteBarWriter.open(minute_path, end_session=end_session)
        start_session = BcolzMinuteBarMetadata.read(minute_path).start_session
        print("  minute: reopened existing store (start %s)" % start_session.date())
    else:
        if os.path.exists(minute_path):
            shutil.rmtree(minute_path)
        os.makedirs(minute_path)
        start_session = sessions[0]
        writer = BcolzMinuteBarWriter(
            minute_path, cal, start_session, end_session, MINUTES_PER_DAY,
        )
        print("  minute: created new store, backfilling %d sessions from %s"
              % (len(sessions), start_session.date()))

    # Per-sid watermark: a symbol added to the universe later starts from
    # scratch while the rest only top up.  NaT means nothing written yet.
    watermarks = {}
    for symbol, sid in sid_map.items():
        try:
            last = writer.last_date_in_output_for_sid(sid)
        except Exception:
            last = pd.NaT
        watermarks[symbol] = last

    def first_needed(symbol):
        last = watermarks[symbol]
        if last is pd.NaT or pd.isnull(last):
            return start_session
        idx = sessions.searchsorted(last, side="right")
        return sessions[idx] if idx < len(sessions) else None

    pending = {s: first_needed(s) for s in sid_map}
    pending = {s: d for s, d in pending.items() if d is not None and d <= end_session}
    if not pending:
        print("  minute: already up to date through %s" % end_session.date())
        return

    # Fetch per distinct start date, NOT from min(pending).  A single symbol
    # Alpaca has no bars for keeps a null watermark forever, and taking the
    # global minimum let that one symbol drag the whole request range back to
    # the start of the store on every run -- re-downloading a year of history
    # nightly just to append nothing.
    groups = collections.defaultdict(list)
    for symbol, needed in pending.items():
        groups[needed].append(symbol)

    n_written = 0
    got_data = set()
    for start_at in sorted(groups):
        symbols = sorted(groups[start_at])
        todo = sessions[sessions.searchsorted(start_at):]
        if len(todo) == 0:
            continue

        valid_minutes = cal.minutes_for_sessions_in_range(todo[0], todo[-1])
        session_closes = pd.DatetimeIndex(
            [cal.session_close(s) for s in todo]
        ).tz_convert("UTC")
        print("  minute: fetching %d sessions x %d symbols (%s -> %s)"
              % (len(todo), len(symbols), todo[0].date(), todo[-1].date()))

        # Chronological chunks: BcolzMinuteBarWriter requires each sid's appends
        # to be strictly increasing, and it zero-fills any gap between them.
        for i in range(0, len(todo), MINUTE_CHUNK_DAYS):
            chunk = todo[i:i + MINUTE_CHUNK_DAYS]
            c_start, c_end = chunk[0], chunk[-1]
            c_minutes = valid_minutes[
                (valid_minutes >= c_start)
                & (valid_minutes <= c_end + pd.Timedelta(days=1))
            ]
            for j in range(0, len(symbols), SYMBOL_BATCH):
                batch = symbols[j:j + SYMBOL_BATCH]
                frames = fetch_minute_chunk(
                    batch,
                    c_start,
                    c_end + pd.Timedelta(days=1),
                    c_minutes,
                    session_closes,
                )
                for symbol, frame in frames.items():
                    frame = frame[frame.index >= start_at]
                    if frame.empty:
                        continue
                    writer.write_sid(sid_map[symbol], frame,
                                     invalid_data_behavior="warn")
                    got_data.add(symbol)
                    n_written += len(frame)
            print("    ... through %s (%d bars)" % (c_end.date(), n_written))

    # Symbols the API returned nothing at all for get zero-filled up to the
    # current session so their watermark advances too.  Without this their
    # window is re-requested in full every night, forever.  Reaching this point
    # means every request succeeded (get_bars raises rather than returning empty
    # on failure), so an empty result really does mean "Alpaca has no bars" --
    # and all-zero OHLCV is already how zipline represents a session with no
    # trading, which is what the writer would fill a gap with anyway.
    #
    # Deliberately NOT restricted to symbols that never had data: a symbol whose
    # watermark is merely stale (delisted, halted, or dropped by the data feed)
    # would otherwise be re-requested every night just as expensively.  Symbols
    # that got *partial* data are excluded, so their remaining sessions are
    # still picked up on the next run.
    no_data = sorted(s for s in pending if s not in got_data)
    for symbol in no_data:
        writer.pad(sid_map[symbol], end_session)
    if no_data:
        print("  minute: no data available for %s; padded to %s"
              % (", ".join(no_data), end_session.date()))

    print("  minute: appended %d bars through %s" % (n_written, end_session.date()))


# ---------------------------------------------------------------------------
# assets / adjustments
# ---------------------------------------------------------------------------

def write_assets_and_adjustments(bundle_dir, metadata, cal):
    assets_path = os.path.join(bundle_dir, "assets-7.sqlite")
    adjustments_path = os.path.join(bundle_dir, "adjustments.sqlite")
    daily_path = os.path.join(bundle_dir, "daily_equities.bcolz")

    if os.path.exists(assets_path):
        os.remove(assets_path)

    # Without an explicit exchanges frame the asset writer defaults country_code
    # to '??', which makes the bundle unusable by any US_EQUITIES pipeline domain.
    exchanges = pd.DataFrame(
        data=[["NYSE", "NYSE", "US"]],
        columns=["exchange", "canonical_name", "country_code"],
    )
    AssetDBWriter(assets_path).write(equities=metadata, exchanges=exchanges)

    with SQLiteAdjustmentWriter(adjustments_path,
                                BcolzDailyBarReader(daily_path),
                                overwrite=True) as adj:
        adj.write()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true",
                        help="delete the central bundle and build it from scratch")
    parser.add_argument("--minute-sessions", type=int,
                        default=MINUTE_LOOKBACK_SESSIONS,
                        help="sessions of minute history to keep (default %d)"
                             % MINUTE_LOOKBACK_SESSIONS)
    parser.add_argument("--no-minute", action="store_true",
                        help="skip the minute half entirely")
    args = parser.parse_args()

    started = time.time()
    cal = trading_calendars.get_calendar("NYSE")  # type: TradingCalendar

    end_session = last_completed_session(cal)

    # The daily store has to extend one session PAST the last completed one.
    # SimplePipelineEngine computes the row for session T out of data through
    # T-1, so dailyReport.py's row for the next session -- the one carrying
    # tomorrow's factors, built from today's close -- only exists if that
    # session has a slot in the bcolz index.  The bar written into that slot is
    # a forward-filled placeholder (the API has no data for a session that has
    # not happened yet); the pipeline never reads it as data, only as an index
    # entry.  Minute bars stop at the last completed session as before.
    daily_end_session = cal.next_session_label(end_session)
    start_session = daily_start_session(cal, daily_end_session)

    initialize_client()
    symbols = list_assets()

    root = bundle_root()
    if not os.path.isdir(root):
        os.makedirs(root)

    bundle_dir = find_central_dir(root)
    if bundle_dir is not None and args.rebuild:
        print("--rebuild: removing %s" % bundle_dir)
        shutil.rmtree(bundle_dir)
        bundle_dir = None

    if bundle_dir is None:
        # Dot-prefixed on purpose.  ingestions_for_bundle() runs
        # pd.Timestamp() over every non-hidden entry in the bundle root, so a
        # staging directory left behind by a crashed first run would raise and
        # break bundles.load() for the whole bundle; pth.hidden() skips names
        # starting with '.'.
        bundle_dir = os.path.join(root, ".central.building")
        if os.path.exists(bundle_dir):
            shutil.rmtree(bundle_dir)
        os.makedirs(bundle_dir)
        open(os.path.join(bundle_dir, MARKER), "w").close()
        print("creating central bundle at %s" % bundle_dir)
    else:
        print("updating central bundle at %s" % bundle_dir)

    sid_map_path = os.path.join(bundle_dir, SID_MAP_FILE)
    sid_map = load_sid_map(sid_map_path, symbols)
    save_sid_map(sid_map_path, sid_map)

    metadata = write_daily(bundle_dir, cal, start_session, daily_end_session,
                           sid_map)

    if args.no_minute:
        ensure_minute_metadata(bundle_dir, cal, end_session, args.minute_sessions)
    else:
        update_minute(bundle_dir, cal, end_session, sid_map, args.minute_sessions)

    write_assets_and_adjustments(bundle_dir, metadata, cal)

    # Rename to a fresh ingest timestamp so bundles.load()'s most-recent-wins
    # lookup picks this directory up without any engine changes.
    stamp = pd.Timestamp.utcnow().tz_convert("utc").tz_localize(None)
    final_dir = os.path.join(root, bundles_core.to_bundle_ingest_dirname(stamp))
    if os.path.abspath(final_dir) != os.path.abspath(bundle_dir):
        if os.path.exists(final_dir):
            shutil.rmtree(final_dir)
        os.rename(bundle_dir, final_dir)

    print("central bundle -> %s" % final_dir)
    print("--- It took %s ---" % timedelta(seconds=time.time() - started))


if __name__ == "__main__":
    main()

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame, TimeFrameUnit
from zipline.gens.brokers.broker import Broker
import zipline.protocol as zp
from zipline.finance.order import (Order as ZPOrder,
                                   ORDER_STATUS as ZP_ORDER_STATUS)
from zipline.finance.execution import (MarketOrder,
                                       LimitOrder,
                                       StopOrder,
                                       StopLimitOrder)
from zipline.finance.transaction import Transaction
from zipline.api import symbol as symbol_lookup
from zipline.errors import SymbolNotFound
import pandas as pd
from collections import OrderedDict

import numpy as np
import uuid

from logbook import Logger
import sys

if sys.version_info > (3,):
    long = int

log = Logger('Alpaca Broker')
NY = 'America/New_York'


class ALPACABroker(Broker):
    '''
    Broker class for Alpaca.
    The uri parameter is not used. Instead, the API key must be
    set via environment variables (APCA_API_KEY_ID and APCA_API_SECRET_KEY).
    Orders are identified by the UUID (v4) generated here and
    associated in the broker side using client_order_id attribute.
    Currently this class makes use of REST API only, but websocket
    streaming can possibly used too.
    '''

    def __init__(self):
        self._api = tradeapi.REST()
        self._subscribed = OrderedDict()

    def subscribe_to_market_data(self, asset):
        self._subscribed[asset.symbol] = asset

    @property
    def subscribed_assets(self):
        # Must be a property: Broker declares it as one and ib_broker
        # implements it as one. Without the decorator this returns the bound
        # method itself, and LiveTradingAlgorithm.on_exit() -- which runs
        # whenever --realtime-bar-target is set -- hands that function to
        # get_realtime_bars and dies with
        # "AttributeError: 'function' object has no attribute 'symbol'".
        #
        # Nothing in zipline calls subscribe_to_market_data, so the set is
        # populated from the assets the algorithm actually asks about instead
        # (see get_spot_value). Returning a bare [] met the interface and made
        # the realtime bar dump silently write nothing.
        return list(self._subscribed.values())


    def set_metrics_tracker(self, metrics_tracker):
        self.metrics_tracker = metrics_tracker

    @property
    def positions(self):
        self._get_positions_from_broker()
        return self.metrics_tracker.positions


    @property
    def portfolio(self):
        account = self._api.get_account()
        z_portfolio = zp.Portfolio()
        # Portfolio.__setattr__ raises; mutate through MutableView, the same way
        # ib_broker.py does. Direct assignment here has never worked -- the guard
        # in protocol.py predates this file.
        editable_portfolio = zp.MutableView(z_portfolio)
        editable_portfolio.cash = float(account.cash)
        editable_portfolio.positions = self.positions
        editable_portfolio.positions_value = float(
            account.portfolio_value) - float(account.cash)
        editable_portfolio.portfolio_value = float(account.portfolio_value)
        return z_portfolio

    @property
    def account(self):
        account = self._api.get_account()
        z_account = zp.Account()
        # Account is immutable for the same reason as Portfolio, see above.
        editable_account = zp.MutableView(z_account)
        editable_account.buying_power = float(account.cash)
        editable_account.total_position_value = float(
            account.portfolio_value) - float(account.cash)
        editable_account.net_liquidation = float(account.portfolio_value)
        return z_account

    @property
    def time_skew(self):
        return pd.Timedelta('0 sec')  # TODO: use clock API

    def is_alive(self):
        try:
            self._api.get_account()
            return True
        except BaseException:
            return False

    def _order2zp(self, order):
        zp_order = ZPOrder(
            id=order.client_order_id,
            asset=symbol_lookup(order.symbol),
            amount=int(order.qty) if order.side == 'buy' else -int(order.qty),
            stop=float(order.stop_price) if order.stop_price else None,
            limit=float(order.limit_price) if order.limit_price else None,
            dt=order.submitted_at,
            commission=0,
        )
        zp_order.status = ZP_ORDER_STATUS.OPEN
        if order.canceled_at:
            zp_order.status = ZP_ORDER_STATUS.CANCELLED
        if order.failed_at:
            zp_order.status = ZP_ORDER_STATUS.REJECTED
        if order.filled_at:
            zp_order.status = ZP_ORDER_STATUS.FILLED
            zp_order.filled = int(order.filled_qty)
        return zp_order

    def _new_order_id(self):
        return uuid.uuid4().hex

    def order(self, asset, amount, style):
        symbol = asset.symbol
        qty = amount if amount > 0 else -amount
        side = 'buy' if amount > 0 else 'sell'
        order_type = 'market'
        if isinstance(style, MarketOrder):
            order_type = 'market'
        elif isinstance(style, LimitOrder):
            order_type = 'limit'
        elif isinstance(style, StopOrder):
            order_type = 'stop'
        elif isinstance(style, StopLimitOrder):
            order_type = 'stop_limit'

        limit_price = style.get_limit_price(side == 'buy') or None
        stop_price = style.get_stop_price(side == 'buy') or None

        zp_order_id = self._new_order_id()
        dt = pd.to_datetime('now', utc=True)
        zp_order = ZPOrder(
            dt=dt,
            asset=asset,
            amount=amount,
            stop=stop_price,
            limit=limit_price,
            id=zp_order_id,
        )

        order = self._api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type=order_type,
            time_in_force='day',
            limit_price=limit_price,
            stop_price=stop_price,
            client_order_id=zp_order.id,
        )
        zp_order = self._order2zp(order)
        return zp_order

    @property
    def orders(self):
        orders = {}
        for o in self._api.list_orders('all'):
            try:
                orders[o.client_order_id] = self._order2zp(o)
            except:
                continue
        return orders

    @property
    def transactions(self):
        orders = self._api.list_orders(status='closed')
        results = {}
        for order in orders:
            if order.filled_at is None:
                continue
            try:
                asset = symbol_lookup(order.symbol)
            except SymbolNotFound:
                # Non-equity history (crypto pairs like 'BTC/USD') or symbols
                # that were never ingested. The `orders` property already skips
                # these; without the same guard here a single such fill in the
                # account's history breaks every fill-processing pass in
                # blotter_live.
                log.warning('Skipping transaction for unknown symbol %s'
                            % order.symbol)
                continue
            tx = Transaction(
                asset=asset,
                amount=int(order.filled_qty),
                dt=order.filled_at,
                price=float(order.filled_avg_price),
                order_id=order.client_order_id)
            results[order.client_order_id] = tx
        return results

    def cancel_order(self, zp_order_id):
        try:
            order = self._api.get_order_by_client_order_id(zp_order_id)
            self._api.cancel_order(order.id)
        except Exception as e:
            log.error(e)
            return

    def get_last_traded_dt(self, asset):
        quote = self._api.get_quote(asset.symbol)
        return pd.Timestamp(quote.last_timestamp)

    def get_spot_value(self, assets, field, dt, data_frequency):
        assert(field in (
            'open', 'high', 'low', 'close', 'volume', 'price', 'last_traded'))
        assets_is_scalar = not isinstance(assets, (list, set, tuple))
        if assets_is_scalar:
            symbols = [assets.symbol]
            self.subscribe_to_market_data(assets)
        else:
            symbols = [asset.symbol for asset in assets]
            for asset in assets:
                self.subscribe_to_market_data(asset)
        if field in ('price', 'last_traded'):
            try:
                last_trade = self._api.get_latest_trade(symbols[0])
                return last_trade.price
            except:
                return np.nan

        # bars = self._api.get_barset(symbols, '1Min', limit=1).df
        bars = self._api.get_bars(symbols, TimeFrame(1, TimeFrameUnit.Minute), limit=1).df
        if bars.empty:
            return np.nan
        if not np.isnan(bars[assets.symbol][field]).all():
            return float(bars[assets.symbol][field])
        # if assets_is_scalar:
        #     if len(bars_list) == 0:
        #         return np.nan
        #     return bars_list[0].bars[-1]._raw[field]
        # bars_map = {a.symbol: a for a in bars_list}
        # return [
        #     bars_map[symbol].bars[-1]._raw[field]
        #     for symbol in symbols
        # ]

    def _get_positions_from_broker(self):
        """
        get the positions from the broker and update zipline objects ( the ledger )
        should be used once at startup and once every time we want to refresh the positions array
        """
        cur_pos_in_tracker = self.metrics_tracker.positions
        positions = self._api.list_positions()
        for ap_position in positions:
            # ap_position = positions[symbol]
            try:
                z_position = zp.Position(zp.InnerPosition(symbol_lookup(ap_position.symbol)))
                editable_position = zp.MutableView(z_position)
            except SymbolNotFound:
                # The symbol might not have been ingested to the db therefore
                # it needs to be skipped.
                log.warning('Wanted to subscribe to %s, but this asset is probably not ingested' % ap_position.symbol)
                continue
            if int(ap_position.qty) == 0:
                continue
            editable_position._underlying_position.amount = int(ap_position.qty)
            editable_position._underlying_position.cost_basis = float(ap_position.avg_entry_price)
            editable_position._underlying_position.last_sale_price = float(ap_position.current_price)
            editable_position._underlying_position.last_sale_date = self._api.get_latest_trade(ap_position.symbol).timestamp
            
            self.metrics_tracker.update_position(z_position.asset,
                                                 amount=z_position.amount,
                                                 last_sale_price=z_position.last_sale_price,
                                                 last_sale_date=z_position.last_sale_date,
                                                 cost_basis=z_position.cost_basis)

        # now let's sync the positions in the internal zipline objects
        position_names = [p.symbol for p in positions]
        assets_to_update = []  # separate list to not change list while iterating
        for asset in cur_pos_in_tracker:
            if asset.symbol not in position_names:
                assets_to_update.append(asset)
        for asset in assets_to_update:
            # deleting object from the metrics_tracker as its not in the portfolio
            self.metrics_tracker.update_position(asset,
                                                 amount=0)
        # for some reason, the metrics tracker has self.positions AND self.portfolio.positions. let's make sure
        # these objects are consistent
        # _ledger._portfolio is a protocol.Portfolio, so it has to be mutated
        # through MutableView like everything else in this file.
        zp.MutableView(self.metrics_tracker._ledger._portfolio).positions = \
            self.metrics_tracker.positions

    def get_realtime_bars(self, assets, data_frequency):
        # TODO: cache the result. The caller
        # (DataPortalLive#get_history_window) makes use of only one
        # column at a time.
        assets_is_scalar = not isinstance(assets, (list, set, tuple, pd.Index))
        is_daily = 'd' in data_frequency  # 'daily' or '1d'
        if assets_is_scalar:
            symbols = [assets.symbol]
            self.subscribe_to_market_data(assets)
        else:
            symbols = [asset.symbol for asset in assets]
            for asset in assets:
                self.subscribe_to_market_data(asset)
        timeframe = TimeFrame(1, TimeFrameUnit.Day) if is_daily else TimeFrame(1, TimeFrameUnit.Minute)
        # df = self._api.get_barset(symbols, timeframe, limit=500).df
        # Alpaca's limit counts rows across ALL requested symbols, not per
        # symbol. A flat limit=500 for a 3-name request came back with 500 bars
        # of the first symbol and nothing for the others, so a multi-asset
        # history window silently lost every asset but one.
        df = self._api.get_bars(symbols, timeframe,
                                limit=500 * len(symbols)).df
        if df.empty:
            return df

        if not is_daily:
            # between_time() reads the index's own clock. Alpaca returns UTC, so
            # filtering on "09:30"-"16:00" directly selected 05:30-12:00 New
            # York -- keeping pre-market and throwing away the afternoon. Convert
            # first, then filter, then convert back so callers still see UTC.
            tz = df.index.tz
            df = df.tz_convert(NY).between_time("09:30", "16:00").tz_convert(tz)

        # Both consumers want asset on column level 0 and OHLCV on level 1:
        # DataPortalLive.get_history_window swaplevel()s to pick a field, and
        # LiveTradingAlgorithm.on_exit indexes with the Equity itself. The raw
        # frame is long-form with a 'symbol' column, so neither worked -- history
        # raised on swaplevel and the realtime-bar dump raised KeyError(Equity).
        if 'symbol' in df.columns:
            df = df.pivot(columns='symbol').swaplevel(axis=1)

        by_symbol = {a.symbol: a for a in ([assets] if assets_is_scalar
                                           else assets)}
        if isinstance(df.columns, pd.MultiIndex):
            # key level 0 by the Asset, and keep the caller's ordering
            df.columns = df.columns.set_levels(
                [by_symbol.get(s, s) for s in df.columns.levels[0]], level=0)
            present = [a for a in by_symbol.values() if a in df.columns.levels[0]]
            df = df.reindex(columns=present, level=0)
        else:
            # single symbol comes back flat; give it the same two-level shape
            asset = list(by_symbol.values())[0]
            df.columns = pd.MultiIndex.from_product([[asset], df.columns])
        return df

"""Real-venue adapter through ccxt. Two modes, both behind explicit gates:

* PREVIEW - real credentials, but every order goes to the venue's order-preview endpoint.
  Nothing is ever placed. This is the "testnet first" stage for Coinbase Advanced, which has
  no ccxt sandbox (verified 2026-11-10: `set_sandbox_mode` is unsupported on `ccxt.coinbase`).
* LIVE    - real orders. Constructing this adapter in live mode fails unless the live gate in
  `config.live_trading_approved` holds (mode=live + TRADEBOT_LIVE_ACK phrase + keys).

STATUS: implemented and unit-tested against a fake client only. It has NOT been exercised
against a real account, because no credentials exist yet. Before any live approval, run
`tradebot preflight` in preview mode and confirm what the venue returns for balances,
positions and previews for this account. Two things in particular are unverified:
  1. how the EU futures account reports equity and positions through `fetch_balance` /
     `fetch_positions`;
  2. exchange-native protective stops. Until that is verified the stop is enforced by the
     software risk tick only, so a host outage leaves an open position without a stop.
     That is a LIVE BLOCKER recorded in PROGRESS.md.

Keys are read from the environment, never logged, never stored.
"""
from __future__ import annotations

import os
import time

from ..config import Settings, live_trading_approved
from ..data.feed import make_ccxt
from ..types import (AccountState, Fill, Instrument, Mode, OrderIntent, OrderResult, OrderStatus, Position, Side)

DAY_MS = 86_400_000


class LiveTradingNotApproved(RuntimeError):
    pass


class CcxtExchange:
    name = "ccxt"
    simulated = False

    def __init__(self, settings: Settings, instruments: dict[str, Instrument], recorder,
                 env: dict[str, str] | None = None, client=None, poll_s: float = 0.5, poll_attempts: int = 10):
        env = os.environ if env is None else env
        if settings.mode is Mode.PAPER:
            raise ValueError("paper mode uses PaperExchange, not the ccxt adapter")
        if settings.mode is Mode.LIVE and not live_trading_approved(settings, env):
            raise LiveTradingNotApproved(
                "live trading is not approved: need mode=live, TRADEBOT_LIVE_ACK set to the approval phrase, "
                "and exchange keys in the environment")
        key, secret = env.get("EXCHANGE_API_KEY", ""), env.get("EXCHANGE_API_SECRET", "")
        if client is None and not (key and secret):
            raise LiveTradingNotApproved("EXCHANGE_API_KEY / EXCHANGE_API_SECRET are not set")
        self.preview_only = settings.mode is not Mode.LIVE
        self.settings = settings
        self._instruments = instruments
        self.recorder = recorder
        self.ex = client if client is not None else make_ccxt(settings.exchange.id, {"apiKey": key, "secret": secret})
        self.quote = settings.exchange.quote
        self.poll_s, self.poll_attempts = poll_s, poll_attempts
        self.last_closed = None
        self.previews: list[dict] = []

    # ---- helpers ------------------------------------------------------------
    def _meta(self) -> dict:
        return dict(self.recorder.get("live_positions", {}))

    def _save_meta(self, meta: dict) -> None:
        self.recorder.set("live_positions", meta)

    def instruments(self) -> dict[str, Instrument]:
        return self._instruments

    def _venue_positions(self) -> dict[str, dict]:
        out = {}
        for p in self.ex.fetch_positions(list(self._instruments)) or []:
            contracts = float(p.get("contracts") or 0.0)
            if contracts == 0:
                continue
            sign = -1.0 if (p.get("side") == "short") else 1.0
            out[p["symbol"]] = {"qty": sign * contracts, "entry": float(p.get("entryPrice") or 0.0),
                                "mark": float(p.get("markPrice") or p.get("entryPrice") or 0.0),
                                "upnl": float(p.get("unrealizedPnl") or 0.0)}
        return out

    def open_position_qty(self, symbol: str) -> float:
        return self._venue_positions().get(symbol, {}).get("qty", 0.0)

    def account(self, now_ms: int) -> AccountState:
        bal = self.ex.fetch_balance()
        cash = float((bal.get("total") or {}).get(self.quote) or 0.0)
        venue = self._venue_positions()
        meta = self._meta()
        positions: dict[str, Position] = {}
        gross, upnl = 0.0, 0.0
        for sym, v in venue.items():
            m = meta.get(sym, {})
            # A position with no local metadata has no known stop: stop_price=entry marks it,
            # and reconciliation (venue vs local) blocks new risk until a human looks.
            positions[sym] = Position(symbol=sym, qty=v["qty"], entry_price=v["entry"],
                                      stop_price=float(m.get("stop_price", v["entry"])),
                                      take_profit=m.get("take_profit"), opened_ts=int(m.get("opened_ts", now_ms)),
                                      setup_id=m.get("setup_id", ""), entry_bar_ts=int(m.get("entry_bar_ts", now_ms)),
                                      decision_id=m.get("decision_id"))
            inst = self._instruments.get(sym)
            if inst is not None:
                gross += inst.notional(v["qty"], v["mark"])
            upnl += v["upnl"]
        equity = cash + upnl
        peak = max(float(self.recorder.get("live_peak_equity", equity)), equity)
        self.recorder.set("live_peak_equity", peak)
        day = now_ms // DAY_MS
        ds = self.recorder.get("live_day_start", None)
        if not ds or ds.get("day") != day:
            ds = {"day": day, "equity": equity}
            self.recorder.set("live_day_start", ds)
        return AccountState(ts=now_ms, cash=cash, equity=equity, peak_equity=peak, day_start_equity=float(ds["equity"]),
                            positions=positions, gross_notional=gross)

    # ---- orders ---------------------------------------------------------------
    def submit(self, intent: OrderIntent) -> OrderResult:
        side = intent.side.value
        if self.preview_only:
            # The venue validates the order (permissions, instrument, margin) and places nothing.
            resp = self.ex.create_order(intent.symbol, "market", side, intent.qty, None, {"preview": True})
            self.previews.append({"client_id": intent.client_id, "symbol": intent.symbol, "side": side,
                                  "qty": intent.qty, "ok": True})
            return OrderResult(OrderStatus.REJECTED, intent.client_id,
                               message="preview only: validated by the venue, not placed" + (" " + str(resp.get("status"))
                                                                                             if isinstance(resp, dict) and resp.get("status") else ""))
        if intent.reduce_only:
            order = self.ex.close_position(intent.symbol, None, {"clientOrderId": intent.client_id, "size": intent.qty})
        else:
            # snake_case on purpose: ccxt.coinbase drops `clientOrderId` but passes `client_order_id` through
            order = self.ex.create_order(intent.symbol, "market", side, intent.qty, None,
                                         {"client_order_id": intent.client_id})
        return self._confirm(intent, order)

    def _confirm(self, intent: OrderIntent, order: dict) -> OrderResult:
        oid = order.get("id")
        last = order
        for _ in range(self.poll_attempts):
            if last.get("status") in ("closed", "canceled", "rejected") and (last.get("filled") is not None):
                break
            time.sleep(self.poll_s)
            try:
                last = self.ex.fetch_order(oid, intent.symbol)
            except Exception:
                continue
        filled = float(last.get("filled") or 0.0)
        if last.get("status") in ("canceled", "rejected") and filled == 0:
            return OrderResult(OrderStatus.REJECTED, intent.client_id, message=str(last.get("status")))
        if filled <= 0 or not last.get("average"):
            return OrderResult(OrderStatus.UNKNOWN, intent.client_id, message="no fill confirmation yet")
        fee = float(((last.get("fee") or {}).get("cost")) or 0.0)
        fill = Fill(client_id=intent.client_id, symbol=intent.symbol, side=intent.side, qty=filled,
                    price=float(last["average"]), fee=fee, ts=int(last.get("timestamp") or intent.ts),
                    reduce_only=intent.reduce_only, order_id=str(oid))
        meta = self._meta()
        if intent.reduce_only:
            meta.pop(intent.symbol, None)
        else:
            meta[intent.symbol] = {"stop_price": intent.stop_price, "take_profit": intent.take_profit,
                                   "setup_id": intent.setup_id, "entry_bar_ts": intent.ts, "opened_ts": fill.ts,
                                   "decision_id": intent.decision_id}
        self._save_meta(meta)
        status = OrderStatus.FILLED if filled >= intent.qty - 1e-9 else OrderStatus.PARTIAL
        return OrderResult(status, intent.client_id, (fill,))

    def fetch_order(self, client_id: str) -> OrderResult | None:
        """Find an order by OUR client id among recent orders. None = the venue never saw it."""
        since = int(time.time() * 1000) - DAY_MS
        seen = []
        for fn in ("fetch_open_orders", "fetch_closed_orders"):
            try:
                seen.extend(getattr(self.ex, fn)(None, since) or [])
            except Exception:
                raise
        for o in seen:
            if o.get("clientOrderId") == client_id:
                filled = float(o.get("filled") or 0.0)
                if filled > 0 and o.get("average"):
                    side = Side.BUY if o.get("side") == "buy" else Side.SELL
                    f = Fill(client_id, o["symbol"], side, filled, float(o["average"]),
                             float(((o.get("fee") or {}).get("cost")) or 0.0), int(o.get("timestamp") or 0),
                             order_id=str(o.get("id")))
                    return OrderResult(OrderStatus.FILLED, client_id, (f,))
                if o.get("status") in ("canceled", "rejected"):
                    return OrderResult(OrderStatus.REJECTED, client_id, message=str(o.get("status")))
                return OrderResult(OrderStatus.UNKNOWN, client_id, message="order is still open")
        return None

    def cancel_all(self) -> None:
        for o in self.ex.fetch_open_orders() or []:
            self.ex.cancel_order(o["id"], o.get("symbol"))

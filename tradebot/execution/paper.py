"""Paper exchange: simulated fills with fees, slippage, market impact and injected failures.

It is the default venue, the backtest venue, and the fallback whenever a real exchange is
unreachable. It never touches the network. Fills are always taker fills at the quote the
driver supplies:

* live paper mode: the current best bid/ask from the public order book;
* backtest: the NEXT bar's open, set by the backtest driver, so a decision made on bar k's
  close can never be filled at a price from inside bar k.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..config import FeeModel
from ..types import AccountState, Fill, Instrument, OrderIntent, OrderResult, OrderStatus, Side
from .ledger import ClosedTrade, Ledger


@dataclass
class Quote:
    bid: float
    ask: float
    ts: int
    bar_dollar_volume: float | None = None


class PaperExchange:
    name = "paper"
    simulated = True     # accepts trigger-price fills for protective orders

    def __init__(self, cash: float, ts: int, instruments: dict[str, Instrument], fees: FeeModel):
        self._instruments = instruments
        self.fees = fees
        self.ledger = Ledger(cash, ts, instruments)
        self.quotes: dict[str, Quote] = {}
        self._orders: dict[str, OrderResult] = {}
        self._fail: list[str] = []          # queued failure modes for the next submits
        self.last_closed: ClosedTrade | None = None
        self.submit_count = 0

    # ---- driver hooks --------------------------------------------------------
    def set_quote(self, symbol: str, bid: float, ask: float, ts: int, bar_dollar_volume: float | None = None) -> None:
        self.quotes[symbol] = Quote(bid, ask, ts, bar_dollar_volume)

    def mark(self, symbol: str, price: float) -> None:
        self.ledger.mark(symbol, price)

    def inject_failure(self, *modes: str) -> None:
        """Queue failures: 'reject', 'timeout_filled', 'timeout_unfilled', 'partial'."""
        self._fail.extend(modes)

    # ---- Exchange protocol ---------------------------------------------------
    def instruments(self) -> dict[str, Instrument]:
        return self._instruments

    def account(self, now_ms: int) -> AccountState:
        return self.ledger.state(now_ms)

    def open_position_qty(self, symbol: str) -> float:
        p = self.ledger.positions.get(symbol)
        return p.qty if p else 0.0

    def fetch_order(self, client_id: str) -> OrderResult | None:
        return self._orders.get(client_id)

    def cancel_all(self) -> None:
        return None  # market orders only: nothing rests on the book

    def _slip_frac(self, notional: float, q: Quote) -> float:
        impact = 0.0
        if q.bar_dollar_volume and q.bar_dollar_volume > 0:
            impact = self.fees.impact_coeff * math.sqrt(notional / q.bar_dollar_volume)
        return self.fees.slippage_bps / 1e4 + impact

    def submit(self, intent: OrderIntent, at_price: float | None = None, passive: bool = False) -> OrderResult:
        """`at_price` fills a protective order at its trigger level (stop / target).
        `passive=True` is a resting target: no slippage is added."""
        self.submit_count += 1
        prior = self._orders.get(intent.client_id)
        if prior is not None:
            return prior                                  # idempotent on client id
        mode = self._fail.pop(0) if self._fail else None
        if mode == "reject":
            res = OrderResult(OrderStatus.REJECTED, intent.client_id, message="injected reject")
            self._orders[intent.client_id] = res
            return res
        if mode == "timeout_unfilled":
            return OrderResult(OrderStatus.UNKNOWN, intent.client_id, message="injected timeout")

        inst = self._instruments[intent.symbol]
        q = self.quotes.get(intent.symbol)
        if at_price is None and (q is None or q.bid <= 0 or q.ask <= 0):
            res = OrderResult(OrderStatus.REJECTED, intent.client_id, message="no quote")
            self._orders[intent.client_id] = res
            return res

        qty = intent.qty
        if intent.reduce_only:
            held = abs(self.open_position_qty(intent.symbol))
            qty = min(qty, held)
            if qty <= 0:
                res = OrderResult(OrderStatus.REJECTED, intent.client_id, message="nothing to reduce")
                self._orders[intent.client_id] = res
                return res
        if mode == "partial":
            qty = max(inst.qty_step, math.floor(qty / 2 / inst.qty_step) * inst.qty_step)

        base = at_price if at_price is not None else (q.ask if intent.side is Side.BUY else q.bid)
        notional = inst.notional(qty, base)
        slip = 0.0 if passive else self._slip_frac(notional, q or Quote(base, base, intent.ts))
        price = base * (1 + slip) if intent.side is Side.BUY else base * (1 - slip)
        fee = inst.notional(qty, price) * self.fees.taker
        fill = Fill(client_id=intent.client_id, symbol=intent.symbol, side=intent.side, qty=qty, price=price,
                    fee=fee, ts=intent.ts, reduce_only=intent.reduce_only, order_id=f"paper-{self.submit_count}")
        self.last_closed = self.ledger.apply_fill(fill, intent)
        status = OrderStatus.PARTIAL if qty < intent.qty - 1e-12 else OrderStatus.FILLED
        res = OrderResult(status, intent.client_id, (fill,))
        self._orders[intent.client_id] = res
        if mode == "timeout_filled":
            # the order went through but the caller never heard back
            return OrderResult(OrderStatus.UNKNOWN, intent.client_id, message="injected timeout after fill")
        return res

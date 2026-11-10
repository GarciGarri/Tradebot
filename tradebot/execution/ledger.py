"""Account bookkeeping from fills, funding and marks.

Futures-style accounting: cash moves only on realised PnL, fees and funding. Equity is cash
plus unrealised PnL at the last marks. Peak equity and the UTC day's starting equity are
tracked here because the drawdown kill switch and the daily loss limit are computed from
them, in code.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..types import AccountState, Fill, Instrument, OrderIntent, Position, Side

DAY_MS = 86_400_000


@dataclass(frozen=True)
class ClosedTrade:
    symbol: str
    setup_id: str
    side: Side
    qty: float
    entry_ts: int
    entry_price: float
    exit_ts: int
    exit_price: float
    pnl: float               # net of fees and funding
    gross_pnl: float
    fees: float
    funding: float
    reason: str
    risk_at_entry: float     # $ at risk to the stop, for R multiples
    decision_id: int | None = None

    @property
    def r_multiple(self) -> float:
        return self.pnl / self.risk_at_entry if self.risk_at_entry > 0 else 0.0


class Ledger:
    def __init__(self, cash: float, ts: int, instruments: dict[str, Instrument]):
        self.cash = float(cash)
        self.instruments = instruments
        self.positions: dict[str, Position] = {}
        self.marks: dict[str, float] = {}
        self.peak_equity = float(cash)
        self.day = ts // DAY_MS
        self.day_start_equity = float(cash)
        self.fees_total = 0.0
        self.funding_total = 0.0
        self.closed: list[ClosedTrade] = []
        self._risk_at_entry: dict[str, float] = {}

    # ---- marks & equity ------------------------------------------------------
    def mark(self, symbol: str, price: float) -> None:
        if price > 0:
            self.marks[symbol] = price

    def unrealized(self) -> float:
        total = 0.0
        for sym, pos in self.positions.items():
            m = self.marks.get(sym, pos.entry_price)
            total += pos.qty * self.instruments[sym].contract_size * (m - pos.entry_price)
        return total

    def gross_notional(self) -> float:
        return sum(self.instruments[s].notional(p.qty, self.marks.get(s, p.entry_price))
                   for s, p in self.positions.items())

    def equity(self) -> float:
        return self.cash + self.unrealized()

    def state(self, ts: int) -> AccountState:
        day = ts // DAY_MS
        eq = self.equity()
        if day != self.day:                 # new UTC day: reset the daily loss reference
            self.day = day
            self.day_start_equity = eq
        if eq > self.peak_equity:
            self.peak_equity = eq
        return AccountState(ts=ts, cash=self.cash, equity=eq, peak_equity=self.peak_equity,
                            day_start_equity=self.day_start_equity, positions=dict(self.positions),
                            gross_notional=self.gross_notional())

    # ---- fills ----------------------------------------------------------------
    def apply_fill(self, fill: Fill, intent: OrderIntent) -> ClosedTrade | None:
        inst = self.instruments[fill.symbol]
        self.cash -= fill.fee
        self.fees_total += fill.fee
        signed = fill.qty * fill.side.sign
        pos = self.positions.get(fill.symbol)
        self.mark(fill.symbol, fill.price)

        if pos is None:
            if fill.reduce_only:
                return None            # nothing to reduce; ignore rather than open a position
            stop = intent.stop_price if intent.stop_price is not None else fill.price
            self.positions[fill.symbol] = Position(
                symbol=fill.symbol, qty=signed, entry_price=fill.price, stop_price=stop,
                take_profit=intent.take_profit, opened_ts=fill.ts, setup_id=intent.setup_id,
                entry_bar_ts=intent.ts, fees_paid=fill.fee, decision_id=intent.decision_id)
            self._risk_at_entry[fill.symbol] = abs(signed) * inst.contract_size * abs(fill.price - stop)
            return None

        if pos.qty * signed > 0:
            # adding to a position: the risk layer forbids it, so treat it as a hard error
            raise RuntimeError(f"fill would pyramid {fill.symbol}; refusing to book it")

        close_qty = min(abs(signed), abs(pos.qty))
        direction = 1 if pos.qty > 0 else -1
        gross = direction * close_qty * inst.contract_size * (fill.price - pos.entry_price)
        self.cash += gross
        share = close_qty / abs(pos.qty)
        entry_fee_share = pos.fees_paid * share
        funding_share = pos.funding_paid * share
        risk_share = self._risk_at_entry.get(fill.symbol, 0.0) * share
        trade = ClosedTrade(
            symbol=fill.symbol, setup_id=pos.setup_id, side=pos.side, qty=close_qty,
            entry_ts=pos.opened_ts, entry_price=pos.entry_price, exit_ts=fill.ts, exit_price=fill.price,
            pnl=gross - entry_fee_share - fill.fee - funding_share, gross_pnl=gross,
            fees=entry_fee_share + fill.fee, funding=funding_share, reason=intent.reason,
            risk_at_entry=risk_share, decision_id=pos.decision_id)
        self.closed.append(trade)
        remaining = abs(pos.qty) - close_qty
        if remaining <= 1e-12:
            del self.positions[fill.symbol]
            self._risk_at_entry.pop(fill.symbol, None)
        else:
            pos.qty = direction * remaining
            pos.fees_paid -= entry_fee_share
            pos.funding_paid -= funding_share
            self._risk_at_entry[fill.symbol] = self._risk_at_entry.get(fill.symbol, 0.0) - risk_share
        return trade

    # ---- funding ----------------------------------------------------------------
    def apply_funding(self, symbol: str, rate: float, pure_cost: bool = False) -> float:
        """Charge one funding interval. rate>0 means longs pay shorts.

        `pure_cost=True` charges |rate| whichever side is held: the conservative assumption
        used when no real funding history is available.
        """
        pos = self.positions.get(symbol)
        if pos is None or rate == 0:
            return 0.0
        inst = self.instruments[symbol]
        notional = inst.notional(pos.qty, self.marks.get(symbol, pos.entry_price))
        paid = notional * abs(rate) if pure_cost else notional * rate * (1 if pos.qty > 0 else -1)
        self.cash -= paid
        pos.funding_paid += paid
        self.funding_total += paid
        return paid

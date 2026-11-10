"""Shared value types. Everything here is plain data: no I/O, no model calls."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Regime(str, Enum):
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOL = "high_vol"
    CRISIS = "crisis"


class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class RiskState(str, Enum):
    SAFE = "safe"
    NEAR_LIMIT = "near_limit"
    REDUCE = "reduce"


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"

    @property
    def sign(self) -> int:
        return 1 if self is Side.BUY else -1

    @property
    def opposite(self) -> "Side":
        return Side.SELL if self is Side.BUY else Side.BUY


class Mode(str, Enum):
    PAPER = "paper"      # simulated fills on live public data (default)
    PREVIEW = "preview"  # real credentials, exchange order-preview endpoint only, no orders
    LIVE = "live"        # real orders; requires the explicit live gate


@dataclass(frozen=True)
class Bar:
    """One candle. `ts` is the OPEN time in ms; the bar is only knowable at `close_ts`."""

    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    tf_ms: int

    @property
    def close_ts(self) -> int:
        return self.ts + self.tf_ms


@dataclass(frozen=True)
class Book:
    """Top-of-book levels as (price, size) with the time the exchange produced them."""

    ts: int
    bids: tuple[tuple[float, float], ...]
    asks: tuple[tuple[float, float], ...]

    @property
    def best_bid(self) -> float:
        return self.bids[0][0]

    @property
    def best_ask(self) -> float:
        return self.asks[0][0]

    @property
    def mid(self) -> float:
        return (self.best_bid + self.best_ask) / 2.0


@dataclass(frozen=True)
class Instrument:
    """A tradable contract. Quantities are in contracts; one contract = `contract_size` base units."""

    symbol: str            # ccxt unified symbol, e.g. "BTC/USD:USD"
    base: str              # e.g. "BTC"
    exchange_id: str       # venue product id, e.g. "BIP-20DEC30-CDE"
    contract_size: float   # base units per contract
    price_tick: float
    qty_step: float = 1.0
    min_qty: float = 1.0
    max_qty: float = 1e9

    def notional(self, qty: float, price: float) -> float:
        return abs(qty) * self.contract_size * price

    def round_qty_down(self, qty: float) -> float:
        if qty <= 0:
            return 0.0
        steps = int(qty / self.qty_step + 1e-9)
        q = steps * self.qty_step
        return q if q >= self.min_qty else 0.0

    def round_price(self, price: float) -> float:
        if self.price_tick <= 0:
            return price
        return round(round(price / self.price_tick) * self.price_tick, 10)


@dataclass
class Position:
    symbol: str
    qty: float                    # signed contracts: >0 long, <0 short
    entry_price: float
    stop_price: float             # protective stop, always set for an open position
    take_profit: float | None
    opened_ts: int
    setup_id: str
    entry_bar_ts: int = 0
    fees_paid: float = 0.0
    funding_paid: float = 0.0     # >0 means we paid
    decision_id: int | None = None

    @property
    def side(self) -> Side:
        return Side.BUY if self.qty > 0 else Side.SELL


@dataclass(frozen=True)
class OrderIntent:
    """What the code wants to do. Produced by code only; a model never emits one."""

    client_id: str
    symbol: str
    side: Side
    qty: float                      # contracts, positive
    reduce_only: bool
    reason: str
    ts: int
    setup_id: str = ""
    stop_price: float | None = None     # required for every non-reduce-only order
    take_profit: float | None = None
    ref_price: float | None = None      # price the code sized against (for sanity checks)
    decision_id: int | None = None


class OrderStatus(str, Enum):
    FILLED = "filled"
    PARTIAL = "partial"
    REJECTED = "rejected"
    UNKNOWN = "unknown"   # timeout / no confirmation: must be reconciled before any retry


@dataclass(frozen=True)
class Fill:
    client_id: str
    symbol: str
    side: Side
    qty: float
    price: float
    fee: float
    ts: int
    reduce_only: bool = False
    order_id: str = ""


@dataclass(frozen=True)
class OrderResult:
    status: OrderStatus
    client_id: str
    fills: tuple[Fill, ...] = ()
    message: str = ""

    @property
    def filled_qty(self) -> float:
        return sum(f.qty for f in self.fills)


@dataclass
class AccountState:
    """Account numbers computed by code from fills and marks. Never by a model."""

    ts: int
    cash: float                               # realized balance incl. fees and funding
    equity: float                             # cash + unrealized PnL at marks
    peak_equity: float
    day_start_equity: float
    positions: dict[str, Position] = field(default_factory=dict)
    gross_notional: float = 0.0

    @property
    def drawdown(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, 1.0 - self.equity / self.peak_equity)

    @property
    def day_pnl_frac(self) -> float:
        if self.day_start_equity <= 0:
            return 0.0
        return self.equity / self.day_start_equity - 1.0

    @property
    def leverage(self) -> float:
        if self.equity <= 0:
            return float("inf") if self.gross_notional > 0 else 0.0
        return self.gross_notional / self.equity

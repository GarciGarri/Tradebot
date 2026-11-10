"""State engine: turns feeds into one causal numeric snapshot.

Rules enforced here, in code:

* Causality. A snapshot for decision time `now` is built only from bars whose CLOSE time is
  <= now and from feed items stamped <= now. Later data sitting in the store is ignored, and
  a book stamped in the future is treated as a clock problem, not as information.
* Data guards. Missing bars, gaps, stale feeds, crossed or absurdly wide books, exchange
  errors and clock skew all produce a non-OK status. A non-OK snapshot is never shown to the
  REFLEX and never leads to a new order.
* Arithmetic stays in code. Mid, spread, imbalance, volatility, inventory, exposure and
  drawdown are computed here; the model only reads the results.
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

import numpy as np

from ..config import DataGuards, RiskLimits
from ..types import AccountState, Bar, Book, RiskState
from . import features as F
from .tokens import estimate_tokens, render


class CausalityError(RuntimeError):
    """Raised when code tries to use data from after the decision time."""


class SnapshotStatus(str, Enum):
    OK = "ok"
    EXCHANGE_ERROR = "exchange_error"
    CLOCK_SKEW = "clock_skew"
    MISSING_DATA = "missing_data"
    STALE = "stale"
    NO_BOOK = "no_book"
    WIDE_SPREAD = "wide_spread"


@dataclass(frozen=True)
class SetupView:
    """The numeric part of a research setup that the REFLEX is allowed to see."""

    setup_id: str = ""
    bias: int = 0                      # +1 long thesis, -1 short thesis, 0 none
    quality_prior: int = 0             # research conviction 0-3
    invalidation_price: float | None = None
    next_catalyst_ts: int | None = None  # confirmed catalysts only


@dataclass(frozen=True)
class Snapshot:
    ts: int
    symbol: str
    status: SnapshotStatus
    issues: tuple[str, ...]
    fields: dict
    text: str
    tokens_est: int
    last_bar_ts: int = 0
    price: float = float("nan")            # last closed-bar close
    mid: float | None = None
    spread_bps: float | None = None
    atr_pct: float | None = None
    bar_dollar_volume: float | None = None  # mean $ volume of recent bars (liquidity cap)
    code_risk_state: RiskState = RiskState.SAFE
    setup_id: str = ""

    @property
    def ok(self) -> bool:
        return self.status is SnapshotStatus.OK


@dataclass
class _Series:
    ts: list[int] = field(default_factory=list)
    bars: list[Bar] = field(default_factory=list)
    synthetic: set[int] = field(default_factory=set)
    _np: dict | None = None

    def arrays(self) -> dict:
        if self._np is None:
            b = self.bars
            self._np = {
                "ts": np.array(self.ts, dtype=np.int64),
                "o": np.array([x.open for x in b], dtype=float),
                "h": np.array([x.high for x in b], dtype=float),
                "l": np.array([x.low for x in b], dtype=float),
                "c": np.array([x.close for x in b], dtype=float),
                "v": np.array([x.volume for x in b], dtype=float),
            }
        return self._np


class MarketStore:
    """In-memory market data. Everything is timestamped so reads can be made causal."""

    def __init__(self, tf_ms: int):
        self.tf_ms = tf_ms
        self._series: dict[str, _Series] = {}
        self._books: dict[str, Book] = {}
        self._funding: dict[str, list[tuple[int, float]]] = {}
        self._oi: dict[str, list[tuple[int, float]]] = {}
        self._index: dict[str, list[tuple[int, float]]] = {}
        self._errors: dict[str, tuple[int, str]] = {}
        self.revisions = 0      # bars whose values changed after first being stored
        self.version = 0        # bumps on every bar mutation; feature caches key off it

    # ---- writes -------------------------------------------------------------
    def add_bar(self, symbol: str, bar: Bar, synthetic: bool = False) -> None:
        if bar.tf_ms != self.tf_ms:
            raise ValueError(f"bar timeframe {bar.tf_ms} != store timeframe {self.tf_ms}")
        if bar.ts % self.tf_ms != 0:
            raise ValueError(f"bar ts {bar.ts} is not aligned to the timeframe")
        s = self._series.setdefault(symbol, _Series())
        i = bisect.bisect_left(s.ts, bar.ts)
        if i < len(s.ts) and s.ts[i] == bar.ts:
            if s.bars[i] != bar:
                self.revisions += 1
                self.version += 1
                s.bars[i] = bar
                s._np = None
        else:
            s.ts.insert(i, bar.ts)
            s.bars.insert(i, bar)
            s._np = None
            self.version += 1
        if synthetic:
            s.synthetic.add(bar.ts)
        else:
            s.synthetic.discard(bar.ts)

    def add_bars(self, symbol: str, bars: list[Bar]) -> None:
        for b in bars:
            self.add_bar(symbol, b)

    def set_book(self, symbol: str, book: Book) -> None:
        self._books[symbol] = book

    @staticmethod
    def _push(series: dict[str, list[tuple[int, float]]], symbol: str, ts: int, value: float, keep: int = 5000) -> None:
        lst = series.setdefault(symbol, [])
        i = bisect.bisect_left(lst, (ts, -math.inf))
        if i < len(lst) and lst[i][0] == ts:
            lst[i] = (ts, value)
        else:
            lst.insert(i, (ts, value))
        if len(lst) > keep:
            del lst[: len(lst) - keep]

    def set_funding(self, symbol: str, rate_per_hour: float, ts: int) -> None:
        self._push(self._funding, symbol, ts, rate_per_hour)

    def set_open_interest(self, symbol: str, oi: float, ts: int) -> None:
        self._push(self._oi, symbol, ts, oi)

    def set_index(self, symbol: str, price: float, ts: int) -> None:
        self._push(self._index, symbol, ts, price)

    def mark_error(self, symbol: str, message: str, ts: int) -> None:
        self._errors[symbol] = (ts, message)

    def clear_error(self, symbol: str) -> None:
        self._errors.pop(symbol, None)

    # ---- causal reads -------------------------------------------------------
    def n_closed(self, symbol: str, now_ms: int) -> int:
        s = self._series.get(symbol)
        if s is None:
            return 0
        # a bar with open ts T closes at T+tf; closed iff T + tf <= now
        return bisect.bisect_right(s.ts, now_ms - self.tf_ms)

    def window(self, symbol: str, now_ms: int, n: int) -> dict | None:
        s = self._series.get(symbol)
        if s is None:
            return None
        end = self.n_closed(symbol, now_ms)
        if end == 0:
            return None
        start = max(0, end - n)
        a = s.arrays()
        out = {k: v[start:end] for k, v in a.items()}
        if out["ts"][-1] + self.tf_ms > now_ms:  # belt and braces
            raise CausalityError(f"{symbol}: bar closing after decision time leaked into window")
        out["synthetic"] = sum(1 for t in out["ts"] if int(t) in s.synthetic) if s.synthetic else 0
        return out

    def last_closed_ts(self, symbol: str, now_ms: int) -> int | None:
        s = self._series.get(symbol)
        end = self.n_closed(symbol, now_ms)
        return s.ts[end - 1] if s is not None and end > 0 else None

    def last_bar(self, symbol: str, now_ms: int) -> Bar | None:
        s = self._series.get(symbol)
        end = self.n_closed(symbol, now_ms)
        return s.bars[end - 1] if s is not None and end > 0 else None

    def book(self, symbol: str) -> Book | None:
        return self._books.get(symbol)

    @staticmethod
    def _latest(series: dict[str, list[tuple[int, float]]], symbol: str, now_ms: int) -> tuple[int, float] | None:
        lst = series.get(symbol)
        if not lst:
            return None
        i = bisect.bisect_right(lst, (now_ms, math.inf))
        return lst[i - 1] if i > 0 else None

    def funding(self, symbol: str, now_ms: int) -> tuple[int, float] | None:
        return self._latest(self._funding, symbol, now_ms)

    def open_interest(self, symbol: str, now_ms: int) -> tuple[int, float] | None:
        return self._latest(self._oi, symbol, now_ms)

    def index(self, symbol: str, now_ms: int) -> tuple[int, float] | None:
        return self._latest(self._index, symbol, now_ms)

    def error(self, symbol: str) -> tuple[int, str] | None:
        return self._errors.get(symbol)

    def symbols(self) -> list[str]:
        return list(self._series)


def code_risk_state(account: AccountState, limits: RiskLimits) -> tuple[RiskState, dict[str, float]]:
    """Deterministic risk state. The gate uses the stricter of this and the model's view."""
    usage = {
        "ddu": account.drawdown / limits.kill_switch_drawdown,
        "dayu": max(0.0, -account.day_pnl_frac) / limits.max_daily_loss,
        "levu": (account.leverage / limits.max_leverage) if math.isfinite(account.leverage) else 9.99,
        "posu": len(account.positions) / limits.max_open_positions,
    }
    # Daily loss and leverage are recoverable states (a new day, a closed position), so they
    # can block entries at 70%. Drawdown is not: blocking entries at 70% of the kill level
    # would silently freeze the account below the kill switch for ever. Drawdown therefore
    # only blocks at the `reduce` level; between 70% and 90% sizing is cut instead (see core).
    exposure = max(usage["dayu"], usage["levu"])
    if max(exposure, usage["ddu"]) >= limits.reduce_frac:
        state = RiskState.REDUCE
    elif exposure >= limits.near_limit_frac or usage["posu"] >= 1.0:
        state = RiskState.NEAR_LIMIT
    else:
        state = RiskState.SAFE
    return state, usage


class StateEngine:
    WINDOW = 200  # bars used for features

    def __init__(self, store: MarketStore, guards: DataGuards, limits: RiskLimits,
                 context_symbols: tuple[str, ...] = (), token_budget: int = 400):
        self.store = store
        self.guards = guards
        self.limits = limits
        self.context_symbols = context_symbols
        self.token_budget = token_budget
        self._cache: dict[tuple[str, int], dict] = {}
        self._cache_version = store.version

    # ---- market features from closed bars only (cacheable, account independent) ----
    def _bar_features(self, symbol: str, now_ms: int) -> dict | None:
        if self._cache_version != self.store.version:
            self._cache.clear()
            self._cache_version = self.store.version
        last_ts = self.store.last_closed_ts(symbol, now_ms)
        if last_ts is None:
            return None
        key = (symbol, last_ts)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        w = self.store.window(symbol, now_ms, self.WINDOW)
        if w is None:
            return None
        c, h, l, v = w["c"], w["h"], w["l"], w["v"]
        e20, e50 = F.ema(c, 20), F.ema(c, 50)
        rv24, rv168 = F.realized_vol(c, 24), F.realized_vol(c, 168)
        dollar = c * v
        out = {
            "n": len(c),
            "last_ts": int(w["ts"][-1]),
            "contiguous": bool(np.all(np.diff(w["ts"]) == self.store.tf_ms)) if len(c) > 1 else True,
            "synthetic": int(w["synthetic"]),
            "finite": bool(np.all(np.isfinite(c)) and np.all(c > 0) and np.all(h >= l)),
            "close": float(c[-1]),
            "r1": F.pct_return(c, 1), "r4": F.pct_return(c, 4),
            "r24": F.pct_return(c, 24), "r72": F.pct_return(c, 72),
            "rv": rv24,
            "rvr": (rv24 / rv168) if rv24 is not None and rv168 not in (None, 0.0) else None,
            "atr": F.atr_pct(h, l, c, 14),
            "ema": ((e20 / e50 - 1.0) * 100.0) if e20 is not None and e50 else None,
            "px50": ((float(c[-1]) / e50 - 1.0) * 100.0) if e50 else None,
            "rng": F.range_position(h, l, c, 48),
            "hi72": F.below_high_pct(h, c, 72),
            "vz": F.zscore_last(v, 48),
            "dollar_vol": float(np.mean(dollar[-24:])) if len(c) >= 24 else None,
        }
        if len(self._cache) > 2_000_000:
            self._cache.clear()
        self._cache[key] = out
        return out

    def snapshot(self, symbol: str, now_ms: int, account: AccountState, setup: SetupView | None = None,
                 exchange_time_ms: int | None = None, contract_size: float = 1.0) -> Snapshot:
        g = self.guards
        tf = self.store.tf_ms
        setup = setup or SetupView()
        issues: list[str] = []
        status = SnapshotStatus.OK

        def fail(st: SnapshotStatus, msg: str) -> None:
            nonlocal status
            issues.append(msg)
            if status is SnapshotStatus.OK:
                status = st

        err = self.store.error(symbol)
        if err is not None:
            fail(SnapshotStatus.EXCHANGE_ERROR, f"exchange error: {err[1]}")
        if exchange_time_ms is not None and abs(now_ms - exchange_time_ms) > g.max_clock_skew_ms:
            fail(SnapshotStatus.CLOCK_SKEW, f"clock skew {now_ms - exchange_time_ms} ms")

        bf = self._bar_features(symbol, now_ms)
        expected_last = (now_ms // tf) * tf - tf
        if bf is None or bf["n"] < min(g.min_bars, self.WINDOW):
            fail(SnapshotStatus.MISSING_DATA, f"only {0 if bf is None else bf['n']} closed bars")
        else:
            if not bf["finite"]:
                fail(SnapshotStatus.EXCHANGE_ERROR, "non-finite or inconsistent bar values")
            if not bf["contiguous"]:
                fail(SnapshotStatus.MISSING_DATA, "gap in bar history")
            if bf["synthetic"] > 0.10 * bf["n"]:
                fail(SnapshotStatus.MISSING_DATA, f"{bf['synthetic']} synthetic bars: market too thin")
            if bf["last_ts"] < expected_last:
                fail(SnapshotStatus.STALE, f"newest closed bar is {(expected_last - bf['last_ts']) // tf} bar(s) behind")
            elif bf["last_ts"] > expected_last:
                raise CausalityError(f"{symbol}: bar {bf['last_ts']} is newer than decision time {now_ms}")

        book = self.store.book(symbol)
        bk: dict | None = None
        if book is not None:
            if book.ts > now_ms + g.max_clock_skew_ms:
                fail(SnapshotStatus.CLOCK_SKEW, "order book stamped in the future")
            elif not book.bids or not book.asks or book.best_bid >= book.best_ask or book.best_bid <= 0:
                fail(SnapshotStatus.EXCHANGE_ERROR, "empty or crossed order book")
            elif now_ms - book.ts > g.max_book_age_s * 1000:
                if g.require_book:
                    fail(SnapshotStatus.NO_BOOK, f"order book is {(now_ms - book.ts) / 1000:.0f}s old")
            else:
                bk = F.book_features(book)
                if bk["spread_bps"] > g.max_spread_bps:
                    fail(SnapshotStatus.WIDE_SPREAD, f"spread {bk['spread_bps']:.1f} bps")
        elif g.require_book:
            fail(SnapshotStatus.NO_BOOK, "no order book")

        risk_state, usage = code_risk_state(account, self.limits)
        price = bf["close"] if bf else float("nan")
        mid = bk["mid"] if bk else None
        ref = mid if mid is not None else price

        fields: dict = {}
        fields["t"] = datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M")
        fields["sym"] = symbol.split("/")[0]
        fields["mid"] = ref if math.isfinite(ref) else None
        fields["spr"] = bk["spread_bps"] if bk else None
        fields["imb"] = bk["imbalance"] if bk else None
        depth_contracts = (bk["bid_sz"] + bk["ask_sz"]) if bk else None
        fields["dep"] = (depth_contracts * contract_size * ref / 1000.0) if bk else None  # $k, top levels
        for k in ("r1", "r4", "r24", "r72", "rv", "rvr", "atr", "ema", "px50", "rng", "hi72", "vz"):
            fields[k] = bf[k] if bf else None

        fnd = self.store.funding(symbol, now_ms)
        fields["fnd"] = fnd[1] * 1e4 if fnd else None            # bps per hour
        oi_now = self.store.open_interest(symbol, now_ms)
        oi_then = self.store.open_interest(symbol, now_ms - 24 * 3_600_000)
        fields["oi24"] = ((oi_now[1] / oi_then[1] - 1.0) * 100.0) if (oi_now and oi_then and oi_then[1] > 0
                                                                      and oi_now[0] != oi_then[0]) else None
        idx = self.store.index(symbol, now_ms)
        fields["bas"] = ((ref / idx[1] - 1.0) * 1e4) if (idx and idx[1] > 0 and math.isfinite(ref)) else None

        ctx = [s for s in self.context_symbols]
        btc = self._bar_features(ctx[0], now_ms) if len(ctx) > 0 else None
        eth = self._bar_features(ctx[1], now_ms) if len(ctx) > 1 else None
        fresh = lambda x: x if (x is not None and x["last_ts"] == expected_last) else None  # noqa: E731
        btc, eth = fresh(btc), fresh(eth)
        fields["b24"] = btc["r24"] if btc else None
        fields["b72"] = btc["r72"] if btc else None
        fields["bema"] = btc["ema"] if btc else None
        fields["e24"] = eth["r24"] if eth else None

        pos = account.positions.get(symbol)
        eq = account.equity if account.equity > 0 else float("nan")
        if pos is not None and math.isfinite(ref):
            fields["pos"] = pos.qty * contract_size * ref / eq
            fields["upl"] = pos.qty * contract_size * (ref - pos.entry_price) / eq * 100.0
        else:
            fields["pos"] = 0.0
            fields["upl"] = 0.0
        fields["ddu"] = usage["ddu"]
        fields["dayu"] = usage["dayu"]
        fields["levu"] = usage["levu"]
        fields["posu"] = usage["posu"]

        fields["bias"] = setup.bias
        fields["q"] = setup.quality_prior
        if setup.invalidation_price and math.isfinite(ref) and ref > 0 and setup.bias != 0:
            # positive = room left before the thesis is invalidated; negative = already breached
            fields["inv"] = setup.bias * (ref / setup.invalidation_price - 1.0) * 100.0
        else:
            fields["inv"] = None
        if setup.next_catalyst_ts is not None and setup.next_catalyst_ts >= now_ms:
            fields["cat"] = (setup.next_catalyst_ts - now_ms) / 86_400_000.0
        else:
            fields["cat"] = None

        text = render(fields)
        tokens = estimate_tokens(text)
        if tokens >= self.token_budget:
            # Never truncate silently: an oversized snapshot is a bug, so refuse to use it.
            fail(SnapshotStatus.EXCHANGE_ERROR, f"snapshot {tokens} tokens exceeds budget {self.token_budget}")

        return Snapshot(
            ts=now_ms, symbol=symbol, status=status, issues=tuple(issues), fields=fields, text=text,
            tokens_est=tokens, last_bar_ts=bf["last_ts"] if bf else 0, price=price, mid=mid,
            spread_bps=bk["spread_bps"] if bk else None, atr_pct=bf["atr"] if bf else None,
            bar_dollar_volume=(bf["dollar_vol"] * contract_size) if (bf and bf["dollar_vol"] is not None) else None,
            code_risk_state=risk_state,
            setup_id=setup.setup_id,
        )

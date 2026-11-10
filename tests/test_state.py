"""State engine: causality, data guards, code-computed account numbers, token budget."""
from __future__ import annotations

import pytest

from tradebot.config import DataGuards, RiskLimits
from tradebot.data.feed import CcxtFeed, fill_gaps
from tradebot.reflex.prompt import SYSTEM_PROMPT
from tradebot.state.engine import MarketStore, SetupView, SnapshotStatus, StateEngine, code_risk_state
from tradebot.state.tokens import estimate_tokens, fmt, render
from tradebot.types import AccountState, Bar, Book, Position, RiskState

from .conftest import ETH, SYM, T0, TF, fresh_book, make_bars

N = 260
NOW = T0 + N * TF


def acct(equity=1000.0, peak=1000.0, day=1000.0, positions=None, gross=0.0) -> AccountState:
    return AccountState(ts=NOW, cash=equity, equity=equity, peak_equity=peak, day_start_equity=day,
                        positions=positions or {}, gross_notional=gross)


def build(guards=None, bars=None, n=N, budget=400):
    store = MarketStore(TF)
    store.add_bars(SYM, bars if bars is not None else make_bars(n))
    store.add_bars(ETH, make_bars(n, price=50.0))
    eng = StateEngine(store, guards or DataGuards(require_book=False), RiskLimits(), (SYM, ETH), budget)
    return store, eng


def test_snapshot_ok_and_under_token_budget():
    store, eng = build()
    store.set_book(SYM, fresh_book(100.0, NOW - 1000))
    store.set_funding(SYM, 0.00001, NOW - 60_000)
    snap = eng.snapshot(SYM, NOW, acct(), SetupView("s", 1, 3, 80.0, NOW + 5 * 86_400_000), contract_size=0.01)
    assert snap.ok, snap.issues
    assert snap.tokens_est < 400
    assert snap.text == render(snap.fields)
    assert snap.fields["sym"] == "BTC" and snap.fields["bias"] == 1 and snap.fields["cat"] == pytest.approx(5.0)
    assert snap.fields["spr"] == pytest.approx(1.0, rel=0.05)


def test_worst_case_snapshot_still_fits_budget():
    """Large prices, negative values and every optional field present must stay under 400 tokens."""
    bars = make_bars(N, price=123456.78, wiggle=0.011)
    store, eng = build(bars=bars)
    px = bars[-1].close
    store.set_book(SYM, fresh_book(px, NOW - 500, spread_bps=12.3456))
    store.set_funding(SYM, -0.000123, NOW - 1)
    store.set_open_interest(SYM, 123456.0, NOW - 25 * TF)
    store.set_open_interest(SYM, 98765.0, NOW - 1)
    store.set_index(SYM, px * 1.00123, NOW - 1)
    pos = Position(SYM, -3.0, px * 1.01, px * 1.05, px * 0.9, NOW - TF, "s")
    a = acct(equity=873.21, peak=1000.0, day=910.0, positions={SYM: pos}, gross=1500.0)
    snap = eng.snapshot(SYM, NOW, a, SetupView("s", -1, 3, px * 1.2, NOW + 12_345_678), contract_size=0.01)
    assert snap.ok, snap.issues
    assert all(v is not None for v in snap.fields.values()), [k for k, v in snap.fields.items() if v is None]
    assert snap.tokens_est < 400


def test_snapshot_ignores_future_bars():
    """Causality: bars that close after the decision time must not change the snapshot."""
    bars = make_bars(N + 50, wiggle=0.004)
    _, eng_past = build(bars=bars[:N])
    store_full, eng_full = build(bars=bars)
    # also make the future wildly different to be sure nothing leaks through
    store_full.add_bar(SYM, Bar(bars[N].ts, 1.0, 9e9, 0.5, 5e8, 1e9, TF))
    a = acct()
    assert eng_past.snapshot(SYM, NOW, a).text == eng_full.snapshot(SYM, NOW, a).text


def test_forming_bar_is_not_used():
    store, eng = build()
    before = eng.snapshot(SYM, NOW, acct()).text
    store.add_bar(SYM, Bar(NOW, 100.0, 500.0, 1.0, 400.0, 1.0, TF))   # opens at NOW, closes at NOW+TF
    assert eng.snapshot(SYM, NOW + TF // 2, acct()).fields["r1"] == eng.snapshot(SYM, NOW, acct()).fields["r1"]
    assert eng.snapshot(SYM, NOW, acct()).text == before


def test_stale_when_newest_closed_bar_is_missing():
    _, eng = build()
    snap = eng.snapshot(SYM, NOW + TF, acct())      # one hour later, no new bar arrived
    assert snap.status is SnapshotStatus.STALE and not snap.ok


def test_missing_data_on_gap_and_on_short_history():
    bars = make_bars(N)
    del bars[-10]
    _, eng = build(bars=bars)
    assert eng.snapshot(SYM, NOW, acct()).status is SnapshotStatus.MISSING_DATA
    _, eng2 = build(n=50)
    assert eng2.snapshot(SYM, T0 + 50 * TF, acct()).status is SnapshotStatus.MISSING_DATA


def test_too_many_synthetic_bars_is_missing_data():
    store = MarketStore(TF)
    for i, b in enumerate(make_bars(N)):
        store.add_bar(SYM, b, synthetic=(i % 4 == 0))
    eng = StateEngine(store, DataGuards(require_book=False), RiskLimits(), ())
    assert eng.snapshot(SYM, NOW, acct()).status is SnapshotStatus.MISSING_DATA


def test_clock_skew_and_future_book():
    store, eng = build()
    assert eng.snapshot(SYM, NOW, acct(), exchange_time_ms=NOW - 5000).status is SnapshotStatus.CLOCK_SKEW
    assert eng.snapshot(SYM, NOW, acct(), exchange_time_ms=NOW - 500).ok
    store.set_book(SYM, fresh_book(100.0, NOW + 60_000))
    assert eng.snapshot(SYM, NOW, acct()).status is SnapshotStatus.CLOCK_SKEW


def test_book_guards():
    g = DataGuards(require_book=True)
    store, eng = build(guards=g)
    assert eng.snapshot(SYM, NOW, acct()).status is SnapshotStatus.NO_BOOK            # no book at all
    store.set_book(SYM, fresh_book(100.0, NOW - 60_000))
    assert eng.snapshot(SYM, NOW, acct()).status is SnapshotStatus.NO_BOOK            # stale book
    store.set_book(SYM, Book(NOW, ((101.0, 1.0),), ((100.0, 1.0),)))
    assert eng.snapshot(SYM, NOW, acct()).status is SnapshotStatus.EXCHANGE_ERROR     # crossed
    store.set_book(SYM, fresh_book(100.0, NOW, spread_bps=80.0))
    assert eng.snapshot(SYM, NOW, acct()).status is SnapshotStatus.WIDE_SPREAD
    store.set_book(SYM, fresh_book(100.0, NOW))
    assert eng.snapshot(SYM, NOW, acct()).ok


def test_exchange_error_flag_and_bad_values():
    store, eng = build()
    store.mark_error(SYM, "503", NOW)
    assert eng.snapshot(SYM, NOW, acct()).status is SnapshotStatus.EXCHANGE_ERROR
    store.clear_error(SYM)
    assert eng.snapshot(SYM, NOW, acct()).ok
    bars = make_bars(N)
    bars[-3] = Bar(bars[-3].ts, 100.0, 100.0, 100.0, float("nan"), 1.0, TF)
    _, eng2 = build(bars=bars)
    assert eng2.snapshot(SYM, NOW, acct()).status is SnapshotStatus.EXCHANGE_ERROR


def test_oversized_snapshot_is_refused_not_truncated():
    _, eng = build(budget=50)
    snap = eng.snapshot(SYM, NOW, acct())
    assert not snap.ok and "exceeds budget" in snap.issues[-1]


def test_account_numbers_are_computed_in_code():
    store, eng = build()
    px = store.last_bar(SYM, NOW).close
    pos = Position(SYM, 10.0, px * 0.9, px * 0.8, None, NOW - TF, "s")
    a = acct(equity=900.0, peak=1000.0, day=950.0, positions={SYM: pos}, gross=10 * 0.01 * px)
    f = eng.snapshot(SYM, NOW, a, contract_size=0.01).fields
    assert f["pos"] == pytest.approx(10 * 0.01 * px / 900.0)
    assert f["upl"] == pytest.approx(10 * 0.01 * (px - px * 0.9) / 900.0 * 100)
    assert f["ddu"] == pytest.approx(0.10 / 0.15)
    assert f["dayu"] == pytest.approx((1 - 900 / 950) / 0.05)
    assert f["posu"] == pytest.approx(1 / 3)


def test_code_risk_state_levels():
    lim = RiskLimits()
    assert code_risk_state(acct(), lim)[0] is RiskState.SAFE
    # daily loss at 70% of its limit: no new entries
    assert code_risk_state(acct(equity=964.0, day=1000.0), lim)[0] is RiskState.NEAR_LIMIT
    # drawdown at 80% of the kill level must NOT freeze the account: it stays tradable (with reduced size)
    assert code_risk_state(acct(equity=880.0, peak=1000.0, day=880.0), lim)[0] is RiskState.SAFE
    # drawdown at 90% of the kill level: no new risk
    assert code_risk_state(acct(equity=865.0, peak=1000.0, day=865.0), lim)[0] is RiskState.REDUCE
    assert code_risk_state(acct(gross=1900.0), lim)[0] is RiskState.REDUCE


def test_every_snapshot_key_is_documented_in_the_prompt():
    _, eng = build()
    for key in eng.snapshot(SYM, NOW, acct()).fields:
        assert f"{key} " in SYSTEM_PROMPT or f" {key}:" in SYSTEM_PROMPT or f" {key}," in SYSTEM_PROMPT, key


def test_token_estimator_and_formatting():
    assert fmt(None) == "na" and fmt(0.0) == "0" and fmt(-0.0001) == "0" and fmt(12345.678) == "12346"
    assert fmt(1.50) == "1.5" and fmt(0.1234) == "0.123" and fmt(True) == "1"
    assert estimate_tokens("r24=-1.37") >= 7
    assert estimate_tokens("") == 0


def test_fill_gaps_marks_synthetic_bars():
    bars = make_bars(10)
    del bars[4:6]
    filled, syn = fill_gaps(bars, TF)
    assert len(filled) == 10 and len(syn) == 2
    assert all(filled[i].volume == 0 and filled[i].close == filled[3].close for i in (4, 5))


class _FakeCcxt:
    markets = {}

    def __init__(self, rows, now):
        self.rows, self.now = rows, now

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
        return self.rows

    def fetch_time(self):
        return self.now

    def load_markets(self):
        return {}


def test_feed_drops_the_forming_bar():
    """Coinbase returns the still-forming candle last; it must never reach the store."""
    rows = [[T0 + i * TF, 1, 2, 0.5, 1.5, 10] for i in range(5)]
    now = T0 + 4 * TF + 60_000                       # one minute into the 5th bar
    feed = CcxtFeed("coinbase", "1h", TF, client=_FakeCcxt(rows, now))
    bars = feed.closed_bars(SYM)
    assert [b.ts for b in bars] == [T0 + i * TF for i in range(4)]
    assert all(b.close_ts <= now for b in bars)

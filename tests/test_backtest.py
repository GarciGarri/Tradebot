"""Backtest engine: no look-ahead, costs on, cutoff guard, walk-forward plumbing."""
from __future__ import annotations

import numpy as np
import pytest

from tradebot.backtest.baseline import technical_setups
from tradebot.backtest.cutoff import CutoffViolation, assert_backtest_allowed
from tradebot.backtest.engine import MarketData, run_backtest
from tradebot.backtest.metrics import buy_and_hold, curve_metrics
from tradebot.backtest.walkforward import Candidate, default_grid, objective, walk_forward
from tradebot.config import FeeModel
from tradebot.reflex.mock import MockReflex
from tradebot.reflex.rules import RuleParams, RuleReflex
from tradebot.types import Bar

from .conftest import ETH, SYM, T0, TF, inst, make_bars, out

INSTS = {SYM: inst(SYM, 0.01), ETH: inst(ETH, 0.1)}


def data_for(settings, limits, bars):
    return MarketData.build(bars, {}, INSTS, settings, limits)


def trending(n=700, drift=0.0008):
    return {SYM: make_bars(n, drift=drift, wiggle=0.001), ETH: make_bars(n, price=50.0, drift=drift, wiggle=0.001)}


def run(settings, limits, policy, bars, reflex, start=T0 + 250 * TF, end=None, cash=100_000.0, **kw):
    end = end or T0 + len(bars[SYM]) * TF
    return run_backtest(data=data_for(settings, limits, bars), settings=settings, limits=limits, policy=policy,
                        setups=technical_setups([SYM]), reflex=reflex, start_ms=start, end_ms=end, cash=cash, **kw)


def test_entry_fills_at_next_bar_open_never_inside_the_signal_bar(settings, limits, policy):
    bars = trending()
    r = run(settings, limits, policy, bars, MockReflex([out("long", 0.9, 3)]), keep_decisions=True, flatten_at_end=False)
    first = next(d for d in r.decisions if d.get("action") == "enter")
    k = (first["ts"] - T0) // TF                       # index of the bar that OPENS at the decision time
    nxt_open = bars[SYM][k].open
    entry = r.trades[0].entry_price if r.trades else None
    fees = settings.exchange.fees
    # reconstruct: ask = next open * (1 + half spread), plus slippage and impact (> 0)
    lo = nxt_open * (1 + 1.5e-4) * (1 + fees.slippage_bps / 1e4)
    assert entry is None or lo <= entry <= lo * 1.003
    assert first["price"] == bars[SYM][k - 1].close    # the decision saw the previous close, nothing later


def test_decisions_do_not_change_when_the_future_changes(settings, limits, policy):
    bars = trending(600)
    cut = 450
    other = {s: bs[:cut] + [Bar(b.ts, b.open * 3, b.high * 3, b.low * 3, b.close * 3, b.volume, TF) for b in bs[cut:]]
             for s, bs in bars.items()}
    end = T0 + cut * TF
    a = run(settings, limits, policy, bars, RuleReflex(), end=end, keep_decisions=True, flatten_at_end=False)
    b = run(settings, limits, policy, other, RuleReflex(), end=end, keep_decisions=True, flatten_at_end=False)
    key = lambda d: (d["ts"], d.get("snapshot"), d.get("direction"), d.get("confidence"), d.get("action"))  # noqa: E731
    assert [key(d) for d in a.decisions[:-1]] == [key(d) for d in b.decisions[:-1]]
    assert len(a.decisions) > 100


def test_backtest_is_deterministic(settings, limits, policy):
    bars = trending()
    a = run(settings, limits, policy, bars, RuleReflex())
    b = run(settings, limits, policy, bars, RuleReflex())
    assert np.array_equal(a.equity, b.equity) and len(a.trades) == len(b.trades) and a.fees == b.fees


def test_costs_are_always_charged(settings, limits, policy):
    bars = {SYM: make_bars(700, drift=0.0, wiggle=0.0), ETH: make_bars(700, price=50.0, drift=0.0, wiggle=0.0)}
    r = run(settings, limits, policy, bars, MockReflex([out("long", 0.9, 3)]))
    assert len(r.trades) >= 1 and r.fees > 0 and r.funding > 0
    assert r.equity[-1] < 100_000.0                     # a flat market can only lose the costs
    assert all(t.pnl < 0 for t in r.trades)
    free = settings.model_copy(update={"exchange": settings.exchange.model_copy(update={"fees": FeeModel(
        taker=0, maker=0, slippage_bps=0, impact_coeff=0, funding_bps_per_hour=0)})})
    r0 = run(free, limits, policy, bars, MockReflex([out("long", 0.9, 3)]))
    assert r0.equity[-1] > r.equity[-1]                 # the only difference is costs


def test_model_backends_cannot_be_backtested_before_their_cutoff(settings, limits, policy):
    from datetime import datetime, timezone
    pre = int(datetime(2026, 3, 1, tzinfo=timezone.utc).timestamp() * 1000)
    post = int(datetime(2026, 7, 1, tzinfo=timezone.utc).timestamp() * 1000)
    assert_backtest_allowed("rules", None, pre)
    assert_backtest_allowed("haiku", "claude-haiku-5-5", post)
    with pytest.raises(CutoffViolation):
        assert_backtest_allowed("haiku", "claude-haiku-5-5", pre)
    with pytest.raises(CutoffViolation):
        assert_backtest_allowed("haiku", "some-unknown-model", post)
    with pytest.raises(CutoffViolation):
        assert_backtest_allowed("jev", None, post)

    class FakeModel(MockReflex):
        name = "haiku"

    with pytest.raises(CutoffViolation):               # T0 is in 2025: before the cutoff
        run(settings, limits, policy, trending(), FakeModel([out()]), reflex_model="claude-haiku-5-5")


def test_metrics():
    ts = np.arange(0, 5) * TF
    m = curve_metrics(ts, np.array([100.0, 110.0, 99.0, 120.0, 108.0]), 8760)
    assert m["total_return"] == pytest.approx(0.08) and m["max_drawdown"] == pytest.approx(0.1)
    bh = buy_and_hold(ts[1:], np.array([100.0, 90.0, 130.0, 120.0]), 100.0, 0.001, 8760)
    assert bh["total_return"] == pytest.approx(120.0 / 100.1 - 1) and bh["max_drawdown"] > 0.09
    assert objective(0.10, 0.05, 20, 8) == pytest.approx(2.0) and objective(0.5, 0.01, 3, 8) == float("-inf")
    assert objective(0.03, 0.001, 20, 8) == pytest.approx(1.0)       # drawdown floor stops tiny samples winning


def test_walk_forward_selects_on_train_and_reports_only_test(settings, limits, policy):
    bars = trending(1400, drift=0.0006)
    data = data_for(settings, limits, bars)
    grid = [Candidate("lo", RuleParams(allow_short=False)), Candidate("ls_wide", RuleParams(), {"stop_atr_mult": 3.5})]
    r = walk_forward(data=data, settings=settings, limits=limits, base_policy=policy, setups=technical_setups([SYM]),
                     grid=grid, start_ms=T0 + 250 * TF, end_ms=T0 + 1400 * TF, train_bars=400, test_bars=250,
                     cash=100_000.0, benchmark_symbol=SYM, min_trades=1)
    assert len(r.folds) == 3
    assert r.ts[0] == r.folds[0].test_start and r.ts[-1] == T0 + 1400 * TF      # OOS only, stitched end to end
    assert all(f.train_start + 400 * TF == f.test_start for f in r.folds)
    assert len(r.ts) == len(r.equity) and np.all(np.diff(r.ts) == TF)
    assert r.benchmark["total_return"] > 0 and set(f.chosen for f in r.folds) <= {"lo", "ls_wide", "flat"}
    replay = walk_forward(data=data, settings=settings, limits=limits, base_policy=policy,
                          setups=technical_setups([SYM]), grid=grid, start_ms=T0 + 250 * TF, end_ms=T0 + 1400 * TF,
                          train_bars=400, test_bars=250, cash=100_000.0, benchmark_symbol=SYM, min_trades=1,
                          fixed_choices=[f.chosen for f in r.folds])
    assert np.allclose(replay.equity, r.equity)


def test_losing_rule_in_training_is_not_deployed(settings, limits, policy):
    bars = {SYM: make_bars(900, drift=0.0, wiggle=0.004), ETH: make_bars(900, price=50.0, drift=0.0, wiggle=0.004)}
    data = data_for(settings, limits, bars)
    r = walk_forward(data=data, settings=settings, limits=limits, base_policy=policy, setups=technical_setups([SYM]),
                     grid=default_grid()[:2], start_ms=T0 + 250 * TF, end_ms=T0 + 900 * TF, train_bars=300,
                     test_bars=175, cash=100_000.0, benchmark_symbol=SYM)
    assert all(f.chosen == "flat" for f in r.folds) and r.equity[-1] == 100_000.0 and not r.trades

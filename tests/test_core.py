"""Trading core, end to end on synthetic data: who decides what, and what survives failure."""
from __future__ import annotations

from datetime import date

import pytest

from tradebot.config import RiskLimits
from tradebot.policy.policy import ExitCfg, Policy
from tradebot.reflex.mock import MockReflex
from tradebot.runtime.core import PriceRange
from tradebot.storage.recorder import MemoryRecorder, SqliteRecorder
from tradebot.types import Direction, Side

from .conftest import ETH, SYM, TF, make_setup, out


def last(rig):
    return rig.recorder.decisions[-1]


def test_entry_when_every_gate_passes(rig_factory):
    rig = rig_factory(MockReflex([out("long", 0.9, 3)]))
    rig.step()
    pos = rig.exchange.ledger.positions[SYM]
    acct = rig.exchange.account(rig.now)
    assert pos.qty > 0 and pos.stop_price < pos.entry_price < pos.take_profit
    risk = pos.qty * 0.01 * (pos.entry_price - pos.stop_price)
    assert risk <= rig.limits.risk_per_trade * 100_000 * 1.001
    assert acct.leverage <= rig.limits.max_leverage
    assert last(rig)["action"] == "enter" and last(rig)["gate_allow"] == 1
    assert rig.recorder.orders[-1]["status"] == "filled" and rig.recorder.orders[-1]["decision_id"] == last(rig)["id"]


def test_short_entry_has_stop_above(rig_factory):
    rig = rig_factory(MockReflex([out("short", 0.9, 3)]))
    rig.step()
    pos = rig.exchange.ledger.positions[SYM]
    assert pos.qty < 0 and pos.take_profit < pos.entry_price < pos.stop_price


@pytest.mark.parametrize("o,reason", [
    (out(conf=0.80), "direction_confidence<=0.80"), (out(quality=1), "setup_quality<2"),
    (out(risk="near_limit"), "model_risk_state_near_limit"), (out(direction="neutral"), "direction_neutral"),
    (out(toxic=True), "toxic_flow")])
def test_gate_blocks_are_recorded_as_misses(rig_factory, o, reason):
    rig = rig_factory(MockReflex([o]))
    rig.step()
    assert not rig.exchange.ledger.positions
    assert reason in last(rig)["gate_reasons"] and last(rig)["action"] == "none"
    assert last(rig)["confidence"] == o.direction_confidence      # misses keep their inputs for calibration


def test_model_cannot_override_code_risk_state(rig_factory):
    """REFLEX insists everything is safe; code sees the daily loss near its limit and refuses."""
    rig = rig_factory(MockReflex([out("long", 0.99, 3, risk="safe")]))
    rig.exchange.ledger.day_start_equity = 100_000.0 / 0.96       # down 4% today = 80% of the 5% limit
    rig.step()
    assert last(rig)["risk_state"] == "safe" and last(rig)["code_risk_state"] == "near_limit"
    assert last(rig)["gate_allow"] == 0 and "code_risk_state_near_limit" in last(rig)["gate_reasons"]
    assert not rig.exchange.ledger.positions


def test_entry_never_puts_the_account_into_reduce_by_itself(rig_factory):
    """Regression: sizing to the hard leverage limit made the bot cut its own fresh position."""
    rig = rig_factory(MockReflex([out("long", 0.9, 3)]), setups={"a": make_setup(SYM, id="a"), "b": make_setup(ETH, id="b")})
    for _ in range(6):
        rig.step()
    acct = rig.exchange.account(rig.now)
    assert acct.leverage <= rig.limits.near_limit_frac * rig.limits.max_leverage * 1.01
    assert not rig.exchange.ledger.closed                          # nothing was churned
    assert "exit_risk_reduce" not in rig.core.counters


def test_slow_reflex_skips_the_candle_but_stops_still_fire(rig_factory):
    """A decision that misses the deadline is dropped; protective exits never wait for it."""
    fast = MockReflex([out("long", 0.9, 3)])
    rig = rig_factory(fast, inline=False, deadline_ms=2000)
    rig.step()
    pos = rig.exchange.ledger.positions[SYM]
    fast.delay_s, rig.runner.deadline_ms = 0.4, 50           # the model becomes too slow
    stop = pos.stop_price
    bar = rig.add_bar(stop * 0.99, low=stop * 0.98)
    rig.add_bar(rig.bars[ETH][-1].close, ETH)
    rig.now += TF
    rig.quote(bar.close)
    rig.core.risk_tick(rig.now, {SYM: PriceRange(bar.open, bar.low, bar.high)})   # no model involved
    assert SYM not in rig.exchange.ledger.positions
    assert rig.exchange.ledger.closed[-1].reason == "stop"
    rig.core.on_candle(rig.now)
    assert last(rig)["status"] == "skip" and last(rig)["skip_reason"] == "deadline"
    assert SYM not in rig.exchange.ledger.positions          # and the skipped candle opened nothing
    rig.runner.close()


def test_failing_or_garbage_reflex_never_trades(rig_factory):
    rig = rig_factory(MockReflex([out()], fail=True))
    rig.step()
    assert last(rig)["skip_reason"] == "error" and not rig.exchange.ledger.positions
    rig2 = rig_factory(MockReflex([{"direction": "long", "size": 1e9}]))
    rig2.step()
    assert last(rig2)["skip_reason"] == "invalid" and not rig2.exchange.ledger.positions


def test_stale_data_skips_without_calling_the_model(rig_factory):
    m = MockReflex([out()])
    rig = rig_factory(m)
    rig.now += 3 * TF                               # the feed stopped: no new bars
    rig.core.on_candle(rig.now)
    assert m.calls == 0 and last(rig)["skip_reason"] == "stale" and not rig.exchange.ledger.positions


def test_stop_and_take_profit(rig_factory):
    rig = rig_factory(MockReflex([out("long", 0.9, 3)]))
    rig.step()
    pos = rig.exchange.ledger.positions[SYM]
    tp = pos.take_profit
    bar = rig.add_bar(tp * 1.001, high=tp * 1.002)
    rig.now += TF
    rig.core.risk_tick(rig.now, {SYM: PriceRange(bar.open, bar.low, bar.high)})
    t = rig.exchange.ledger.closed[-1]
    assert t.reason == "take_profit" and t.exit_price == pytest.approx(tp) and t.pnl > 0


def test_stop_wins_when_both_levels_are_inside_one_bar_and_gaps_fill_at_the_open(rig_factory):
    rig = rig_factory(MockReflex([out("long", 0.9, 3)]))
    rig.step()
    pos = rig.exchange.ledger.positions[SYM]
    rig.core.risk_tick(rig.now + TF, {SYM: PriceRange(pos.entry_price, pos.stop_price * 0.99, pos.take_profit * 1.01)})
    assert rig.exchange.ledger.closed[-1].reason == "stop"
    rig2 = rig_factory(MockReflex([out("long", 0.9, 3)]))
    rig2.step()
    p2 = rig2.exchange.ledger.positions[SYM]
    gap_open = p2.stop_price * 0.95                 # opens far below the stop
    rig2.core.risk_tick(rig2.now + TF, {SYM: PriceRange(gap_open, gap_open * 0.99, gap_open)})
    assert rig2.exchange.ledger.closed[-1].exit_price < p2.stop_price * 0.951   # filled at the gap, not the stop


def test_kill_switch_flattens_latches_and_survives_restart(rig_factory, settings, limits, policy, tmp_path):
    rec = SqliteRecorder(tmp_path / "state.db")
    rig = rig_factory(MockReflex([out("long", 0.9, 3)]), recorder=rec)
    rig.step()
    assert SYM in rig.exchange.ledger.positions
    rig.exchange.ledger.cash -= 16_000.0            # an external loss pushes drawdown past 15%
    rig.core.risk_tick(rig.now, {SYM: PriceRange(rig.price, rig.price, rig.price)})
    assert rig.core.kill and not rig.exchange.ledger.positions
    assert rig.exchange.ledger.closed[-1].reason == "kill_switch"
    assert rig.alerts.sent[-1][0] == "critical" and rig.alerts.sent[-1][1] == "kill_switch"
    rig.step()
    assert not rig.exchange.ledger.positions        # the model still says "long 0.9": nothing opens
    reasons = rig.recorder.query("SELECT gate_reasons FROM decisions ORDER BY id DESC LIMIT 1")[0]["gate_reasons"]
    assert "code_risk_state_reduce" in reasons
    # a restarted process reads the latch back from disk; even with a fresh, healthy account
    # and a passing gate, the hard risk layer refuses the order
    rig2 = rig_factory(MockReflex([out("long", 0.9, 3)]), recorder=SqliteRecorder(tmp_path / "state.db"))
    assert rig2.core.kill
    rig2.step()
    assert not rig2.exchange.ledger.positions
    msg = rig2.recorder.query("SELECT message FROM orders ORDER BY rowid DESC LIMIT 1")[0]["message"]
    assert "kill_switch_active" in msg


def test_daily_loss_limit_flattens_and_blocks_until_next_utc_day(rig_factory):
    rig = rig_factory(MockReflex([out("long", 0.9, 3)]))
    rig.step()
    rig.exchange.ledger.cash -= 5_500.0             # more than 5% of the day's starting equity
    rig.core.risk_tick(rig.now, {SYM: PriceRange(rig.price, rig.price, rig.price)})
    assert not rig.exchange.ledger.positions and not rig.core.kill
    assert rig.exchange.ledger.closed[-1].reason == "daily_loss_limit"
    rig.step()
    assert not rig.exchange.ledger.positions        # same UTC day: blocked
    while (rig.now // 86_400_000) == rig.core.daily_halt_day:
        rig.step()
    assert SYM in rig.exchange.ledger.positions     # new UTC day: trading resumes


def test_crisis_pauses_all_new_risk_and_escalates(rig_factory):
    script = [out("long", 0.9, 3, regime="crisis")] + [out("long", 0.9, 3)]
    rig = rig_factory(MockReflex(script))
    rig.step()
    assert not rig.exchange.ledger.positions and rig.core.paused(ETH, rig.now)
    assert any(k == "escalate:crisis" for _, k, _ in rig.alerts.sent)
    rig.step()
    assert "paused" in last(rig)["gate_reasons"] and not rig.exchange.ledger.positions
    for _ in range(rig.policy.pause.cooldown_bars):
        rig.step()
    assert SYM in rig.exchange.ledger.positions     # the pause expires on its own


def test_low_confidence_on_a_held_position_pauses_that_symbol_once(rig_factory):
    script = [out("long", 0.9, 3), out("long", 0.55, 2), out("long", 0.55, 2), out("long", 0.55, 2)]
    rig = rig_factory(MockReflex(script))
    rig.step()
    rig.step()
    assert rig.core.paused(SYM, rig.now) and not rig.core.paused(ETH, rig.now)
    rig.step()
    assert sum(1 for _, k, _ in rig.alerts.sent if k.startswith("escalate:confidence")) == 1   # no alert storm
    assert SYM in rig.exchange.ledger.positions     # the pause stops new risk; the stop still guards the old


def test_thesis_invalidation_closes_without_the_model(rig_factory):
    setup = make_setup(SYM, bias="long", price_below=99.9)
    m = MockReflex([out("long", 0.9, 3)])
    rig = rig_factory(m, setups={"t": setup})
    rig.step()
    pos = rig.exchange.ledger.positions[SYM]
    pos.stop_price = 1.0                            # take the stop out of the way: only the thesis level acts
    m.fail = True                                   # and the model is down
    rig.step(close=rig.price * 0.990)               # a small drop: below the thesis level, far from any limit
    assert rig.price < 99.9
    assert SYM not in rig.exchange.ledger.positions
    assert rig.exchange.ledger.closed[-1].reason == "thesis_invalidated"


def test_time_stop_and_expired_setup(rig_factory, policy):
    pol = policy.model_copy(update={"exits": ExitCfg(max_hold_bars=3)})
    rig = rig_factory(MockReflex([out("long", 0.9, 3), out("long", 0.9, 2)]), policy=pol)
    for _ in range(5):
        rig.step()
    assert rig.exchange.ledger.closed[0].reason == "time_stop"
    exp = make_setup(SYM, expires=date(2025, 1, 1))
    rig2 = rig_factory(MockReflex([out("long", 0.9, 3)]), setups={"t": exp})
    rig2.step()
    assert not rig2.exchange.ledger.positions and "setup_expired" in last(rig2)["gate_reasons"]


def test_signal_flip_exits_but_never_reverses_in_one_step(rig_factory):
    rig = rig_factory(MockReflex([out("long", 0.9, 3), out("short", 0.9, 3), out("neutral", 0.5, 0)]))
    rig.step()
    rig.step()
    assert SYM not in rig.exchange.ledger.positions and rig.exchange.ledger.closed[-1].reason == "signal_flip"


def test_reflex_output_cannot_set_size_price_or_side_effects(rig_factory):
    """Same judgement, very different account: the size follows the account, not the model."""
    small = rig_factory(MockReflex([out("long", 0.9, 3)]), cash=10_000.0)
    big = rig_factory(MockReflex([out("long", 0.9, 3)]), cash=100_000.0)
    small.step()
    big.step()
    q_small, q_big = small.exchange.ledger.positions[SYM].qty, big.exchange.ledger.positions[SYM].qty
    assert 8 * q_small < q_big <= 10.01 * q_small    # ~10x the account -> ~10x the size (less market impact)


def test_drawdown_caution_halves_risk_instead_of_freezing(rig_factory):
    rig = rig_factory(MockReflex([out("long", 0.9, 3)]))
    rig.exchange.ledger.peak_equity = 100_000.0 / (1 - 0.12)      # 12% drawdown = 80% of the kill level
    rig.step()
    pos = rig.exchange.ledger.positions[SYM]                      # still trades...
    risk = pos.qty * 0.01 * (pos.entry_price - pos.stop_price)
    assert risk <= 0.5 * rig.limits.risk_per_trade * 100_000 * 1.001    # ...at half the risk budget
    deep = rig_factory(MockReflex([out("long", 0.9, 3)]))
    deep.exchange.ledger.peak_equity = 100_000.0 / (1 - 0.14)     # 14% = 93% of the kill level: no new risk
    deep.step()
    assert not deep.exchange.ledger.positions and "code_risk_state_reduce" in last(deep)["gate_reasons"]


def test_two_active_setups_on_one_symbol_are_rejected(rig_factory):
    with pytest.raises(ValueError, match="two active setups"):
        rig_factory(MockReflex([out()]), setups={"a": make_setup(SYM, id="a"), "b": make_setup(SYM, id="b")})


def test_reconcile_mismatch_blocks_new_risk(rig_factory):
    rig = rig_factory(MockReflex([out("long", 0.9, 3)]))
    rig.exchange.open_position_qty = lambda s: 5.0 if s == SYM else 0.0     # venue reports a ghost position
    assert rig.core.reconcile(rig.now)
    assert rig.alerts.sent[-1][1] == "reconcile_mismatch"
    rig.step()
    assert not rig.exchange.ledger.positions
    assert "reconciliation_pending" in rig.recorder.orders[-1]["message"]

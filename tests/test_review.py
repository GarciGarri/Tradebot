"""Nightly review: labelling, Brier, proposals, gates, versioning, no silent live promotion."""
from __future__ import annotations

import random

import pytest

from tradebot.config import FeeModel
from tradebot.policy.policy import Policy, PolicyStore
from tradebot.review.nightly import (MIN_LABELED, label_decisions, measure, propose, render_markdown, run_nightly,
                                     triple_barrier)
from tradebot.types import Bar

from .conftest import SYM, T0, TF, make_bars, make_setup

FEES = FeeModel()


def bar(i, o, h, l, c):
    return Bar(T0 + i * TF, o, h, l, c, 1.0, TF)


def test_triple_barrier_outcomes():
    up = [bar(0, 100, 100.5, 99.8, 100.4), bar(1, 100.4, 102.5, 100.2, 102.2)]
    assert triple_barrier("long", 100.0, 0.01, 2.0, 10, up, 0.0) == (1, 2.0)
    down = [bar(0, 100, 100.2, 98.5, 98.8)]
    win, r = triple_barrier("long", 100.0, 0.01, 2.0, 10, down, 0.0)
    assert win == 0 and r == pytest.approx(-1.0)
    both = [bar(0, 100, 103, 98, 100)]                      # stop and target in one bar: stop first
    assert triple_barrier("long", 100.0, 0.01, 2.0, 10, both, 0.0)[0] == 0
    gap = [bar(0, 97, 97.5, 96, 96.5)]                      # gap through the stop fills at the open
    assert triple_barrier("long", 100.0, 0.01, 2.0, 10, gap, 0.0)[1] == pytest.approx(-3.0)
    short = [bar(0, 100, 100.2, 97.5, 98)]
    assert triple_barrier("short", 100.0, 0.01, 2.0, 10, short, 0.0) == (1, 2.0)
    flat = [bar(i, 100, 100.3, 99.7, 100.2) for i in range(3)]
    assert triple_barrier("long", 100.0, 0.01, 2.0, 3, flat, 0.001)[1] == pytest.approx(0.2 - 0.1)   # time exit, net of costs
    assert triple_barrier("long", 100.0, 0.01, 2.0, 5, flat, 0.0) is None                            # not resolved yet


def synthetic_rows(n, skill=0.0, seed=1):
    """Decisions whose win chance rises with confidence by `skill` (0 = the model knows nothing)."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        conf = rng.uniform(0.5, 1.0)
        win = 1 if rng.random() < 0.40 + skill * (conf - 0.5) else 0
        rows.append({"id": i + 1, "ts": T0 + i * TF, "symbol": SYM, "status": "decision", "direction": "long",
                     "confidence": conf, "setup_quality": rng.choice([1, 2, 3]), "risk_state": "safe",
                     "code_risk_state": "safe", "gate_allow": int(conf > 0.8), "gate_reasons": "" if conf > 0.8 else
                     "direction_confidence<=0.80", "action": "enter" if conf > 0.8 else "none", "latency_ms": 900.0,
                     "stop_frac": 0.01, "take_profit_r": 2.0, "max_hold_bars": 24, "outcome": win,
                     "outcome_r": 1.9 if win else -1.1})
    return rows


def test_label_decisions_uses_only_later_bars_and_skips_unresolved():
    bars = {SYM: make_bars(100, drift=0.002, wiggle=0.0)}
    rows = [{"id": 1, "ts": bars[SYM][10].ts, "symbol": SYM, "status": "decision", "direction": "long",
             "confidence": 0.9, "stop_frac": 0.01, "take_profit_r": 2.0, "max_hold_bars": 24},
            {"id": 2, "ts": bars[SYM][95].ts, "symbol": SYM, "status": "decision", "direction": "short",
             "confidence": 0.9, "stop_frac": 0.5, "take_profit_r": 2.0, "max_hold_bars": 24},     # future unknown
            {"id": 3, "ts": bars[SYM][10].ts, "symbol": SYM, "status": "decision", "direction": "neutral", "confidence": 0.5},
            {"id": 4, "ts": bars[SYM][10].ts, "symbol": SYM, "status": "skip", "skip_reason": "deadline"}]
    out = label_decisions(rows, bars, FEES)
    assert [r["id"] for r in out] == [1] and out[0]["outcome"] == 1 and out[0]["outcome_r"] > 1.5


def test_measure_reports_brier_misses_and_health():
    rows = synthetic_rows(400) + [{"id": 999, "ts": T0, "symbol": SYM, "status": "skip", "skip_reason": "deadline"}]
    trades = [{"pnl": 5.0, "r_multiple": 1.0, "fees": 0.2, "reason": "take_profit"},
              {"pnl": -3.0, "r_multiple": -1.0, "fees": 0.2, "reason": "stop"}]
    s = measure([r for r in rows if r.get("outcome") is not None], rows, trades, Policy(version=1))
    assert s["n_labeled"] == 400 and s["skips"] == {"deadline": 1}
    assert 0.2 < s["brier_raw"] < 0.5 and s["brier_calibrated"] < s["brier_raw"]      # shrinking helps a clueless model
    assert s["misses_by_gate_reason"]["direction_confidence<=0.80"]["n"] > 100
    assert s["fills"]["n_trades"] == 2 and s["fills"]["pnl"] == 2.0 and s["latency_ms"]["p50"] == 900.0
    assert s["risk_state_disagreement"] == 0.0


def test_no_proposal_without_enough_evidence():
    p = propose(Policy(version=1), synthetic_rows(MIN_LABELED - 1))
    assert p.candidate is None and not p.passed and p.gates == {"enough_samples": False}


def test_clueless_model_never_gets_looser_thresholds():
    p = propose(Policy(version=1), synthetic_rows(600, skill=0.0))
    assert p.candidate is not None
    t = p.details["thresholds"]
    if t["candidate"] != t["current"]:
        assert not p.passed          # a change picked on noise must fail out-of-sample gates
    assert 0.65 <= p.candidate.gate.min_direction_confidence <= 0.95 and p.candidate.gate.min_setup_quality >= 2


def test_candidate_calibration_learns_overconfidence():
    p = propose(Policy(version=1), synthetic_rows(800, skill=0.2, seed=5))
    probs = [b.p for b in p.candidate.calibration.bins]
    assert probs == sorted(probs) and max(probs) < 0.6      # true win rates are 40-50%, not 80-100%
    assert p.gates["calibration_not_worse"] and p.gates["inside_tuning_bounds"]


def test_run_nightly_versions_candidate_and_never_touches_live(tmp_path):
    store = PolicyStore(tmp_path / "policies")
    store.save_new(Policy(version=1))
    store.promote(1, "paper", by="bootstrap", reason="init")
    rows = synthetic_rows(500, skill=0.2, seed=9)
    setup = make_setup(SYM, bias="long", price_below=1.0)
    setup.invalidation.conditions.append("monthly revenue below $45M")
    rep = run_nightly(rows=rows, trades=[], bars={SYM: make_bars(10)}, fees=FEES, store=store,
                      reports_dir=tmp_path / "reports", events=[{"key": "escalate:crisis", "level": "critical", "message": "x"}],
                      setups={"s": setup}, run_tests=False, now_ms=T0)
    assert rep["candidate_version"] == 2 and store.versions() == [1, 2]
    assert store.pointer()["live"] is None                              # never promoted to live
    assert rep["promoted_to_paper"] == all(rep["gates"].values())
    assert store.active("paper").version == (2 if rep["promoted_to_paper"] else 1)
    assert any("monthly revenue" in m for m in rep["manual_checks"]) and rep["escalations"]
    md = (tmp_path / "reports" / "nightly" / f"{rep['date']}.md").read_text()
    assert "Brier" in md and "Misses" in md and "Nothing here can promote to live" in md
    if rep["promoted_to_paper"]:
        assert store.rollback("paper", by="human", reason="test") == 1   # the old policy is one step away
    assert render_markdown(rep) == md


def test_failed_gate_keeps_the_current_policy(tmp_path):
    store = PolicyStore(tmp_path / "policies")
    store.save_new(Policy(version=1))
    store.promote(1, "paper", by="bootstrap", reason="init")
    rows = synthetic_rows(500, skill=0.2, seed=9)
    import tradebot.review.nightly as n
    orig = n.regression_tests
    n.regression_tests = lambda timeout_s=600: (False, "1 failed")
    try:
        rep = run_nightly(rows=rows, trades=[], bars={SYM: make_bars(10)}, fees=FEES, store=store,
                          reports_dir=tmp_path / "reports", run_tests=True, now_ms=T0)
    finally:
        n.regression_tests = orig
    assert rep["gates"]["regression_tests"] is False and not rep["promoted_to_paper"]
    assert store.active("paper").version == 1 and store.versions() == [1, 2]    # candidate kept on file, not used

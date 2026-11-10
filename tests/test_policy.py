"""Gate, sizing, calibration and the versioned policy store."""
from __future__ import annotations

import random
from datetime import date

import pytest
from pydantic import ValidationError

from tradebot.config import FeeModel, RiskLimits
from tradebot.policy.calibration import MAX_CALIBRATED_P, Calibrator, brier_score, fit_bins, reliability_table
from tradebot.policy.gate import evaluate_gate
from tradebot.policy.policy import (CalibrationCfg, GateCfg, LivePromotionRefused, Policy, PolicyStore, SizingCfg)
from tradebot.policy.sizing import kelly_full, round_trip_cost_frac, size_position
from tradebot.setups.schema import SetupReflexSpec, SetupSchema
from tradebot.types import Direction, Regime, RiskState

from .conftest import SYM, inst, make_setup, out

NOW = 1_800_000_000_000
POL = Policy(version=1)


def gate(o, setup=None, code_risk=RiskState.SAFE, price=100.0, paused=False, policy=POL):
    return evaluate_gate(o, policy, setup or make_setup(), code_risk, NOW, price, paused)


# ---- gate --------------------------------------------------------------------------------
def test_gate_passes_only_above_all_thresholds():
    assert gate(out(conf=0.81, quality=2)).allow
    assert not gate(out(conf=0.80, quality=2)).allow             # strictly greater than 0.80
    assert not gate(out(conf=0.95, quality=1)).allow
    assert gate(out(conf=0.95, quality=1)).reasons == ("setup_quality<2",)


def test_gate_blocks_on_risk_state_from_model_or_code():
    assert not gate(out(risk="near_limit")).allow
    assert not gate(out(risk="reduce")).allow
    # the model saying "safe" cannot override the code's own reading
    r = gate(out(risk="safe"), code_risk=RiskState.NEAR_LIMIT)
    assert not r.allow and "code_risk_state_near_limit" in r.reasons


def test_gate_blocks_neutral_crisis_toxic_and_paused():
    assert "direction_neutral" in gate(out(direction="neutral")).reasons
    assert "regime_crisis" in gate(out(regime="crisis")).reasons
    assert "toxic_flow" in gate(out(toxic=True)).reasons
    assert gate(out(toxic=None)).allow                           # unknown flow is not a block by itself
    assert "paused" in gate(out(), paused=True).reasons


def test_gate_respects_setup_constraints():
    long_only = make_setup(bias="long", reflex=SetupReflexSpec(allowed_directions=[Direction.LONG]))
    assert "direction_not_allowed_by_setup" in gate(out(direction="short"), long_only).reasons
    strict = make_setup(reflex=SetupReflexSpec(allowed_directions=[Direction.LONG], min_setup_quality=3,
                                               min_direction_confidence=0.9, allowed_regimes=[Regime.TRENDING]))
    assert not gate(out(conf=0.85, quality=3), strict).allow     # a setup may be stricter than the policy
    assert not gate(out(conf=0.95, quality=2), strict).allow
    assert "regime_not_allowed_by_setup" in gate(out(conf=0.95, regime="high_vol"), strict).reasons
    assert gate(out(conf=0.95, quality=3), strict).allow
    loose = make_setup(reflex=SetupReflexSpec(allowed_directions=[Direction.LONG], min_setup_quality=0,
                                              min_direction_confidence=0.5))
    assert not gate(out(conf=0.7, quality=1), loose).allow       # ...but never looser


def test_gate_blocks_inactive_expired_and_invalidated_setups():
    assert "setup_status_watch" in gate(out(), make_setup(status="watch")).reasons
    assert "setup_expired" in gate(out(), make_setup(expires=date(2021, 1, 1))).reasons
    s = make_setup(bias="long", price_below=90.0)
    assert gate(out(), s, price=95.0).allow
    assert "setup_invalidated" in gate(out(), s, price=89.0).reasons
    sh = make_setup(bias="short", price_above=110.0)
    assert "setup_invalidated" in gate(out(direction="short"), sh, price=111.0).reasons


def test_setup_schema_requires_an_invalidation_level():
    with pytest.raises(ValidationError):
        SetupSchema(id="x", symbol=SYM, base="BTC", bias=Direction.LONG, quality_prior=2, thesis="t", bear_case="b",
                    invalidation={}, reflex=SetupReflexSpec(allowed_directions=[Direction.LONG]),
                    created=date(2026, 1, 1), expires=date(2026, 2, 1))


# ---- sizing ------------------------------------------------------------------------------
LIM = RiskLimits()
FEES = FeeModel()


def size(**kw):
    base = dict(p=0.65, stop_frac=0.02, take_profit_r=2.0, equity=100_000.0, price=100.0, instrument=inst(SYM, 1.0),
                limits=LIM, fees=FEES, kelly_fraction=0.25, gross_notional=0.0, bar_dollar_volume=1e9)
    base.update(kw)
    return size_position(**base)


def test_kelly_formula():
    assert kelly_full(0.6, 2.0) == pytest.approx(0.4)
    assert kelly_full(0.5, 1.0) == pytest.approx(0.0)
    assert kelly_full(0.4, 1.0) < 0


def test_size_is_capped_at_risk_per_trade():
    r = size()
    assert "capped_risk_per_trade" in r.reasons
    assert r.risk_fraction <= LIM.risk_per_trade + 1e-12
    cost = round_trip_cost_frac(FEES, r.notional, 1e9)            # costs priced at the real order size
    assert r.notional * (0.02 + cost) <= 2000.0 + 1e-6            # rounded DOWN: never above the budget
    assert (r.qty + 2) * 100.0 * (0.02 + cost) > 2000.0           # ...and as close to it as contracts allow


def test_quarter_kelly_binds_when_edge_is_thin():
    cost = round_trip_cost_frac(FEES, 100.0, 1e9)
    b = (2 * 0.02 - cost) / (0.02 + cost)
    p = 1 / (1 + b) + 0.02                                        # two points above breakeven
    r = size(p=p)
    assert "capped_risk_per_trade" not in r.reasons
    assert r.risk_fraction <= 0.25 * kelly_full(p, b) + 1e-9
    assert 0 < r.risk_fraction < LIM.risk_per_trade


def test_kelly_fraction_can_never_exceed_a_quarter():
    with pytest.raises(ValidationError):
        SizingCfg(kelly_fraction=0.5)
    assert size(kelly_fraction=5.0, p=0.52).risk_fraction <= 0.25 * size(kelly_fraction=5.0, p=0.52).kelly_full + 1e-9


def test_no_trade_without_edge_or_when_costs_eat_the_target():
    assert size(p=0.30).qty == 0 and "no_edge_after_costs" in size(p=0.30).reasons
    assert size(stop_frac=0.0005, take_profit_r=1.0).reasons == ("costs_exceed_target",)


def test_leverage_and_liquidity_caps():
    # entries may use leverage only up to near_limit_frac * max_leverage = 1.4x, so 40k of room is left
    r = size(stop_frac=0.008, gross_notional=100_000.0)
    assert "capped_leverage" in r.reasons and r.notional <= 40_000.0
    assert size(stop_frac=0.008, gross_notional=150_000.0).qty == 0
    r2 = size(bar_dollar_volume=100_000.0)                        # 2% participation -> 2k
    assert "capped_liquidity" in r2.reasons and r2.notional <= 2_000.0 + 1e-9


def test_below_one_contract_means_no_trade():
    big = inst(SYM, 1.0)                                          # one contract = $75,000
    r = size(equity=1000.0, price=75_000.0, instrument=big)
    assert r.qty == 0 and "below_one_contract" in r.reasons


def test_risk_scale_only_shrinks():
    assert size(risk_scale=0.5).risk_fraction <= 0.5 * LIM.risk_per_trade + 1e-12
    assert size(risk_scale=3.0).risk_fraction <= LIM.risk_per_trade + 1e-12


def test_realised_risk_never_exceeds_budget_property():
    rng = random.Random(7)
    for _ in range(2000):
        eq = rng.uniform(200, 200_000)
        price = rng.uniform(0.01, 90_000)
        cs = rng.choice([0.01, 0.1, 1, 5, 100, 1000])
        stop = rng.uniform(0.004, 0.06)
        r = size(p=rng.uniform(0.3, 0.9), stop_frac=stop, take_profit_r=rng.uniform(1.0, 3.0), equity=eq, price=price,
                 instrument=inst(SYM, cs), gross_notional=rng.uniform(0, 2 * eq), bar_dollar_volume=rng.uniform(1e4, 1e9),
                 kelly_fraction=rng.uniform(0.05, 1.0))
        assert r.qty >= 0 and r.qty == int(r.qty)
        assert r.risk_fraction <= LIM.risk_per_trade + 1e-9
        if r.qty:
            assert r.notional <= LIM.near_limit_frac * LIM.max_leverage * eq + 1e-6
            assert r.qty * cs * price * stop <= LIM.risk_per_trade * eq + 1e-6


# ---- calibration -------------------------------------------------------------------------
def test_shrink_calibration_is_conservative():
    c = Calibrator(CalibrationCfg())
    assert c.prob(0.9) == pytest.approx(0.7)
    assert c.prob(0.5) == pytest.approx(0.5)
    assert Calibrator(CalibrationCfg(shrink=1.0)).prob(1.0) == MAX_CALIBRATED_P


def test_brier_and_reliability():
    assert brier_score([1.0, 0.0], [1, 0]) == 0.0
    assert brier_score([0.5, 0.5], [1, 0]) == 0.25
    assert brier_score([], []) is None
    rows = reliability_table([0.55, 0.85, 0.86], [0, 1, 0])
    assert rows[0]["n"] == 1 and rows[3]["n"] == 2 and rows[3]["hit_rate"] == 0.5


def test_fit_bins_is_monotone_capped_and_shrunk_on_small_samples():
    rng = random.Random(3)
    conf = [rng.uniform(0.5, 1.0) for _ in range(600)]
    wins = [1 if rng.random() < (0.45 + 0.2 * (c - 0.5)) else 0 for c in conf]   # an overconfident model
    cfg = fit_bins(conf, wins)
    ps = [b.p for b in cfg.bins]
    assert ps == sorted(ps) and max(ps) <= MAX_CALIBRATED_P
    assert Calibrator(cfg).prob(0.95) < 0.70                      # learned that 0.95 is not really 0.95
    lucky = fit_bins([0.95] * 5, [1] * 5)                         # five lucky wins must not move sizing much
    assert Calibrator(lucky).prob(0.95) < 0.80


# ---- policy store -------------------------------------------------------------------------
def test_policy_store_versions_promotion_and_rollback(tmp_path):
    store = PolicyStore(tmp_path)
    p1 = store.save_new(Policy(version=1, note="first"))
    p2 = store.save_new(Policy(version=1, parent=p1.version, gate=GateCfg(min_direction_confidence=0.85)))
    assert (p1.version, p2.version) == (1, 2) and store.versions() == [1, 2]
    store.promote(1, "paper", by="bootstrap", reason="init")
    store.promote(2, "paper", by="auto-nightly", reason="passed gates")
    assert store.active("paper").version == 2
    assert store.rollback("paper", by="human", reason="worse live fills") == 1
    assert store.active("paper").version == 1
    assert [h["to"] for h in store.pointer()["history"]] == [1, 2, 1]


def test_policy_never_reaches_live_without_a_named_human(tmp_path):
    store = PolicyStore(tmp_path)
    store.save_new(Policy(version=1))
    for kwargs in (dict(by="auto-nightly", reason="x"), dict(by="auto-nightly", reason="x", human_approved=True),
                   dict(by="", reason="x", human_approved=True), dict(by="human-operator", reason="x", human_approved=False)):
        with pytest.raises(LivePromotionRefused):
            store.promote(1, "live", **kwargs)
    assert store.pointer()["live"] is None
    store.promote(1, "live", by="human-operator", reason="approved after 4 weeks of paper", human_approved=True)
    assert store.active("live").version == 1


def test_policy_cannot_carry_risk_limits():
    with pytest.raises(ValidationError):
        Policy(version=1, risk_per_trade=0.5)
    with pytest.raises(ValidationError):
        Policy(version=1, gate={"min_setup_quality": 2, "max_leverage": 10})
    assert not (set(Policy.model_fields) | set(GateCfg.model_fields)) & set(RiskLimits.model_fields)


def test_repo_policy_and_risk_files_load():
    from tradebot.config import REPO_ROOT, load_risk_limits, load_settings
    lim = load_risk_limits()
    assert (lim.risk_per_trade, lim.max_daily_loss, lim.max_open_positions, lim.max_leverage,
            lim.kill_switch_drawdown) == (0.02, 0.05, 3, 2.0, 0.15)
    pol = PolicyStore(REPO_ROOT / "config" / "policies").active("paper")
    assert pol.gate.min_setup_quality == 2 and pol.gate.min_direction_confidence == 0.80
    assert pol.sizing.kelly_fraction == 0.25 and pol.pause.confidence_floor == 0.60
    s = load_settings()
    assert s.mode.value == "paper" and s.paper_balance == 1000.0 and s.reflex.snapshot_token_budget == 400

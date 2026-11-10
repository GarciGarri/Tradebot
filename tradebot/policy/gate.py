"""The trade gate. Pure code: thresholds come from the policy and the setup, never a model.

The REFLEX supplies judgements. This function decides whether those judgements are enough
to even consider an entry. Sizing and the hard risk layer still come after it.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..reflex.schema import ReflexOutput
from ..setups.schema import SetupSchema
from ..types import Direction, Regime, RiskState
from .policy import Policy


@dataclass(frozen=True)
class GateResult:
    allow: bool
    reasons: tuple[str, ...]      # every failed check, for the nightly "misses" analysis


def evaluate_gate(out: ReflexOutput, policy: Policy, setup: SetupSchema, code_risk: RiskState,
                  now_ms: int, price: float, paused: bool) -> GateResult:
    reasons: list[str] = []
    ok, why = setup.tradable(now_ms, price)
    if not ok:
        reasons.append(why)
    if paused:
        reasons.append("paused")
    if out.direction is Direction.NEUTRAL:
        reasons.append("direction_neutral")
    elif out.direction not in setup.reflex.allowed_directions:
        reasons.append("direction_not_allowed_by_setup")
    if out.regime is Regime.CRISIS:
        reasons.append("regime_crisis")
    elif out.regime not in setup.reflex.allowed_regimes:
        reasons.append("regime_not_allowed_by_setup")
    min_q = max(policy.gate.min_setup_quality, setup.reflex.min_setup_quality)
    if out.setup_quality < min_q:
        reasons.append(f"setup_quality<{min_q}")
    min_c = max(policy.gate.min_direction_confidence, setup.reflex.min_direction_confidence)
    if not out.direction_confidence > min_c:
        reasons.append(f"direction_confidence<={min_c:.2f}")
    if out.risk_state is not RiskState.SAFE:
        reasons.append(f"model_risk_state_{out.risk_state.value}")
    if code_risk is not RiskState.SAFE:
        reasons.append(f"code_risk_state_{code_risk.value}")
    if policy.gate.block_on_toxic_flow and out.toxic_flow is True:
        reasons.append("toxic_flow")
    return GateResult(allow=not reasons, reasons=tuple(reasons))

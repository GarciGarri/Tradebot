"""Deterministic REFLEX: plain code rules behind the same interface as the model.

Why it exists:
* It is the only decision-maker that may be backtested on data from before a model's training
  cutoff (a model remembers that period; code does not).
* It is the baseline every model-backed REFLEX has to beat.
* It lets the whole pipeline run with no API key.

The rule is a simple trend-following score. Its parameters are hypotheses that walk-forward
selects on training windows only. It is not claimed to be an edge.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..state.engine import Snapshot
from ..types import Direction, Regime, RiskState
from .schema import ReflexOutput


@dataclass(frozen=True)
class RuleParams:
    trend_ema_min: float = 0.30      # |EMA20-EMA50 gap| in % needed to call a trend
    score_threshold: float = 0.35    # |score| needed to take a side
    conf_gain: float = 0.90          # how fast confidence rises with the score
    vol_expansion: float = 1.80      # rvr above this = high_vol
    crisis_btc_r24: float = -10.0    # BTC 24-bar return that (with expanding vol) means crisis
    allow_short: bool = True
    max_confidence: float = 0.95


def _num(v, default: float = 0.0) -> float:
    return default if v is None or (isinstance(v, float) and not math.isfinite(v)) else float(v)


class RuleReflex:
    name = "rules"

    def __init__(self, params: RuleParams | None = None):
        self.params = params or RuleParams()

    def decide(self, snapshot: Snapshot) -> ReflexOutput:
        p, f = self.params, snapshot.fields
        ema, px50 = _num(f.get("ema")), _num(f.get("px50"))
        r24, r72 = _num(f.get("r24")), _num(f.get("r72"))
        rvr = _num(f.get("rvr"), 1.0)
        b24, bema = _num(f.get("b24")), _num(f.get("bema"))
        vz = _num(f.get("vz"))
        rng = _num(f.get("rng"), 0.5)
        bias = int(_num(f.get("bias")))
        inv = f.get("inv")

        # --- regime
        if (b24 <= p.crisis_btc_r24 or r24 <= 1.5 * p.crisis_btc_r24) and rvr >= 2.0:
            regime = Regime.CRISIS
        elif rvr >= p.vol_expansion:
            regime = Regime.HIGH_VOL
        elif abs(ema) >= p.trend_ema_min and ema * px50 > 0:
            regime = Regime.TRENDING
        else:
            regime = Regime.MEAN_REVERTING

        # --- direction score in [-1, 1]
        score = (0.35 * math.tanh(ema / 1.0) + 0.25 * math.tanh(px50 / 2.5)
                 + 0.20 * math.tanh(r24 / 4.0) + 0.20 * math.tanh(r72 / 8.0))
        if regime is Regime.TRENDING and score > p.score_threshold:
            direction = Direction.LONG
        elif regime is Regime.TRENDING and score < -p.score_threshold and p.allow_short:
            direction = Direction.SHORT
        else:
            direction = Direction.NEUTRAL
        if direction is Direction.NEUTRAL:
            confidence = 0.5
        else:
            excess = (abs(score) - p.score_threshold) / max(1e-9, 1.0 - p.score_threshold)
            confidence = min(p.max_confidence, 0.5 + 0.5 * min(1.0, excess * p.conf_gain + 0.35))

        # --- setup quality 0-3
        sign = 1 if direction is Direction.LONG else -1 if direction is Direction.SHORT else 0
        if sign == 0 or (inv is not None and _num(inv) <= 0):
            quality = 0
        else:
            quality = 1
            if bias == sign or (bias == 0 and regime is Regime.TRENDING):
                quality += 1
            backdrop_ok = (bema * sign > 0) and (b24 * sign > -2.0)
            not_stretched = (rng < 0.97) if sign > 0 else (rng > 0.03)
            if backdrop_ok and not_stretched and vz > -1.0:
                quality += 1
            if bias == -sign:
                quality = min(quality, 1)   # fighting the research thesis is never a good setup

        # --- order-flow view
        spr, imb = f.get("spr"), f.get("imb")
        if spr is None and imb is None:
            toxic = None
        else:
            toxic = bool(_num(spr) > 15.0 or (_num(imb) * sign < -0.6 and sign != 0))

        # --- risk state from usage ratios
        exposure = max(_num(f.get("dayu")), _num(f.get("levu")))
        if max(exposure, _num(f.get("ddu"))) >= 0.9:
            risk = RiskState.REDUCE
        elif exposure >= 0.7 or _num(f.get("posu")) >= 1.0:
            risk = RiskState.NEAR_LIMIT
        else:
            risk = RiskState.SAFE

        return ReflexOutput(regime=regime, direction=direction, direction_confidence=round(confidence, 4),
                            toxic_flow=toxic, setup_quality=quality, risk_state=risk)

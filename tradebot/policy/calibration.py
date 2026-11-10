"""Confidence calibration and Brier scoring.

The REFLEX reports a confidence. Sizing needs a probability. Until there is evidence the two
agree, the confidence is shrunk toward 0.5. Once outcomes exist, the nightly job fits a
binned mapping with a prior that pulls small samples back to the shrunk value, so ten lucky
trades cannot talk the sizing code into betting more.
"""
from __future__ import annotations

from typing import Sequence

from .policy import CalibrationBin, CalibrationCfg

MAX_CALIBRATED_P = 0.90  # no mapping may claim more certainty than this


def shrink_prob(confidence: float, shrink: float) -> float:
    return 0.5 + shrink * (confidence - 0.5)


class Calibrator:
    def __init__(self, cfg: CalibrationCfg):
        self.cfg = cfg

    def prob(self, confidence: float) -> float:
        c = min(1.0, max(0.0, confidence))
        if self.cfg.method == "bins":
            for b in self.cfg.bins:
                if b.lo <= c < b.hi or (c == 1.0 and b.hi >= 1.0 and b.lo <= c):
                    return min(MAX_CALIBRATED_P, b.p)
        return min(MAX_CALIBRATED_P, shrink_prob(c, self.cfg.shrink))


def brier_score(probs: Sequence[float], outcomes: Sequence[int]) -> float | None:
    if not probs:
        return None
    return sum((p - y) ** 2 for p, y in zip(probs, outcomes)) / len(probs)


def reliability_table(confidences: Sequence[float], outcomes: Sequence[int],
                      edges: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0001)) -> list[dict]:
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = [i for i, c in enumerate(confidences) if lo <= c < hi]
        if not idx:
            rows.append({"lo": lo, "hi": min(hi, 1.0), "n": 0, "mean_conf": None, "hit_rate": None})
            continue
        rows.append({
            "lo": lo, "hi": min(hi, 1.0), "n": len(idx),
            "mean_conf": sum(confidences[i] for i in idx) / len(idx),
            "hit_rate": sum(outcomes[i] for i in idx) / len(idx),
        })
    return rows


def fit_bins(confidences: Sequence[float], outcomes: Sequence[int], shrink: float = 0.5,
             prior_strength: float = 30.0,
             edges: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0001)) -> CalibrationCfg:
    """Binned calibration with a Beta-style prior centred on the shrunk confidence.

    The result is forced to be monotone and is capped at MAX_CALIBRATED_P.
    """
    bins: list[CalibrationBin] = []
    for row in reliability_table(confidences, outcomes, edges):
        centre = (row["lo"] + row["hi"]) / 2.0
        p0 = shrink_prob(centre, shrink)
        n = row["n"]
        wins = (row["hit_rate"] or 0.0) * n
        p = (wins + prior_strength * p0) / (n + prior_strength)
        bins.append(CalibrationBin(lo=row["lo"], hi=row["hi"], p=min(MAX_CALIBRATED_P, p), n=n))
    # pool adjacent violators so a higher confidence never maps to a lower probability
    ps = [b.p for b in bins]
    ws = [max(1, b.n) for b in bins]
    i = 0
    blocks = [[p, w, 1] for p, w in zip(ps, ws)]
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0] + 1e-12:
            p = (blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]) / (blocks[i][1] + blocks[i + 1][1])
            blocks[i] = [p, blocks[i][1] + blocks[i + 1][1], blocks[i][2] + blocks[i + 1][2]]
            del blocks[i + 1]
            i = max(0, i - 1)
        else:
            i += 1
    flat: list[float] = []
    for p, _w, k in blocks:
        flat.extend([p] * k)
    bins = [b.model_copy(update={"p": round(p, 4)}) for b, p in zip(bins, flat)]
    return CalibrationCfg(method="bins", shrink=shrink, bins=bins)

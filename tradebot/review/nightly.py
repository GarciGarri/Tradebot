"""Nightly review: read every fill and miss, measure calibration, propose, gate, version.

What it does, in order:

1. LABEL  every directional decision (taken or not) with what would have happened: a triple
   barrier (stop, target, time limit) on the bars that followed, entered at the next bar's
   open and net of estimated costs. A "miss" is a decision the gate blocked; it is labelled
   exactly like a fill, so the gate itself can be judged.
2. MEASURE calibration (Brier score, reliability by confidence bucket), fills, misses by gate
   reason, skipped candles, REFLEX latency and how often the model's risk_state disagreed
   with the code's.
3. PROPOSE at most one candidate policy: a recalibrated confidence mapping and, if the
   evidence supports it, new gate thresholds inside fixed tuning bounds. Thresholds are
   selected on the older 70% of the labelled decisions and judged on the newest 30%.
4. GATE the candidate: enough samples, better out-of-sample expectancy than the current
   policy, no worse calibration, bounds respected, regression tests green.
5. VERSION it as a new policy file. If every gate passes, the PAPER pointer moves to it (the
   old version stays one `rollback` away). The LIVE pointer is never touched here.

Hard risk limits are out of scope: this module cannot read or write them.
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..config import REPO_ROOT, FeeModel
from ..policy.calibration import Calibrator, brier_score, fit_bins, reliability_table
from ..policy.policy import TUNE_MIN_CONFIDENCE, TUNE_MIN_QUALITY, GateCfg, Policy, PolicyStore
from ..policy.sizing import round_trip_cost_frac
from ..types import Bar

MIN_LABELED = 60          # below this, nothing is proposed: too little evidence
MIN_SELECTED = 12         # a threshold pair must pass at least this many validation decisions
IMPROVE_MARGIN_R = 0.02   # required out-of-sample improvement in mean R


# ---------------------------------------------------------------- labelling
def triple_barrier(direction: str, entry: float, stop_frac: float, take_profit_r: float, max_hold: int,
                   future: list[Bar], cost_frac: float) -> tuple[int, float] | None:
    """Outcome of a hypothetical trade. Returns (win 0/1, R multiple net of costs) or None if
    the future is not known yet. If one bar contains both levels the stop is assumed first."""
    if not future or stop_frac <= 0:
        return None
    sign = 1 if direction == "long" else -1
    stop = entry * (1 - sign * stop_frac)
    target = entry * (1 + sign * stop_frac * take_profit_r)
    cost_r = cost_frac / stop_frac
    for i, b in enumerate(future[:max_hold]):
        hit_stop = b.low <= stop if sign > 0 else b.high >= stop
        hit_target = b.high >= target if sign > 0 else b.low <= target
        if hit_stop:
            gap = min(b.open, stop) if sign > 0 else max(b.open, stop)
            return 0, sign * (gap / entry - 1.0) / stop_frac - cost_r
        if hit_target:
            return 1, take_profit_r - cost_r
    if len(future) < max_hold:
        return None                      # still open: label later
    r = sign * (future[max_hold - 1].close / entry - 1.0) / stop_frac - cost_r
    return (1 if r > 0 else 0), r


def label_decisions(rows: list[dict], bars: dict[str, list[Bar]], fees: FeeModel) -> list[dict]:
    """Attach `outcome` and `outcome_r` to directional decision rows that can be resolved."""
    index = {s: {b.ts: i for i, b in enumerate(bs)} for s, bs in bars.items()}
    cost = round_trip_cost_frac(fees, 0.0, None, 24)
    out = []
    for r in rows:
        if r.get("status") != "decision" or r.get("direction") not in ("long", "short"):
            continue
        if r.get("outcome") is not None:
            out.append(r)
            continue
        sym, sf = r["symbol"], r.get("stop_frac")
        if sym not in bars or not bars[sym]:
            continue
        tf = bars[sym][0].tf_ms
        # Live decisions are stamped a few seconds after the candle close; floor to the bar
        # that opened at that close. That bar is the first one the decision could trade in.
        i = index[sym].get((r["ts"] // tf) * tf)
        if i is None or not sf:
            continue
        series = bars[sym]
        res = triple_barrier(r["direction"], series[i].open, sf, r.get("take_profit_r") or 2.0,
                             int(r.get("max_hold_bars") or 72), series[i:], cost)
        if res is None:
            continue
        out.append({**r, "outcome": res[0], "outcome_r": res[1]})
    return out


# ---------------------------------------------------------------- measurement
def _passes(row: dict, min_conf: float, min_quality: int) -> bool:
    return (row["confidence"] > min_conf and (row.get("setup_quality") or 0) >= min_quality)


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def measure(labeled: list[dict], all_rows: list[dict], trades: list[dict], policy: Policy) -> dict:
    conf = [r["confidence"] for r in labeled]
    wins = [r["outcome"] for r in labeled]
    cal = Calibrator(policy.calibration)
    base_rate = _mean([float(w) for w in wins])
    misses: dict[str, dict] = {}
    for r in labeled:
        if r.get("gate_allow"):
            continue
        for reason in (r.get("gate_reasons") or "unknown").split(","):
            m = misses.setdefault(reason, {"n": 0, "wins": 0, "sum_r": 0.0})
            m["n"] += 1
            m["wins"] += r["outcome"]
            m["sum_r"] += r["outcome_r"]
    for m in misses.values():
        m["hit_rate"] = m["wins"] / m["n"]
        m["mean_r"] = m["sum_r"] / m["n"]
    skips: dict[str, int] = {}
    lat = sorted(r["latency_ms"] for r in all_rows if r.get("status") == "decision" and r.get("latency_ms") is not None)
    for r in all_rows:
        if r.get("status") == "skip":
            skips[r.get("skip_reason") or "unknown"] = skips.get(r.get("skip_reason") or "unknown", 0) + 1
    decided = [r for r in all_rows if r.get("status") == "decision"]
    disagree = sum(1 for r in decided if r.get("risk_state") != r.get("code_risk_state"))
    taken = [r for r in labeled if r.get("action") == "enter"]
    pnl = [t["pnl"] for t in trades]
    return {
        "n_rows": len(all_rows), "n_decisions": len(decided), "n_labeled": len(labeled),
        "brier_raw": brier_score(conf, wins), "brier_calibrated": brier_score([cal.prob(c) for c in conf], wins),
        "brier_base_rate": brier_score([base_rate] * len(wins), wins) if base_rate is not None else None,
        "base_rate": base_rate, "reliability": reliability_table(conf, wins),
        "taken": {"n": len(taken), "hit_rate": _mean([float(r["outcome"]) for r in taken]),
                  "mean_r": _mean([r["outcome_r"] for r in taken])},
        "fills": {"n_trades": len(trades), "pnl": sum(pnl), "win_rate": _mean([1.0 if p > 0 else 0.0 for p in pnl]),
                  "mean_r": _mean([t.get("r_multiple") or 0.0 for t in trades]),
                  "fees": sum(t.get("fees") or 0.0 for t in trades),
                  "by_reason": {k: sum(1 for t in trades if t.get("reason") == k) for k in {t.get("reason") for t in trades}}},
        "misses_by_gate_reason": misses, "skips": skips,
        "latency_ms": {"p50": lat[len(lat) // 2] if lat else None, "p95": lat[int(len(lat) * 0.95)] if lat else None},
        "risk_state_disagreement": (disagree / len(decided)) if decided else None,
    }


# ---------------------------------------------------------------- proposal + gates
@dataclass
class Proposal:
    candidate: Policy | None
    rationale: list[str] = field(default_factory=list)
    gates: dict[str, bool] = field(default_factory=dict)
    details: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.candidate is not None and bool(self.gates) and all(self.gates.values())


def _expectancy(rows: list[dict], min_conf: float, min_quality: int) -> tuple[float | None, int]:
    sel = [r["outcome_r"] for r in rows if _passes(r, min_conf, min_quality)]
    return _mean(sel), len(sel)


def propose(policy: Policy, labeled: list[dict]) -> Proposal:
    prop = Proposal(candidate=None)
    n = len(labeled)
    prop.gates["enough_samples"] = n >= MIN_LABELED
    if n < MIN_LABELED:
        prop.rationale.append(f"only {n} labelled decisions (need {MIN_LABELED}): no change proposed")
        return prop
    rows = sorted(labeled, key=lambda r: r["ts"])
    cut = int(n * 0.7)
    train, valid = rows[:cut], rows[cut:]

    # 1) recalibrate on the training part only
    new_cal = fit_bins([r["confidence"] for r in train], [r["outcome"] for r in train], shrink=policy.calibration.shrink)
    b_old = brier_score([Calibrator(policy.calibration).prob(r["confidence"]) for r in valid], [r["outcome"] for r in valid])
    b_new = brier_score([Calibrator(new_cal).prob(r["confidence"]) for r in valid], [r["outcome"] for r in valid])
    prop.details["brier_validation"] = {"current": b_old, "candidate": b_new}

    # 2) gate thresholds: choose on train, judge on validation, stay inside the tuning bounds
    cur = (policy.gate.min_direction_confidence, policy.gate.min_setup_quality)
    best, best_score = cur, _expectancy(train, *cur)[0]
    lo, hi = TUNE_MIN_CONFIDENCE
    grid_c = [round(lo + 0.05 * k, 2) for k in range(int(round((hi - lo) / 0.05)) + 1)]
    for c in grid_c:
        for q in range(TUNE_MIN_QUALITY[0], TUNE_MIN_QUALITY[1] + 1):
            e, k = _expectancy(train, c, q)
            if e is not None and k >= MIN_SELECTED * 2 and (best_score is None or e > best_score + 1e-9):
                best, best_score = (c, q), e
    e_cur, n_cur = _expectancy(valid, *cur)
    e_new, n_new = _expectancy(valid, *best)
    prop.details["thresholds"] = {"current": cur, "candidate": best, "valid_mean_r_current": e_cur, "valid_n_current": n_cur,
                                  "valid_mean_r_candidate": e_new, "valid_n_candidate": n_new}
    gate_cfg = policy.gate
    thresholds_changed = best != cur
    if thresholds_changed:
        gate_cfg = GateCfg(min_setup_quality=best[1], min_direction_confidence=best[0],
                           block_on_toxic_flow=policy.gate.block_on_toxic_flow)
        prop.rationale.append(f"gate {cur} -> {best}: train expectancy {best_score:+.3f}R")
    prop.rationale.append(f"recalibrated confidence on {len(train)} decisions: validation Brier {b_old:.4f} -> {b_new:.4f}")

    prop.candidate = policy.model_copy(update={
        "parent": policy.version, "gate": gate_cfg, "calibration": new_cal,
        "note": f"nightly candidate from v{policy.version}: " + "; ".join(prop.rationale)})
    prop.gates["calibration_not_worse"] = b_new is not None and b_old is not None and b_new <= b_old + 0.002
    if thresholds_changed:
        prop.gates["validation_sample"] = n_new >= MIN_SELECTED
        prop.gates["validation_expectancy_positive"] = e_new is not None and e_new > 0
        prop.gates["validation_beats_current"] = (e_new is not None and
                                                  (e_cur is None or e_new >= e_cur + IMPROVE_MARGIN_R))
    prop.gates["inside_tuning_bounds"] = (lo <= gate_cfg.min_direction_confidence <= hi and
                                          TUNE_MIN_QUALITY[0] <= gate_cfg.min_setup_quality <= TUNE_MIN_QUALITY[1])
    return prop


def regression_tests(timeout_s: int = 600) -> tuple[bool, str]:
    """Run the test suite. Fail closed: if it cannot run, the gate does not pass."""
    tests = REPO_ROOT / "tests"
    if not tests.exists():
        return False, "tests directory not present"
    try:
        p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", str(tests)], cwd=REPO_ROOT, capture_output=True,
                           text=True, timeout=timeout_s)
    except Exception as exc:
        return False, f"could not run tests: {type(exc).__name__}"
    tail = (p.stdout.strip().splitlines() or [""])[-1]
    return p.returncode == 0, tail[:200]


# ---------------------------------------------------------------- orchestration
def run_nightly(*, rows: list[dict], trades: list[dict], bars: dict[str, list[Bar]], fees: FeeModel,
                store: PolicyStore, reports_dir: str | Path, events: list[dict] | None = None,
                setups: dict | None = None, now_ms: int | None = None, run_tests: bool = True,
                recorder=None) -> dict:
    now = datetime.fromtimestamp((now_ms or 0) / 1000, tz=timezone.utc) if now_ms else datetime.now(timezone.utc)
    policy = store.active("paper")
    labeled = label_decisions(rows, bars, fees)
    if recorder is not None and hasattr(recorder, "label_decision"):
        for r in labeled:
            recorder.label_decision(r["id"], r["outcome"], r["outcome_r"])
    stats = measure(labeled, rows, trades, policy)
    prop = propose(policy, labeled)
    promoted, new_version, tests_tail = False, None, ""
    if prop.candidate is not None:
        if run_tests:
            ok, tests_tail = regression_tests()
            prop.gates["regression_tests"] = ok
        saved = store.save_new(prop.candidate)          # every candidate is versioned, passed or not
        new_version = saved.version
        if prop.passed:
            store.promote(saved.version, "paper", by="auto-nightly",
                          reason="all nightly gates passed: " + "; ".join(prop.rationale))
            promoted = True
    escalations = [e for e in (events or []) if str(e.get("key", "")).startswith("escalate")
                   or e.get("level") == "critical"]
    manual = []
    for s in (setups or {}).values():
        days_left = (s.expires - now.date()).days
        if s.status == "active" and days_left <= 7:
            manual.append(f"{s.id}: expires in {days_left} day(s), review the thesis")
        for cond in s.invalidation.conditions:
            if s.status == "active":
                manual.append(f"{s.id}: check invalidation condition: {cond}")
    report = {
        "date": now.strftime("%Y-%m-%d"), "generated": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "policy_before": policy.version, "candidate_version": new_version, "promoted_to_paper": promoted,
        "live_policy": store.pointer().get("live"), "stats": stats, "rationale": prop.rationale, "gates": prop.gates,
        "details": prop.details, "regression_tests": tests_tail, "escalations": escalations[-50:],
        "manual_checks": manual,
    }
    out = Path(reports_dir) / "nightly"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{report['date']}.json").write_text(json.dumps(report, indent=1, default=str))
    (out / f"{report['date']}.md").write_text(render_markdown(report))
    return report


def render_markdown(r: dict) -> str:
    s = r["stats"]

    def f(x, spec=".3f"):
        return "n/a" if x is None else format(x, spec)

    lines = [
        f"# Nightly review {r['date']}", "",
        f"Policy in paper before this run: v{r['policy_before']}. Live policy: {r['live_policy'] or 'none (not live)'}.",
        f"Candidate: {('v' + str(r['candidate_version'])) if r['candidate_version'] else 'none'}; "
        f"promoted to paper: {'yes' if r['promoted_to_paper'] else 'no'}. Nothing here can promote to live.", "",
        "## Calibration",
        f"- Labelled directional decisions: {s['n_labeled']} of {s['n_decisions']} decisions ({s['n_rows']} candles).",
        f"- Brier raw {f(s['brier_raw'], '.4f')}, calibrated {f(s['brier_calibrated'], '.4f')}, "
        f"always-the-base-rate {f(s['brier_base_rate'], '.4f')} (base rate {f(s['base_rate'])}).",
        "", "| confidence | n | mean confidence | hit rate |", "|---|---|---|---|"]
    for row in s["reliability"]:
        lines.append(f"| {row['lo']:.1f}-{row['hi']:.1f} | {row['n']} | {f(row['mean_conf'])} | {f(row['hit_rate'])} |")
    fl = s["fills"]
    lines += ["", "## Fills",
              f"- Trades closed: {fl['n_trades']}, PnL {fl['pnl']:+.2f}, win rate {f(fl['win_rate'])}, mean R {f(fl['mean_r'])}, "
              f"fees {fl['fees']:.2f}. Exits: {fl['by_reason']}",
              f"- Entries taken (labelled): {s['taken']['n']}, hit rate {f(s['taken']['hit_rate'])}, mean R {f(s['taken']['mean_r'])}.",
              "", "## Misses (what the gate blocked, and what would have happened)",
              "| gate reason | n | hit rate | mean R |", "|---|---|---|---|"]
    for reason, m in sorted(s["misses_by_gate_reason"].items(), key=lambda kv: -kv[1]["n"]):
        lines.append(f"| {reason} | {m['n']} | {m['hit_rate']:.3f} | {m['mean_r']:+.3f} |")
    lines += ["", "## Health",
              f"- Skipped candles: {s['skips'] or 'none'}",
              f"- REFLEX latency p50 {f(s['latency_ms']['p50'], '.0f')} ms, p95 {f(s['latency_ms']['p95'], '.0f')} ms.",
              f"- Model vs code risk_state disagreement: {f(s['risk_state_disagreement'])}.",
              "", "## Proposal"]
    lines += [f"- {x}" for x in (r["rationale"] or ["no proposal"])]
    lines += ["", "| gate | passed |", "|---|---|"] + [f"| {k} | {'yes' if v else 'NO'} |" for k, v in r["gates"].items()]
    if r["regression_tests"]:
        lines.append(f"\nRegression tests: `{r['regression_tests']}`")
    if r["manual_checks"]:
        lines += ["", "## Needs a human or the BRAIN"] + [f"- {x}" for x in r["manual_checks"]]
    if r["escalations"]:
        lines += ["", "## Escalations and critical events"] + [f"- {e.get('key')}: {e.get('message')}" for e in r["escalations"][-15:]]
    return "\n".join(lines) + "\n"

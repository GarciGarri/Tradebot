"""Training-cutoff guard for backtests.

A language model has seen the market up to its training cutoff. Backtesting its decisions on
that period measures memory, not skill. So: a model-backed REFLEX may only be evaluated on
candles strictly after its cutoff. Code rules have no memory and may be tested on any period.
"""
from __future__ import annotations

from datetime import datetime, timezone

# First instant that is safely after each model's training data (cutoff month + 1).
MODEL_SAFE_START = {
    "claude-haiku-5-5": datetime(2026, 7, 1, tzinfo=timezone.utc),
    "claude-fable-5-2": datetime(2026, 7, 1, tzinfo=timezone.utc),
}
CODE_BACKENDS = {"rules", "mock"}


class CutoffViolation(ValueError):
    pass


def assert_backtest_allowed(backend: str, model: str | None, start_ms: int) -> None:
    """Raise if a model would be judged on data it may remember."""
    if backend in CODE_BACKENDS:
        return
    if not model or model not in MODEL_SAFE_START:
        raise CutoffViolation(
            f"backend {backend!r} model {model!r}: unknown training cutoff. Add it to MODEL_SAFE_START "
            "before backtesting, or the result cannot be trusted.")
    safe = int(MODEL_SAFE_START[model].timestamp() * 1000)
    if start_ms < safe:
        raise CutoffViolation(
            f"{model} may only be backtested on data from {MODEL_SAFE_START[model].date()} onward; "
            f"requested start is {datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).date()}.")

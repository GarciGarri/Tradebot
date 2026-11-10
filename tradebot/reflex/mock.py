"""Test and replay backends."""
from __future__ import annotations

import time
from typing import Callable, Iterable

from ..state.engine import Snapshot
from .schema import ReflexError, ReflexOutput


class MockReflex:
    """Returns scripted outputs. `delay_s` simulates a slow model; `fail` simulates an error."""

    name = "mock"

    def __init__(self, outputs: Iterable[ReflexOutput | dict] | Callable[[Snapshot], ReflexOutput | dict],
                 delay_s: float = 0.0, fail: bool = False):
        self._fn = outputs if callable(outputs) else None
        self._queue = None if callable(outputs) else list(outputs)
        self.delay_s = delay_s
        self.fail = fail
        self.calls = 0
        self.seen: list[str] = []

    def decide(self, snapshot: Snapshot):
        self.calls += 1
        self.seen.append(snapshot.text)
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.fail:
            raise ReflexError("mock failure")
        if self._fn is not None:
            return self._fn(snapshot)
        if not self._queue:
            raise ReflexError("mock script exhausted")
        return self._queue.pop(0) if len(self._queue) > 1 else self._queue[0]


class ReplayReflex:
    """Replays decisions recorded earlier, keyed by (symbol, decision ts).

    Used to backtest model decisions that were collected once (and paid for once) on
    post-cutoff data. A missing key is a skipped candle, exactly like a missed deadline.
    """

    name = "replay"

    def __init__(self, decisions: dict[tuple[str, int], ReflexOutput], source: str = "replay"):
        self._decisions = decisions
        self.name = source

    def decide(self, snapshot: Snapshot) -> ReflexOutput:
        try:
            return self._decisions[(snapshot.symbol, snapshot.ts)]
        except KeyError as exc:
            raise ReflexError("no recorded decision for this candle") from exc

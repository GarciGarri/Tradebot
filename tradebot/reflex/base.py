"""REFLEX interface and the runner that enforces the candle deadline.

Every backend (Haiku, Jev, code rules, mocks) implements `Reflex`. `ReflexRunner` measures
latency and turns a late, failed or malformed answer into a `ReflexSkip`: the candle is
skipped, nothing is retried, and no order logic depends on waiting for a model.
"""
from __future__ import annotations

import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Protocol, runtime_checkable

from pydantic import ValidationError

from ..state.engine import Snapshot
from .schema import ReflexDecision, ReflexOutput, ReflexSkip


@runtime_checkable
class Reflex(Protocol):
    name: str

    def decide(self, snapshot: Snapshot) -> ReflexOutput: ...


class ReflexRunner:
    def __init__(self, reflex: Reflex, deadline_ms: int, inline: bool = False):
        """`inline=True` runs the backend on the calling thread (deterministic backends in
        backtests). A model backend must use the pool so a slow call can be abandoned."""
        self.reflex = reflex
        self.deadline_ms = deadline_ms
        self.inline = inline
        self._pool = None if inline else ThreadPoolExecutor(max_workers=4, thread_name_prefix="reflex")
        self.stats: Counter[str] = Counter()
        self.latencies_ms: deque[float] = deque(maxlen=5000)

    def run(self, snapshot: Snapshot) -> ReflexDecision | ReflexSkip:
        if not snapshot.ok:
            self.stats["snapshot_not_ok"] += 1
            return ReflexSkip("snapshot_not_ok", snapshot.ts, snapshot.symbol, detail="; ".join(snapshot.issues))
        t0 = time.perf_counter()
        try:
            if self.inline:
                raw = self.reflex.decide(snapshot)
            else:
                fut = self._pool.submit(self.reflex.decide, snapshot)
                try:
                    raw = fut.result(timeout=self.deadline_ms / 1000.0)
                except FutureTimeout:
                    fut.cancel()  # a late answer is discarded, never acted on
                    latency = (time.perf_counter() - t0) * 1000.0
                    self.stats["deadline"] += 1
                    return ReflexSkip("deadline", snapshot.ts, snapshot.symbol, latency,
                                      f"no decision within {self.deadline_ms} ms")
        except Exception as exc:  # backend failure of any kind: skip, do not crash the loop
            latency = (time.perf_counter() - t0) * 1000.0
            self.stats["error"] += 1
            return ReflexSkip("error", snapshot.ts, snapshot.symbol, latency, f"{type(exc).__name__}: {exc}"[:300])
        latency = (time.perf_counter() - t0) * 1000.0
        self.latencies_ms.append(latency)
        if latency > self.deadline_ms:
            self.stats["deadline"] += 1
            return ReflexSkip("deadline", snapshot.ts, snapshot.symbol, latency, "decision arrived after the deadline")
        try:
            out = raw if isinstance(raw, ReflexOutput) else ReflexOutput.model_validate(raw)
        except ValidationError as exc:
            self.stats["invalid"] += 1
            return ReflexSkip("invalid", snapshot.ts, snapshot.symbol, latency, str(exc)[:300])
        self.stats["ok"] += 1
        return ReflexDecision(output=out, ts=snapshot.ts, symbol=snapshot.symbol, setup_id=snapshot.setup_id,
                              backend=self.reflex.name, latency_ms=latency)

    def latency_summary(self) -> dict[str, float]:
        if not self.latencies_ms:
            return {"n": 0}
        xs = sorted(self.latencies_ms)
        pick = lambda q: xs[min(len(xs) - 1, int(q * len(xs)))]  # noqa: E731
        return {"n": len(xs), "p50_ms": pick(0.50), "p95_ms": pick(0.95), "max_ms": xs[-1]}

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)

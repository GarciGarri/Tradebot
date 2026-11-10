"""Order manager: the single doorway to the exchange.

Every order passes through `execute`, which

1. runs the hard risk check immediately before sending (no exceptions, no caching);
2. is idempotent on the client order id;
3. never blindly retries an order whose outcome is unknown: it asks the venue first;
4. records the order and its fills.

`reconcile` compares the venue's positions with the local view. Any disagreement, or any
order whose fate is still unknown, blocks new risk until it is resolved.
"""
from __future__ import annotations

from dataclasses import replace

from ..alerts.base import Alerter
from ..risk.manager import Halts, RiskManager, RiskVerdict
from ..types import AccountState, Instrument, OrderIntent, OrderResult, OrderStatus


class OrderManager:
    def __init__(self, exchange, risk: RiskManager, recorder, alerts: Alerter, resolve_attempts: int = 3):
        self.exchange = exchange
        self.risk = risk
        self.recorder = recorder
        self.alerts = alerts
        self.resolve_attempts = resolve_attempts
        self._seen: dict[str, OrderResult] = {}
        self.unresolved: dict[str, OrderIntent] = {}
        self.last_verdict: RiskVerdict | None = None

    @property
    def blocked(self) -> bool:
        return bool(self.unresolved)

    def _record(self, intent: OrderIntent, status: str, message: str = "") -> None:
        self.recorder.order({
            "client_id": intent.client_id, "ts": intent.ts, "symbol": intent.symbol, "side": intent.side.value,
            "qty": intent.qty, "reduce_only": int(intent.reduce_only), "reason": intent.reason, "status": status,
            "message": message, "setup_id": intent.setup_id, "decision_id": intent.decision_id})

    def _record_fills(self, res: OrderResult) -> None:
        for f in res.fills:
            self.recorder.fill({"client_id": f.client_id, "ts": f.ts, "symbol": f.symbol, "side": f.side.value,
                                "qty": f.qty, "price": f.price, "fee": f.fee, "reduce_only": int(f.reduce_only)})

    def execute(self, intent: OrderIntent, account: AccountState, instrument: Instrument,
                mark_price: float | None, bar_dollar_volume: float | None, data_ok: bool, halts: Halts,
                trigger_price: float | None = None, passive: bool = False) -> OrderResult:
        if intent.client_id in self._seen:
            return self._seen[intent.client_id]
        if self.blocked and not intent.reduce_only:
            halts = replace(halts, reconcile_block=True)
        verdict = self.risk.check_order(intent, account, instrument, mark_price, bar_dollar_volume, data_ok, halts)
        self.last_verdict = verdict
        if not verdict.ok:
            res = OrderResult(OrderStatus.REJECTED, intent.client_id, message="risk: " + ",".join(verdict.violations))
            self._seen[intent.client_id] = res
            self._record(intent, "risk_rejected", res.message)
            return res

        try:
            if getattr(self.exchange, "simulated", False):
                res = self.exchange.submit(intent, at_price=trigger_price, passive=passive)
            else:
                res = self.exchange.submit(intent)
        except Exception as exc:  # transport error: outcome unknown until the venue is asked
            res = OrderResult(OrderStatus.UNKNOWN, intent.client_id, message=f"{type(exc).__name__}")

        if res.status is OrderStatus.UNKNOWN:
            res = self._resolve(intent, res)
        self._seen[intent.client_id] = res
        self._record(intent, res.status.value, res.message)
        self._record_fills(res)
        return res

    def _resolve(self, intent: OrderIntent, res: OrderResult) -> OrderResult:
        """An order timed out. Ask the venue what happened; never resend blindly."""
        lookups_ok, found = 0, None
        for _ in range(self.resolve_attempts):
            try:
                found = self.exchange.fetch_order(intent.client_id)
            except Exception:
                continue
            lookups_ok += 1
            if found is not None and found.status is not OrderStatus.UNKNOWN:
                return found
        if lookups_ok > 0 and lookups_ok == self.resolve_attempts and found is None:
            # Every lookup worked and the venue has no record: the order never arrived.
            # A ghost fill arriving later is still caught by position reconciliation.
            return OrderResult(OrderStatus.REJECTED, intent.client_id, message="timeout: venue never received the order")
        self.unresolved[intent.client_id] = intent
        self.alerts.alert("critical", "order_unresolved",
                          f"order {intent.client_id} {intent.symbol} {intent.side.value} {intent.qty} has unknown status; new risk is blocked",
                          force=True)
        return res

    def retry_unresolved(self) -> None:
        for cid, intent in list(self.unresolved.items()):
            try:
                found = self.exchange.fetch_order(cid)
            except Exception:
                continue
            if found is None:
                found = OrderResult(OrderStatus.REJECTED, cid, message="timeout: venue never received the order")
            if found.status is not OrderStatus.UNKNOWN:
                self._seen[cid] = found
                self._record(intent, found.status.value, "resolved after timeout")
                self._record_fills(found)
                del self.unresolved[cid]

    def reconcile(self, local_positions: dict[str, float], symbols: list[str]) -> list[str]:
        """Compare signed position sizes. Returns human-readable mismatches (empty = clean)."""
        problems: list[str] = []
        for sym in symbols:
            try:
                venue = self.exchange.open_position_qty(sym)
            except Exception as exc:
                problems.append(f"{sym}: venue position unavailable ({type(exc).__name__})")
                continue
            local = local_positions.get(sym, 0.0)
            if abs(venue - local) > 1e-9:
                problems.append(f"{sym}: venue {venue:g} vs local {local:g}")
        return problems

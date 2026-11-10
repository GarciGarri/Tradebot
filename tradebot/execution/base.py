"""Exchange interface shared by the paper simulator and the ccxt adapter."""
from __future__ import annotations

from typing import Protocol

from ..types import AccountState, Instrument, OrderIntent, OrderResult


class Exchange(Protocol):
    name: str

    def instruments(self) -> dict[str, Instrument]: ...

    def account(self, now_ms: int) -> AccountState: ...

    def submit(self, intent: OrderIntent) -> OrderResult:
        """Send one order. Must be idempotent on `intent.client_id`."""

    def fetch_order(self, client_id: str) -> OrderResult | None:
        """Look an order up by client id. None means the venue has never seen it."""

    def open_position_qty(self, symbol: str) -> float:
        """Signed position as the VENUE reports it (used for reconciliation)."""

    def cancel_all(self) -> None: ...

"""Alerting. Always logs; sends to Telegram only if a bot token and chat id are configured.

Alerts are rate-limited per key so a stuck condition cannot flood the channel, and a failed
send never raises into the trading loop. Message bodies never contain credentials.
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.parse
import urllib.request
from typing import Callable

log = logging.getLogger("tradebot.alerts")


class Alerter:
    def __init__(self, min_interval_s: float = 300.0, sender: Callable[[str], bool] | None = None,
                 clock: Callable[[], float] = time.time):
        self.min_interval_s = min_interval_s
        self._sender = sender
        self._clock = clock
        self._last: dict[str, float] = {}
        self.sent: list[tuple[str, str, str]] = []      # (level, key, message) for tests and status

    def alert(self, level: str, key: str, message: str, force: bool = False) -> bool:
        """Returns True if the alert was emitted (not suppressed by the rate limit)."""
        now = self._clock()
        if not force and now - self._last.get(key, -1e18) < self.min_interval_s:
            return False
        self._last[key] = now
        self.sent.append((level, key, message))
        log.log(logging.ERROR if level in ("critical", "error") else logging.WARNING if level == "warning" else logging.INFO,
                "[%s] %s: %s", level, key, message)
        if self._sender is not None:
            try:
                self._sender(f"[{level.upper()}] {key}\n{message}")
            except Exception as exc:  # never let alerting break trading
                log.error("alert delivery failed: %s", type(exc).__name__)
        return True


def telegram_sender_from_env(env: dict[str, str] | None = None, timeout: float = 10.0) -> Callable[[str], bool] | None:
    """Build a Telegram sender from TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID, or None if unset."""
    env = os.environ if env is None else env
    token, chat = env.get("TELEGRAM_BOT_TOKEN", ""), env.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat:
        return None

    def send(text: str) -> bool:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = urllib.parse.urlencode({"chat_id": chat, "text": text[:3900]}).encode()
        req = urllib.request.Request(url, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return bool(json.load(resp).get("ok"))

    return send

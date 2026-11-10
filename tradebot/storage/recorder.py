"""Persistence: every decision, miss, order, fill, trade, equity point and event.

`SqliteRecorder` is the production store (one file, survives restarts). `MemoryRecorder`
keeps the same records in lists for backtests and tests. Both implement the same methods,
so the trading core never knows which one it is talking to.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS decisions (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER NOT NULL, symbol TEXT NOT NULL, setup_id TEXT,
  policy_version INTEGER, prompt_version TEXT, backend TEXT, latency_ms REAL,
  status TEXT NOT NULL, skip_reason TEXT, snapshot TEXT,
  regime TEXT, direction TEXT, confidence REAL, toxic_flow INTEGER, setup_quality INTEGER, risk_state TEXT,
  code_risk_state TEXT, gate_allow INTEGER, gate_reasons TEXT, p_calibrated REAL, action TEXT,
  price REAL, stop_frac REAL, take_profit_r REAL, max_hold_bars INTEGER,
  outcome INTEGER, outcome_r REAL, labeled INTEGER DEFAULT 0);
CREATE INDEX IF NOT EXISTS decisions_ts ON decisions(ts);
CREATE TABLE IF NOT EXISTS orders (
  client_id TEXT PRIMARY KEY, ts INTEGER, symbol TEXT, side TEXT, qty REAL, reduce_only INTEGER,
  reason TEXT, status TEXT, message TEXT, setup_id TEXT, decision_id INTEGER);
CREATE TABLE IF NOT EXISTS fills (
  id INTEGER PRIMARY KEY AUTOINCREMENT, client_id TEXT, ts INTEGER, symbol TEXT, side TEXT,
  qty REAL, price REAL, fee REAL, reduce_only INTEGER);
CREATE TABLE IF NOT EXISTS trades (
  id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, setup_id TEXT, side TEXT, qty REAL,
  entry_ts INTEGER, entry_price REAL, exit_ts INTEGER, exit_price REAL, pnl REAL, gross_pnl REAL,
  fees REAL, funding REAL, reason TEXT, r_multiple REAL, decision_id INTEGER);
CREATE TABLE IF NOT EXISTS equity (ts INTEGER PRIMARY KEY, equity REAL, cash REAL, gross_notional REAL, drawdown REAL);
CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER, level TEXT, key TEXT, message TEXT);
CREATE TABLE IF NOT EXISTS bars (symbol TEXT, ts INTEGER, open REAL, high REAL, low REAL, close REAL, volume REAL,
  PRIMARY KEY (symbol, ts));
"""


class MemoryRecorder:
    def __init__(self, keep_decisions: bool = True):
        self.keep_decisions = keep_decisions
        self.kv: dict[str, Any] = {}
        self.decisions: list[dict] = []
        self.orders: list[dict] = []
        self.fills: list[dict] = []
        self.trades: list[dict] = []
        self.equity: list[tuple[int, float]] = []
        self.events: list[dict] = []
        self._n = 0

    def get(self, key: str, default: Any = None) -> Any:
        return self.kv.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.kv[key] = value

    def decision(self, row: dict) -> int:
        self._n += 1
        if self.keep_decisions:
            self.decisions.append({"id": self._n, **row})
        return self._n

    def label_decision(self, decision_id: int, outcome: int, outcome_r: float) -> None:
        for d in self.decisions:
            if d["id"] == decision_id:
                d.update(outcome=outcome, outcome_r=outcome_r, labeled=1)
                return

    def order(self, row: dict) -> None:
        self.orders.append(row)

    def fill(self, row: dict) -> None:
        self.fills.append(row)

    def trade(self, row: dict) -> None:
        self.trades.append(row)

    def equity_point(self, ts: int, equity: float, cash: float, gross: float, dd: float) -> None:
        self.equity.append((ts, equity))

    def event(self, ts: int, level: str, key: str, message: str) -> None:
        self.events.append({"ts": ts, "level": level, "key": key, "message": message})

    def bar(self, symbol: str, ts: int, o: float, h: float, l: float, c: float, v: float) -> None:
        return None

    def close(self) -> None:
        return None


class SqliteRecorder:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.executescript("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;" + SCHEMA)
            self._db.commit()

    def _exec(self, sql: str, args: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._db.execute(sql, args)
            self._db.commit()
            return cur

    def _insert(self, table: str, row: dict) -> int:
        cols = ",".join(row)
        marks = ",".join("?" for _ in row)
        return int(self._exec(f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({marks})", tuple(row.values())).lastrowid)

    def query(self, sql: str, args: tuple = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._db.execute(sql, args).fetchall()]

    def get(self, key: str, default: Any = None) -> Any:
        rows = self.query("SELECT value FROM kv WHERE key=?", (key,))
        return json.loads(rows[0]["value"]) if rows else default

    def set(self, key: str, value: Any) -> None:
        self._exec("INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)", (key, json.dumps(value)))

    def decision(self, row: dict) -> int:
        return self._insert("decisions", row)

    def label_decision(self, decision_id: int, outcome: int, outcome_r: float) -> None:
        self._exec("UPDATE decisions SET outcome=?, outcome_r=?, labeled=1 WHERE id=?", (outcome, outcome_r, decision_id))

    def order(self, row: dict) -> None:
        self._insert("orders", row)

    def fill(self, row: dict) -> None:
        self._insert("fills", row)

    def trade(self, row: dict) -> None:
        self._insert("trades", row)

    def equity_point(self, ts: int, equity: float, cash: float, gross: float, dd: float) -> None:
        self._exec("INSERT OR REPLACE INTO equity (ts, equity, cash, gross_notional, drawdown) VALUES (?,?,?,?,?)",
                   (ts, equity, cash, gross, dd))

    def event(self, ts: int, level: str, key: str, message: str) -> None:
        self._exec("INSERT INTO events (ts, level, key, message) VALUES (?,?,?,?)", (ts, level, key, message))

    def bar(self, symbol: str, ts: int, o: float, h: float, l: float, c: float, v: float) -> None:
        self._exec("INSERT OR REPLACE INTO bars (symbol, ts, open, high, low, close, volume) VALUES (?,?,?,?,?,?,?)",
                   (symbol, ts, o, h, l, c, v))

    def close(self) -> None:
        with self._lock:
            self._db.close()


def now_ms() -> int:
    return int(time.time() * 1000)

"""Trading core: one code path for backtests, paper trading and live.

The driver (backtest loop or the 24/7 runtime) owns the clock and the data feed and calls:

* `risk_tick`  - on every tick. Pure code: protective stops and targets, the drawdown kill
  switch, the daily loss halt, unresolved orders, reconciliation. It never calls the REFLEX,
  so stopping risk cannot be delayed by a model.
* `on_candle`  - once per closed candle: code-only exits first, then for each setup one
  snapshot -> REFLEX judgement -> code gate -> code sizing -> hard risk check -> order.

The REFLEX only judges. Every threshold, size, price and side effect is decided here.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from ..alerts.base import Alerter
from ..config import RiskLimits, Settings, timeframe_ms
from ..execution.orders import OrderManager
from ..policy.calibration import Calibrator
from ..policy.gate import evaluate_gate
from ..policy.policy import Policy
from ..policy.sizing import size_position
from ..reflex.base import ReflexRunner
from ..reflex.schema import ReflexDecision, ReflexSkip
from ..risk.manager import AccountAction, Halts, RiskManager
from ..setups.schema import SetupSchema
from ..state.engine import Snapshot, StateEngine
from ..types import (AccountState, Direction, Instrument, OrderIntent, OrderStatus, Position, Regime,
                     RiskState, Side)

log = logging.getLogger("tradebot.core")
DAY_MS = 86_400_000
GLOBAL = "*"


@dataclass(frozen=True)
class PriceRange:
    open: float
    low: float
    high: float


class TradingCore:
    def __init__(self, *, settings: Settings, limits: RiskLimits, policy: Policy,
                 setups: dict[str, SetupSchema], instruments: dict[str, Instrument], engine: StateEngine,
                 runner: ReflexRunner, exchange, recorder, alerts: Alerter):
        self.settings = settings
        self.limits = limits
        self.policy = policy
        self.instruments = instruments
        self.engine = engine
        self.runner = runner
        self.exchange = exchange
        self.recorder = recorder
        self.alerts = alerts
        self.tf_ms = timeframe_ms(settings.exchange.timeframe)
        self.fees = settings.exchange.fees
        self.risk = RiskManager(limits, self.fees)
        self.orders = OrderManager(exchange, self.risk, recorder, alerts)
        self.calibrator = Calibrator(policy.calibration)
        self.keep_rows = getattr(recorder, "keep_decisions", True)

        self.by_symbol: dict[str, SetupSchema] = {}
        for s in setups.values():
            if s.status != "active":
                continue
            if s.symbol not in instruments:
                raise ValueError(f"setup {s.id}: instrument {s.symbol} is not available on the venue")
            if s.symbol in self.by_symbol:
                raise ValueError(f"two active setups for {s.symbol}: {self.by_symbol[s.symbol].id} and {s.id}")
            self.by_symbol[s.symbol] = s

        # persistent, code-owned control state
        self.kill: bool = bool(recorder.get("kill", False))
        self.daily_halt_day: int = int(recorder.get("daily_halt_day", -1))
        self.pause_until: dict[str, int] = dict(recorder.get("pause_until", {}))
        self.reconcile_block: bool = False
        self.counters: dict[str, int] = {}

    # ------------------------------------------------------------------ helpers
    def _count(self, key: str) -> None:
        self.counters[key] = self.counters.get(key, 0) + 1

    def account(self, now_ms: int) -> AccountState:
        return self.exchange.account(now_ms)

    def paused(self, symbol: str, now_ms: int) -> bool:
        return now_ms < max(self.pause_until.get(GLOBAL, 0), self.pause_until.get(symbol, 0))

    def halts(self, symbol: str, now_ms: int) -> Halts:
        return Halts(kill=self.kill, daily_halt=self.daily_halt_day == now_ms // DAY_MS,
                     paused=self.paused(symbol, now_ms),
                     reconcile_block=self.reconcile_block or self.orders.blocked)

    def _event(self, now_ms: int, level: str, key: str, message: str, force: bool = False) -> None:
        self.recorder.event(now_ms, level, key, message)
        self.alerts.alert(level, key, message, force=force)

    # ------------------------------------------------------------------ closing
    def _close(self, pos: Position, now_ms: int, reason: str, mark: float | None,
               trigger_price: float | None = None, passive: bool = False) -> bool:
        inst = self.instruments[pos.symbol]
        intent = OrderIntent(
            client_id=f"c:{inst.base}:{now_ms}:{reason}", symbol=pos.symbol, side=pos.side.opposite,
            qty=abs(pos.qty), reduce_only=True, reason=reason, ts=now_ms, setup_id=pos.setup_id,
            ref_price=trigger_price or mark, decision_id=pos.decision_id)
        account = self.account(now_ms)
        res = self.orders.execute(intent, account, inst, mark, None, True, self.halts(pos.symbol, now_ms),
                                  trigger_price=trigger_price, passive=passive)
        if res.status in (OrderStatus.FILLED, OrderStatus.PARTIAL):
            self._count(f"exit_{reason}")
            trade = getattr(self.exchange, "last_closed", None)
            if trade is not None:
                self.recorder.trade({
                    "symbol": trade.symbol, "setup_id": trade.setup_id, "side": trade.side.value, "qty": trade.qty,
                    "entry_ts": trade.entry_ts, "entry_price": trade.entry_price, "exit_ts": trade.exit_ts,
                    "exit_price": trade.exit_price, "pnl": trade.pnl, "gross_pnl": trade.gross_pnl,
                    "fees": trade.fees, "funding": trade.funding, "reason": trade.reason,
                    "r_multiple": trade.r_multiple, "decision_id": trade.decision_id})
            return True
        self._event(now_ms, "error", "close_failed", f"{pos.symbol} close ({reason}) failed: {res.message}")
        return False

    def flatten(self, now_ms: int, reason: str, marks: dict[str, float] | None = None) -> None:
        account = self.account(now_ms)
        for sym, pos in list(account.positions.items()):
            self._close(pos, now_ms, reason, (marks or {}).get(sym))

    # ------------------------------------------------------------------ risk tick (no model)
    def risk_tick(self, now_ms: int, ranges: dict[str, PriceRange]) -> None:
        """Protective exits and account-level limits. `ranges` holds what price did since the
        last tick: a whole bar in backtests, a single price in live operation."""
        account = self.account(now_ms)
        for sym, pos in list(account.positions.items()):
            rng = ranges.get(sym)
            if rng is None:
                continue
            long = pos.qty > 0
            stop_hit = rng.low <= pos.stop_price if long else rng.high >= pos.stop_price
            tp = pos.take_profit
            tp_hit = tp is not None and (rng.high >= tp if long else rng.low <= tp)
            if stop_hit:   # if both levels sit inside one bar, assume the stop came first
                trigger = min(pos.stop_price, rng.open) if long else max(pos.stop_price, rng.open)
                self._close(pos, now_ms, "stop", rng.open, trigger_price=trigger)
            elif tp_hit:
                trigger = max(tp, rng.open) if long else min(tp, rng.open)
                self._close(pos, now_ms, "take_profit", rng.open, trigger_price=trigger, passive=True)

        account = self.account(now_ms)
        action = self.risk.account_action(account)
        marks = {s: r.open for s, r in ranges.items()}
        if action is AccountAction.KILL and not self.kill:
            self.kill = True
            self.recorder.set("kill", True)
            self.flatten(now_ms, "kill_switch", marks)
            self._event(now_ms, "critical", "kill_switch",
                        f"drawdown {account.drawdown:.1%} reached the kill level "
                        f"{self.limits.kill_switch_drawdown:.0%}: flattened and halted. Manual resume required.",
                        force=True)
        elif action is AccountAction.DAILY_HALT and self.daily_halt_day != now_ms // DAY_MS:
            self.daily_halt_day = now_ms // DAY_MS
            self.recorder.set("daily_halt_day", self.daily_halt_day)
            self.flatten(now_ms, "daily_loss_limit", marks)
            self._event(now_ms, "critical", "daily_loss_limit",
                        f"day PnL {account.day_pnl_frac:.1%} hit the daily loss limit "
                        f"{self.limits.max_daily_loss:.0%}: flattened, no new risk until the next UTC day.",
                        force=True)

        if self.orders.unresolved:
            self.orders.retry_unresolved()

    def reconcile(self, now_ms: int) -> list[str]:
        account = self.account(now_ms)
        local = {s: p.qty for s, p in account.positions.items()}
        symbols = sorted(set(self.by_symbol) | set(local))
        problems = self.orders.reconcile(local, symbols)
        if problems and not self.reconcile_block:
            self._event(now_ms, "critical", "reconcile_mismatch", "; ".join(problems), force=True)
        self.reconcile_block = bool(problems)
        return problems

    # ------------------------------------------------------------------ candle
    def on_candle(self, now_ms: int, exchange_time_ms: int | None = None) -> None:
        account = self.account(now_ms)
        # 1) code-only exits: they need no model and run before any model call
        for sym, pos in list(account.positions.items()):
            setup = self.by_symbol.get(sym)
            last = self.engine.store.last_bar(sym, now_ms)
            price = last.close if last else None
            exits = self._exit_params(setup)
            held_bars = (now_ms - pos.entry_bar_ts) // self.tf_ms
            if setup is None:
                self._close(pos, now_ms, "setup_removed", price)
            elif price is not None and setup.invalidated(price):
                self._close(pos, now_ms, "thesis_invalidated", price)
                self._event(now_ms, "warning", f"invalidated:{setup.id}",
                            f"{setup.id}: price {price:g} crossed the invalidation level; position closed")
            elif setup.expired(now_ms):
                self._close(pos, now_ms, "setup_expired", price)
            elif held_bars >= exits["max_hold_bars"]:
                self._close(pos, now_ms, "time_stop", price)

        # 2) one judgement per active setup
        for sym in self.by_symbol:
            self._decide(sym, now_ms, exchange_time_ms)

        account = self.account(now_ms)
        self.recorder.equity_point(now_ms, account.equity, account.cash, account.gross_notional, account.drawdown)

    def _exit_params(self, setup: SetupSchema | None) -> dict:
        e = self.policy.exits
        return {
            "stop_atr_mult": (setup.exits.stop_atr_mult if setup and setup.exits.stop_atr_mult else e.stop_atr_mult),
            "take_profit_r": (setup.exits.take_profit_r if setup and setup.exits.take_profit_r else e.take_profit_r),
            "max_hold_bars": (setup.exits.max_hold_bars if setup and setup.exits.max_hold_bars else e.max_hold_bars),
        }

    def _row(self, snap: Snapshot, setup: SetupSchema, **extra) -> dict:
        row = {
            "ts": snap.ts, "symbol": snap.symbol, "setup_id": setup.id, "policy_version": self.policy.version,
            "prompt_version": self.policy.prompt_version, "backend": self.runner.reflex.name,
            "snapshot": snap.text if self.keep_rows else None, "price": snap.price,
            "code_risk_state": snap.code_risk_state.value,
        }
        row.update(extra)
        return row

    def _pause_rules(self, dec: ReflexDecision, symbol: str, pos: Position | None, now_ms: int) -> None:
        """Pause new risk and escalate. Code decides; the model only supplied the reading."""
        out = dec.output
        until = now_ms + self.policy.pause.cooldown_bars * self.tf_ms
        if out.regime is Regime.CRISIS:
            newly = now_ms >= self.pause_until.get(GLOBAL, 0)
            self.pause_until[GLOBAL] = max(self.pause_until.get(GLOBAL, 0), until)
            self.recorder.set("pause_until", self.pause_until)
            if not newly:
                return
            self._count("pause_crisis")
            self._event(now_ms, "critical", "escalate:crisis",
                        f"REFLEX reads regime=crisis on {symbol}. New risk is paused on every symbol for "
                        f"{self.policy.pause.cooldown_bars} bars. Existing stops stay in force.")
        elif pos is not None:
            pos_dir = Direction.LONG if pos.qty > 0 else Direction.SHORT
            support = out.direction_confidence if out.direction is pos_dir else 0.0
            if support < self.policy.pause.confidence_floor:
                newly = not self.paused(symbol, now_ms)
                self.pause_until[symbol] = max(self.pause_until.get(symbol, 0), until)
                self.recorder.set("pause_until", self.pause_until)
                if newly:   # escalate once per pause, not on every candle of it
                    self._count("pause_low_confidence")
                    self._event(now_ms, "warning", f"escalate:confidence:{symbol}",
                                f"REFLEX support for the open {pos_dir.value} in {symbol} is {support:.2f}, below "
                                f"the {self.policy.pause.confidence_floor:.2f} floor. No new risk in {symbol} for "
                                f"{self.policy.pause.cooldown_bars} bars.")

    def _decide(self, symbol: str, now_ms: int, exchange_time_ms: int | None) -> None:
        setup = self.by_symbol[symbol]
        inst = self.instruments[symbol]
        account = self.account(now_ms)
        snap = self.engine.snapshot(symbol, now_ms, account, setup.view(now_ms), exchange_time_ms, inst.contract_size)
        if not snap.ok:
            self._count(f"skip_{snap.status.value}")
            self.recorder.decision(self._row(snap, setup, status="skip", skip_reason=snap.status.value))
            return
        res = self.runner.run(snap)
        if isinstance(res, ReflexSkip):
            # The candle is skipped: no entry, no retry. Stops and limits still ran in risk_tick.
            self._count(f"skip_{res.reason}")
            self.recorder.decision(self._row(snap, setup, status="skip", skip_reason=res.reason,
                                             latency_ms=res.latency_ms))
            if res.reason != "snapshot_not_ok":
                self._event(now_ms, "warning", f"reflex_skip:{res.reason}",
                            f"{symbol}: candle skipped ({res.reason}) {res.detail}")
            return

        out = res.output
        pos = account.positions.get(symbol)
        self._pause_rules(res, symbol, pos, now_ms)
        exits = self._exit_params(setup)
        base = dict(status="decision", latency_ms=res.latency_ms, regime=out.regime.value,
                    direction=out.direction.value, confidence=out.direction_confidence,
                    toxic_flow=None if out.toxic_flow is None else int(out.toxic_flow),
                    setup_quality=out.setup_quality, risk_state=out.risk_state.value,
                    take_profit_r=exits["take_profit_r"], max_hold_bars=exits["max_hold_bars"])
        stop_frac = None
        if snap.atr_pct is not None:
            e = self.policy.exits
            stop_frac = min(e.max_stop_pct, max(e.min_stop_pct, exits["stop_atr_mult"] * snap.atr_pct)) / 100.0
        base["stop_frac"] = stop_frac

        # ---- an open position: the REFLEX can only take risk off
        if pos is not None:
            pos_dir = Direction.LONG if pos.qty > 0 else Direction.SHORT
            action = "hold"
            flipped = (out.direction not in (pos_dir, Direction.NEUTRAL)
                       and out.direction_confidence >= self.policy.exits.flip_exit_confidence)
            losing = (snap.price - pos.entry_price) * (1 if pos.qty > 0 else -1) < 0
            if flipped:
                action = "exit:signal_flip" if self._close(pos, now_ms, "signal_flip", snap.price) else "exit_failed"
            elif snap.code_risk_state is RiskState.REDUCE and losing:
                action = "exit:risk_reduce" if self._close(pos, now_ms, "risk_reduce", snap.price) else "exit_failed"
            self.recorder.decision(self._row(snap, setup, **base, gate_allow=0, gate_reasons="in_position", action=action))
            return

        # ---- flat: gate -> size -> hard risk check -> order
        dd_usage = account.drawdown / self.limits.kill_switch_drawdown
        if dd_usage >= self.limits.reduce_frac:
            # Not silent: this close to the kill level no new risk is opened, so the account
            # cannot trade its way back. A human has to review and resume.
            self.alerts.alert("critical", f"drawdown_soft_halt:{now_ms // DAY_MS}",
                              f"drawdown {account.drawdown:.1%} is within {1 - self.limits.reduce_frac:.0%} of the kill "
                              f"level ({self.limits.kill_switch_drawdown:.0%}). New risk is blocked until a human "
                              "reviews and runs `tradebot resume --reset-peak`.")
        gate = evaluate_gate(out, self.policy, setup, snap.code_risk_state, now_ms, snap.price,
                             self.paused(symbol, now_ms))
        if not gate.allow:
            self._count("gate_blocked")
            self.recorder.decision(self._row(snap, setup, **base, gate_allow=0,
                                             gate_reasons=",".join(gate.reasons), action="none"))
            return
        p = self.calibrator.prob(out.direction_confidence)
        ref = snap.mid if snap.mid is not None else snap.price
        # deep in drawdown: keep trading (so recovery is possible) but with a smaller risk budget
        in_caution = account.drawdown / self.limits.kill_switch_drawdown >= self.limits.near_limit_frac
        risk_scale = self.limits.drawdown_caution_scale if in_caution else 1.0
        if in_caution:
            self.alerts.alert("warning", "drawdown_caution",
                              f"drawdown {account.drawdown:.1%} is past {self.limits.near_limit_frac:.0%} of the kill "
                              f"level: risk per trade scaled by {risk_scale:g}")
        if stop_frac is None:
            self.recorder.decision(self._row(snap, setup, **base, gate_allow=1, gate_reasons="", p_calibrated=p,
                                             action="none:no_atr"))
            return
        size = size_position(
            p=p, stop_frac=stop_frac, take_profit_r=exits["take_profit_r"], equity=account.equity, price=ref,
            instrument=inst, limits=self.limits, fees=self.fees, kelly_fraction=self.policy.sizing.kelly_fraction,
            gross_notional=account.gross_notional, bar_dollar_volume=snap.bar_dollar_volume,
            hold_bars=min(24, exits["max_hold_bars"]), risk_scale=risk_scale)
        if size.qty <= 0:
            self._count("size_zero")
            self.recorder.decision(self._row(snap, setup, **base, gate_allow=1, gate_reasons="", p_calibrated=p,
                                             action="none:" + ",".join(size.reasons)))
            return

        long = out.direction is Direction.LONG
        stop_price = inst.round_price(ref * (1 - stop_frac) if long else ref * (1 + stop_frac))
        tp_dist = exits["take_profit_r"] * stop_frac
        take_profit = inst.round_price(ref * (1 + tp_dist) if long else ref * (1 - tp_dist))
        decision_id = self.recorder.decision(self._row(snap, setup, **base, gate_allow=1, gate_reasons="",
                                                       p_calibrated=p, action="enter"))
        intent = OrderIntent(
            client_id=f"e:{inst.base}:{now_ms}", symbol=symbol, side=Side.BUY if long else Side.SELL, qty=size.qty,
            reduce_only=False, reason="entry", ts=now_ms, setup_id=setup.id, stop_price=stop_price,
            take_profit=take_profit, ref_price=ref, decision_id=decision_id)
        result = self.orders.execute(intent, account, inst, snap.price, snap.bar_dollar_volume, snap.ok,
                                     self.halts(symbol, now_ms))
        if result.status in (OrderStatus.FILLED, OrderStatus.PARTIAL):
            self._count("entries")
        else:
            self._count("entry_rejected")
            self._event(now_ms, "warning", "entry_rejected", f"{symbol}: {result.message}")

"""Executable risk guardrails for Plan B+."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Literal

from src.config.settings import Settings
from src.config.tokens import assert_tradable_symbol


@dataclass(frozen=True)
class TradeRecord:
    """Recorded trade used for daily limits and realized PnL tracking."""

    symbol: str
    side: Literal["buy", "sell"]
    value_usdc: float
    realized_pnl_usdc: float
    timestamp: datetime


class RiskState(Enum):
    NORMAL = "normal"
    REDUCED_RISK = "reduced_risk"
    PAUSED_STREAK = "paused_streak"
    PAUSED_DAILY = "paused_daily"
    KILL_SWITCH = "kill_switch"


@dataclass(frozen=True)
class RiskDecision:
    state: RiskState
    allow_new_entries: bool
    position_multiplier: float
    max_slippage_pct: float
    max_daily_trades: int
    base_risk_per_trade_pct: float
    reasons: list[str]


class Guardrails:
    """Enforce non-negotiable trading limits."""

    def __init__(self, settings: Settings, state_path: str | Path | None = None) -> None:
        self.settings = settings
        self.state_path = Path(state_path or settings.guardrail_state_path)
        self.trade_records: list[TradeRecord] = []
        self._daily_date = self._now().date()
        self._daily_trade_count = 0
        self._daily_realized_loss_usdc = 0.0
        self._paused_until: datetime | None = None
        self._all_time_high_usdc = 0.0
        self._kill_switch = False
        self._state = RiskState.NORMAL
        self._daily_loss_pct = 0.0
        self._loss_streak = 0
        self._last_trade_day: datetime | None = None
        self._load_state()
        self._reset_daily_if_needed()

    def validate_new_trade(
        self,
        symbol: str,
        position_value_usdc: float,
        portfolio_value_usdc: float,
        estimated_slippage_pct: float,
    ) -> None:
        """Raise if a new trade violates the configured guardrails."""

        self._reset_daily_if_needed()
        assert_tradable_symbol(symbol)
        if self._kill_switch:
            raise RuntimeError("drawdown kill switch is active")
        if not self.can_open_new_trade():
            seconds = self.seconds_until_trading_resumes()
            raise RuntimeError(f"new trades are paused for {seconds} more seconds")
        max_position_value = portfolio_value_usdc * self.settings.max_position_pct
        if position_value_usdc > max_position_value:
            raise ValueError(
                f"position value {position_value_usdc:.2f} exceeds max {max_position_value:.2f}"
            )
        if estimated_slippage_pct < 0:
            raise ValueError("estimated slippage must not be negative")
        if estimated_slippage_pct > self.settings.max_slippage_pct:
            raise ValueError(
                f"estimated slippage {estimated_slippage_pct:.4f} exceeds cap {self.settings.max_slippage_pct:.4f}"
            )
        max_loss = portfolio_value_usdc * self.settings.max_daily_loss_pct
        if self._daily_realized_loss_usdc >= max_loss and max_loss > 0:
            raise RuntimeError("daily realized loss limit has been reached")

    def record_trade(self, record: TradeRecord, portfolio_value_usdc: float) -> None:
        """Record a trade and update daily counters."""

        self._reset_daily_if_needed(record.timestamp)
        assert_tradable_symbol(record.symbol)
        self.trade_records.append(record)
        if record.side == "buy":
            self._daily_trade_count += 1
        if record.realized_pnl_usdc < 0:
            self._daily_realized_loss_usdc += abs(record.realized_pnl_usdc)
        max_loss = portfolio_value_usdc * self.settings.max_daily_loss_pct
        if self._daily_realized_loss_usdc >= max_loss and max_loss > 0:
            self._paused_until = self._now() + timedelta(hours=24)
        self._save_state()

    def update_ath(self, portfolio_value_usdc: float) -> None:
        if portfolio_value_usdc > self._all_time_high_usdc:
            self._all_time_high_usdc = portfolio_value_usdc
            self._save_state()

    def evaluate(
        self,
        portfolio_value: float,
        regime_result: object,
        volatility_breaker: bool | None = None,
    ) -> RiskDecision:
        self._reset_daily_if_needed()
        self._reset_recorded_loss_day_if_needed()
        self.update_ath(portfolio_value)
        drawdown = self._drawdown_pct(portfolio_value)
        reasons: list[str] = []
        inferred_breaker = "volatility_breaker_reported" in getattr(regime_result, "reasons", [])
        has_breaker = inferred_breaker if volatility_breaker is None else volatility_breaker

        state = RiskState.NORMAL
        if drawdown >= self.settings.drawdown_kill_switch_pct:
            state = RiskState.KILL_SWITCH
            reasons.append("drawdown_kill_switch")
            self._kill_switch = True
        elif self._daily_loss_limit_hit(portfolio_value):
            state = RiskState.PAUSED_DAILY
            reasons.append("daily_loss_limit")
        elif self._loss_streak >= self.settings.loss_streak_pause:
            state = RiskState.PAUSED_STREAK
            reasons.append("loss_streak_pause")
        else:
            reduced_reasons = self._reduced_risk_reasons(drawdown, regime_result, has_breaker)
            if reduced_reasons:
                state = RiskState.REDUCED_RISK
                reasons.extend(reduced_reasons)

        self._state = state
        self._save_state()
        return self._risk_decision(state, reasons)

    def record_trade_result(self, realized_pnl_pct: float) -> None:
        now = self._now()
        self._reset_recorded_loss_day_if_needed(now)
        self._last_trade_day = now
        if realized_pnl_pct < 0:
            self._daily_loss_pct += abs(realized_pnl_pct)
        if realized_pnl_pct < -0.005:
            self._loss_streak += 1
        elif realized_pnl_pct > 0:
            # Only a profitable exit resets the streak. Entry bookkeeping calls
            # this with exactly 0.0 and must not wipe a loss streak built from
            # real exits, otherwise the streak pause can never trigger.
            self._loss_streak = 0
        self._save_state()

    def record_compliance_trade(self) -> None:
        """Count a non-directional compliance trade (e.g. stable-to-stable swap).

        Stable symbols are settlement tokens and fail ``assert_tradable_symbol``,
        but a tiny stable swap is still an on-chain trade for the competition's
        one-trade-per-day minimum, so it increments the daily counter directly.
        """

        self._reset_daily_if_needed()
        self._daily_trade_count += 1
        self._save_state()

    def can_open_new_trade(self) -> bool:
        """Return whether a new position may be opened now."""

        self._reset_daily_if_needed()
        if self._kill_switch:
            return False
        if self._paused_until is not None and self._paused_until > self._now():
            return False
        return self._daily_trade_count < self.settings.max_daily_trades

    def update_portfolio_value(self, portfolio_value_usdc: float) -> bool:
        """Update all-time high tracking and return whether the kill switch is active."""

        if portfolio_value_usdc <= 0:
            return self._kill_switch
        if portfolio_value_usdc > self._all_time_high_usdc:
            self._all_time_high_usdc = portfolio_value_usdc
            self._save_state()
        drawdown_trigger = self._all_time_high_usdc * (1 - self.settings.drawdown_kill_switch_pct)
        if self._all_time_high_usdc > 0 and portfolio_value_usdc <= drawdown_trigger:
            self._kill_switch = True
        return self._kill_switch

    def should_kill_switch(self) -> bool:
        """Return whether the drawdown kill switch has fired."""

        return self._kill_switch

    def seconds_until_trading_resumes(self) -> int:
        """Return seconds remaining in the daily-loss pause window."""

        if self._paused_until is None:
            return 0
        remaining = self._paused_until - self._now()
        return max(0, int(remaining.total_seconds()))

    def _reset_daily_if_needed(self, current_time: datetime | None = None) -> None:
        now = current_time or self._now()
        if now.date() == self._daily_date:
            self._reset_recorded_loss_day_if_needed(now)
            return
        self._daily_date = now.date()
        self._daily_trade_count = 0
        self._daily_realized_loss_usdc = 0.0
        self._daily_loss_pct = 0.0
        if self._paused_until is not None and self._paused_until <= now:
            self._paused_until = None
        self._save_state()

    def _load_state(self) -> None:
        if not self.state_path.exists():
            self._save_state()
            return
        with self.state_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid guardrail state file: {self.state_path}")

        self._daily_trade_count = int(payload.get("daily_trade_count", 0))
        self._daily_realized_loss_usdc = float(payload.get("daily_realized_loss", 0.0))
        self._all_time_high_usdc = float(payload.get("portfolio_ath", 0.0))
        self._daily_date = self._date_from_state(payload.get("last_reset_date"))
        self._daily_loss_pct = float(payload.get("daily_loss_pct", 0.0))
        self._loss_streak = int(payload.get("loss_streak", 0))
        self._kill_switch = bool(payload.get("kill_switch", False))
        last_trade_day = payload.get("last_trade_day")
        if last_trade_day:
            parsed_day = datetime.fromisoformat(str(last_trade_day))
            self._last_trade_day = parsed_day if parsed_day.tzinfo else parsed_day.replace(tzinfo=timezone.utc)

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "daily_trade_count": self._daily_trade_count,
            "daily_realized_loss": self._daily_realized_loss_usdc,
            "daily_loss_pct": self._daily_loss_pct,
            "loss_streak": self._loss_streak,
            "kill_switch": self._kill_switch,
            "last_trade_day": self._last_trade_day.isoformat() if self._last_trade_day else None,
            "portfolio_ath": self._all_time_high_usdc,
            "last_reset_date": self._daily_date.isoformat(),
        }
        with self.state_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    def _date_from_state(self, raw_value: object) -> date:
        if raw_value is None:
            return self._now().date()
        try:
            return date.fromisoformat(str(raw_value))
        except ValueError as exc:
            raise ValueError(f"Invalid last_reset_date in {self.state_path}") from exc

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    def _drawdown_pct(self, portfolio_value: float) -> float:
        if self._all_time_high_usdc <= 0:
            return 0.0
        return max(0.0, (self._all_time_high_usdc - portfolio_value) / self._all_time_high_usdc)

    def _daily_loss_limit_hit(self, portfolio_value: float) -> bool:
        if self._daily_loss_pct >= self.settings.max_daily_loss_pct > 0:
            return True
        max_loss_usdc = portfolio_value * self.settings.max_daily_loss_pct
        return max_loss_usdc > 0 and self._daily_realized_loss_usdc >= max_loss_usdc

    def _reduced_risk_reasons(
        self,
        drawdown: float,
        regime_result: object,
        volatility_breaker: bool,
    ) -> list[str]:
        reasons: list[str] = []
        if drawdown >= self.settings.drawdown_soft_stop_pct:
            reasons.append("drawdown_soft_stop")
        if self._loss_streak >= self.settings.loss_streak_reduce_size:
            reasons.append("loss_streak_reduce")
        fragility = getattr(regime_result, "sentiment_fragility", "NONE")
        if fragility in {"CROWDED_LONG", "EXTREME_GREED", "GAS_FOMO"}:
            reasons.append(f"sentiment_{fragility}")
        if volatility_breaker:
            reasons.append("volatility_breaker")
        return reasons

    def _risk_decision(self, state: RiskState, reasons: list[str]) -> RiskDecision:
        base_risk = self.settings.base_risk_per_trade_pct
        strict_slippage = min(self.settings.max_slippage_pct, self.settings.risk_off_max_slippage_pct)
        if state == RiskState.NORMAL:
            return RiskDecision(state, True, 1.0, self.settings.max_slippage_pct, self.settings.max_daily_trades, base_risk, reasons)
        if state == RiskState.REDUCED_RISK:
            return RiskDecision(state, True, 0.5, strict_slippage, 1, base_risk * 0.5, reasons)
        return RiskDecision(state, False, 0.0, strict_slippage, 0, 0.0, reasons)

    def _reset_recorded_loss_day_if_needed(self, current_time: datetime | None = None) -> None:
        now = current_time or self._now()
        if self._last_trade_day is not None and self._last_trade_day.date() != now.date():
            self._daily_loss_pct = 0.0

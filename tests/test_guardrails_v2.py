"""Tests for v2 risk state machine behavior."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.config.settings import Settings
from src.strategy.guardrails import Guardrails, RiskState
from src.strategy.regime_detector import MarketRegime, RegimeResult


def _settings(tmp_path: Path) -> Settings:
    return Settings(guardrail_state_path=str(tmp_path / "guardrail_state.json"))


def _regime(fragility: str = "NONE", reasons: list[str] | None = None) -> RegimeResult:
    return RegimeResult(
        regime=MarketRegime.TRENDING_UP,
        score=4.0,
        reasons=reasons or [],
        position_multiplier=1.0,
        min_entry_factors=4,
        max_slippage_pct=0.01,
        sentiment_delta=0.0,
        sentiment_fragility=fragility,
    )


def test_kill_switch_at_18_percent_drawdown(tmp_path: Path) -> None:
    guardrails = Guardrails(_settings(tmp_path))
    guardrails.update_ath(100.0)
    decision = guardrails.evaluate(82.0, _regime())
    assert decision.state == RiskState.KILL_SWITCH
    assert decision.allow_new_entries is False


def test_reduced_risk_at_10_percent_drawdown(tmp_path: Path) -> None:
    guardrails = Guardrails(_settings(tmp_path))
    guardrails.update_ath(100.0)
    decision = guardrails.evaluate(89.0, _regime())
    assert decision.state == RiskState.REDUCED_RISK
    assert decision.position_multiplier == 0.5


def test_sentiment_extreme_greed_reduces_to_reduced_risk(tmp_path: Path) -> None:
    decision = Guardrails(_settings(tmp_path)).evaluate(100.0, _regime("EXTREME_GREED"))
    assert decision.state == RiskState.REDUCED_RISK
    assert decision.position_multiplier == 0.5


def test_sentiment_crowded_long_reduces_to_reduced_risk(tmp_path: Path) -> None:
    decision = Guardrails(_settings(tmp_path)).evaluate(100.0, _regime("CROWDED_LONG"))
    assert decision.state == RiskState.REDUCED_RISK


def test_sentiment_none_keeps_normal(tmp_path: Path) -> None:
    decision = Guardrails(_settings(tmp_path)).evaluate(100.0, _regime("NONE"))
    assert decision.state == RiskState.NORMAL


def test_paused_streak_after_three_losses(tmp_path: Path) -> None:
    guardrails = Guardrails(_settings(tmp_path))
    guardrails.record_trade_result(-0.006)
    guardrails.record_trade_result(-0.006)
    guardrails.record_trade_result(-0.006)
    decision = guardrails.evaluate(100.0, _regime())
    assert decision.state == RiskState.PAUSED_STREAK
    assert not decision.allow_new_entries


def test_daily_loss_reset_on_new_day(tmp_path: Path) -> None:
    guardrails = Guardrails(_settings(tmp_path))
    guardrails.record_trade_result(-0.04)
    guardrails._last_trade_day = datetime.now(timezone.utc) - timedelta(days=1)
    decision = guardrails.evaluate(100.0, _regime())
    assert decision.state == RiskState.NORMAL


def test_precedence_kill_over_daily(tmp_path: Path) -> None:
    guardrails = Guardrails(_settings(tmp_path))
    guardrails.update_ath(100.0)
    guardrails.record_trade_result(-0.04)
    decision = guardrails.evaluate(82.0, _regime())
    assert decision.state == RiskState.KILL_SWITCH


def test_volatility_circuit_breaker_reduces_size(tmp_path: Path) -> None:
    decision = Guardrails(_settings(tmp_path)).evaluate(100.0, _regime(), volatility_breaker=True)
    assert decision.state == RiskState.REDUCED_RISK


def test_reduced_risk_uses_stricter_slippage(tmp_path: Path) -> None:
    decision = Guardrails(_settings(tmp_path)).evaluate(100.0, _regime("GAS_FOMO"))
    assert decision.max_slippage_pct == 0.005

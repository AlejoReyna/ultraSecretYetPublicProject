"""Tests for risk guardrails."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.config.settings import Settings
from src.strategy.guardrails import Guardrails, TradeRecord


def _settings(tmp_path: Path) -> Settings:
    return Settings(guardrail_state_path=str(tmp_path / "guardrail_state.json"))


def test_rejects_symbol_not_in_target_allowlist(tmp_path: Path) -> None:
    guardrails = Guardrails(_settings(tmp_path))
    with pytest.raises(ValueError):
        guardrails.validate_new_trade("BNB", 100.0, 10000.0, 0.001)


def test_rejects_stablecoin_as_directional_trade(tmp_path: Path) -> None:
    guardrails = Guardrails(_settings(tmp_path))
    with pytest.raises(ValueError):
        guardrails.validate_new_trade("USDC", 100.0, 10000.0, 0.001)


def test_rejects_position_over_five_percent(tmp_path: Path) -> None:
    guardrails = Guardrails(_settings(tmp_path))
    with pytest.raises(ValueError):
        guardrails.validate_new_trade("CAKE", 501.0, 10000.0, 0.001)


def test_rejects_slippage_over_one_percent(tmp_path: Path) -> None:
    guardrails = Guardrails(_settings(tmp_path))
    with pytest.raises(ValueError):
        guardrails.validate_new_trade("CAKE", 100.0, 10000.0, 0.011)


def test_rejects_negative_slippage(tmp_path: Path) -> None:
    guardrails = Guardrails(_settings(tmp_path))
    with pytest.raises(ValueError, match="slippage"):
        guardrails.validate_new_trade("CAKE", 100.0, 10000.0, -0.001)


def test_accepts_zero_slippage_from_dex_price_impact(tmp_path: Path) -> None:
    guardrails = Guardrails(_settings(tmp_path))
    guardrails.validate_new_trade("CAKE", 100.0, 10000.0, 0.0)


def test_max_daily_trades_blocks_fourth_trade(tmp_path: Path) -> None:
    guardrails = Guardrails(_settings(tmp_path))
    now = datetime.now(timezone.utc)
    for _ in range(3):
        guardrails.record_trade(
            TradeRecord("CAKE", "buy", 100.0, 0.0, now),
            portfolio_value_usdc=10000.0,
        )
    assert guardrails.can_open_new_trade() is False
    with pytest.raises(RuntimeError):
        guardrails.validate_new_trade("CAKE", 100.0, 10000.0, 0.001)


def test_drawdown_kill_switch_triggers_at_eighteen_percent(tmp_path: Path) -> None:
    guardrails = Guardrails(_settings(tmp_path))
    assert guardrails.update_portfolio_value(10000.0) is False
    assert guardrails.update_portfolio_value(8200.0) is True
    assert guardrails.should_kill_switch() is True


def test_guardrail_state_initializes_and_persists(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    state_path = Path(settings.guardrail_state_path)

    guardrails = Guardrails(settings)
    assert state_path.exists()

    guardrails.record_trade(
        TradeRecord("CAKE", "buy", 100.0, 0.0, datetime.now(timezone.utc)),
        portfolio_value_usdc=10000.0,
    )
    guardrails.record_trade(
        TradeRecord("CAKE", "sell", 90.0, -10.0, datetime.now(timezone.utc)),
        portfolio_value_usdc=10000.0,
    )
    guardrails.update_portfolio_value(12345.0)

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["daily_trade_count"] == 1
    assert payload["daily_realized_loss"] == 10.0
    assert payload["portfolio_ath"] == 12345.0
    assert payload["last_reset_date"] == datetime.now(timezone.utc).date().isoformat()

    reloaded = Guardrails(settings)
    with pytest.raises(RuntimeError, match="daily realized loss"):
        reloaded.validate_new_trade("CAKE", 1.0, 100.0, 0.001)


def test_guardrail_state_resets_daily_counts_after_date_change(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    state_path = Path(settings.guardrail_state_path)
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat()
    state_path.write_text(
        json.dumps(
            {
                "daily_trade_count": 3,
                "daily_realized_loss": 50.0,
                "portfolio_ath": 10000.0,
                "last_reset_date": yesterday,
            }
        ),
        encoding="utf-8",
    )

    guardrails = Guardrails(settings)

    assert guardrails.can_open_new_trade() is True
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["daily_trade_count"] == 0
    assert payload["daily_realized_loss"] == 0.0
    assert payload["portfolio_ath"] == 10000.0

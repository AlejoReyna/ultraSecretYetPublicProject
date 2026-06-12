"""Tests for decision logging from the trading loop."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src import main as main_module
from src.config.settings import Settings


def _settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "paper_trade": True,
        "position_state_path": str(tmp_path / "positions.json"),
        "guardrail_state_path": str(tmp_path / "guardrail_state.json"),
        "execution_log_path": str(tmp_path / "execution_log.jsonl"),
        "decision_log_path": str(tmp_path / "decision_log.jsonl"),
        # No health server in tests: a real bind on a fixed port collides
        # across run_agent invocations within one pytest process.
        "health_check_port": 0,
    }
    values.update(overrides)
    return Settings(**values)


def _patch_run_agent_dependencies(
    monkeypatch: Any,
    *,
    portfolio_value: float = 10000.0,
    decision: main_module.BreakoutDecision | None = None,
) -> None:
    class FakeToolkit:
        def __init__(self, settings: Settings) -> None:
            self.settings = settings

        def get_balance(self, symbol: str) -> dict[str, object]:
            return {"portfolio_value_usdc": portfolio_value, "symbol": symbol}

    class FakeTWAK:
        def __init__(self, paper_trade: bool = False) -> None:
            self.paper_trade = paper_trade

    class FakeRouter:
        def __init__(self, twak_interface: FakeTWAK) -> None:
            self.twak_interface = twak_interface

    class FakeEngine:
        def __init__(self, settings: Settings, twak_interface: FakeTWAK) -> None:
            self.settings = settings
            self.twak_interface = twak_interface

        def evaluate_universe(
            self,
            market_snapshot: dict[str, dict[str, object]],
            portfolio_value_usdc: float,
        ) -> main_module.BreakoutDecision:
            if decision is None:
                # The telemetry path (_telemetry_candidate_for_log) evaluates
                # the engine every cycle by design; return a no-enter decision.
                return main_module.BreakoutDecision(
                    should_enter=False,
                    symbol=None,
                    position_size_usdc=0.0,
                    factor_scores={},
                    true_factor_count=0,
                    reason="no signal evaluated",
                )
            return decision

    monkeypatch.setattr(main_module, "BnbToolkitWrapper", FakeToolkit)
    monkeypatch.setattr(main_module, "TWAKInterface", FakeTWAK)
    monkeypatch.setattr(main_module, "PancakeSwapRouter", FakeRouter)
    monkeypatch.setattr(main_module, "BreakoutEngine", FakeEngine)
    monkeypatch.setattr(
        main_module,
        "_fetch_snapshot",
        lambda settings, cmc_client, in_position=False: {"CAKE": {"symbol": "CAKE", "price": 2.0}},
    )


def _read_decision(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    return json.loads(lines[0])


def test_run_agent_logs_evaluated_wait_decision(
    monkeypatch: Any,
    tmp_path: Path,
    caplog: Any,
) -> None:
    settings = _settings(tmp_path)
    _patch_run_agent_dependencies(monkeypatch, decision=None)
    caplog.set_level(logging.INFO, logger=main_module.LOGGER.name)

    main_module.run_agent(settings, max_cycles=1)

    record = _read_decision(Path(settings.decision_log_path))
    assert record["action"] == "WAIT"
    assert record["symbol"] is None
    assert record["factor_scores"] == {}
    assert record["true_factor_count"] == 0
    assert record["priced_target_count"] == 1
    assert record["reason"] == "No candidate passed gates"
    assert 'Decision cycle=1 action=WAIT symbol=- factors=- slippage=- reason="' in caplog.text


def test_run_agent_logs_guardrail_blocked_cycle(monkeypatch: Any, tmp_path: Path) -> None:
    settings = _settings(tmp_path, max_daily_trades=0)
    _patch_run_agent_dependencies(monkeypatch, decision=None)

    main_module.run_agent(settings, max_cycles=1)

    record = _read_decision(Path(settings.decision_log_path))
    assert record["action"] == "BLOCKED"
    assert record["entries_allowed"] is False
    assert record["symbol"] is None
    assert record["reason"] == "daily trade limit reached"


def test_run_agent_logs_drawdown_halt_cycle(monkeypatch: Any, tmp_path: Path) -> None:
    guardrail_path = tmp_path / "guardrail_state.json"
    guardrail_path.write_text(
        json.dumps(
            {
                "daily_trade_count": 0,
                "daily_realized_loss": 0.0,
                "portfolio_ath": 10000.0,
                "last_reset_date": datetime.now(timezone.utc).date().isoformat(),
            }
        ),
        encoding="utf-8",
    )
    settings = _settings(tmp_path, guardrail_state_path=str(guardrail_path))
    _patch_run_agent_dependencies(monkeypatch, portfolio_value=8000.0, decision=None)

    main_module.run_agent(settings, max_cycles=1)

    record = _read_decision(Path(settings.decision_log_path))
    assert record["action"] == "HALT"
    assert record["entries_allowed"] is False
    assert record["portfolio_value_usdc"] == 8000.0
    assert record["reason"] == "drawdown kill switch"

"""Integration tests for the v2.5 live orchestrator wiring."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest import mock

from src import main as main_module
from src.config.settings import Settings
from src.execution.liquidity_analyzer import LiquidityResult
from src.strategy.guardrails import Guardrails, RiskDecision, RiskState
from src.strategy.regime_detector import MarketRegime, RegimeResult


def _settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "paper_trade": True,
        "position_state_path": str(tmp_path / "positions.json"),
        "guardrail_state_path": str(tmp_path / "guardrail_state.json"),
        "execution_log_path": str(tmp_path / "execution_log.jsonl"),
        "decision_log_path": str(tmp_path / "decision_log.jsonl"),
        "loop_seconds": 0,
        # No health server in tests: a real bind on a fixed port collides
        # across run_agent invocations within one pytest process.
        "health_check_port": 0,
    }
    values.update(overrides)
    return Settings(**values)


def _install_fast_fakes(
    monkeypatch: Any,
    tmp_path: Path,
    *,
    portfolio_value: float = 10000.0,
    delta: float = 0.0,
    fake_sentiment: bool = True,
) -> type:
    monkeypatch.chdir(tmp_path)

    class FakeToolkit:
        def __init__(self, settings: Settings) -> None:
            self.settings = settings

        def get_balance(self, symbol: str) -> dict[str, object]:
            normalized = symbol.upper()
            if normalized == "USDC":
                return {
                    "symbol": "USDC",
                    "balances": {"USDC": portfolio_value},
                    "portfolio_value_usdc": portfolio_value,
                }
            return {"symbol": normalized, "balances": {normalized: 0}}

    class FakeTWAK:
        def __init__(self, paper_trade: bool = False) -> None:
            self.paper_trade = paper_trade

        def estimate_slippage_pct(
            self,
            amount: float,
            from_token: str,
            to_token: str,
        ) -> float:
            return 0.002

    class FakeRouter:
        calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def __init__(self, twak_interface: FakeTWAK) -> None:
            self.twak_interface = twak_interface

        def swap_exact_in(self, *args: object, **kwargs: object) -> dict[str, object]:
            self.calls.append((args, kwargs))
            return {"tx_hash": "0xabc", "status": 1, "receipt": {"status": 1}}

    class FakeSentiment:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def compute_sentiment(self) -> main_module.SentimentResult:
            return main_module.SentimentResult(
                fear_greed_index=55,
                fear_greed_classification="Neutral",
                funding_rate_btc=0.0,
                open_interest_btc=100.0,
                gas_price_gwei=0.1,
                gas_avg_24h_gwei=None,
                sentiment_delta=delta,
                regime_fragility="NONE",
            )

    monkeypatch.setattr(main_module, "BnbToolkitWrapper", FakeToolkit)
    monkeypatch.setattr(main_module, "TWAKInterface", FakeTWAK)
    monkeypatch.setattr(main_module, "PancakeSwapRouter", FakeRouter)
    if fake_sentiment:
        monkeypatch.setattr(main_module, "SentimentTier1", FakeSentiment)
    return FakeRouter


def _read_first_jsonl(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8").splitlines()[0])


def test_full_cycle_paper_trade_with_all_modules(monkeypatch: Any, tmp_path: Path) -> None:
    _install_fast_fakes(monkeypatch, tmp_path)

    main_module.run_agent(_settings(tmp_path), max_cycles=1)

    decision = _read_first_jsonl(tmp_path / "logs" / "decision_live.jsonl")
    assert decision["schema_version"] == "2.6.0"
    assert decision["action"] == "ENTER"
    assert decision["symbol"] == "CAKE"
    assert (tmp_path / "logs" / "sentiment_live.jsonl").exists()
    assert (tmp_path / "logs" / "portfolio_snapshots.jsonl").exists()


def test_sentiment_delta_appears_in_decision_log(monkeypatch: Any, tmp_path: Path) -> None:
    _install_fast_fakes(monkeypatch, tmp_path, delta=-1.0)

    main_module.run_agent(_settings(tmp_path), max_cycles=1)

    line = _read_first_jsonl(tmp_path / "logs" / "decision_live.jsonl")
    assert "sentiment_delta" in line
    assert "sentiment_fragility" in line
    assert line["sentiment_delta"] == -1.0


def test_execution_reconciler_called_before_open_position(monkeypatch: Any, tmp_path: Path) -> None:
    _install_fast_fakes(monkeypatch, tmp_path)
    opened = {"value": False}
    original_open = main_module.PositionManager.open_position

    with mock.patch("src.execution.execution_reconciler.ExecutionReconciler.reconcile") as mock_reconcile:
        mock_reconcile.return_value = main_module.ReconciliationResult(
            status="SUCCESS",
            tx_hash="0xabc",
            token_out="CAKE",
            amount_out_expected=Decimal("10"),
            amount_out_actual=Decimal("10"),
            effective_slippage_pct=Decimal("0"),
            gas_used=0,
            block_number=0,
            receipt_status=1,
            balance_delta_confirmed=True,
        )

        def checked_open(self: Any, *args: object, **kwargs: object) -> object:
            assert mock_reconcile.called
            opened["value"] = True
            return original_open(self, *args, **kwargs)

        monkeypatch.setattr(main_module.PositionManager, "open_position", checked_open)
        main_module.run_agent(_settings(tmp_path), max_cycles=1)

    assert mock_reconcile.called
    assert opened["value"] is True


def test_guardrails_evaluate_called_not_legacy_methods(monkeypatch: Any, tmp_path: Path) -> None:
    _install_fast_fakes(monkeypatch, tmp_path)
    decision = RiskDecision(
        state=RiskState.NORMAL,
        allow_new_entries=True,
        position_multiplier=1.0,
        max_slippage_pct=0.01,
        max_daily_trades=3,
        base_risk_per_trade_pct=0.0035,
        reasons=[],
    )

    with mock.patch("src.strategy.guardrails.Guardrails.evaluate", return_value=decision) as mock_eval:
        with mock.patch("src.strategy.guardrails.Guardrails.can_open_new_trade", side_effect=AssertionError("legacy")):
            main_module.run_agent(_settings(tmp_path), max_cycles=1)

    assert mock_eval.called


def test_daily_minimum_compliance_activates(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    guardrails = Guardrails(settings)
    regime = RegimeResult(
        regime=MarketRegime.RISK_OFF,
        score=0.0,
        reasons=[],
        position_multiplier=0.1,
        min_entry_factors=5,
        max_slippage_pct=0.005,
        sentiment_delta=0.0,
        sentiment_fragility="NONE",
    )

    decision = main_module.check_daily_minimum_compliance(
        guardrails,
        regime,
        cycle_id=1,
        now_utc=datetime(2026, 6, 5, 21, tzinfo=timezone.utc),
        settings=settings,
    )

    assert decision is not None
    assert decision.size_pct == 0.005
    assert decision.reason == "daily_minimum_compliance_risk_off"


def test_kill_switch_stops_loop(monkeypatch: Any, tmp_path: Path) -> None:
    _install_fast_fakes(monkeypatch, tmp_path, portfolio_value=8400.0)
    guardrail_path = tmp_path / "guardrail_state.json"
    guardrail_path.write_text(
        json.dumps(
            {
                "daily_trade_count": 0,
                "daily_realized_loss": 0.0,
                "daily_loss_pct": 0.0,
                "loss_streak": 0,
                "last_trade_day": None,
                "portfolio_ath": 10000.0,
                "last_reset_date": datetime.now(timezone.utc).date().isoformat(),
            }
        ),
        encoding="utf-8",
    )
    calls = {"liquidate": 0}

    def fake_liquidate(*args: object, **kwargs: object) -> None:
        calls["liquidate"] += 1

    monkeypatch.setattr(main_module, "emergency_liquidate", fake_liquidate)
    monkeypatch.setattr(
        main_module, "_ensure_daily_minimum_trade", lambda *args, **kwargs: False
    )

    main_module.run_agent(_settings(tmp_path, guardrail_state_path=str(guardrail_path)), max_cycles=3)

    # Competition behavior: with no open positions there is nothing to
    # liquidate, and the loop must stay alive in capital-preservation mode
    # (halting would silently fail the one-trade-per-day minimum).
    assert calls["liquidate"] == 0
    lines = (tmp_path / "logs" / "decision_live.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3, "kill switch must not break the loop"
    decision = json.loads(lines[0])
    assert decision["action"] == "HALT"
    assert decision["risk_state"] == "kill_switch"


def test_no_crash_when_sentiment_api_fails(monkeypatch: Any, tmp_path: Path) -> None:
    _install_fast_fakes(monkeypatch, tmp_path, fake_sentiment=False)

    with mock.patch("urllib.request.urlopen", side_effect=Exception("API down")):
        main_module.run_agent(_settings(tmp_path), max_cycles=1)

    decision = _read_first_jsonl(tmp_path / "logs" / "decision_live.jsonl")
    assert "sentiment_delta" in decision


def test_position_size_uses_atr_and_regime() -> None:
    low_atr = main_module.calculate_position_pct(10000.0, 0.02, 1.0, 1.0, 0)
    high_atr = main_module.calculate_position_pct(10000.0, 0.08, 1.0, 1.0, 0)
    reduced_regime = main_module.calculate_position_pct(10000.0, 0.02, 0.5, 1.0, 0)

    assert low_atr > high_atr
    assert reduced_regime < low_atr


def test_liquidity_reject_blocks_trade(monkeypatch: Any, tmp_path: Path) -> None:
    FakeRouter = _install_fast_fakes(monkeypatch, tmp_path)

    with mock.patch("src.execution.liquidity_analyzer.LiquidityAnalyzer.analyze_liquidity") as mock_liq:
        mock_liq.return_value = LiquidityResult(
            symbol="CAKE",
            liquidity_score=0.0,
            slippage_small=0.002,
            slippage_normal=0.02,
            slippage_curve_convex=False,
            recommendation="REJECT",
        )
        main_module.run_agent(_settings(tmp_path), max_cycles=1)

    assert FakeRouter.calls == []

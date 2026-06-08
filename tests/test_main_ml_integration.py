"""Integration tests for ML wiring in main loop."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.config.settings import Settings
from src.main import _evaluate_universe_v25
from src.strategy.guardrails import RiskDecision, RiskState
from src.strategy.regime_detector import MarketRegime, RegimeResult


def test_evaluate_universe_v25_forwards_ml_bundle() -> None:
    settings = Settings(paper_trade=True, strategy_mode="breakout")
    regime = RegimeResult(
        regime=MarketRegime.RANGING,
        score=1.0,
        reasons=[],
        position_multiplier=1.0,
        min_entry_factors=4,
        max_slippage_pct=0.01,
        sentiment_delta=0.0,
        sentiment_fragility="NONE",
    )
    risk = RiskDecision(
        state=RiskState.NORMAL,
        allow_new_entries=True,
        position_multiplier=1.0,
        max_slippage_pct=0.01,
        max_daily_trades=3,
        base_risk_per_trade_pct=0.0035,
        reasons=[],
    )
    ml_bundle = MagicMock()
    ml_bundle.build_contexts.return_value = {}

    snapshot = {
        "CAKE": {
            "symbol": "CAKE",
            "price": 2.0,
            "volume_24h": 1_000_000.0,
            "market_cap": 10_000_000.0,
        }
    }

    with patch("src.main.scoring") as scoring_mock:
        scoring_mock.evaluate_universe = MagicMock(return_value=None)
        _evaluate_universe_v25(
            snapshot,
            10_000.0,
            regime,
            risk,
            settings,
            ml_bundle=ml_bundle,
        )
        kwargs = scoring_mock.evaluate_universe.call_args.kwargs
        assert kwargs.get("ml_bundle") is ml_bundle

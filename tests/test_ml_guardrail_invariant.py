"""Tests proving ML cannot bypass core factor gate."""

from __future__ import annotations

import importlib

from src.config.settings import Settings
from src.execution.twak_interface import TWAKInterface
from src.ml.types import MLContext

_breakout = importlib.import_module("src.strategy.6falgorithm.breakout_engine")
BreakoutEngine = _breakout.BreakoutEngine


def test_high_ml_confidence_cannot_enter_without_core_factors() -> None:
    engine = BreakoutEngine(Settings(paper_trade=True), TWAKInterface(paper_trade=True))
    token_data = {
        "symbol": "CAKE",
        "price": 2.0,
        "volume_24h": 10_000_000.0,
        "market_cap": 100_000_000.0,
        "volume_1h": 100.0,
        "rolling_24h_hourly_volume_avg": 1000.0,
        "high_6h": 3.0,
        "percent_change_1h": -0.05,
        "percent_change_24h": -0.10,
        "bnb_1h_trend_pct": -0.05,
    }
    ml_contexts = {
        "CAKE": MLContext("CAKE", "momentum", 0.99, 1.0, 0.99, {}),
    }
    decision = engine.evaluate_token(token_data, 10_000.0, ml_contexts.get("CAKE"))
    assert decision.should_enter is False
    assert decision.ml_context is not None

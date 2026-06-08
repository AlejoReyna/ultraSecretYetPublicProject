"""Tests for ML core types."""

from __future__ import annotations

from src.ml.types import (
    MLContext,
    RegimePrediction,
    attach_symbol,
    chop_fallback,
    context_from_prediction,
)


def test_regime_prediction_construction() -> None:
    pred = RegimePrediction(
        regime="momentum",
        confidence=0.72,
        feature_vector={"ret_1": 0.01},
        model_version="v1",
    )
    assert pred.regime == "momentum"
    assert pred.confidence == 0.72


def test_ml_context_is_frozen() -> None:
    ctx = MLContext(
        symbol="CAKE",
        regime="chop",
        confidence=0.4,
        position_size_multiplier=0.5,
        rank_score=0.4,
        feature_vector={},
    )
    assert ctx.symbol == "CAKE"


def test_chop_fallback_defaults() -> None:
    ctx = chop_fallback("ETH")
    assert ctx.regime == "chop"
    assert ctx.confidence == 0.0
    assert ctx.position_size_multiplier == 0.5


def test_context_from_prediction_momentum() -> None:
    pred = RegimePrediction("momentum", 0.8, {}, "v1")
    ctx = context_from_prediction(pred, momentum_multiplier=1.0, chop_multiplier=0.5)
    assert ctx.position_size_multiplier == 1.0


def test_attach_symbol() -> None:
    ctx = chop_fallback("cake")
    attached = attach_symbol(ctx, "CAKE")
    assert attached.symbol == "CAKE"

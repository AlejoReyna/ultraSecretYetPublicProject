"""Core ML types for regime prediction and candidate ranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from src.config.settings import Settings


RegimeLabel = Literal["momentum", "chop"]


@dataclass(frozen=True)
class RegimePrediction:
    """Output of the LightGBM regime classifier for one token."""

    regime: RegimeLabel
    confidence: float
    feature_vector: dict[str, float]
    model_version: str


@dataclass(frozen=True)
class MLContext:
    """Per-token ML state passed into strategy evaluation."""

    symbol: str
    regime: RegimeLabel
    confidence: float
    position_size_multiplier: float
    rank_score: float
    feature_vector: dict[str, float]


@dataclass(frozen=True)
class MLRankingAudit:
    """Audit payload when multiple tokens pass core factors."""

    candidates: list[str]
    confidences: dict[str, float]
    selected: str | None


def ranking_audit_to_dict(audit: MLRankingAudit | None) -> dict[str, Any] | None:
    """Serialize MLRankingAudit for JSONL decision logs."""

    if audit is None:
        return None
    return {
        "candidates": list(audit.candidates),
        "confidences": dict(audit.confidences),
        "selected": audit.selected,
    }


def chop_fallback(
    symbol: str,
    model_version: str = "fallback",
    *,
    position_multiplier: float = 0.5,
) -> MLContext:
    """Conservative default when inference fails or data is missing."""

    return MLContext(
        symbol=symbol.upper(),
        regime="chop",
        confidence=0.0,
        position_size_multiplier=position_multiplier,
        rank_score=0.0,
        feature_vector={},
    )


def context_from_prediction(
    prediction: RegimePrediction,
    *,
    momentum_multiplier: float,
    chop_multiplier: float,
) -> MLContext:
    """Build MLContext from a RegimePrediction and sizing settings."""

    multiplier = momentum_multiplier if prediction.regime == "momentum" else chop_multiplier
    return MLContext(
        symbol="",
        regime=prediction.regime,
        confidence=prediction.confidence,
        position_size_multiplier=multiplier,
        rank_score=prediction.confidence,
        feature_vector=dict(prediction.feature_vector),
    )


def attach_symbol(context: MLContext, symbol: str) -> MLContext:
    """Return a copy of context with symbol set."""

    return MLContext(
        symbol=symbol.upper(),
        regime=context.regime,
        confidence=context.confidence,
        position_size_multiplier=context.position_size_multiplier,
        rank_score=context.rank_score,
        feature_vector=dict(context.feature_vector),
    )

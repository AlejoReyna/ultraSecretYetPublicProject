"""Shared entry candidate types for strategy evaluators."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any


@dataclass(frozen=True)
class EntryCandidate:
    """Normalized entry candidate selected by a strategy evaluator."""

    symbol: str
    price: float
    position_size_usdc: float
    expected_amount_out: Decimal
    slippage_small: float | None
    slippage_normal: float | None
    reason: str
    factor_scores: dict[str, bool]
    true_factor_count: int
    source: str = "scoring_v25"
    entry_score: float | None = None
    strategy_mode: str = "breakout"
    ml_context: Any | None = None
    ml_ranking: dict[str, Any] | None = None

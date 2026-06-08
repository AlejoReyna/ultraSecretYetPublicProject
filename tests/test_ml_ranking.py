"""Tests for ML candidate ranking."""

from __future__ import annotations

import importlib

from src.ml.candidate_ranker import CandidateRanker
from src.ml.types import MLContext

_breakout = importlib.import_module("src.strategy.6falgorithm.breakout_engine")
BreakoutDecision = _breakout.BreakoutDecision


def _decision(symbol: str, factors: int = 4) -> BreakoutDecision:
    return BreakoutDecision(
        should_enter=True,
        symbol=symbol,
        position_size_usdc=100.0,
        factor_scores={"volume_breakout": True},
        true_factor_count=factors,
        reason="test",
        estimated_slippage_pct=0.005,
    )


def test_ranker_picks_highest_confidence() -> None:
    decisions = [_decision("CAKE"), _decision("ETH"), _decision("DOGE")]
    contexts = {
        "CAKE": MLContext("CAKE", "momentum", 0.72, 1.0, 0.72, {}),
        "ETH": MLContext("ETH", "momentum", 0.61, 1.0, 0.61, {}),
        "DOGE": MLContext("DOGE", "chop", 0.58, 0.5, 0.58, {}),
    }
    ranked, audit = CandidateRanker().rank(decisions, contexts)
    assert ranked is not None
    assert ranked.symbol == "CAKE"
    assert audit is not None
    assert audit.selected == "CAKE"
    assert set(audit.candidates) == {"CAKE", "ETH", "DOGE"}

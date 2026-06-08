"""ML-based ranking among tokens that passed core factors."""

from __future__ import annotations

import importlib
from typing import Any

from src.ml.types import MLContext, MLRankingAudit

_breakout = importlib.import_module("src.strategy.6falgorithm.breakout_engine")
BreakoutDecision = _breakout.BreakoutDecision


class CandidateRanker:
    """Pick the best entry among multiple core-factor passers."""

    def rank(
        self,
        decisions: list[BreakoutDecision],
        ml_contexts: dict[str, MLContext],
        volume_by_symbol: dict[str, float] | None = None,
    ) -> tuple[BreakoutDecision | None, MLRankingAudit | None]:
        passers = [decision for decision in decisions if decision.should_enter and decision.symbol]
        if not passers:
            return None, None
        if len(passers) == 1:
            return passers[0], None

        volumes = volume_by_symbol or {}

        def sort_key(decision: BreakoutDecision) -> tuple[float, int, float]:
            symbol = (decision.symbol or "").upper()
            ctx = ml_contexts.get(symbol)
            confidence = ctx.confidence if ctx is not None else 0.0
            volume = volumes.get(symbol, 0.0)
            return (confidence, decision.true_factor_count, volume)

        confidences = {
            (decision.symbol or "").upper(): (
                ml_contexts.get((decision.symbol or "").upper()).confidence
                if ml_contexts.get((decision.symbol or "").upper()) is not None
                else 0.0
            )
            for decision in passers
        }
        selected = max(passers, key=sort_key)
        audit = MLRankingAudit(
            candidates=[(decision.symbol or "").upper() for decision in passers],
            confidences=confidences,
            selected=(selected.symbol or "").upper() if selected.symbol else None,
        )
        return selected, audit

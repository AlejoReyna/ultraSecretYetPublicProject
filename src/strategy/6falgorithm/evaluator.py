"""Breakout strategy universe evaluator."""

from __future__ import annotations

import importlib
import logging
from typing import Any

from src.config.settings import Settings
from src.execution.twak_interface import TWAKInterface
from src.ml.types import ranking_audit_to_dict
from src.strategy.candidate_adapter import breakout_decision_to_candidate

_fallback = importlib.import_module("src.strategy.6falgorithm.fallback_scorer")
fallback_scoring_evaluate_universe = _fallback.fallback_scoring_evaluate_universe
from src.strategy.entry_types import EntryCandidate
from src.strategy.guardrails import RiskDecision
from src.strategy.regime_detector import RegimeResult

_breakout_module = importlib.import_module("src.strategy.6falgorithm.breakout_engine")
BreakoutEngine = _breakout_module.BreakoutEngine

LOGGER = logging.getLogger(__name__)


def evaluate_universe_breakout(
    snapshot: dict[str, dict[str, Any]],
    portfolio_value: float,
    regime_result: RegimeResult,
    risk_decision: RiskDecision,
    *,
    settings: Settings,
    twak_interface: TWAKInterface | None = None,
    exclude_symbols: set[str] | None = None,
    use_breakout_engine: bool = True,
    ml_bundle: Any | None = None,
) -> EntryCandidate | None:
    """Evaluate the universe using the 6-factor BreakoutEngine or legacy fallback."""

    if not use_breakout_engine:
        return fallback_scoring_evaluate_universe(
            snapshot,
            portfolio_value,
            regime_result,
            risk_decision,
            settings=settings,
            twak_interface=twak_interface,
            exclude_symbols=exclude_symbols,
        )

    engine = BreakoutEngine(settings, twak_interface)
    filtered_snapshot = {
        symbol: data
        for symbol, data in snapshot.items()
        if symbol.upper() not in {item.upper() for item in (exclude_symbols or set())}
    }

    ml_contexts: dict[str, Any] = {}
    if ml_bundle is not None:
        ml_contexts = ml_bundle.build_contexts(filtered_snapshot)

    decisions = engine.evaluate_all(filtered_snapshot, portfolio_value, ml_contexts)
    passers = [decision for decision in decisions if decision.should_enter]
    if not passers:
        return None

    selected = passers[0]
    ml_ranking: dict[str, object] | None = None
    if len(passers) > 1:
        from src.ml.candidate_ranker import CandidateRanker

        volume_by_symbol = {
            symbol.upper(): float(data.get("volume_24h") or 0.0)
            for symbol, data in filtered_snapshot.items()
            if isinstance(data, dict)
        }
        ranked, audit = CandidateRanker().rank(passers, ml_contexts, volume_by_symbol=volume_by_symbol)
        ranking_active = getattr(ml_bundle, "is_ranking_active", False) if ml_bundle is not None else False

        if ranking_active and ranked is not None:
            selected = ranked
        elif ranked is not None:
            LOGGER.info(
                "SHADOW: ML would pick %s; executing rule-based %s (AUC gate/shadow mode)",
                ranked.symbol,
                selected.symbol,
            )

        ml_ranking = ranking_audit_to_dict(audit) or {}
        if ml_bundle is not None:
            shadow_fields = ml_bundle.shadow_audit_fields(
                passers,
                ml_contexts,
                selected.symbol,
            )
            ml_ranking.update(shadow_fields)

    candidate = breakout_decision_to_candidate(
        selected,
        snapshot,
        portfolio_value,
        settings,
        risk_decision,
    )
    if candidate is None or ml_ranking is None:
        return candidate
    return EntryCandidate(
        symbol=candidate.symbol,
        price=candidate.price,
        position_size_usdc=candidate.position_size_usdc,
        expected_amount_out=candidate.expected_amount_out,
        slippage_small=candidate.slippage_small,
        slippage_normal=candidate.slippage_normal,
        reason=candidate.reason,
        factor_scores=candidate.factor_scores,
        true_factor_count=candidate.true_factor_count,
        source=candidate.source,
        entry_score=candidate.entry_score,
        strategy_mode=candidate.strategy_mode,
        ml_context=candidate.ml_context,
        ml_ranking=ml_ranking,
    )

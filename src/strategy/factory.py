"""Strategy factory for breakout vs scalping modes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from src.config.settings import Settings
from src.execution.twak_interface import TWAKInterface
import importlib

from src.config.settings import Settings
from src.execution.twak_interface import TWAKInterface
from src.strategy.entry_types import EntryCandidate
from src.strategy.guardrails import RiskDecision
from src.strategy.regime_detector import RegimeResult

_evaluator = importlib.import_module("src.strategy.6falgorithm.evaluator")
_fallback = importlib.import_module("src.strategy.6falgorithm.fallback_scorer")
evaluate_universe_breakout = _evaluator.evaluate_universe_breakout
fallback_scoring_evaluate_universe = _fallback.fallback_scoring_evaluate_universe
from src.strategy.guardrails import Guardrails
from src.strategy.position_manager import PositionManager
from src.strategy.scalping_engine import ScalpingEngine
from src.strategy.scalping_guardrails import ScalpingGuardrails
from src.strategy.scalping_position_manager import ScalpingPositionManager
from src.strategy.volatility import PriceCache


@dataclass(frozen=True)
class StrategyBundle:
    """Runtime strategy components selected by settings.strategy_mode."""

    evaluate_universe: Callable[..., Any]
    guardrails: Guardrails
    position_manager: PositionManager
    strategy_mode: str
    scalping_engine: ScalpingEngine | None = None


def create_strategy_bundle(
    settings: Settings,
    price_cache: PriceCache,
    twak_interface: TWAKInterface | None = None,
) -> StrategyBundle:
    """Instantiate guardrails, position manager, and universe evaluator."""

    if settings.strategy_mode == "scalping":
        guardrails = ScalpingGuardrails(settings)
        position_manager = ScalpingPositionManager(settings)
        engine = ScalpingEngine(settings, price_cache, twak_interface)

        def evaluate_universe(
            snapshot: dict[str, dict[str, Any]],
            portfolio_value: float,
            regime_result: object,
            risk_decision: object,
            *,
            settings: Settings | None = None,
            twak_interface: TWAKInterface | None = None,
            exclude_symbols: set[str] | None = None,
            sentiment_result: object | None = None,
            **_: Any,
        ) -> Any:
            del settings, twak_interface
            return engine.evaluate_universe(
                snapshot,
                portfolio_value,
                regime_result,  # type: ignore[arg-type]
                risk_decision,  # type: ignore[arg-type]
                sentiment_result=sentiment_result,  # type: ignore[arg-type]
                exclude_symbols=exclude_symbols,
                cooldown_checker=position_manager.is_symbol_on_cooldown,
            )

        return StrategyBundle(
            evaluate_universe=evaluate_universe,
            guardrails=guardrails,
            position_manager=position_manager,
            strategy_mode="scalping",
            scalping_engine=engine,
        )

    guardrails = Guardrails(settings)
    position_manager = PositionManager(settings)

    def evaluate_universe(
        snapshot: dict[str, dict[str, Any]],
        portfolio_value: float,
        regime_result: object,
        risk_decision: object,
        *,
        settings: Settings | None = None,
        twak_interface: TWAKInterface | None = None,
        exclude_symbols: set[str] | None = None,
        ml_bundle: Any | None = None,
        **_: Any,
    ) -> Any:
        active_settings = settings or guardrails.settings
        return evaluate_universe_breakout(
            snapshot,
            portfolio_value,
            regime_result,  # type: ignore[arg-type]
            risk_decision,  # type: ignore[arg-type]
            settings=active_settings,
            twak_interface=twak_interface,
            exclude_symbols=exclude_symbols,
            use_breakout_engine=True,
            ml_bundle=ml_bundle,
        )

    return StrategyBundle(
        evaluate_universe=evaluate_universe,
        guardrails=guardrails,
        position_manager=position_manager,
        strategy_mode="breakout",
    )


def fallback_evaluate_universe(*args: Any, **kwargs: Any) -> Any:
    """Legacy fallback scorer exposed for tests and compatibility."""

    return fallback_scoring_evaluate_universe(*args, **kwargs)

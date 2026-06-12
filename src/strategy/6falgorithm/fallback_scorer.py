"""Legacy v2.5 inline fallback scorer extracted from main.py."""

from __future__ import annotations

from typing import Any

from src.config.settings import Settings
from src.config.tokens import has_bsc_contract, is_liquid, is_momentum_candidate_symbol
from src.strategy.candidate_adapter import (
    coerce_entry_candidate,
    decimal_div,
    first_market_number,
    maybe_number,
)
from src.strategy.entry_types import EntryCandidate
from src.strategy.guardrails import RiskDecision
from src.strategy.regime_detector import MarketRegime, RegimeResult


def fallback_scoring_evaluate_universe(
    snapshot: dict[str, dict[str, Any]],
    portfolio_value: float,
    regime_result: RegimeResult,
    risk_decision: RiskDecision,
    *,
    settings: Settings,
    twak_interface: object | None = None,
    exclude_symbols: set[str] | None = None,
) -> EntryCandidate | None:
    """Small local scoring adapter used when BreakoutEngine is not selected."""

    del twak_interface
    excluded = {symbol.upper() for symbol in (exclude_symbols or set())}
    ranked: list[tuple[int, float, EntryCandidate]] = []
    for raw_symbol, raw_data in snapshot.items():
        symbol = str(raw_symbol).upper()
        if symbol in excluded or not is_momentum_candidate_symbol(symbol) or not has_bsc_contract(symbol):
            continue
        data = {"symbol": symbol, **raw_data}
        if not is_liquid(data):
            continue
        price = maybe_number(data.get("price"))
        if price is None or price <= 0:
            continue

        slippage_normal = maybe_number(data.get("estimated_slippage_pct"))
        slippage_small = maybe_number(data.get("estimated_slippage_small_pct"))
        if slippage_small is None and slippage_normal is not None:
            slippage_small = max(0.0, slippage_normal * 0.5)

        volume_1h = maybe_number(data.get("volume_1h"))
        rolling_hourly = maybe_number(data.get("rolling_24h_hourly_volume_avg"))
        volume_breakout = (
            volume_1h is not None
            and rolling_hourly is not None
            and rolling_hourly > 0
            and volume_1h > 2.0 * rolling_hourly
        )
        high_3h = maybe_number(data.get("high_3h"))
        high_6h = maybe_number(data.get("high_6h"))
        reference_high = high_3h if high_3h is not None else high_6h
        high_break = reference_high is not None and price > reference_high * (1 + settings.breakout_buffer)
        token_1h = first_market_number(data, ("token_percent_change_1h", "percent_change_1h"), 0.0)
        token_24h = first_market_number(data, ("token_percent_change_24h", "percent_change_24h"), 0.0)
        trend_ok = token_1h > settings.token_regime_1h_min and token_24h > settings.token_regime_24h_min
        regime_ok = regime_result.regime != MarketRegime.RISK_OFF
        slippage_ok = slippage_normal is not None and slippage_normal <= risk_decision.max_slippage_pct

        factor_scores = {
            "volume_breakout": volume_breakout,
            "high_break": high_break,
            "trend_ok": trend_ok,
            "regime_not_risk_off": regime_ok,
            "slippage_under_cap": slippage_ok,
        }
        true_factor_count = sum(1 for passed in factor_scores.values() if passed)
        entry_factors = (volume_breakout, high_break, trend_ok, slippage_ok)
        entry_factor_count = sum(1 for passed in entry_factors if passed)
        required = min(len(entry_factors), max(settings.min_entry_factors, regime_result.min_entry_factors))
        if entry_factor_count < required:
            continue

        regime_size_modifier = 1.0 if regime_ok else float(getattr(settings, "regime_size_multiplier", 0.5))
        provisional_size = portfolio_value * settings.max_position_pct * regime_result.position_multiplier
        provisional_size *= regime_size_modifier
        provisional_size *= risk_decision.position_multiplier
        expected_amount_out = decimal_div(provisional_size, price)
        reason = f"{entry_factor_count}/{len(entry_factors)} v2.5 entry gates passed"
        ranked.append(
            (
                true_factor_count,
                first_market_number(data, ("volume_24h", "market_cap"), 0.0),
                EntryCandidate(
                    symbol=symbol,
                    price=price,
                    position_size_usdc=provisional_size,
                    expected_amount_out=expected_amount_out,
                    slippage_small=slippage_small,
                    slippage_normal=slippage_normal,
                    reason=reason,
                    factor_scores=factor_scores,
                    true_factor_count=true_factor_count,
                    source="fallback_scorer",
                    strategy_mode="breakout",
                ),
            )
        )
    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return ranked[0][2]


def fallback_best_near_miss(
    snapshot: dict[str, dict[str, Any]],
    portfolio_value: float,
    regime_result: RegimeResult,
    risk_decision: RiskDecision,
    *,
    settings: Settings,
    exclude_symbols: set[str] | None = None,
) -> EntryCandidate | None:
    """Return the best-scoring fallback candidate for dashboard telemetry."""

    excluded = {symbol.upper() for symbol in (exclude_symbols or set())}
    ranked: list[tuple[int, float, EntryCandidate]] = []
    for raw_symbol, raw_data in snapshot.items():
        symbol = str(raw_symbol).upper()
        if symbol in excluded or not is_momentum_candidate_symbol(symbol) or not has_bsc_contract(symbol):
            continue
        data = {"symbol": symbol, **raw_data}
        if not is_liquid(data):
            continue
        price = maybe_number(data.get("price"))
        if price is None or price <= 0:
            continue

        slippage_normal = maybe_number(data.get("estimated_slippage_pct"))
        slippage_small = maybe_number(data.get("estimated_slippage_small_pct"))
        if slippage_small is None and slippage_normal is not None:
            slippage_small = max(0.0, slippage_normal * 0.5)

        volume_1h = maybe_number(data.get("volume_1h"))
        rolling_hourly = maybe_number(data.get("rolling_24h_hourly_volume_avg"))
        volume_breakout = (
            volume_1h is not None
            and rolling_hourly is not None
            and rolling_hourly > 0
            and volume_1h > 2.0 * rolling_hourly
        )
        high_3h = maybe_number(data.get("high_3h"))
        high_6h = maybe_number(data.get("high_6h"))
        reference_high = high_3h if high_3h is not None else high_6h
        high_break = reference_high is not None and price > reference_high * (1 + settings.breakout_buffer)
        token_1h = first_market_number(data, ("token_percent_change_1h", "percent_change_1h"), 0.0)
        token_24h = first_market_number(data, ("token_percent_change_24h", "percent_change_24h"), 0.0)
        trend_ok = token_1h > settings.token_regime_1h_min and token_24h > settings.token_regime_24h_min
        regime_ok = regime_result.regime != MarketRegime.RISK_OFF
        slippage_ok = slippage_normal is not None and slippage_normal <= risk_decision.max_slippage_pct

        factor_scores = {
            "volume_breakout": volume_breakout,
            "high_break": high_break,
            "trend_ok": trend_ok,
            "regime_not_risk_off": regime_ok,
            "slippage_under_cap": slippage_ok,
        }
        true_factor_count = sum(1 for passed in factor_scores.values() if passed)
        entry_factors = (volume_breakout, high_break, trend_ok, slippage_ok)
        entry_factor_count = sum(1 for passed in entry_factors if passed)
        required = min(len(entry_factors), max(settings.min_entry_factors, regime_result.min_entry_factors))
        regime_size_modifier = 1.0 if regime_ok else float(getattr(settings, "regime_size_multiplier", 0.5))
        provisional_size = portfolio_value * settings.max_position_pct * regime_result.position_multiplier
        provisional_size *= regime_size_modifier
        provisional_size *= risk_decision.position_multiplier
        reason = f"{entry_factor_count}/{len(entry_factors)} v2.5 entry gates passed (need {required})"
        ranked.append(
            (
                true_factor_count,
                first_market_number(data, ("volume_24h", "market_cap"), 0.0),
                EntryCandidate(
                    symbol=symbol,
                    price=price,
                    position_size_usdc=0.0,
                    expected_amount_out=decimal_div(provisional_size, price),
                    slippage_small=slippage_small,
                    slippage_normal=slippage_normal,
                    reason=reason,
                    factor_scores=factor_scores,
                    true_factor_count=true_factor_count,
                    source="fallback_scorer",
                    strategy_mode="breakout",
                ),
            )
        )
    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return ranked[0][2]

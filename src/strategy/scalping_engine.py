"""Scalping v1.0 signal engine with weighted 0-100 score."""

from __future__ import annotations

from typing import Any

from src.config.settings import Settings
from src.config.tokens import has_verified_bsc_contract, is_liquid, is_momentum_candidate_symbol
from src.execution.twak_interface import TWAKInterface
from src.strategy.candidate_adapter import decimal_div, first_market_number, maybe_number
from src.strategy.entry_types import EntryCandidate
from src.strategy.guardrails import RiskDecision
from src.strategy.regime_detector import MarketRegime, RegimeResult
from src.strategy.sentiment_tier1 import SentimentResult
from src.strategy.volatility import PriceCache

# MEV RISK DISCLOSURE (academic finding, Econstor 2024 / Han et al. 2022)
# PancakeSwap/BSC is susceptible to sandwich attacks and front-running by MEV bots.
# This strategy accepts MEV cost as implicit slippage. Mitigation:
#   1. Small position sizes (1% of portfolio, ~$5-$50 in production)
#   2. Fixed TP/SL (no trailing stops that could be gamed)
#   3. Slippage filter < 0.3% pre-trade
# Expected MEV leakage: 0.1-0.5% per trade in high-volatility periods.
# This is NOT modeled in backtests; live performance may underperform by this margin.

SCALPING_FACTOR_WEIGHTS = {
    "micro_momentum": 30,
    "slippage_ok": 25,
    "regime_neutro": 20,
    "no_whale_dump": 15,
    "gas_viable": 10,
}


def _true_factor_count(factor_scores: dict[str, bool]) -> int:
    return sum(1 for passed in factor_scores.values() if passed)


class ScalpingEngine:
    """Evaluate tokens using a weighted score instead of boolean gates."""

    def __init__(
        self,
        settings: Settings,
        price_cache: PriceCache,
        twak_interface: TWAKInterface | None = None,
    ) -> None:
        self.settings = settings
        self.price_cache = price_cache
        self.twak_interface = twak_interface

    def evaluate_universe(
        self,
        snapshot: dict[str, dict[str, Any]],
        portfolio_value: float,
        regime_result: RegimeResult,
        risk_decision: RiskDecision,
        *,
        sentiment_result: SentimentResult | None = None,
        exclude_symbols: set[str] | None = None,
        cooldown_checker: Any | None = None,
    ) -> EntryCandidate | None:
        excluded = {symbol.upper() for symbol in (exclude_symbols or set())}
        ranked: list[tuple[float, float, EntryCandidate]] = []

        for raw_symbol, raw_data in snapshot.items():
            symbol = str(raw_symbol).upper()
            if symbol in excluded:
                continue
            if cooldown_checker is not None and cooldown_checker(symbol):
                continue
            candidate = self._evaluate_symbol(
                symbol,
                {"symbol": symbol, **raw_data},
                portfolio_value,
                regime_result,
                risk_decision,
                sentiment_result,
            )
            if candidate is None:
                continue
            ranked.append(
                (
                    candidate.entry_score or 0.0,
                    first_market_number(raw_data, ("volume_24h", "market_cap"), 0.0),
                    candidate,
                )
            )

        if not ranked:
            return None
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        best = ranked[0][2]
        if (best.entry_score or 0.0) < self.settings.scalping_entry_score_min:
            return None
        return best

    def best_near_miss(
        self,
        snapshot: dict[str, dict[str, Any]],
        portfolio_value: float,
        regime_result: RegimeResult,
        risk_decision: RiskDecision,
        *,
        sentiment_result: SentimentResult | None = None,
        exclude_symbols: set[str] | None = None,
        cooldown_checker: Any | None = None,
    ) -> EntryCandidate | None:
        """Return the highest-scoring symbol for operator telemetry when no entry triggers."""

        excluded = {symbol.upper() for symbol in (exclude_symbols or set())}
        ranked: list[tuple[float, float, EntryCandidate]] = []

        for raw_symbol, raw_data in snapshot.items():
            symbol = str(raw_symbol).upper()
            if symbol in excluded:
                continue
            if cooldown_checker is not None and cooldown_checker(symbol):
                continue
            candidate = self._score_symbol_for_telemetry(
                symbol,
                {"symbol": symbol, **raw_data},
                portfolio_value,
                regime_result,
                risk_decision,
                sentiment_result,
            )
            if candidate is None:
                continue
            ranked.append(
                (
                    candidate.entry_score or 0.0,
                    first_market_number(raw_data, ("volume_24h", "market_cap"), 0.0),
                    candidate,
                )
            )

        if not ranked:
            return None

        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        best = ranked[0][2]
        score = best.entry_score or 0.0
        minimum = self.settings.scalping_entry_score_min
        if score >= minimum:
            return best
        return EntryCandidate(
            symbol=best.symbol,
            price=best.price,
            position_size_usdc=0.0,
            expected_amount_out=best.expected_amount_out,
            slippage_small=best.slippage_small,
            slippage_normal=best.slippage_normal,
            reason=f"best {best.symbol} scalping score {score:.0f}/100 < {minimum:.0f}",
            factor_scores=best.factor_scores,
            true_factor_count=best.true_factor_count,
            source=best.source,
            entry_score=score,
            strategy_mode=best.strategy_mode,
        )

    def _score_symbol_for_telemetry(
        self,
        symbol: str,
        data: dict[str, Any],
        portfolio_value: float,
        regime_result: RegimeResult,
        risk_decision: RiskDecision,
        sentiment_result: SentimentResult | None,
    ) -> EntryCandidate | None:
        # Telemetry: score momentum-eligible hack symbols with a price. Entry keeps liquidity gates.
        if not is_momentum_candidate_symbol(symbol):
            return None
        price = maybe_number(data.get("price"))
        if price is None or price <= 0:
            return None

        slippage_normal = maybe_number(data.get("estimated_slippage_pct"))
        score, factor_scores = self.score_token(
            symbol,
            data,
            regime_result,
            sentiment_result,
            slippage_normal=slippage_normal,
        )
        position_size = portfolio_value * self.settings.scalping_position_pct * risk_decision.position_multiplier
        slippage_small = maybe_number(data.get("estimated_slippage_small_pct"))
        if slippage_small is None and slippage_normal is not None:
            slippage_small = max(0.0, slippage_normal * 0.5)

        return EntryCandidate(
            symbol=symbol,
            price=price,
            position_size_usdc=position_size,
            expected_amount_out=decimal_div(position_size, price),
            slippage_small=slippage_small,
            slippage_normal=slippage_normal,
            reason=f"scalping score {score:.0f}/100",
            factor_scores=factor_scores,
            true_factor_count=_true_factor_count(factor_scores),
            source="scalping_engine",
            entry_score=score,
            strategy_mode="scalping",
        )

    def score_token(
        self,
        symbol: str,
        token_data: dict[str, Any],
        regime_result: RegimeResult,
        sentiment_result: SentimentResult | None = None,
        slippage_normal: float | None = None,
    ) -> tuple[float, dict[str, bool]]:
        """Return weighted score and factor booleans for one token."""

        price = maybe_number(token_data.get("price"))
        ema_9 = self.price_cache.get_ema(symbol, 9) if price is not None else None
        micro_momentum = self._micro_momentum(symbol, token_data, price, ema_9)
        if slippage_normal is None:
            slippage_normal = maybe_number(token_data.get("estimated_slippage_pct"))
        slippage_ok = slippage_normal is not None and slippage_normal < 0.003
        regime_neutro = regime_result.regime != MarketRegime.RISK_OFF
        change_1h = first_market_number(token_data, ("percent_change_1h", "token_percent_change_1h"), 0.0)
        no_whale_dump = change_1h > -0.02
        gas_gwei = sentiment_result.gas_price_gwei if sentiment_result is not None else None
        gas_viable = gas_gwei is not None and gas_gwei < self.settings.scalping_max_gas_gwei

        factor_scores = {
            "micro_momentum": micro_momentum,
            "slippage_ok": slippage_ok,
            "regime_neutro": regime_neutro,
            "no_whale_dump": no_whale_dump,
            "gas_viable": gas_viable,
        }
        score = sum(
            weight for key, weight in SCALPING_FACTOR_WEIGHTS.items() if factor_scores.get(key)
        )
        return float(score), factor_scores

    def _evaluate_symbol(
        self,
        symbol: str,
        data: dict[str, Any],
        portfolio_value: float,
        regime_result: RegimeResult,
        risk_decision: RiskDecision,
        sentiment_result: SentimentResult | None,
    ) -> EntryCandidate | None:
        if not is_momentum_candidate_symbol(symbol) or not has_verified_bsc_contract(symbol) or not is_liquid(data):
            return None
        price = maybe_number(data.get("price"))
        if price is None or price <= 0:
            return None

        market_cap = maybe_number(data.get("market_cap"))
        if market_cap is not None and market_cap < self.settings.scalping_min_market_cap_usd:
            return None

        slippage_normal = maybe_number(data.get("estimated_slippage_pct"))
        if slippage_normal is not None and slippage_normal > self.settings.scalping_max_slippage_pct:
            return None

        if self._pumped_recently(symbol, price):
            return None

        score, factor_scores = self.score_token(
            symbol,
            data,
            regime_result,
            sentiment_result,
            slippage_normal=slippage_normal,
        )
        if score < self.settings.scalping_entry_score_min:
            return None

        position_size = portfolio_value * self.settings.scalping_position_pct
        position_size *= risk_decision.position_multiplier
        slippage_small = maybe_number(data.get("estimated_slippage_small_pct"))
        if slippage_small is None and slippage_normal is not None:
            slippage_small = max(0.0, slippage_normal * 0.5)

        return EntryCandidate(
            symbol=symbol,
            price=price,
            position_size_usdc=position_size,
            expected_amount_out=decimal_div(position_size, price),
            slippage_small=slippage_small,
            slippage_normal=slippage_normal,
            reason=f"scalping score {score:.0f}/100 >= {self.settings.scalping_entry_score_min:.0f}",
            factor_scores=factor_scores,
            true_factor_count=_true_factor_count(factor_scores),
            source="scalping_engine",
            entry_score=score,
            strategy_mode="scalping",
        )

    def _micro_momentum(
        self,
        symbol: str,
        token_data: dict[str, Any],
        price: float | None,
        ema_9: float | None,
    ) -> bool:
        if price is None or ema_9 is None or price <= ema_9:
            return False

        volume_1h = maybe_number(token_data.get("volume_1h"))
        volume_24h = maybe_number(token_data.get("volume_24h"))
        if volume_1h is not None and volume_24h is not None and volume_24h > 0:
            hourly_avg = volume_24h / 24
            return hourly_avg > 0 and volume_1h > hourly_avg * 1.5

        volume_latest = self._latest_volume(symbol)
        avg_volume = self._average_hourly_volume(symbol)
        if volume_latest is not None and avg_volume is not None and avg_volume > 0:
            return volume_latest > avg_volume * 1.05

        return False

    def _latest_volume(self, symbol: str) -> float | None:
        points = list(self.price_cache._data.get(symbol.upper(), ()))
        if not points:
            return None
        return float(points[-1].volume)

    def _average_hourly_volume(self, symbol: str) -> float | None:
        points = list(self.price_cache._data.get(symbol.upper(), ()))
        if len(points) < 2:
            return None
        recent = points[-12:]
        volumes = [point.volume for point in recent if point.volume > 0]
        if not volumes:
            return None
        return sum(volumes) / len(volumes)

    def _pumped_recently(self, symbol: str, current_price: float) -> bool:
        points = list(self.price_cache._data.get(symbol.upper(), ()))
        if len(points) < 3:
            return False
        reference = points[-3].close
        if reference <= 0:
            return False
        change = (current_price - reference) / reference
        return change > self.settings.scalping_pump_filter_15m_pct

"""Six-factor momentum breakout strategy engine adapted to 4-factor core."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.config.settings import Settings
from src.config.tokens import is_liquid, is_tradable_symbol
from src.execution.twak_interface import TWAKInterface

if TYPE_CHECKING:
    from src.ml.types import MLContext

CORE_FACTOR_COUNT = 4
TOTAL_FACTOR_COUNT = 6


@dataclass(frozen=True)
class BreakoutDecision:
    """Decision returned by the breakout engine."""

    should_enter: bool
    symbol: str | None
    position_size_usdc: float
    factor_scores: dict[str, bool]
    true_factor_count: int
    reason: str
    estimated_slippage_pct: float | None = None
    ml_context: Any | None = None


@dataclass(frozen=True)
class _CheapCandidate:
    """Candidate factors that can be evaluated without a TWAK quote."""

    symbol: str
    token_data: dict[str, Any]
    position_size_usdc: float
    volume_24h: float
    volume_breakout: bool
    six_hour_high_break: bool
    regime_not_risk_off: bool
    rsi_in_range: bool
    derivatives_risk_clear: bool
    cheap_core_pass_count: int
    true_factor_count_without_slippage: int


MAX_UNIVERSE_TWAK_QUOTES = 2


class LocalCache:
    """Simple JSON file cache for time-series data."""

    def __init__(self, filename: str) -> None:
        self.path = Path(filename)
        self.data = self._load()

    def _load(self) -> dict[str, Any]:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def save(self) -> None:
        try:
            self.path.write_text(json.dumps(self.data), encoding="utf-8")
        except OSError:
            pass

    def add_data_point(self, symbol: str, value: float, max_age_hours: float) -> None:
        now = time.time()
        if symbol not in self.data:
            self.data[symbol] = []
        self.data[symbol].append({"timestamp": now, "value": value})

        cutoff = now - (max_age_hours * 3600)
        self.data[symbol] = [pt for pt in self.data[symbol] if pt["timestamp"] >= cutoff]

    def get_max_value(self, symbol: str, max_age_hours: float | None = None) -> float | None:
        points = self.data.get(symbol, [])
        if max_age_hours is not None:
            cutoff = time.time() - (max_age_hours * 3600)
            points = [pt for pt in points if pt.get("timestamp", 0) >= cutoff]
        values = [value for point in points if (value := self._point_value(point)) is not None]
        if not values:
            return None
        return max(values)

    def get_average_value(self, symbol: str) -> float | None:
        points = self.data.get(symbol, [])
        values = [value for point in points if (value := self._point_value(point)) is not None]
        if not values:
            return None
        return sum(values) / len(values)

    @staticmethod
    def _point_value(point: dict[str, Any]) -> float | None:
        try:
            return float(point.get("value", point.get("price")))
        except (TypeError, ValueError):
            return None


class BreakoutEngine:
    """Evaluate BSC tokens against a 4-factor core entry filter."""

    def __init__(
        self,
        settings: Settings,
        twak_interface: TWAKInterface | None = None,
    ) -> None:
        self.settings = settings
        self.twak_interface = twak_interface or TWAKInterface()
        self.price_cache = LocalCache("price_cache.json")
        self.volume_cache = LocalCache("volume_cache.json")

    def evaluate_token(
        self,
        token_data: dict[str, Any],
        portfolio_value_usdc: float,
        ml_context: Any | None = None,
    ) -> BreakoutDecision:
        """Evaluate one token against the entry filter."""

        symbol = str(token_data.get("symbol", "")).upper()
        if not is_liquid({"symbol": symbol, **token_data}):
            return BreakoutDecision(
                should_enter=False,
                symbol=symbol or None,
                position_size_usdc=0.0,
                factor_scores={},
                true_factor_count=0,
                reason="token failed liquidity filter",
            )
        if not is_tradable_symbol(symbol):
            return BreakoutDecision(
                should_enter=False,
                symbol=symbol or None,
                position_size_usdc=0.0,
                factor_scores={},
                true_factor_count=0,
                reason="symbol outside tradable target allowlist",
            )

        candidate = self._evaluate_cheap_candidate(token_data, portfolio_value_usdc)
        estimated_slippage: float | None = None
        if candidate.cheap_core_pass_count >= self._min_cheap_core_for_slippage_quote:
            estimated_slippage = self._estimate_candidate_slippage(candidate)
        decision = self._decision_from_candidate(candidate, estimated_slippage)
        return self._attach_ml_context(decision, {symbol: ml_context} if ml_context else None)

    def _evaluate_cheap_candidate(
        self,
        token_data: dict[str, Any],
        portfolio_value_usdc: float,
    ) -> _CheapCandidate:
        """Evaluate all candidate factors that do not require TWAK."""

        symbol = str(token_data.get("symbol", "")).upper()
        position_size = portfolio_value_usdc * self.settings.max_position_pct
        price = self._positive_number(token_data.get("price"))
        volume_24h = self._positive_number(token_data.get("volume_24h"))
        market_cap = self._positive_number(token_data.get("market_cap"))
        rsi = self._positive_number(token_data.get("rsi"))
        funding_rate = self._nonzero_number(token_data.get("funding_rate"))
        open_interest_change = self._nonzero_number(token_data.get("open_interest_change_pct"))

        volume_breakout = self._volume_breakout(symbol, token_data, volume_24h, market_cap)

        six_hour_high_break = self._breakout_high_break(symbol, token_data, price)

        regime_not_risk_off = self.check_regime(token_data)

        rsi_in_range = True
        if rsi is not None:
            rsi_in_range = 55.0 <= rsi <= 75.0

        derivatives_risk_clear = True
        if funding_rate is not None and open_interest_change is not None:
            derivatives_risk_clear = not (abs(funding_rate) > 0.0015 or open_interest_change < -10.0)

        cheap_core_pass_count = sum(
            1 for passed in (volume_breakout, six_hour_high_break, regime_not_risk_off) if passed
        )
        true_factor_count_without_slippage = sum(
            1
            for passed in (
                volume_breakout,
                six_hour_high_break,
                regime_not_risk_off,
                rsi_in_range,
                derivatives_risk_clear,
            )
            if passed
        )

        return _CheapCandidate(
            symbol=symbol,
            token_data=token_data,
            position_size_usdc=position_size,
            volume_24h=volume_24h or 0.0,
            volume_breakout=volume_breakout,
            six_hour_high_break=six_hour_high_break,
            regime_not_risk_off=regime_not_risk_off,
            rsi_in_range=rsi_in_range,
            derivatives_risk_clear=derivatives_risk_clear,
            cheap_core_pass_count=cheap_core_pass_count,
            true_factor_count_without_slippage=true_factor_count_without_slippage,
        )

    def _estimate_candidate_slippage(self, candidate: _CheapCandidate) -> float | None:
        return self.twak_interface.estimate_slippage_pct(
            amount=candidate.position_size_usdc,
            from_token=self.settings.default_stable_symbol,
            to_token=candidate.symbol,
        )

    def _decision_from_candidate(
        self,
        candidate: _CheapCandidate,
        estimated_slippage: float | None,
    ) -> BreakoutDecision:
        """Build a full decision after optional TWAK slippage evaluation."""

        slippage_under_cap = (
            estimated_slippage is not None
            and estimated_slippage >= 0
            and estimated_slippage < self.settings.max_slippage_pct
        )

        factor_scores = {
            "volume_breakout": candidate.volume_breakout,
            "six_hour_high_break": candidate.six_hour_high_break,
            "regime_not_risk_off": candidate.regime_not_risk_off,
            "slippage_under_cap": slippage_under_cap,
            "rsi_in_range": candidate.rsi_in_range,
            "derivatives_risk_clear": candidate.derivatives_risk_clear,
        }

        passing_core_count = candidate.cheap_core_pass_count + int(slippage_under_cap)
        true_factor_count = sum(1 for passed in factor_scores.values() if passed)

        min_core = self.settings.min_entry_factors
        should_enter = passing_core_count >= min_core and slippage_under_cap

        if should_enter:
            reason = (
                f"{passing_core_count}/{CORE_FACTOR_COUNT} core factors passed "
                f"({true_factor_count}/{TOTAL_FACTOR_COUNT} total)"
            )
        elif candidate.cheap_core_pass_count < self._min_cheap_core_for_slippage_quote:
            reason = (
                f"insufficient signal: {passing_core_count}/{CORE_FACTOR_COUNT} "
                f"core factors passed (need {min_core})"
            )
        elif not slippage_under_cap:
            reason = "slippage estimate missing, negative, or above cap"
        elif passing_core_count < min_core:
            reason = (
                f"insufficient signal: {passing_core_count}/{CORE_FACTOR_COUNT} "
                f"core factors passed (need {min_core})"
            )
        else:
            reason = (
                f"insufficient signal: {passing_core_count}/{CORE_FACTOR_COUNT} "
                f"core factors passed (need {min_core})"
            )

        return BreakoutDecision(
            should_enter=should_enter,
            symbol=candidate.symbol,
            position_size_usdc=candidate.position_size_usdc if should_enter else 0.0,
            factor_scores=factor_scores,
            true_factor_count=true_factor_count,
            reason=reason,
            estimated_slippage_pct=estimated_slippage,
        )

    def _attach_ml_context(self, decision: BreakoutDecision, ml_contexts: dict[str, Any] | None) -> BreakoutDecision:
        if not ml_contexts or decision.symbol is None:
            return decision
        ctx = ml_contexts.get(decision.symbol.upper())
        if ctx is None:
            return decision
        return BreakoutDecision(
            should_enter=decision.should_enter,
            symbol=decision.symbol,
            position_size_usdc=decision.position_size_usdc,
            factor_scores=decision.factor_scores,
            true_factor_count=decision.true_factor_count,
            reason=decision.reason,
            estimated_slippage_pct=decision.estimated_slippage_pct,
            ml_context=ctx,
        )

    def evaluate_all(
        self,
        market_snapshot: dict[str, dict[str, Any]],
        portfolio_value_usdc: float,
        ml_contexts: dict[str, Any] | None = None,
    ) -> list[BreakoutDecision]:
        """Scan target symbols and return all slippage-confirmed entry decisions."""

        candidates: list[_CheapCandidate] = []
        best_decision: BreakoutDecision | None = None
        best_volume = -1.0
        saw_target_symbol = False
        for symbol, token_data in market_snapshot.items():
            if not is_tradable_symbol(symbol):
                continue
            saw_target_symbol = True
            enriched_data = {"symbol": symbol.upper(), **token_data}
            if not is_liquid(enriched_data):
                continue
            candidate = self._evaluate_cheap_candidate(enriched_data, portfolio_value_usdc)
            candidates.append(candidate)

            unquoted_decision = self._decision_from_candidate(candidate, estimated_slippage=None)
            if self._is_better_decision(unquoted_decision, candidate.volume_24h, best_decision, best_volume):
                best_decision = unquoted_decision
                best_volume = candidate.volume_24h

        quote_candidates = sorted(
            (
                candidate
                for candidate in candidates
                if candidate.cheap_core_pass_count >= self._min_cheap_core_for_slippage_quote
            ),
            key=self._cheap_candidate_rank,
            reverse=True,
        )

        passers: list[BreakoutDecision] = []
        for candidate in quote_candidates[:MAX_UNIVERSE_TWAK_QUOTES]:
            decision = self._decision_from_candidate(
                candidate,
                self._estimate_candidate_slippage(candidate),
            )
            decision = self._attach_ml_context(decision, ml_contexts)
            if decision.should_enter:
                passers.append(decision)
            if self._is_better_decision(decision, candidate.volume_24h, best_decision, best_volume):
                best_decision = decision
                best_volume = candidate.volume_24h

        self.price_cache.save()
        self.volume_cache.save()

        if passers:
            return passers

        if best_decision is None:
            return [
                BreakoutDecision(
                    should_enter=False,
                    symbol=None,
                    position_size_usdc=0.0,
                    factor_scores={},
                    true_factor_count=0,
                    reason="no liquid target symbols available" if saw_target_symbol else "no target symbols available",
                )
            ]
        return [self._attach_ml_context(best_decision, ml_contexts)]

    def evaluate_universe(
        self,
        market_snapshot: dict[str, dict[str, Any]],
        portfolio_value_usdc: float,
        ml_contexts: dict[str, Any] | None = None,
    ) -> BreakoutDecision:
        """Scan target symbols and pick the highest-scoring candidate."""

        decisions = self.evaluate_all(market_snapshot, portfolio_value_usdc, ml_contexts)
        passers = [decision for decision in decisions if decision.should_enter]
        if passers:
            return passers[0]
        return decisions[0]

    _min_cheap_core_for_slippage_quote = 2

    def _volume_breakout(
        self,
        symbol: str,
        token_data: dict[str, Any],
        volume_24h: float | None,
        market_cap: float | None,
    ) -> bool:
        volume_1h = self._positive_number(token_data.get("volume_1h"))
        rolling_hourly_avg = self._positive_number(token_data.get("rolling_24h_hourly_volume_avg"))
        if rolling_hourly_avg is None and volume_1h is not None and volume_24h is not None:
            rolling_hourly_avg = volume_24h / 24.0
        breakout_mult = self.settings.ml_volume_breakout_multiplier
        cache_mult = self.settings.ml_volume_cache_multiplier
        if volume_1h is not None and rolling_hourly_avg is not None and rolling_hourly_avg > 0:
            return volume_1h > breakout_mult * rolling_hourly_avg

        if volume_24h is not None:
            self.volume_cache.add_data_point(symbol, volume_24h, max_age_hours=24)
            avg_vol = self.volume_cache.get_average_value(symbol)
            if avg_vol is not None and avg_vol > 0:
                return volume_24h > cache_mult * avg_vol
            if market_cap is not None:
                return volume_24h > 0.05 * market_cap
        return False

    def _breakout_high_break(
        self,
        symbol: str,
        token_data: dict[str, Any],
        price: float | None,
    ) -> bool:
        if price is None:
            return False

        high_3h = self._positive_number(token_data.get("high_3h"))
        buffer_multiplier = 1.0 + self.settings.breakout_buffer
        if high_3h is not None:
            broke = price > high_3h * buffer_multiplier
        else:
            max_price = self.price_cache.get_max_value(
                symbol,
                max_age_hours=self.settings.breakout_lookback_hours,
            )
            broke = max_price is not None and price > max_price * buffer_multiplier
        self.price_cache.add_data_point(symbol, price, max_age_hours=self.settings.breakout_lookback_hours)
        return broke

    def check_regime(self, token_data: dict[str, Any], bnb_data: dict[str, Any] | None = None) -> bool:
        bnb_source = bnb_data if bnb_data is not None else token_data
        bnb_change_1h = self._bnb_change_1h_fraction(bnb_source, separate_bnb_data=bnb_data is not None)
        token_change_1h = self._token_change_fraction(token_data, hours=1)
        token_change_24h = self._token_change_fraction(token_data, hours=24)
        bnb_ok = bnb_change_1h is not None and bnb_change_1h > self.settings.bnb_regime_threshold
        token_1h_ok = token_change_1h is not None and token_change_1h > self.settings.token_regime_1h_min
        token_24h_ok = token_change_24h is not None and token_change_24h > self.settings.token_regime_24h_min
        return bnb_ok and token_1h_ok and token_24h_ok

    def _bnb_change_1h_fraction(self, data: dict[str, Any], separate_bnb_data: bool) -> float | None:
        if separate_bnb_data:
            return self._first_change_fraction(
                data,
                (
                    ("percent_change_1h", "fraction"),
                    ("bnb_percent_change_1h", "fraction"),
                    ("price_change_percentage_1h", "percent_points"),
                    ("change_1h", "percent_points"),
                    ("bnb_1h_trend_pct", "percent_points"),
                ),
            )
        return self._first_change_fraction(
            data,
            (
                ("bnb_percent_change_1h", "fraction"),
                ("bnb_1h_trend_pct", "percent_points"),
                ("bnb_1h_change_pct", "percent_points"),
            ),
        )

    def _token_change_fraction(self, data: dict[str, Any], hours: int) -> float | None:
        return self._first_change_fraction(
            data,
            (
                (f"token_percent_change_{hours}h", "fraction"),
                (f"token_change_{hours}h", "fraction"),
                (f"percent_change_{hours}h", "percent_points"),
                (f"price_change_percentage_{hours}h", "percent_points"),
                (f"change_{hours}h", "percent_points"),
            ),
        )

    def _first_change_fraction(
        self,
        data: dict[str, Any],
        fields: tuple[tuple[str, str], ...],
    ) -> float | None:
        for key, mode in fields:
            number = self._number(data.get(key))
            if number is None:
                continue
            if mode == "percent_points":
                return number / 100.0
            return number
        return None

    @staticmethod
    def _cheap_candidate_rank(candidate: _CheapCandidate) -> tuple[int, int, float]:
        return (
            candidate.cheap_core_pass_count,
            candidate.true_factor_count_without_slippage,
            candidate.volume_24h,
        )

    @staticmethod
    def _is_better_decision(
        candidate: BreakoutDecision,
        candidate_volume: float,
        best: BreakoutDecision | None,
        best_volume: float,
    ) -> bool:
        if best is None:
            return True
        if candidate.true_factor_count > best.true_factor_count:
            return True
        return candidate.true_factor_count == best.true_factor_count and candidate_volume > best_volume

    @staticmethod
    def _positive_number(value: Any) -> float | None:
        number = BreakoutEngine._number(value)
        if number is None or number <= 0:
            return None
        return number

    @staticmethod
    def _nonzero_number(value: Any) -> float | None:
        number = BreakoutEngine._number(value)
        if number is None or number == 0:
            return None
        return number

    @staticmethod
    def _number(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

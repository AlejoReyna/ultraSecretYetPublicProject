"""Integration tests for scalping v1.0 strategy."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import pytest

from src.config.settings import Settings
from src.strategy.entry_types import EntryCandidate
from src.strategy.guardrails import TradeRecord
from src.strategy.regime_detector import MarketRegime, RegimeResult
from src.strategy.scalping_engine import ScalpingEngine
from src.strategy.scalping_guardrails import ScalpingGuardrails
from src.strategy.scalping_position_manager import ScalpingPositionManager
from src.strategy.sentiment_tier1 import SentimentResult
from src.strategy.volatility import PriceCache


def _settings(**overrides: Any) -> Settings:
    base = Settings(strategy_mode="scalping")
    return base.model_copy(update=overrides)


def _regime() -> RegimeResult:
    return RegimeResult(
        regime=MarketRegime.TRENDING_UP,
        score=0.8,
        position_multiplier=1.0,
        min_entry_factors=4,
        max_slippage_pct=0.01,
        reasons=[],
        sentiment_delta=0.0,
        sentiment_fragility="NONE",
    )


def _sentiment(**overrides: Any) -> SentimentResult:
    payload = {
        "fear_greed_index": 55,
        "fear_greed_classification": "Neutral",
        "funding_rate_btc": 0.0001,
        "open_interest_btc": 1_000_000.0,
        "gas_price_gwei": 1.0,
        "gas_avg_24h_gwei": 1.0,
        "sentiment_delta": 0.0,
        "regime_fragility": "NONE",
    }
    payload.update(overrides)
    return SentimentResult(**payload)


def _risk_decision() -> Any:
    from src.strategy.guardrails import RiskDecision, RiskState

    return RiskDecision(
        state=RiskState.NORMAL,
        allow_new_entries=True,
        position_multiplier=1.0,
        max_slippage_pct=0.01,
        max_daily_trades=10,
        base_risk_per_trade_pct=0.01,
        reasons=[],
    )


def _seed_price_cache(cache: PriceCache, symbol: str, *, price: float, volume: float, count: int = 12) -> None:
    now = datetime.now(timezone.utc)
    for index in range(count):
        close = price * (0.995 + (index / max(count - 1, 1)) * 0.01)
        cache.add_ohlcv(
            symbol,
            close * 0.99,
            close * 1.01,
            close * 0.98,
            close,
            volume if index == count - 1 else volume * 0.5,
            now - timedelta(minutes=5 * (count - index)),
        )


def _token_payload(**overrides: Any) -> dict[str, Any]:
    payload = {
        "price": 10.0,
        "market_cap": 55_000_000.0,
        "estimated_slippage_pct": 0.001,
        "percent_change_1h": 0.01,
        "volume_24h": 6_000_000.0,
    }
    payload.update(overrides)
    return payload


def test_scalping_score_60_enters() -> None:
    settings = _settings()
    cache = PriceCache()
    _seed_price_cache(cache, "CAKE", price=10.0, volume=200_000.0)
    engine = ScalpingEngine(settings, cache)
    score, factors = engine.score_token(
        "CAKE",
        _token_payload(),
        _regime(),
        _sentiment(gas_price_gwei=1.0),
    )
    assert score >= 60
    assert factors["gas_viable"] is True


def test_scalping_score_59_waits() -> None:
    settings = _settings()
    cache = PriceCache()
    _seed_price_cache(cache, "CAKE", price=10.0, volume=100.0)
    engine = ScalpingEngine(settings, cache)
    score, _ = engine.score_token(
        "CAKE",
        _token_payload(estimated_slippage_pct=0.004),
        _regime(),
        _sentiment(gas_price_gwei=1.0),
    )
    assert score <= 59


def test_scalping_tp_exit() -> None:
    manager = ScalpingPositionManager(_settings())
    manager.open_position("CAKE", 10.0, 100.0, 1000.0)
    manager.check_exits({"CAKE": {"price": 101.6}}, None)
    signal = manager.pop_pending_exit()
    assert signal is not None
    assert signal.reason == "tp"


def test_scalping_sl_exit() -> None:
    manager = ScalpingPositionManager(_settings())
    position = manager.open_position("CAKE", 10.0, 100.0, 1000.0)
    position.opened_at = datetime.now(timezone.utc) - timedelta(minutes=5)
    manager.check_exits({"CAKE": {"price": 99.1}}, None)
    signal = manager.pop_pending_exit()
    assert signal is not None
    assert signal.reason == "sl"


def test_scalping_time_stop_20min() -> None:
    manager = ScalpingPositionManager(_settings())
    position = manager.open_position("CAKE", 10.0, 100.0, 1000.0)
    position.opened_at = datetime.now(timezone.utc) - timedelta(minutes=21)
    manager.check_exits({"CAKE": {"price": 100.2}}, None)
    signal = manager.pop_pending_exit()
    assert signal is not None
    assert signal.reason == "time_stop"


def test_scalping_time_stop_30min_absolute() -> None:
    manager = ScalpingPositionManager(_settings())
    position = manager.open_position("CAKE", 10.0, 100.0, 1000.0)
    position.opened_at = datetime.now(timezone.utc) - timedelta(minutes=31)
    manager.check_exits({"CAKE": {"price": 100.4}}, None)
    signal = manager.pop_pending_exit()
    assert signal is not None
    assert signal.reason == "max_hold"


def test_scalping_daily_loss_cap_pauses() -> None:
    settings = _settings()
    guardrails = ScalpingGuardrails(settings)
    guardrails._scalping_daily_pnl_pct = -0.025
    assert guardrails.scalping_entries_allowed(10_000.0) is False


def test_scalping_cooldown_blocks_same_symbol() -> None:
    manager = ScalpingPositionManager(_settings())
    manager.open_position("CAKE", 10.0, 100.0, 1000.0)
    manager.close_position("CAKE")
    assert manager.is_symbol_on_cooldown("CAKE") is True


def test_scalping_no_entry_when_position_open() -> None:
    settings = _settings()
    cache = PriceCache()
    _seed_price_cache(cache, "CAKE", price=10.0, volume=200_000.0)
    engine = ScalpingEngine(settings, cache)
    candidate = engine.evaluate_universe(
        {"CAKE": _token_payload()},
        10_000.0,
        _regime(),
        _risk_decision(),
        sentiment_result=_sentiment(gas_price_gwei=1.0),
        exclude_symbols={"CAKE"},
    )
    assert candidate is None


def test_scalping_gas_filter_blocks() -> None:
    settings = _settings()
    cache = PriceCache()
    _seed_price_cache(cache, "CAKE", price=10.0, volume=200_000.0)
    engine = ScalpingEngine(settings, cache)
    score, factors = engine.score_token(
        "CAKE",
        _token_payload(),
        _regime(),
        _sentiment(gas_price_gwei=8.0),
    )
    assert factors["gas_viable"] is False
    assert score <= 90


def test_scalping_position_size_is_one_percent() -> None:
    settings = _settings(scalping_position_pct=0.01)
    cache = PriceCache()
    _seed_price_cache(cache, "CAKE", price=10.0, volume=200_000.0)
    engine = ScalpingEngine(settings, cache)
    candidate = engine.evaluate_universe(
        {"CAKE": _token_payload()},
        10_000.0,
        _regime(),
        _risk_decision(),
        sentiment_result=_sentiment(gas_price_gwei=1.0),
    )
    assert candidate is not None
    assert candidate.position_size_usdc == pytest.approx(100.0)


def test_scalping_consecutive_loss_cooldown() -> None:
    settings = _settings()
    guardrails = ScalpingGuardrails(settings)
    for _ in range(3):
        guardrails.record_scalping_trade(
            TradeRecord("CAKE", "sell", 100.0, -5.0, datetime.now(timezone.utc)),
            10_000.0,
            exit_reason="sl",
        )
    assert guardrails.check_consecutive_loss_cooldown() is True


def test_scalping_best_near_miss_returns_scored_symbol_when_entry_threshold_not_met() -> None:
    settings = _settings(scalping_entry_score_min=60.0)
    cache = PriceCache()
    _seed_price_cache(cache, "CAKE", price=10.0, volume=200_000.0)
    engine = ScalpingEngine(settings, cache)
    near_miss = engine.best_near_miss(
        {"CAKE": _token_payload(estimated_slippage_pct=0.002, percent_change_1h=0.01)},
        10_000.0,
        _regime(),
        _risk_decision(),
        sentiment_result=_sentiment(gas_price_gwei=8.0),
    )
    assert near_miss is not None
    assert near_miss.symbol == "CAKE"
    assert near_miss.factor_scores
    assert (near_miss.entry_score or 0.0) < 60.0
    assert "scalping score" in near_miss.reason


def test_scalping_best_near_miss_scores_symbols_without_bsc_contract_mapping() -> None:
    settings = _settings(scalping_entry_score_min=60.0)
    cache = PriceCache()
    engine = ScalpingEngine(settings, cache)
    near_miss = engine.best_near_miss(
        {
            "PENGU": {
                "symbol": "PENGU",
                "price": 0.05,
                "volume_24h": 20_000_000.0,
                "market_cap": 500_000_000.0,
                "percent_change_1h": 0.02,
            }
        },
        1_000.0,
        _regime(),
        _risk_decision(),
        sentiment_result=_sentiment(gas_price_gwei=1.0),
    )
    assert near_miss is not None
    assert near_miss.symbol == "PENGU"
    assert near_miss.entry_score is not None

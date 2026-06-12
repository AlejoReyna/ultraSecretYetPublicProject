"""Tests for the breakout engine."""

from __future__ import annotations

import time
from typing import Any

from src.config import tokens as token_config
from src.config.settings import Settings
from src.config.tokens import ELIGIBLE_149_SYMBOLS, is_liquid
from src.strategy.breakout_engine import BreakoutEngine


class FakeTWAKSlippage:
    """Stub TWAK slippage estimator for breakout-engine tests."""

    def __init__(self, slippage: float | None | dict[str, float | None]) -> None:
        self.slippage = slippage
        self.calls: list[tuple[float, str, str]] = []

    def estimate_slippage_pct(
        self,
        amount: float,
        from_token: str,
        to_token: str,
        chain: str = "bsc",
    ) -> float | None:
        self.calls.append((amount, from_token, to_token))
        if isinstance(self.slippage, dict):
            return self.slippage.get(to_token.upper())
        return self.slippage


def _engine(
    settings: Settings | None = None,
    slippage: float | None = 0.005,
    **kwargs: Any,
) -> BreakoutEngine:
    resolved_settings = settings or Settings()
    twak = kwargs.pop("twak", FakeTWAKSlippage(slippage))
    return BreakoutEngine(resolved_settings, twak_interface=twak)  # type: ignore[arg-type]


def _token(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "symbol": "CAKE",
        "price": 10.5,
        "volume_24h": 10_000_000.0,
        "market_cap": 100_000_000.0,
        "bnb_1h_trend_pct": 0.5,
        "token_percent_change_1h": 0.003,
        "token_percent_change_24h": 0.02,
        "rsi": 62.0,
        "estimated_slippage_pct": 0.005,
        "funding_rate": 0.002,
        "open_interest_change_pct": -1.0,
    }
    data.update(overrides)
    return data


def _engine_with_price_high(
    symbol: str,
    prior_high: float,
    slippage: float | None = 0.005,
    settings: Settings | None = None,
) -> BreakoutEngine:
    engine = _engine(slippage=slippage, settings=settings)
    engine.price_cache.data = {symbol: [{"timestamp": time.time(), "value": prior_high}]}
    engine.volume_cache.data = {
        symbol: [{"timestamp": time.time() - 3600, "value": 500_000.0}],
    }
    return engine


def _seed_breakout_caches(engine: BreakoutEngine, symbols: list[str]) -> None:
    engine.price_cache.data = {
        symbol: [{"timestamp": time.time(), "value": 10.0}]
        for symbol in symbols
    }
    engine.volume_cache.data = {
        symbol: [{"timestamp": time.time() - 3600, "value": 500_000.0}]
        for symbol in symbols
    }


def test_three_actionable_core_factors_enters() -> None:
    engine = _engine_with_price_high("CAKE", 10.0)
    decision = engine.evaluate_token(_token(), 10000.0)

    assert decision.should_enter is True
    assert decision.factor_scores["volume_breakout"] is True
    assert decision.factor_scores["six_hour_high_break"] is True
    assert decision.factor_scores["regime_not_risk_off"] is True
    assert decision.factor_scores["slippage_under_cap"] is True
    assert decision.position_size_usdc == 500.0
    assert "3/3 core factors passed" in decision.reason


def test_missing_rsi_and_derivatives_fail_optional_factors_but_do_not_veto_core_entry() -> None:
    engine = _engine_with_price_high("CAKE", 10.0)
    decision = engine.evaluate_token(
        _token(rsi=None, funding_rate=None, open_interest_change_pct=None),
        10000.0,
    )

    assert decision.should_enter is True
    assert decision.factor_scores["rsi_in_range"] is False
    assert decision.factor_scores["derivatives_risk_clear"] is False


def test_stablecoin_targets_are_not_directional_entries() -> None:
    engine = _engine()

    decision = engine.evaluate_token(_token(symbol="USDC", bnb_1h_trend_pct=0.1, funding_rate=0.0), 10000.0)

    assert decision.should_enter is False
    assert decision.reason == "symbol outside tradable target allowlist"


def test_high_rsi_weakens_optional_score_only() -> None:
    normal = _engine_with_price_high("CAKE", 10.0).evaluate_token(_token(), 10000.0)
    hot = _engine_with_price_high("CAKE", 10.0).evaluate_token(_token(rsi=81.0), 10000.0)
    assert hot.factor_scores["rsi_in_range"] is False
    assert hot.should_enter == normal.should_enter
    assert hot.true_factor_count == normal.true_factor_count - 1


def test_universe_chooses_highest_scoring_target_token() -> None:
    engine = _engine()
    engine.volume_cache.data = {
        "LINK": [{"timestamp": time.time() - 3600, "value": 500_000.0}],
    }
    engine.price_cache.data["LINK"] = [{"timestamp": time.time(), "value": 9.0}]
    engine.price_cache.data["CAKE"] = [{"timestamp": time.time(), "value": 10.0}]
    snapshot = {
        "NOTREAL": _token(symbol="NOTREAL", volume_24h=999999.0),
        "CAKE": _token(volume_24h=3000.0, market_cap=100_000.0, estimated_slippage_pct=0.02),
        "LINK": _token(
            symbol="LINK",
            price=10.5,
            volume_24h=12_000_000.0,
            market_cap=120_000_000.0,
            bnb_1h_trend_pct=0.1,
            funding_rate=0.0,
        ),
    }
    decision = engine.evaluate_universe(snapshot, 10000.0)
    assert decision.symbol == "LINK"
    assert decision.should_enter is True


def test_universe_quotes_only_best_ranked_candidate_when_it_enters() -> None:
    twak = FakeTWAKSlippage({"LINK": 0.005, "CAKE": 0.005, "AAVE": 0.005})
    engine = BreakoutEngine(Settings(), twak_interface=twak)  # type: ignore[arg-type]
    _seed_breakout_caches(engine, ["LINK", "CAKE", "AAVE"])
    snapshot = {
        "CAKE": _token(symbol="CAKE", volume_24h=8_000_000.0, funding_rate=0.0001),
        "AAVE": _token(symbol="AAVE", volume_24h=7_000_000.0, funding_rate=0.0001),
        "LINK": _token(symbol="LINK", volume_24h=12_000_000.0, funding_rate=0.0001),
    }

    decision = engine.evaluate_universe(snapshot, 10000.0)

    assert decision.symbol == "LINK"
    assert decision.should_enter is True
    assert decision.estimated_slippage_pct == 0.005
    # evaluate_all quotes up to MAX_UNIVERSE_TWAK_QUOTES candidates (best
    # first) so the ML ranker can choose among multiple quoted passers.
    assert twak.calls[0] == (500.0, "USDC", "LINK")
    assert len(twak.calls) <= 2


def test_universe_quotes_runner_up_only_when_best_slippage_fails() -> None:
    twak = FakeTWAKSlippage({"LINK": 0.02, "CAKE": 0.005, "AAVE": 0.005})
    engine = BreakoutEngine(Settings(), twak_interface=twak)  # type: ignore[arg-type]
    _seed_breakout_caches(engine, ["LINK", "CAKE", "AAVE"])
    snapshot = {
        "AAVE": _token(symbol="AAVE", volume_24h=7_000_000.0, funding_rate=0.0001),
        "LINK": _token(symbol="LINK", volume_24h=12_000_000.0, funding_rate=0.0001),
        "CAKE": _token(symbol="CAKE", volume_24h=8_000_000.0, funding_rate=0.0001),
    }

    decision = engine.evaluate_universe(snapshot, 10000.0)

    assert decision.symbol == "CAKE"
    assert decision.should_enter is True
    assert decision.estimated_slippage_pct == 0.005
    assert twak.calls == [(500.0, "USDC", "LINK"), (500.0, "USDC", "CAKE")]


def test_missing_or_zero_data_fails_closed() -> None:
    engine = _engine(slippage=None)
    decision = engine.evaluate_token(
        _token(
            rsi=None,
            funding_rate=0.0,
            open_interest_change_pct=0.0,
            volume_1h=100.0,
            rolling_24h_hourly_volume_avg=1000.0,
            bnb_1h_trend_pct=None,
        ),
        10000.0,
    )

    assert decision.factor_scores["volume_breakout"] is False
    assert decision.factor_scores["regime_not_risk_off"] is False
    assert decision.factor_scores["rsi_in_range"] is False
    assert decision.factor_scores["slippage_under_cap"] is False
    assert decision.factor_scores["derivatives_risk_clear"] is True
    assert decision.should_enter is False


def test_missing_slippage_blocks_entry_even_when_other_factors_pass() -> None:
    engine = _engine_with_price_high("CAKE", 10.0, slippage=None)
    decision = engine.evaluate_token(
        _token(
            bnb_1h_trend_pct=0.1,
            funding_rate=0.0001,
            open_interest_change_pct=1.0,
        ),
        10000.0,
    )

    assert decision.factor_scores["slippage_under_cap"] is False
    assert decision.should_enter is False
    assert decision.reason == "slippage estimate missing, negative, or above cap"


def test_slippage_factor_with_real_estimate() -> None:
    twak = FakeTWAKSlippage(0.008)
    engine = _engine_with_price_high("CAKE", 10.0)
    engine.twak_interface = twak  # type: ignore[assignment]

    decision = engine.evaluate_token(_token(), 10000.0)

    assert decision.factor_scores["slippage_under_cap"] is True
    assert twak.calls == [(500.0, "USDC", "CAKE")]


def test_skips_twak_quote_when_cheap_core_factors_fail() -> None:
    twak = FakeTWAKSlippage(0.005)
    engine = BreakoutEngine(Settings(), twak_interface=twak)  # type: ignore[arg-type]
    engine.evaluate_token(
        _token(bnb_1h_trend_pct=-5.0, volume_24h=None),
        10000.0,
    )

    assert twak.calls == []


def test_slippage_factor_missing_estimate() -> None:
    twak = FakeTWAKSlippage(None)
    engine = _engine_with_price_high("CAKE", 10.0)
    engine.twak_interface = twak  # type: ignore[assignment]

    decision = engine.evaluate_token(_token(), 10000.0)

    assert decision.factor_scores["slippage_under_cap"] is False
    assert decision.should_enter is False


def test_negative_slippage_blocks_entry() -> None:
    twak = FakeTWAKSlippage(-0.001)
    engine = _engine_with_price_high("CAKE", 10.0)
    engine.twak_interface = twak  # type: ignore[assignment]

    decision = engine.evaluate_token(_token(), 10000.0)

    assert decision.factor_scores["slippage_under_cap"] is False
    assert decision.should_enter is False
    assert decision.reason == "slippage estimate missing, negative, or above cap"


def test_insufficient_core_factors_reports_count() -> None:
    engine = _engine()
    decision = engine.evaluate_token(
        _token(
            bnb_1h_trend_pct=-5.0,
            volume_1h=100.0,
            rolling_24h_hourly_volume_avg=1000.0,
            estimated_slippage_pct=0.005,
        ),
        10000.0,
    )

    assert decision.should_enter is False
    assert decision.reason in {
        "slippage estimate missing, negative, or above cap",
        "insufficient signal: 0/3 core factors passed (need 3)",
        "insufficient signal: 1/3 core factors passed (need 3)",
    }


def test_eligible_rules_list_contains_149_entries() -> None:
    assert len(ELIGIBLE_149_SYMBOLS) == 149


def test_target_symbols_are_deduplicated_eligible_universe() -> None:
    from src.config.tokens import TARGET_SYMBOLS

    assert len(TARGET_SYMBOLS) == 148
    assert len(TARGET_SYMBOLS) == len(set(TARGET_SYMBOLS))


def test_liquidity_blacklist_marks_live_illiquid_symbols_untradeable() -> None:
    assert is_liquid({"symbol": "lisUSD", "volume_24h": 100_000_000.0, "market_cap": 1_000_000_000.0}) is False


def test_liquidity_soft_filter_skips_thin_target_before_quote() -> None:
    twak = FakeTWAKSlippage(0.005)
    engine = BreakoutEngine(Settings(), twak_interface=twak)  # type: ignore[arg-type]

    decision = engine.evaluate_token(
        _token(volume_24h=4_999_999.0, market_cap=100_000_000.0),
        10000.0,
    )

    assert decision.should_enter is False
    assert decision.reason == "token failed liquidity filter"
    assert twak.calls == []


def test_blacklisted_tradable_token_skips_before_quote(monkeypatch: Any) -> None:
    monkeypatch.setattr(token_config, "LIQUIDITY_BLACKLIST", {"CAKE"})
    twak = FakeTWAKSlippage(0.005)
    engine = _engine_with_price_high("CAKE", 10.0)
    engine.twak_interface = twak  # type: ignore[assignment]

    decision = engine.evaluate_token(_token(), 10000.0)

    assert decision.should_enter is False
    assert decision.reason == "token failed liquidity filter"
    assert twak.calls == []


def test_volume_breakout_uses_cmc_1h_fields_when_present() -> None:
    engine = _engine_with_price_high("CAKE", 10.0)
    passing = engine.evaluate_token(
        _token(volume_1h=2600.0, rolling_24h_hourly_volume_avg=1000.0),
        10000.0,
    )
    failing = engine.evaluate_token(
        _token(volume_1h=1500.0, rolling_24h_hourly_volume_avg=1000.0, volume_24h=5_000_000.0),
        10000.0,
    )

    assert passing.factor_scores["volume_breakout"] is True
    assert failing.factor_scores["volume_breakout"] is False


def test_volume_breakout_falls_back_to_24h_cache_without_cmc_hourly_fields() -> None:
    engine = _engine_with_price_high("CAKE", 10.0)
    decision = engine.evaluate_token(_token(volume_24h=10_000_000.0), 10000.0)

    assert decision.factor_scores["volume_breakout"] is True


def test_volume_breakout_derives_hourly_average_from_24h_when_needed() -> None:
    engine = _engine_with_price_high("CAKE", 10.0)
    passing = engine.evaluate_token(
        _token(volume_1h=900_000.0, rolling_24h_hourly_volume_avg=None, volume_24h=10_000_000.0),
        10000.0,
    )
    failing = engine.evaluate_token(
        _token(volume_1h=700_000.0, rolling_24h_hourly_volume_avg=None, volume_24h=10_000_000.0),
        10000.0,
    )

    assert passing.factor_scores["volume_breakout"] is True
    assert failing.factor_scores["volume_breakout"] is False


def test_three_hour_breakout_uses_cmc_high_3h_with_buffer_when_present() -> None:
    engine = _engine()
    engine.price_cache.data = {}
    passing = engine.evaluate_token(_token(price=2.11, high_3h=2.10), 10000.0)
    engine.price_cache.data = {}
    failing = engine.evaluate_token(_token(price=2.104, high_3h=2.10), 10000.0)

    assert passing.factor_scores["six_hour_high_break"] is True
    assert failing.factor_scores["six_hour_high_break"] is False


def test_three_hour_breakout_falls_back_to_price_cache_without_high_3h() -> None:
    engine = _engine_with_price_high("CAKE", 10.0)
    decision = engine.evaluate_token(_token(price=10.03, high_3h=None), 10000.0)

    assert decision.factor_scores["six_hour_high_break"] is True


def test_three_hour_breakout_ignores_stale_cache_points() -> None:
    engine = _engine()
    engine.price_cache.data = {
        "CAKE": [{"timestamp": time.time() - (4 * 3600), "value": 10.0}],
    }
    decision = engine.evaluate_token(_token(price=10.5, high_3h=None), 10000.0)

    assert decision.factor_scores["six_hour_high_break"] is False


def test_flat_bnb_regime_is_not_risk_off() -> None:
    engine = _engine_with_price_high("CAKE", 10.0)
    decision = engine.evaluate_token(_token(bnb_1h_trend_pct=0.0), 10000.0)

    assert decision.factor_scores["regime_not_risk_off"] is True
    assert decision.should_enter is True


def test_bnb_regime_risk_off_halves_size_without_veto() -> None:
    engine = _engine_with_price_high("CAKE", 10.0)
    decision = engine.evaluate_token(_token(bnb_1h_trend_pct=-1.1), 10000.0)

    assert decision.factor_scores["regime_not_risk_off"] is False
    assert decision.should_enter is True
    assert decision.position_size_usdc == 250.0


def test_token_regime_requires_positive_1h_and_24h_guard() -> None:
    engine = _engine_with_price_high("CAKE", 10.0)
    weak_1h = engine.evaluate_token(_token(token_percent_change_1h=0.0), 10000.0)
    weak_24h = engine.evaluate_token(_token(token_percent_change_24h=-0.09), 10000.0)

    assert weak_1h.factor_scores["regime_not_risk_off"] is False
    assert weak_24h.factor_scores["regime_not_risk_off"] is False


def test_regime_accepts_explicit_separate_bnb_data() -> None:
    engine = _engine()

    assert engine.check_regime(
        {"token_percent_change_1h": 0.003, "token_percent_change_24h": 0.0},
        {"percent_change_1h": -0.009},
    ) is True
    assert engine.check_regime(
        {"token_percent_change_1h": 0.003, "token_percent_change_24h": 0.0},
        {"percent_change_1h": -0.011},
    ) is False


def test_regime_factor_does_not_count_against_min_entry_factors() -> None:
    engine = _engine_with_price_high("CAKE", 10.0)
    decision = engine.evaluate_token(
        _token(bnb_1h_trend_pct=-5.0, volume_24h=10_000_000.0),
        10000.0,
    )

    assert decision.factor_scores["regime_not_risk_off"] is False
    assert decision.should_enter is True
    assert decision.position_size_usdc == 250.0


def test_min_entry_factors_three_allows_one_missing_core_when_configured() -> None:
    settings = Settings(min_entry_factors=3)
    engine = _engine_with_price_high("CAKE", 10.0, settings=settings)
    decision = engine.evaluate_token(
        _token(bnb_1h_trend_pct=-5.0, volume_24h=10_000_000.0),
        10000.0,
    )

    assert decision.should_enter is True


def test_gold_tokens_are_excluded_from_momentum_candidates() -> None:
    engine = _engine_with_price_high("CAKE", 10.0)
    engine.price_cache.data["XAUT"] = [{"timestamp": time.time(), "value": 3000.0}]
    snapshot = {
        "XAUT": _token(
            symbol="XAUT",
            price=3100.0,
            volume_24h=1_000_000_000.0,
            market_cap=10_000_000_000.0,
            funding_rate=0.0,
            open_interest_change_pct=0.0,
        ),
        "CAKE": _token(funding_rate=0.0, open_interest_change_pct=0.0),
    }

    decision = engine.evaluate_universe(snapshot, 10000.0)

    assert decision.symbol == "CAKE"

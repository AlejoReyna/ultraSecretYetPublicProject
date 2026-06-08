"""Tests for MLFeatureCache CMC metrics history."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.data.ml_feature_cache import MLFeatureCache


def test_cmc_metrics_prior_and_funding_history(tmp_path) -> None:
    cache = MLFeatureCache(tmp_path / "ml_cache.sqlite")
    now = datetime.now(timezone.utc)
    prior_ts = now - timedelta(hours=25)
    mid_ts = now - timedelta(hours=12)

    cache.record_cmc_metrics(
        {
            "CAKE": {"fear_greed_index": 40, "funding_rate": 0.0001},
        },
        timestamp=prior_ts,
    )
    cache.record_cmc_metrics(
        {
            "CAKE": {"fear_greed_index": 55, "funding_rate": 0.0002},
        },
        timestamp=mid_ts,
    )
    cache.record_cmc_metrics(
        {
            "CAKE": {"fear_greed_index": 60, "funding_rate": 0.0003},
        },
        timestamp=now,
    )

    prior = cache.get_fear_greed_prior(hours_ago=24.0)
    assert prior is not None
    assert abs(prior - 0.40) < 0.01

    history = cache.get_funding_history("CAKE", days=7.0)
    assert len(history) == 3
    assert history[-1] == 0.0003

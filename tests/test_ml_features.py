"""Tests for strategy-specific ML features."""

from __future__ import annotations

import pandas as pd

from src.ml.features.pipeline import FeaturePipeline
from src.ml.features.strategy_features import STRATEGY_FEATURE_NAMES, compute_strategy_features


def _synthetic_ohlcv(rows: int = 60) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=rows, freq="15min", tz="UTC")
    close = pd.Series([1.0 + 0.001 * idx for idx in range(rows)], dtype=float)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close - 0.001,
            "high": close + 0.002,
            "low": close - 0.002,
            "close": close,
            "volume": 1000 + pd.Series(range(rows)),
        }
    )


def test_strategy_features_present_in_pipeline() -> None:
    names = FeaturePipeline.feature_names()
    for required in ("volume_skew_3h_6h", "bnb_residual_1h", "funding_rate_percentile"):
        assert required in names


def test_strategy_features_non_nan() -> None:
    ohlcv = _synthetic_ohlcv()
    bnb = _synthetic_ohlcv()
    row = compute_strategy_features(
        "CAKE",
        ohlcv,
        {"funding_rate": 0.0002, "fear_greed_index": 55},
        bnb_ohlcv=bnb,
        funding_history=[0.0001, 0.0002, 0.0003],
        token_win_rate=0.6,
    )
    for name in STRATEGY_FEATURE_NAMES:
        assert name in row
        assert pd.notna(row[name])

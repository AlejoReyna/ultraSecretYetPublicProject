"""Tests for FeaturePipeline."""

from __future__ import annotations

import pandas as pd

from src.ml.features.pipeline import FeaturePipeline


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


def test_feature_names_stable_order() -> None:
    names = FeaturePipeline.feature_names()
    assert names[0] == "ret_1"
    assert "funding_rate" in names
    assert "bnb_corr_48" in names


def test_build_row_positive_return_on_uptrend() -> None:
    ohlcv = _synthetic_ohlcv()
    row = FeaturePipeline.build_row(
        "CAKE",
        ohlcv,
        {"funding_rate": 0.0001, "volume_1h": 2000, "rolling_24h_hourly_volume_avg": 1000},
    )
    assert row["ret_1"] >= 0.0
    assert row["funding_rate"] == 0.0001


def test_build_matrix_row_matches_feature_names() -> None:
    ohlcv = _synthetic_ohlcv()
    row = FeaturePipeline.build_row("CAKE", ohlcv, {})
    vector = FeaturePipeline.build_matrix_row(row)
    assert len(vector) == len(FeaturePipeline.feature_names())

"""Tests for composite momentum labels."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.ml.labels import compute_label, vectorized_composite_label


def _ohlcv_from_close(closes: list[float]) -> pd.DataFrame:
    close = pd.Series(closes, dtype=float)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=len(closes), freq="15min", tz="UTC"),
            "open": close,
            "high": close + 0.01,
            "low": close - 0.005,
            "close": close,
            "volume": 1000.0,
        }
    )


def test_composite_label_positive_on_sustained_move() -> None:
    closes = [1.0] * 5 + [1.01, 1.02, 1.03, 1.025] + [1.024] * 20
    frame = _ohlcv_from_close(closes)
    frame.loc[5:8, "high"] = [1.03, 1.04, 1.05, 1.04]
    frame.loc[5:8, "low"] = [0.995, 1.0, 1.01, 1.02]
    label = compute_label(frame, 4)
    assert label == 1


def test_positive_class_rate_in_target_band_on_realistic_series() -> None:
    matrix_path = Path("data/historical/feature_matrix.parquet")
    if not matrix_path.exists():
        pytest.skip("feature matrix not built")

    frame = pd.read_parquet(matrix_path)
    rate = float(frame["label"].mean())
    assert 0.20 <= rate <= 0.40

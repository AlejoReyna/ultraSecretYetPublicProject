"""Tests for momentum labels."""

from __future__ import annotations

import pandas as pd

from src.ml.labels import momentum_label, vectorized_momentum_label


def test_momentum_label_positive_when_forward_spike() -> None:
    close = pd.Series([1.0, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09])
    labels = momentum_label(close, horizon=3, threshold=0.02)
    assert labels.iloc[0] == 1.0


def test_momentum_label_zero_when_flat() -> None:
    close = pd.Series([1.0] * 10)
    labels = momentum_label(close, horizon=3, threshold=0.02)
    assert labels.iloc[0] == 0.0


def test_vectorized_label_last_rows_nan() -> None:
    close = pd.Series([1.0, 1.01, 1.02, 1.03, 1.04, 1.05])
    labels = vectorized_momentum_label(close, horizon=2, threshold=0.02)
    assert pd.isna(labels.iloc[-1])
    assert pd.isna(labels.iloc[-2])

"""Purged time-series cross-validation for ML training."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from sklearn.model_selection import TimeSeriesSplit


def purged_cv_split(
    n_samples: int,
    n_splits: int = 5,
    purge_gap: int = 24,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    TimeSeriesSplit with a purge gap between train and test to prevent label leakage.

    Removes the last `purge_gap` indices from each training fold.
    """

    if n_samples < n_splits + 1:
        return

    tscv = TimeSeriesSplit(n_splits=n_splits)
    indices = np.arange(n_samples)
    for train_idx, test_idx in tscv.split(indices):
        if len(train_idx) > purge_gap:
            train_idx = train_idx[:-purge_gap]
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        yield train_idx, test_idx

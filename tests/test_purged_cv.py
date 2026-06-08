"""Tests for purged time-series cross-validation."""

from __future__ import annotations

import numpy as np

from src.ml.cv import purged_cv_split


def test_purged_cv_no_overlap_and_gap() -> None:
    splits = list(purged_cv_split(500, n_splits=5, purge_gap=24))
    assert len(splits) >= 1
    for train_idx, test_idx in splits:
        assert len(set(train_idx) & set(test_idx)) == 0
        if len(train_idx) > 0 and len(test_idx) > 0:
            assert train_idx.max() < test_idx.min()
            gap = test_idx.min() - train_idx.max()
            assert gap >= 24

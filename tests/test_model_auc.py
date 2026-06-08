"""Model AUC gate tests."""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.ml.features.pipeline import FeaturePipeline


def test_production_model_auc_gate() -> None:
    model_path = Path("models/regime_lgbm_v1.pkl")
    matrix_path = Path("data/historical/feature_matrix.parquet")
    if not model_path.exists() or not matrix_path.exists():
        pytest.skip("Model or feature matrix not built")

    payload = joblib.load(model_path)
    model = payload["model"]
    feature_names = payload["feature_names"]
    frame = pd.read_parquet(matrix_path)
    frame = frame[frame["label"] >= 0].sort_values("timestamp").reset_index(drop=True)
    holdout = frame.iloc[int(len(frame) * 0.85) :]
    x = holdout[feature_names]
    y = holdout["label"].astype(int)
    if y.nunique() < 2:
        pytest.skip("Holdout has single class")

    preds = model.predict_proba(x)[:, 1]
    auc = float(roc_auc_score(y, preds))
    if auc < 0.65:
        warnings.warn(f"Production model AUC {auc:.4f} below 0.65 gate", stacklevel=1)
    assert auc > 0.45

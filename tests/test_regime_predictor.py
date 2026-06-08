"""Tests for RegimePredictor."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier

from src.ml.features.pipeline import FeaturePipeline
from src.ml.model_store import ModelArtifact, save_artifact
from src.ml.regime_predictor import RegimePredictor


def _synthetic_ohlcv(rows: int = 60) -> pd.DataFrame:
    close = pd.Series([1.0 + 0.001 * idx for idx in range(rows)], dtype=float)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="15min", tz="UTC"),
            "open": close - 0.001,
            "high": close + 0.002,
            "low": close - 0.002,
            "close": close,
            "volume": 1000 + pd.Series(range(rows)),
        }
    )


def test_regime_predictor_load_and_predict(tmp_path: Path) -> None:
    feature_names = FeaturePipeline.feature_names()
    x = [[0.0] * len(feature_names), [1.0] * len(feature_names)]
    y = [0, 1]
    model = LGBMClassifier(n_estimators=10, max_depth=3, device="cpu")
    model.fit(x, y)
    path = tmp_path / "model.pkl"
    save_artifact(
        path,
        ModelArtifact(
            model=model,
            feature_names=feature_names,
            version="test",
            trained_at="2024-01-01T00:00:00+00:00",
            metrics={"test_auc": 1.0},
        ),
    )

    predictor = RegimePredictor.load(str(path), threshold=0.5)
    result = predictor.predict(_synthetic_ohlcv(), {"funding_rate": 0.0001})
    assert result.regime in {"momentum", "chop"}
    assert 0.0 <= result.confidence <= 1.0


def test_predict_batch_returns_all_symbols(tmp_path: Path) -> None:
    feature_names = FeaturePipeline.feature_names()
    model = LGBMClassifier(n_estimators=10, max_depth=3, device="cpu")
    model.fit([[0.0] * len(feature_names), [1.0] * len(feature_names)], [0, 1])
    path = tmp_path / "model.pkl"
    save_artifact(
        path,
        ModelArtifact(model, feature_names, "test", "2024-01-01T00:00:00+00:00", {}),
    )
    predictor = RegimePredictor.load(str(path))
    ohlcv = _synthetic_ohlcv()
    rows = [("CAKE", ohlcv, {}), ("ETH", ohlcv, {})]
    results = predictor.predict_batch(rows)
    assert set(results.keys()) == {"CAKE", "ETH"}

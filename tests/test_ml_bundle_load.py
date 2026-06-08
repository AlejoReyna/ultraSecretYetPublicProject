"""Tests that MLBundle loads when ML is enabled and model exists."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config.settings import Settings
from src.ml.bundle import MLBundle


def test_ml_bundle_from_settings_loads_model(tmp_path) -> None:
    model_path = Path("models/regime_lgbm_v1.pkl")
    if not model_path.exists():
        pytest.skip("models/regime_lgbm_v1.pkl not built; run scripts/train_regime_model.py")

    settings = Settings(
        paper_trade=True,
        ml_enabled=True,
        ml_model_path=str(model_path),
        ml_ohlcv_cache_db=str(tmp_path / "ml_ohlcv_cache.sqlite"),
    )
    bundle = MLBundle.from_settings(settings)
    assert bundle.predictor is not None
    assert bundle.cache is not None

"""Regime-only fallback when AUC gate fails."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.config.settings import Settings
from src.ml.bundle import MLBundle
from src.ml.regime_predictor import RegimePredictor


def test_regime_fallback_applies_chop_multiplier_when_auc_low(tmp_path) -> None:
    model_path = Path("models/regime_lgbm_v1.pkl")
    if not model_path.exists():
        pytest.skip("models/regime_lgbm_v1.pkl not built")

    settings = Settings(
        paper_trade=True,
        ml_enabled=True,
        ml_min_auc=0.99,
        ml_shadow_mode=True,
        ml_regime_only_chop_multiplier=0.3,
        ml_ohlcv_cache_db=str(tmp_path / "cache.sqlite"),
    )
    predictor = RegimePredictor.load(str(model_path))
    bundle = MLBundle(settings, predictor, None, None, validation_auc=0.56)  # type: ignore[arg-type]

    assert bundle.is_ranking_active is False
    assert bundle.is_regime_only_fallback is True
    assert bundle._chop_multiplier() == 0.3

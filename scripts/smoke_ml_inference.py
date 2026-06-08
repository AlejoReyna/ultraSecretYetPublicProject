#!/usr/bin/env python3
"""Smoke test for ML inference latency and output."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config.settings import load_settings
from src.ml.features.pipeline import FeaturePipeline
from src.ml.regime_predictor import RegimePredictor


def main() -> int:
    settings = load_settings()
    model_path = Path(settings.ml_model_path)
    if not model_path.exists():
        print(f"Model not found: {model_path}. Run scripts/train_regime_model.py --synthetic")
        return 1

    predictor = RegimePredictor.load(str(model_path), threshold=settings.ml_regime_threshold)
    ohlcv = _synthetic_ohlcv(96)
    snapshot = {"funding_rate": 0.0001, "fear_greed_index": 55, "volume_1h": 2000, "rolling_24h_hourly_volume_avg": 1000}

    start = time.perf_counter()
    rows = [(f"SYM{i}", ohlcv, snapshot) for i in range(20)]
    predictions = predictor.predict_batch(rows)
    elapsed_ms = (time.perf_counter() - start) * 1000
    per_row_ms = elapsed_ms / max(len(rows), 1)

    sample = next(iter(predictions.values()))
    print(f"Regime={sample.regime} confidence={sample.confidence:.4f}")
    print(f"Batch latency: {elapsed_ms:.2f}ms total, {per_row_ms:.2f}ms/row")
    if per_row_ms > 5.0:
        print("WARNING: inference exceeds 5ms/row target")
        return 1
    return 0


def _synthetic_ohlcv(rows: int) -> pd.DataFrame:
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


if __name__ == "__main__":
    raise SystemExit(main())

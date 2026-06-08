#!/usr/bin/env python3
"""Build merged feature matrix parquet from historical OHLCV and CMC snapshots."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.settings import load_settings
from src.ml.feature_matrix import build_feature_matrix_from_sources


def main() -> int:
    settings = load_settings()
    ohlcv_dir = Path("data/historical/binance")
    cmc_path = Path("data/historical/cmc/premium_snapshots.parquet")
    out_path = Path("data/historical/feature_matrix.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    matrix = build_feature_matrix_from_sources(
        ohlcv_dir=ohlcv_dir,
        cmc_path=cmc_path,
        symbols=settings.ml_universe_symbols,
        execution_log_path=settings.execution_log_path,
    )
    matrix.to_parquet(out_path, index=False)
    attrs = getattr(matrix, "attrs", {})
    thresholds = attrs.get("label_thresholds", {})
    if thresholds:
        print(f"Label thresholds: {thresholds}")
    print(f"Wrote {len(matrix)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

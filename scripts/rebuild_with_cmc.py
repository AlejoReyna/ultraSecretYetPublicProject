#!/usr/bin/env python3
"""Rebuild feature matrix v3 with CMC premium SQLite cache."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config.settings import load_settings
from src.ml.feature_matrix import build_feature_matrix_from_sources

DB_PATH = Path("data/cmc_premium.db")


def _cmc_snapshots_from_db(db_path: Path) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame(columns=["timestamp", "symbol"])
    rows: list[dict] = []
    with sqlite3.connect(db_path) as conn:
        funding = conn.execute(
            "SELECT token, timestamp, funding_rate, open_interest FROM funding_rates ORDER BY timestamp"
        ).fetchall()
        for token, ts, rate, oi in funding:
            rows.append(
                {
                    "timestamp": ts,
                    "symbol": token,
                    "funding_rate": rate,
                    "open_interest": oi,
                    "open_interest_change_pct": None,
                }
            )
        fgi = conn.execute("SELECT timestamp, value, classification FROM fear_greed ORDER BY timestamp").fetchall()
        for ts, value, classification in fgi:
            rows.append(
                {
                    "timestamp": ts,
                    "symbol": "GLOBAL",
                    "fear_greed_index": value,
                    "fear_greed_classification": classification,
                }
            )
    if not rows:
        return pd.DataFrame(columns=["timestamp", "symbol"])
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.sort_values("timestamp").reset_index(drop=True)


def main() -> int:
    settings = load_settings()
    ohlcv_dir = Path("data/historical/binance")
    cmc_cache = _cmc_snapshots_from_db(DB_PATH)
    cmc_path = Path("data/historical/cmc/premium_snapshots.parquet")
    cmc_path.parent.mkdir(parents=True, exist_ok=True)
    if not cmc_cache.empty:
        cmc_cache.to_parquet(cmc_path, index=False)
        print(f"Wrote merged CMC cache ({len(cmc_cache)} rows) to {cmc_path}")
    else:
        print("No CMC premium DB rows; using existing parquet if present")

    matrix = build_feature_matrix_from_sources(
        ohlcv_dir=ohlcv_dir,
        cmc_path=cmc_path,
        symbols=settings.ml_universe_symbols,
        execution_log_path=settings.execution_log_path,
    )
    out_path = Path("data/historical/feature_matrix_v3.parquet")
    matrix.to_parquet(out_path, index=False)
    attrs = getattr(matrix, "attrs", {})
    print(f"Wrote {len(matrix)} rows to {out_path}")
    if attrs.get("label_thresholds"):
        print(f"Label thresholds: {attrs['label_thresholds']}")

    # Retrain against v3 matrix
    import shutil

    default_matrix = Path("data/historical/feature_matrix.parquet")
    shutil.copy2(out_path, default_matrix)
    print("Copied to feature_matrix.parquet for train_regime_model.py")

    import subprocess

    result = subprocess.run([sys.executable, "scripts/train_regime_model.py", "--allow-low-auc"], check=False)
    report = ROOT / "MODEL_QUALITY_REPORT.md"
    if report.exists():
        v3 = ROOT / "MODEL_QUALITY_REPORT_V3.md"
        shutil.copy2(report, v3)
        print(f"Wrote {v3}")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())

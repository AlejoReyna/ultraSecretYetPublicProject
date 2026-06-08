#!/usr/bin/env python3
"""Fetch historical Binance OHLCV and CMC premium snapshots for ML training."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.settings import load_settings
from src.data.binance_client import BinanceClient


def _load_cmc_client(settings):
    try:
        from src.data.cmc_mcp_client import CMCMCPClient

        return CMCMCPClient(settings)
    except Exception as exc:
        print(f"CMC client unavailable ({exc}); skipping premium snapshots.")
        return None


def main() -> int:
    settings = load_settings()
    client = BinanceClient()
    binance_only = "--binance-only" in sys.argv
    cmc_client = None if binance_only else _load_cmc_client(settings)

    binance_dir = Path("data/historical/binance")
    cmc_dir = Path("data/historical/cmc")
    binance_dir.mkdir(parents=True, exist_ok=True)
    cmc_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "symbols": {},
    }

    for symbol in settings.ml_universe_symbols:
        print(f"Fetching Binance OHLCV for {symbol}...")
        frame = client.fetch_history_days(symbol, days=30, interval="15m")
        out_path = binance_dir / f"ohlcv_15m_{symbol.upper()}.parquet"
        frame.to_parquet(out_path, index=False)
        manifest["symbols"][symbol.upper()] = {
            "rows": len(frame),
            "path": str(out_path),
        }

    snapshot_rows: list[dict[str, object]] = []
    if cmc_client is not None:
        end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        start = end - timedelta(days=30)
        day = start
        while day <= end:
            print(f"Fetching CMC premium snapshot for {day.date()}...")
            try:
                snapshot = cmc_client.fetch_x402_enriched_snapshot(settings.ml_universe_symbols)
            except Exception as exc:
                print(f"  CMC fetch failed: {exc}")
                snapshot = {}
            ts = day.isoformat()
            for symbol, payload in snapshot.items():
                if not isinstance(payload, dict):
                    continue
                row = {"timestamp": ts, "symbol": symbol.upper(), **payload}
                snapshot_rows.append(row)
            day += timedelta(days=1)

    cmc_path = cmc_dir / "premium_snapshots.parquet"
    if snapshot_rows:
        import pandas as pd

        pd.DataFrame(snapshot_rows).to_parquet(cmc_path, index=False)
    manifest["cmc_rows"] = len(snapshot_rows)
    manifest_path = binance_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

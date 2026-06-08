#!/usr/bin/env python3
"""Background CMC premium feature collector (parallel ML experiment)."""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.settings import load_settings
from src.execution.twak_interface import TWAKInterface

DB_PATH = Path("data/cmc_premium.db")
RAW_DIR = Path("data/cmc_premium")


def _init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS funding_rates (
            token TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            funding_rate REAL,
            open_interest REAL,
            PRIMARY KEY (token, timestamp)
        );
        CREATE TABLE IF NOT EXISTS fear_greed (
            timestamp TEXT PRIMARY KEY,
            value REAL,
            classification TEXT
        );
        CREATE TABLE IF NOT EXISTS market_metrics (
            timestamp TEXT PRIMARY KEY,
            btc_dominance REAL,
            altcoin_volume REAL,
            payload_json TEXT
        );
        """
    )


def _store_snapshot(conn: sqlite3.Connection, ts: str, payload: dict) -> None:
    fgi = payload.get("fear_greed_index") or payload.get("fear_greed")
    if fgi is not None:
        try:
            conn.execute(
                "INSERT OR REPLACE INTO fear_greed (timestamp, value, classification) VALUES (?, ?, ?)",
                (ts, float(fgi), str(payload.get("fear_greed_classification") or "")),
            )
        except (TypeError, ValueError):
            pass
    metrics = payload.get("market_metrics") or payload
    if isinstance(metrics, dict):
        conn.execute(
            "INSERT OR REPLACE INTO market_metrics (timestamp, btc_dominance, altcoin_volume, payload_json) VALUES (?, ?, ?, ?)",
            (
                ts,
                float(metrics.get("btc_dominance") or 0.0) if metrics.get("btc_dominance") is not None else None,
                float(metrics.get("altcoin_volume") or 0.0) if metrics.get("altcoin_volume") is not None else None,
                json.dumps(metrics),
            ),
        )
    for symbol, row in payload.items():
        if not isinstance(row, dict):
            continue
        funding = row.get("funding_rate")
        oi = row.get("open_interest")
        if funding is None and oi is None:
            continue
        try:
            conn.execute(
                "INSERT OR REPLACE INTO funding_rates (token, timestamp, funding_rate, open_interest) VALUES (?, ?, ?, ?)",
                (
                    str(symbol).upper(),
                    ts,
                    float(funding) if funding is not None else None,
                    float(oi) if oi is not None else None,
                ),
            )
        except (TypeError, ValueError):
            continue


def main() -> int:
    settings = load_settings()
    if not getattr(settings, "cmc_collector_enabled", True):
        print("CMC collector disabled (CMC_COLLECTOR_ENABLED=false)")
        return 0

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc)
    stamp = ts.strftime("%Y-%m-%d_%H-%M")
    ts_iso = ts.isoformat()

    twak = TWAKInterface(paper_trade=settings.paper_trade)
    snapshot: dict = {"collected_at": ts_iso, "symbols": {}}

    try:
        from src.data.cmc_mcp_client import CMCMCPClient

        client = CMCMCPClient(settings)
        symbols = settings.ml_universe_symbols
        enriched = client.fetch_x402_enriched_snapshot(symbols)
        if isinstance(enriched, dict):
            snapshot["symbols"] = enriched
            snapshot.update({k: v for k, v in enriched.items() if isinstance(v, dict)})
    except Exception as exc:
        snapshot["error"] = str(exc)
        print(f"CMC fetch failed: {exc}")

    raw_path = RAW_DIR / f"{stamp}.json"
    raw_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    with sqlite3.connect(DB_PATH) as conn:
        _init_db(conn)
        _store_snapshot(conn, ts_iso, snapshot.get("symbols", snapshot))
        conn.commit()

    print(f"Wrote {raw_path} and updated {DB_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

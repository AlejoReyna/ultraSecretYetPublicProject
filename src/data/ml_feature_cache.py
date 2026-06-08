"""SQLite cache for live Binance OHLCV used by ML inference."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

OHLCV_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")
DEFAULT_INTERVAL = "15m"
STALE_MINUTES = 15
FGI_PRIOR_HOURS = 24.0
FUNDING_HISTORY_DAYS = 7.0
FUNDING_HISTORY_LIMIT = 500


class MLFeatureCache:
    """Persist and serve recent OHLCV candles for ML feature building."""

    def __init__(self, db_path: str | Path, interval: str = DEFAULT_INTERVAL) -> None:
        self.db_path = Path(db_path)
        self.interval = interval
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ohlcv (
                symbol TEXT NOT NULL,
                interval TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                PRIMARY KEY (symbol, interval, timestamp)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cmc_metrics (
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                fear_greed_index REAL,
                funding_rate REAL,
                PRIMARY KEY (timestamp, symbol)
            )
            """
        )
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.commit()

    def upsert_klines(self, symbol: str, df: pd.DataFrame, interval: str | None = None) -> None:
        """Insert or replace OHLCV rows for a symbol."""

        if df.empty:
            return
        normalized = symbol.upper()
        bucket = interval or self.interval
        with self._connect() as conn:
            for _, row in df.iterrows():
                ts = row["timestamp"]
                if hasattr(ts, "isoformat"):
                    ts_text = ts.isoformat()
                else:
                    ts_text = str(ts)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO ohlcv
                    (symbol, interval, timestamp, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        normalized,
                        bucket,
                        ts_text,
                        float(row["open"]),
                        float(row["high"]),
                        float(row["low"]),
                        float(row["close"]),
                        float(row["volume"]),
                    ),
                )
            conn.commit()

    def get_recent(self, symbol: str, n: int, interval: str | None = None) -> pd.DataFrame:
        """Return the most recent N candles for a symbol."""

        normalized = symbol.upper()
        bucket = interval or self.interval
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT timestamp, open, high, low, close, volume
                FROM ohlcv
                WHERE symbol = ? AND interval = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (normalized, bucket, n),
            ).fetchall()
        if not rows:
            return pd.DataFrame(columns=list(OHLCV_COLUMNS))
        frame = pd.DataFrame(rows, columns=list(OHLCV_COLUMNS))
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        return frame.sort_values("timestamp").reset_index(drop=True)

    def is_stale(self, symbol: str, interval: str | None = None, stale_minutes: int = STALE_MINUTES) -> bool:
        """True when cache is empty or newest candle is older than stale_minutes."""

        frame = self.get_recent(symbol, 1, interval=interval)
        if frame.empty:
            return True
        newest = frame["timestamp"].iloc[-1]
        if newest.tzinfo is None:
            newest = newest.replace(tzinfo=timezone.utc)
        age_minutes = (datetime.now(timezone.utc) - newest).total_seconds() / 60.0
        return age_minutes > stale_minutes

    def latest_timestamp(self, symbol: str, interval: str | None = None) -> datetime | None:
        frame = self.get_recent(symbol, 1, interval=interval)
        if frame.empty:
            return None
        ts = frame["timestamp"].iloc[-1]
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts

    @staticmethod
    def _normalize_fear_greed(value: Any) -> float | None:
        try:
            if value is None:
                return None
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number / 100.0 if number > 1.0 else number

    def record_cmc_metrics(
        self,
        snapshot: dict[str, dict[str, Any]],
        timestamp: datetime | None = None,
    ) -> None:
        """Persist per-cycle CMC premium metrics for delta/z-score features."""

        ts_text = (timestamp or datetime.now(timezone.utc)).isoformat()
        with self._connect() as conn:
            for symbol, payload in snapshot.items():
                if not isinstance(payload, dict):
                    continue
                normalized = symbol.upper()
                fear_greed = self._normalize_fear_greed(payload.get("fear_greed_index"))
                funding_rate = payload.get("funding_rate")
                try:
                    funding_value = float(funding_rate) if funding_rate is not None else None
                except (TypeError, ValueError):
                    funding_value = None
                conn.execute(
                    """
                    INSERT OR REPLACE INTO cmc_metrics
                    (timestamp, symbol, fear_greed_index, funding_rate)
                    VALUES (?, ?, ?, ?)
                    """,
                    (ts_text, normalized, fear_greed, funding_value),
                )
            conn.commit()

    def get_fear_greed_prior(self, hours_ago: float = FGI_PRIOR_HOURS) -> float | None:
        """Return normalized fear/greed index from approximately hours_ago."""

        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT fear_greed_index
                FROM cmc_metrics
                WHERE timestamp <= ? AND fear_greed_index IS NOT NULL
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (cutoff,),
            ).fetchone()
        if row is None:
            return None
        return float(row[0])

    def get_funding_history(
        self,
        symbol: str,
        days: float = FUNDING_HISTORY_DAYS,
        limit: int = FUNDING_HISTORY_LIMIT,
    ) -> list[float]:
        """Return recent funding-rate readings for z-score computation."""

        normalized = symbol.upper()
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT funding_rate
                FROM cmc_metrics
                WHERE symbol = ? AND timestamp >= ? AND funding_rate IS NOT NULL
                ORDER BY timestamp ASC
                LIMIT ?
                """,
                (normalized, cutoff, limit),
            ).fetchall()
        return [float(row[0]) for row in rows]

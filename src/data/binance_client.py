"""Free Binance REST OHLCV feed for ML feature engineering."""

from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

LOGGER = logging.getLogger(__name__)

BINANCE_API_BASE = "https://api.binance.com"
INTERVAL_MS = {
    "15m": 15 * 60 * 1000,
    "1h": 60 * 60 * 1000,
}

# Symbol overrides for Binance USDT pairs.
_SYMBOL_TO_PAIR: dict[str, str] = {
    "BNB": "BNBUSDT",
    "WBNB": "BNBUSDT",
}


class BinanceClient:
    """Fetch OHLCV klines from Binance public API."""

    def __init__(self, base_url: str = BINANCE_API_BASE, request_delay_s: float = 0.1) -> None:
        self.base_url = base_url.rstrip("/")
        self.request_delay_s = request_delay_s

    @staticmethod
    def symbol_to_pair(symbol: str) -> str:
        normalized = symbol.strip().upper()
        if normalized in _SYMBOL_TO_PAIR:
            return _SYMBOL_TO_PAIR[normalized]
        return f"{normalized}USDT"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, min=0.5, max=4))
    def fetch_klines(
        self,
        symbol: str,
        interval: str = "15m",
        limit: int = 1000,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> pd.DataFrame:
        """Fetch one page of klines."""

        params: dict[str, Any] = {
            "symbol": self.symbol_to_pair(symbol),
            "interval": interval,
            "limit": min(limit, 1000),
        }
        if start_ms is not None:
            params["startTime"] = start_ms
        if end_ms is not None:
            params["endTime"] = end_ms

        response = requests.get(f"{self.base_url}/api/v3/klines", params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise ValueError(f"unexpected Binance klines response for {symbol}")

        rows: list[dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, list) or len(item) < 6:
                continue
            rows.append(
                {
                    "timestamp": pd.to_datetime(int(item[0]), unit="ms", utc=True),
                    "open": float(item[1]),
                    "high": float(item[2]),
                    "low": float(item[3]),
                    "close": float(item[4]),
                    "volume": float(item[5]),
                }
            )
        return pd.DataFrame(rows)

    def fetch_history_days(self, symbol: str, days: int = 30, interval: str = "15m") -> pd.DataFrame:
        """Paginate klines to cover approximately `days` of history."""

        interval_ms = INTERVAL_MS.get(interval, INTERVAL_MS["15m"])
        end_ms = int(time.time() * 1000)
        start_ms = end_ms - days * 24 * 60 * 60 * 1000
        frames: list[pd.DataFrame] = []
        cursor = start_ms

        while cursor < end_ms:
            frame = self.fetch_klines(symbol, interval=interval, limit=1000, start_ms=cursor, end_ms=end_ms)
            if frame.empty:
                break
            frames.append(frame)
            last_ts = int(frame["timestamp"].iloc[-1].timestamp() * 1000)
            next_cursor = last_ts + interval_ms
            if next_cursor <= cursor:
                break
            cursor = next_cursor
            time.sleep(self.request_delay_s)

        if not frames:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        merged = pd.concat(frames, ignore_index=True)
        merged = merged.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        return merged

    def get_recent_ohlcv(self, symbol: str, candles: int = 96, interval: str = "15m") -> pd.DataFrame:
        """Fetch the most recent N candles for live inference."""

        limit = min(max(candles, 1), 1000)
        return self.fetch_klines(symbol, interval=interval, limit=limit)

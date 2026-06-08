"""Tests for BinanceClient."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from src.data.binance_client import BinanceClient


def test_symbol_to_pair() -> None:
    assert BinanceClient.symbol_to_pair("CAKE") == "CAKEUSDT"
    assert BinanceClient.symbol_to_pair("BNB") == "BNBUSDT"


@patch("src.data.binance_client.requests.get")
def test_fetch_klines_parses_response(mock_get: MagicMock) -> None:
    mock_get.return_value.json.return_value = [
        [1_700_000_000_000, "1.0", "1.1", "0.9", "1.05", "100.0"],
        [1_700_000_900_000, "1.05", "1.2", "1.0", "1.15", "120.0"],
    ]
    mock_get.return_value.raise_for_status = MagicMock()

    client = BinanceClient()
    frame = client.fetch_klines("CAKE", limit=2)

    assert len(frame) == 2
    assert list(frame.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert frame["close"].iloc[-1] == 1.15

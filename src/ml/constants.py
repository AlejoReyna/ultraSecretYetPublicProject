"""ML layer constants."""

from __future__ import annotations

# 20-token universe: subset of TRADABLE_TARGET_SYMBOLS with verified BSC contracts
# and Binance USDT pairs.
ML_DEFAULT_20: list[str] = [
    "CAKE",
    "ETH",
    "BNB",
    "LINK",
    "DOGE",
    "SHIB",
    "FLOKI",
    "ADA",
    "XRP",
    "DOT",
    "AVAX",
    "ATOM",
    "UNI",
    "AAVE",
    "INJ",
    "FIL",
    "LTC",
    "TRX",
    "BONK",
    "TON",
]

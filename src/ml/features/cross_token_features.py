"""Cross-token features for ML regime detection."""

from __future__ import annotations

import math

import pandas as pd

CROSS_TOKEN_FEATURE_NAMES: list[str] = [
    "bnb_corr_48",
    "bnb_beta_48",
    "sector_momentum_16",
    "universe_breadth_1h",
    "relative_strength_vs_bnb_16",
    "volume_rank_pctile",
]

SECTOR_TAGS: dict[str, str] = {
    "CAKE": "defi",
    "UNI": "defi",
    "AAVE": "defi",
    "LINK": "oracle",
    "FLOKI": "meme",
    "DOGE": "meme",
    "SHIB": "meme",
    "BONK": "meme",
    "BNB": "l1",
    "ETH": "l1",
    "AVAX": "l1",
    "DOT": "l1",
    "ATOM": "l1",
    "TON": "l1",
    "ADA": "l1",
    "XRP": "payments",
    "TRX": "payments",
    "LTC": "payments",
    "INJ": "defi",
    "FIL": "storage",
}


def _returns(close: pd.Series) -> pd.Series:
    return close.pct_change().fillna(0.0)


def compute_cross_token_features(
    symbol: str,
    token_ohlcv: pd.DataFrame,
    bnb_ohlcv: pd.DataFrame,
    universe_returns_4: dict[str, float],
    universe_returns_16: dict[str, float],
    volume_rank_pctile: float,
) -> dict[str, float]:
    """Compute cross-token context features for one symbol."""

    normalized = symbol.upper()
    token_close = token_ohlcv["close"].astype(float)
    bnb_close = bnb_ohlcv["close"].astype(float) if not bnb_ohlcv.empty else token_close

    token_ret = _returns(token_close).tail(48)
    bnb_ret = _returns(bnb_close).tail(48)
    min_len = min(len(token_ret), len(bnb_ret))
    if min_len < 2:
        corr = 0.0
        beta = 0.0
    else:
        aligned_token = token_ret.tail(min_len)
        aligned_bnb = bnb_ret.tail(min_len)
        corr = float(aligned_token.corr(aligned_bnb))
        if math.isnan(corr):
            corr = 0.0
        bnb_var = float(aligned_bnb.var(ddof=0))
        beta = float(aligned_token.cov(aligned_bnb) / bnb_var) if bnb_var > 0 else 0.0

    sector = SECTOR_TAGS.get(normalized, "other")
    sector_rets = [value for key, value in universe_returns_16.items() if SECTOR_TAGS.get(key, "other") == sector]
    sector_momentum = sum(sector_rets) / len(sector_rets) if sector_rets else 0.0

    breadth_values = list(universe_returns_4.values())
    breadth = sum(1 for value in breadth_values if value > 0) / len(breadth_values) if breadth_values else 0.0

    token_ret_16 = universe_returns_16.get(normalized, 0.0)
    bnb_ret_16 = universe_returns_16.get("BNB", 0.0)

    return {
        "bnb_corr_48": corr,
        "bnb_beta_48": beta,
        "sector_momentum_16": sector_momentum,
        "universe_breadth_1h": breadth,
        "relative_strength_vs_bnb_16": token_ret_16 - bnb_ret_16,
        "volume_rank_pctile": volume_rank_pctile,
    }

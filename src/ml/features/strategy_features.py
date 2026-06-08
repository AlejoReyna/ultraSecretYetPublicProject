"""Strategy-specific features for Plan B+ regime detection."""

from __future__ import annotations

import math
from datetime import timezone
from typing import Any

import pandas as pd

STRATEGY_FEATURE_NAMES: list[str] = [
    "volume_skew_3h_6h",
    "price_accel_1h",
    "bnb_residual_1h",
    "funding_rate_percentile",
    "fear_greed_bucket",
    "token_historical_win_rate",
    "range_compression_6h",
    "volume_price_divergence",
    "body_to_range_ratio",
    "hour_of_day",
    "day_of_week",
]

CATEGORICAL_FEATURE_NAMES: list[str] = [
    "fear_greed_bucket",
    "hour_of_day",
    "day_of_week",
]

CANDLES_1H = 4
CANDLES_3H = 12
CANDLES_6H = 24


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0 or math.isnan(denominator):
        return default
    return numerator / denominator


def _ret_n(close: pd.Series, n: int) -> float:
    if len(close) <= n:
        return 0.0
    last = float(close.iloc[-1])
    prior = float(close.iloc[-1 - n])
    if prior <= 0:
        return 0.0
    return last / prior - 1.0


def _atr_value(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    return float(atr) if not math.isnan(float(atr)) else 0.0


def fear_greed_to_bucket(fear_greed_index: float) -> int:
    """Bucket normalized or raw F&G into 5 bins (0-4)."""

    value = fear_greed_index * 100.0 if fear_greed_index <= 1.0 else fear_greed_index
    value = max(0.0, min(100.0, value))
    return min(int(value // 20), 4)


def compute_strategy_features(
    symbol: str,
    ohlcv_df: pd.DataFrame,
    cmc_snapshot: dict[str, Any],
    *,
    bnb_ohlcv: pd.DataFrame | None = None,
    funding_history: list[float] | None = None,
    token_win_rate: float = 0.5,
) -> dict[str, float]:
    """Compute Plan B+ strategy-specific features for one token."""

    defaults = {name: 0.0 for name in STRATEGY_FEATURE_NAMES}
    defaults["token_historical_win_rate"] = token_win_rate
    defaults["fear_greed_bucket"] = 2.0
    defaults["hour_of_day"] = 12.0
    defaults["day_of_week"] = 3.0

    if len(ohlcv_df) < CANDLES_6H:
        return defaults

    frame = ohlcv_df.copy()
    close = frame["close"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    open_ = frame["open"].astype(float)
    volume = frame["volume"].astype(float)

    vol_3h = float(volume.tail(CANDLES_3H).sum())
    vol_6h = float(volume.tail(CANDLES_6H).sum())
    volume_skew = _safe_div(vol_3h, vol_6h)

    ret_1h = _ret_n(close, CANDLES_1H)
    ret_2h_ago = _ret_n(close.iloc[:-CANDLES_1H], CANDLES_1H) if len(close) > CANDLES_1H * 2 else 0.0
    price_accel = ret_1h - ret_2h_ago

    bnb_residual = ret_1h
    if bnb_ohlcv is not None and len(bnb_ohlcv) >= CANDLES_1H:
        bnb_close = bnb_ohlcv["close"].astype(float)
        bnb_residual = ret_1h - _ret_n(bnb_close, CANDLES_1H)

    funding_rate = 0.0
    try:
        funding_rate = float(cmc_snapshot.get("funding_rate") or 0.0)
    except (TypeError, ValueError):
        funding_rate = 0.0

    funding_percentile = 0.0
    if funding_history:
        max_abs = max(abs(item) for item in funding_history) or 1e-9
        funding_percentile = funding_rate / max_abs

    fear_greed_raw = cmc_snapshot.get("fear_greed_index", 50.0)
    try:
        fear_greed_val = float(fear_greed_raw) if fear_greed_raw is not None else 50.0
    except (TypeError, ValueError):
        fear_greed_val = 50.0
    fear_bucket = float(fear_greed_to_bucket(fear_greed_val))

    high_6h = float(high.tail(CANDLES_6H).max())
    low_6h = float(low.tail(CANDLES_6H).min())
    atr = _atr_value(high, low, close, 14)
    range_compression = _safe_div(high_6h - low_6h, atr)

    tail_vol = volume.tail(CANDLES_6H)
    tail_close = close.tail(CANDLES_6H)
    vol_std = float(tail_vol.std(ddof=0))
    close_std = float(tail_close.std(ddof=0))
    if vol_std > 1e-12 and close_std > 1e-12:
        volume_price_div = float(tail_vol.corr(tail_close))
        volume_price_div = 0.0 if math.isnan(volume_price_div) else volume_price_div
    else:
        volume_price_div = 0.0

    body_ratios: list[float] = []
    for idx in range(-3, 0):
        hl = float(high.iloc[idx] - low.iloc[idx])
        body = abs(float(close.iloc[idx] - open_.iloc[idx]))
        body_ratios.append(_safe_div(body, hl))
    body_to_range = sum(body_ratios) / len(body_ratios)

    ts = frame["timestamp"].iloc[-1]
    if hasattr(ts, "tzinfo") and ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    hour = float(ts.hour) if hasattr(ts, "hour") else 12.0
    dow = float(ts.weekday()) if hasattr(ts, "weekday") else 3.0

    return {
        "volume_skew_3h_6h": volume_skew,
        "price_accel_1h": price_accel,
        "bnb_residual_1h": bnb_residual,
        "funding_rate_percentile": funding_percentile,
        "fear_greed_bucket": fear_bucket,
        "token_historical_win_rate": token_win_rate,
        "range_compression_6h": range_compression,
        "volume_price_divergence": volume_price_div,
        "body_to_range_ratio": body_to_range,
        "hour_of_day": hour,
        "day_of_week": dow,
    }

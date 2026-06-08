"""OHLCV-derived features for ML regime detection."""

from __future__ import annotations

import math

import pandas as pd

OHLCV_FEATURE_NAMES: list[str] = [
    "ret_1",
    "ret_4",
    "ret_16",
    "volatility_16",
    "volatility_48",
    "volume_zscore_24",
    "volume_ratio_4_24",
    "rsi_14",
    "rsi_slope_4",
    "hl_range_pct",
    "body_pct",
    "upper_wick_ratio",
    "close_vs_high_16",
    "ema_8_21_spread",
    "atr_pct_14",
]


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0 or math.isnan(denominator):
        return default
    return numerator / denominator


def _series_returns(close: pd.Series) -> pd.Series:
    return close.pct_change()


def _wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _atr_pct(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    last_close = float(close.iloc[-1])
    return _safe_div(float(atr), last_close)


def compute_ohlcv_features(df: pd.DataFrame) -> dict[str, float]:
    """Compute OHLCV features from the last row of a candle history."""

    if len(df) < 48:
        return {name: 0.0 for name in OHLCV_FEATURE_NAMES}

    frame = df.copy()
    close = frame["close"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    open_ = frame["open"].astype(float)
    volume = frame["volume"].astype(float)

    ret = _series_returns(close)
    rsi = _wilder_rsi(close, 14)

    last_open = float(open_.iloc[-1])
    last_high = float(high.iloc[-1])
    last_low = float(low.iloc[-1])
    last_close = float(close.iloc[-1])
    hl_range = last_high - last_low

    vol_mean_24 = float(volume.tail(24).mean())
    vol_std_24 = float(volume.tail(24).std(ddof=0))
    vol_z = _safe_div(float(volume.iloc[-1]) - vol_mean_24, vol_std_24)

    ema8 = float(_ema(close, 8).iloc[-1])
    ema21 = float(_ema(close, 21).iloc[-1])

    return {
        "ret_1": float(ret.iloc[-1]) if not math.isnan(ret.iloc[-1]) else 0.0,
        "ret_4": _safe_div(last_close, float(close.iloc[-5])) - 1.0 if len(close) >= 5 else 0.0,
        "ret_16": _safe_div(last_close, float(close.iloc[-17])) - 1.0 if len(close) >= 17 else 0.0,
        "volatility_16": float(ret.tail(16).std(ddof=0) or 0.0),
        "volatility_48": float(ret.tail(48).std(ddof=0) or 0.0),
        "volume_zscore_24": vol_z,
        "volume_ratio_4_24": _safe_div(float(volume.tail(4).mean()), vol_mean_24),
        "rsi_14": float(rsi.iloc[-1]) if not math.isnan(rsi.iloc[-1]) else 50.0,
        "rsi_slope_4": float(rsi.iloc[-1] - rsi.iloc[-5]) if len(rsi) >= 5 and not math.isnan(rsi.iloc[-1]) else 0.0,
        "hl_range_pct": _safe_div(hl_range, last_close),
        "body_pct": _safe_div(abs(last_close - last_open), last_close),
        "upper_wick_ratio": _safe_div(last_high - max(last_open, last_close), hl_range + 1e-12),
        "close_vs_high_16": _safe_div(last_close, float(high.tail(16).max())) - 1.0,
        "ema_8_21_spread": _safe_div(ema8, ema21) - 1.0,
        "atr_pct_14": _atr_pct(high, low, close, 14),
    }

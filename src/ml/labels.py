"""Label builders for offline ML training."""

from __future__ import annotations

import pandas as pd

LABEL_HORIZON = 24
DEFAULT_MAX_RETURN = 0.02
DEFAULT_MAX_DRAWDOWN = -0.015
DEFAULT_CLOSE_LOCATION = 0.5


def compute_label(
    ohlcv_window: pd.DataFrame,
    entry_candle_idx: int,
    *,
    max_return_thresh: float = DEFAULT_MAX_RETURN,
    max_drawdown_thresh: float = DEFAULT_MAX_DRAWDOWN,
    close_location_thresh: float = DEFAULT_CLOSE_LOCATION,
    horizon: int = LABEL_HORIZON,
) -> int:
    """
    Composite momentum label.

    Label = 1 only if ALL three conditions are met:
    1. Max high in next `horizon` candles exceeds +max_return_thresh from entry close
    2. Max drawdown (lowest low) stays above max_drawdown_thresh
    3. Final close is in upper portion of the forward range (sustainable momentum)
    Returns -1 when insufficient future history.
    """

    if entry_candle_idx + horizon >= len(ohlcv_window):
        return -1

    entry_price = float(ohlcv_window.iloc[entry_candle_idx]["close"])
    if entry_price <= 0:
        return 0

    future = ohlcv_window.iloc[entry_candle_idx + 1 : entry_candle_idx + horizon + 1]
    if len(future) < horizon:
        return -1

    max_return = float(future["high"].max()) / entry_price - 1.0
    max_drawdown = float(future["low"].min()) / entry_price - 1.0
    future_high = float(future["high"].max())
    future_low = float(future["low"].min())
    close_location = (float(future["close"].iloc[-1]) - future_low) / (future_high - future_low + 1e-9)

    is_momentum = (
        max_return > max_return_thresh
        and max_drawdown > max_drawdown_thresh
        and close_location > close_location_thresh
    )
    return int(is_momentum)


def vectorized_composite_label(
    ohlcv: pd.DataFrame,
    *,
    horizon: int = LABEL_HORIZON,
    max_return_thresh: float = DEFAULT_MAX_RETURN,
    max_drawdown_thresh: float = DEFAULT_MAX_DRAWDOWN,
    close_location_thresh: float = DEFAULT_CLOSE_LOCATION,
) -> pd.Series:
    """Vectorized composite labels; -1 for insufficient history."""

    labels = pd.Series(-1, index=ohlcv.index, dtype=int)
    close = ohlcv["close"].astype(float)
    high = ohlcv["high"].astype(float)
    low = ohlcv["low"].astype(float)

    for idx in range(len(ohlcv) - horizon):
        entry_price = float(close.iloc[idx])
        if entry_price <= 0:
            labels.iloc[idx] = 0
            continue
        future_high = high.iloc[idx + 1 : idx + horizon + 1]
        future_low = low.iloc[idx + 1 : idx + horizon + 1]
        future_close = close.iloc[idx + 1 : idx + horizon + 1]
        if len(future_close) < horizon:
            continue
        max_return = float(future_high.max()) / entry_price - 1.0
        max_drawdown = float(future_low.min()) / entry_price - 1.0
        hmax = float(future_high.max())
        lmin = float(future_low.min())
        close_location = (float(future_close.iloc[-1]) - lmin) / (hmax - lmin + 1e-9)
        is_momentum = (
            max_return > max_return_thresh
            and max_drawdown > max_drawdown_thresh
            and close_location > close_location_thresh
        )
        labels.iloc[idx] = int(is_momentum)

    labels.iloc[-horizon:] = -1
    return labels


def tune_label_thresholds(
    ohlcv: pd.DataFrame,
    *,
    target_low: float = 0.25,
    target_high: float = 0.35,
    horizon: int = LABEL_HORIZON,
) -> tuple[float, float, float]:
    """Tighten thresholds until positive class rate falls within target band."""

    max_return = DEFAULT_MAX_RETURN
    close_location = DEFAULT_CLOSE_LOCATION
    max_drawdown = DEFAULT_MAX_DRAWDOWN

    for _ in range(12):
        labels = vectorized_composite_label(
            ohlcv,
            horizon=horizon,
            max_return_thresh=max_return,
            max_drawdown_thresh=max_drawdown,
            close_location_thresh=close_location,
        )
        valid = labels[labels >= 0]
        if valid.empty:
            break
        positive_rate = float(valid.mean())
        if target_low <= positive_rate <= target_high:
            break
        if positive_rate > target_high:
            max_return += 0.0025
            close_location = min(close_location + 0.05, 0.75)
        elif positive_rate < target_low:
            max_return = max(0.01, max_return - 0.0025)
            close_location = max(0.35, close_location - 0.05)
        else:
            break

    return max_return, max_drawdown, close_location


# Legacy helpers kept for backward-compatible tests
def momentum_label(
    close: pd.Series,
    horizon: int = 24,
    threshold: float = 0.02,
) -> pd.Series:
    """Binary label: 1 if max forward close over `horizon` candles exceeds threshold."""

    values = close.astype(float)
    labels = pd.Series(index=values.index, dtype="float64")
    for idx in range(len(values)):
        if idx + horizon >= len(values):
            labels.iloc[idx] = float("nan")
            continue
        current = float(values.iloc[idx])
        if current <= 0:
            labels.iloc[idx] = 0.0
            continue
        forward = values.iloc[idx + 1 : idx + horizon + 1]
        max_forward = float(forward.max())
        labels.iloc[idx] = 1.0 if (max_forward / current - 1.0) > threshold else 0.0
    return labels


def vectorized_momentum_label(
    close: pd.Series,
    horizon: int = 24,
    threshold: float = 0.02,
) -> pd.Series:
    """Vectorized momentum label for training pipelines."""

    values = close.astype(float)
    future_max = values.iloc[::-1].rolling(window=horizon, min_periods=horizon).max().iloc[::-1].shift(-1)
    forward_return = future_max / values - 1.0
    labels = (forward_return > threshold).astype(float)
    labels.iloc[-horizon:] = float("nan")
    return labels

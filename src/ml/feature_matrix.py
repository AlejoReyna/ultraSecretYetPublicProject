"""Shared helpers for building the offline ML feature matrix."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.ml.execution_stats import load_token_win_rates
from src.ml.features.pipeline import FeaturePipeline
from src.ml.labels import (
    LABEL_HORIZON,
    tune_label_thresholds,
    vectorized_composite_label,
)

MIN_LOOKBACK = 48


def load_ohlcv_parquet(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.sort_values("timestamp").reset_index(drop=True)


def load_cmc_snapshots(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["timestamp", "symbol"])
    frame = pd.read_parquet(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.sort_values("timestamp").reset_index(drop=True)


def asof_cmc_row(cmc_rows: pd.DataFrame, symbol: str, candle_ts: pd.Timestamp) -> dict[str, Any]:
    """Return latest CMC snapshot row for symbol at or before candle_ts."""

    if cmc_rows.empty:
        return {"symbol": symbol}
    subset = cmc_rows[(cmc_rows["symbol"] == symbol.upper()) & (cmc_rows["timestamp"] <= candle_ts)]
    if subset.empty:
        return {"symbol": symbol}
    row = subset.iloc[-1].to_dict()
    return {key: value for key, value in row.items() if key not in {"timestamp", "symbol"}}


def _resolve_label_thresholds(ohlcv_dir: Path, symbols: list[str]) -> tuple[float, float, float]:
    for symbol in symbols:
        parquet_path = ohlcv_dir / f"ohlcv_15m_{symbol.upper()}.parquet"
        if not parquet_path.exists():
            continue
        ohlcv = load_ohlcv_parquet(parquet_path)
        if len(ohlcv) >= MIN_LOOKBACK + LABEL_HORIZON + 1:
            return tune_label_thresholds(ohlcv)
    from src.ml.labels import DEFAULT_CLOSE_LOCATION, DEFAULT_MAX_DRAWDOWN, DEFAULT_MAX_RETURN

    return DEFAULT_MAX_RETURN, DEFAULT_MAX_DRAWDOWN, DEFAULT_CLOSE_LOCATION


def build_feature_matrix_from_sources(
    *,
    ohlcv_dir: Path,
    cmc_path: Path,
    symbols: list[str],
    execution_log_path: str | Path | None = None,
) -> pd.DataFrame:
    """Merge OHLCV history with CMC snapshots and compute composite labels."""

    cmc_rows = load_cmc_snapshots(cmc_path)
    token_win_rates = load_token_win_rates(execution_log_path or "execution_log.jsonl")
    max_return, max_drawdown, close_location = _resolve_label_thresholds(ohlcv_dir, symbols)
    all_rows: list[dict[str, Any]] = []

    bnb_ohlcv = pd.DataFrame()
    bnb_path = ohlcv_dir / "ohlcv_15m_BNB.parquet"
    if bnb_path.exists():
        bnb_ohlcv = load_ohlcv_parquet(bnb_path)

    for symbol in symbols:
        parquet_path = ohlcv_dir / f"ohlcv_15m_{symbol.upper()}.parquet"
        if not parquet_path.exists():
            continue
        ohlcv = load_ohlcv_parquet(parquet_path)
        if len(ohlcv) < MIN_LOOKBACK + LABEL_HORIZON + 1:
            continue

        labels = vectorized_composite_label(
            ohlcv,
            horizon=LABEL_HORIZON,
            max_return_thresh=max_return,
            max_drawdown_thresh=max_drawdown,
            close_location_thresh=close_location,
        )
        ohlcv_by_symbol = {symbol.upper(): ohlcv}
        if not bnb_ohlcv.empty:
            ohlcv_by_symbol["BNB"] = bnb_ohlcv

        for idx in range(MIN_LOOKBACK, len(ohlcv) - LABEL_HORIZON):
            label = int(labels.iloc[idx])
            if label < 0:
                continue
            window = ohlcv.iloc[: idx + 1].reset_index(drop=True)
            candle_ts = window["timestamp"].iloc[-1]
            cmc_snapshot = asof_cmc_row(cmc_rows, symbol, candle_ts)
            universe_context = FeaturePipeline.build_universe_context(
                {symbol.upper(): window, **({"BNB": bnb_ohlcv} if not bnb_ohlcv.empty else {})},
                {symbol.upper(): cmc_snapshot},
                token_win_rates=token_win_rates,
            )
            features = FeaturePipeline.build_row(symbol, window, cmc_snapshot, universe_context)
            record = {
                "symbol": symbol.upper(),
                "timestamp": candle_ts,
                "label": label,
                **features,
            }
            all_rows.append(record)

    if not all_rows:
        return pd.DataFrame(columns=["symbol", "timestamp", "label", *FeaturePipeline.feature_names()])
    frame = pd.DataFrame(all_rows)
    frame.attrs["label_thresholds"] = {
        "max_return": max_return,
        "max_drawdown": max_drawdown,
        "close_location": close_location,
        "positive_rate": float(frame["label"].mean()),
    }
    return frame

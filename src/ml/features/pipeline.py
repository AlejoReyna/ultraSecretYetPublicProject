"""Unified feature pipeline for training and live inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.ml.features.cmc_features import CMC_FEATURE_NAMES, compute_cmc_features
from src.ml.features.cross_token_features import CROSS_TOKEN_FEATURE_NAMES, compute_cross_token_features
from src.ml.features.ohlcv_features import OHLCV_FEATURE_NAMES, compute_ohlcv_features
from src.ml.features.strategy_features import (
    CATEGORICAL_FEATURE_NAMES,
    STRATEGY_FEATURE_NAMES,
    compute_strategy_features,
)


@dataclass
class UniverseContext:
    """Precomputed cross-token context for one evaluation cycle."""

    bnb_ohlcv: pd.DataFrame = field(default_factory=pd.DataFrame)
    universe_returns_4: dict[str, float] = field(default_factory=dict)
    universe_returns_16: dict[str, float] = field(default_factory=dict)
    volume_rank_pctiles: dict[str, float] = field(default_factory=dict)
    fear_greed_prior: float | None = None
    funding_history: dict[str, list[float]] = field(default_factory=dict)
    token_win_rates: dict[str, float] = field(default_factory=dict)


class FeaturePipeline:
    """Build stable-order feature rows from OHLCV and CMC snapshots."""

    @staticmethod
    def feature_names() -> list[str]:
        return OHLCV_FEATURE_NAMES + CMC_FEATURE_NAMES + CROSS_TOKEN_FEATURE_NAMES + STRATEGY_FEATURE_NAMES

    @staticmethod
    def categorical_feature_names() -> list[str]:
        return list(CATEGORICAL_FEATURE_NAMES)

    @staticmethod
    def _ret_n(close: pd.Series, n: int) -> float:
        if len(close) <= n:
            return 0.0
        last = float(close.iloc[-1])
        prior = float(close.iloc[-1 - n])
        if prior <= 0:
            return 0.0
        return last / prior - 1.0

    @classmethod
    def build_universe_context(
        cls,
        ohlcv_by_symbol: dict[str, pd.DataFrame],
        cmc_snapshot: dict[str, dict[str, Any]],
        *,
        fear_greed_prior: float | None = None,
        funding_history: dict[str, list[float]] | None = None,
        token_win_rates: dict[str, float] | None = None,
    ) -> UniverseContext:
        """Precompute shared cross-token context for all symbols in one cycle."""

        returns_4: dict[str, float] = {}
        returns_16: dict[str, float] = {}
        volumes: dict[str, float] = {}

        for symbol, frame in ohlcv_by_symbol.items():
            if frame.empty:
                returns_4[symbol.upper()] = 0.0
                returns_16[symbol.upper()] = 0.0
                volumes[symbol.upper()] = 0.0
                continue
            close = frame["close"].astype(float)
            normalized = symbol.upper()
            returns_4[normalized] = cls._ret_n(close, 4)
            returns_16[normalized] = cls._ret_n(close, 16)
            token_data = cmc_snapshot.get(normalized, {})
            vol = token_data.get("volume_24h") if isinstance(token_data, dict) else None
            try:
                volumes[normalized] = float(vol) if vol is not None else float(frame["volume"].tail(96).sum())
            except (TypeError, ValueError):
                volumes[normalized] = 0.0

        ranked = sorted(volumes.items(), key=lambda item: item[1], reverse=True)
        rank_pctiles: dict[str, float] = {}
        total = max(len(ranked), 1)
        for index, (symbol, _) in enumerate(ranked):
            rank_pctiles[symbol] = 1.0 - (index / total)

        bnb_frame = ohlcv_by_symbol.get("BNB", pd.DataFrame())
        if bnb_frame.empty:
            bnb_frame = next(iter(ohlcv_by_symbol.values()), pd.DataFrame())

        return UniverseContext(
            bnb_ohlcv=bnb_frame,
            universe_returns_4=returns_4,
            universe_returns_16=returns_16,
            volume_rank_pctiles=rank_pctiles,
            fear_greed_prior=fear_greed_prior,
            funding_history=funding_history or {},
            token_win_rates=token_win_rates or {},
        )

    @classmethod
    def build_row(
        cls,
        symbol: str,
        ohlcv_df: pd.DataFrame,
        cmc_snapshot: dict[str, Any],
        universe_context: UniverseContext | None = None,
    ) -> dict[str, float]:
        """Build one feature row for a symbol."""

        normalized = symbol.upper()
        ohlcv_features = compute_ohlcv_features(ohlcv_df)
        ctx = universe_context or UniverseContext()
        rank_pctile = ctx.volume_rank_pctiles.get(normalized, 0.5)
        cmc_features = compute_cmc_features(
            cmc_snapshot,
            fear_greed_prior=ctx.fear_greed_prior,
            funding_history=ctx.funding_history.get(normalized),
            rank_pctile=rank_pctile,
        )
        cross_features = compute_cross_token_features(
            normalized,
            ohlcv_df,
            ctx.bnb_ohlcv,
            ctx.universe_returns_4,
            ctx.universe_returns_16,
            rank_pctile,
        )
        strategy_features = compute_strategy_features(
            normalized,
            ohlcv_df,
            cmc_snapshot,
            bnb_ohlcv=ctx.bnb_ohlcv,
            funding_history=ctx.funding_history.get(normalized),
            token_win_rate=ctx.token_win_rates.get(normalized, 0.5),
        )
        row = {**ohlcv_features, **cmc_features, **cross_features, **strategy_features}
        for name in cls.feature_names():
            row.setdefault(name, 0.0)
        return row

    @classmethod
    def build_matrix_row(cls, row: dict[str, float]) -> list[float]:
        """Return feature values in stable column order."""

        return [float(row.get(name, 0.0)) for name in cls.feature_names()]

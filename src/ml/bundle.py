"""Orchestrates live ML inference for the trading loop."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from src.data.binance_client import BinanceClient
from src.data.ml_feature_cache import MLFeatureCache
from src.ml.execution_stats import load_token_win_rates
from src.ml.features.pipeline import FeaturePipeline
from src.ml.regime_predictor import RegimePredictor
from src.ml.types import MLContext, attach_symbol, chop_fallback, context_from_prediction

if TYPE_CHECKING:
    from src.config.settings import Settings

LOGGER = logging.getLogger(__name__)


def _load_validation_auc(settings: Settings, predictor: RegimePredictor) -> float:
    validation_path = Path("models/validation_auc_v2.txt")
    if validation_path.exists():
        try:
            return float(validation_path.read_text(encoding="utf-8").strip())
        except ValueError:
            pass
    metrics = predictor.artifact.metrics
    if "worst_auc" in metrics:
        return float(metrics["worst_auc"])
    if "mean_auc" in metrics:
        return float(metrics["mean_auc"])
    return float(metrics.get("test_auc", 0.0))


class MLBundle:
    """Lazy-loaded ML components for one agent run."""

    def __init__(
        self,
        settings: Settings,
        predictor: RegimePredictor,
        binance_client: BinanceClient,
        cache: MLFeatureCache,
        validation_auc: float,
    ) -> None:
        self.settings = settings
        self.predictor = predictor
        self.binance_client = binance_client
        self.cache = cache
        self.validation_auc = validation_auc

    @classmethod
    def from_settings(cls, settings: Settings) -> MLBundle:
        predictor = RegimePredictor.load(settings.ml_model_path, threshold=settings.ml_regime_threshold)
        validation_auc = _load_validation_auc(settings, predictor)
        binance_client = BinanceClient()
        cache = MLFeatureCache(settings.ml_ohlcv_cache_db)
        bundle = cls(settings, predictor, binance_client, cache, validation_auc)
        LOGGER.info(
            "ML bundle loaded (AUC=%.4f, ranking_active=%s, shadow=%s)",
            validation_auc,
            bundle.is_ranking_active,
            settings.ml_shadow_mode,
        )
        return bundle

    @property
    def is_ranking_active(self) -> bool:
        """True when ML may select among multiple core-factor passers."""

        return self.validation_auc >= self.settings.ml_min_auc and not self.settings.ml_shadow_mode

    @property
    def is_regime_only_fallback(self) -> bool:
        """True when model quality is too low for ranking but regime sizing remains."""

        return not self.is_ranking_active

    def _chop_multiplier(self) -> float:
        if self.is_regime_only_fallback:
            return self.settings.ml_regime_only_chop_multiplier
        return self.settings.ml_chop_size_multiplier

    def refresh_ohlcv_if_stale(self) -> None:
        """Refresh stale OHLCV cache entries once per cycle."""

        lookback = self.settings.ml_ohlcv_lookback_candles
        for symbol in self.settings.ml_universe_symbols:
            normalized = symbol.upper()
            if not self.cache.is_stale(normalized):
                continue
            try:
                frame = self.binance_client.get_recent_ohlcv(normalized, candles=lookback)
                self.cache.upsert_klines(normalized, frame)
            except Exception as exc:
                LOGGER.warning("Failed to refresh OHLCV for %s: %s", normalized, exc)

    def _ohlcv_for_symbol(self, symbol: str) -> pd.DataFrame:
        normalized = symbol.upper()
        frame = self.cache.get_recent(normalized, self.settings.ml_ohlcv_lookback_candles)
        if not frame.empty:
            return frame
        try:
            frame = self.binance_client.get_recent_ohlcv(normalized, self.settings.ml_ohlcv_lookback_candles)
            self.cache.upsert_klines(normalized, frame)
            return frame
        except Exception as exc:
            LOGGER.warning("OHLCV unavailable for %s: %s", normalized, exc)
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    def build_contexts(self, snapshot: dict[str, dict]) -> dict[str, MLContext]:
        """Build MLContext for each symbol in the ML universe present in snapshot."""

        rows: list[tuple[str, pd.DataFrame, dict]] = []
        ohlcv_by_symbol: dict[str, pd.DataFrame] = {}
        for symbol in self.settings.ml_universe_symbols:
            normalized = symbol.upper()
            if normalized not in snapshot and normalized != "BNB":
                continue
            token_data = snapshot.get(normalized, {})
            if not isinstance(token_data, dict):
                continue
            frame = self._ohlcv_for_symbol(normalized)
            ohlcv_by_symbol[normalized] = frame
            rows.append((normalized, frame, token_data))

        if "BNB" not in ohlcv_by_symbol:
            bnb_frame = self._ohlcv_for_symbol("BNB")
            ohlcv_by_symbol["BNB"] = bnb_frame

        if not rows:
            return {}

        self.cache.record_cmc_metrics(snapshot)
        funding_history = {
            symbol.upper(): self.cache.get_funding_history(symbol.upper())
            for symbol in self.settings.ml_universe_symbols
        }
        token_win_rates = load_token_win_rates(self.settings.execution_log_path)
        universe_context = FeaturePipeline.build_universe_context(
            ohlcv_by_symbol,
            snapshot,
            fear_greed_prior=self.cache.get_fear_greed_prior(),
            funding_history=funding_history,
            token_win_rates=token_win_rates,
        )
        predictions = self.predictor.predict_batch(rows, universe_context=universe_context)

        chop_mult = self._chop_multiplier()
        contexts: dict[str, MLContext] = {}
        for symbol, _, _ in rows:
            normalized = symbol.upper()
            prediction = predictions.get(normalized)
            if prediction is None:
                contexts[normalized] = chop_fallback(normalized, position_multiplier=chop_mult)
                continue
            ctx = context_from_prediction(
                prediction,
                momentum_multiplier=self.settings.ml_momentum_size_multiplier,
                chop_multiplier=chop_mult,
            )
            contexts[normalized] = attach_symbol(ctx, normalized)
        return contexts

    def shadow_audit_fields(
        self,
        passers: list,
        ml_contexts: dict[str, MLContext],
        selected_symbol: str | None,
    ) -> dict[str, object]:
        """Build shadow-mode audit payload for decision logs."""

        scores = {
            (getattr(decision, "symbol", "") or "").upper(): (
                ml_contexts.get((getattr(decision, "symbol", "") or "").upper()).confidence
                if ml_contexts.get((getattr(decision, "symbol", "") or "").upper()) is not None
                else 0.0
            )
            for decision in passers
        }
        ml_selected = max(scores, key=scores.get) if scores else None
        return {
            "ml_active": self.is_ranking_active,
            "ml_scores": scores,
            "ml_selected_symbol": ml_selected,
            "executed_symbol": selected_symbol,
            "validation_auc": self.validation_auc,
            "regime_only_fallback": self.is_regime_only_fallback,
        }

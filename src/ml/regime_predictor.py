"""Regime classifier inference wrapper (LightGBM / XGBoost / CatBoost)."""

from __future__ import annotations

import logging

import pandas as pd

from src.ml.features.pipeline import FeaturePipeline
from src.ml.model_store import ModelArtifact, load_artifact
from src.ml.types import RegimeLabel, RegimePrediction

LOGGER = logging.getLogger(__name__)


class RegimePredictor:
    """CPU-only regime classifier with fail-closed defaults."""

    def __init__(self, artifact: ModelArtifact, threshold: float = 0.55) -> None:
        self.artifact = artifact
        self.threshold = threshold
        self._cat_indices = self._resolve_cat_indices()

    @classmethod
    def load(cls, path: str, threshold: float = 0.55) -> RegimePredictor:
        return cls(load_artifact(path), threshold=threshold)

    def _resolve_cat_indices(self) -> list[int]:
        categorical = set(self.artifact.categorical_features or FeaturePipeline.categorical_feature_names())
        return [idx for idx, name in enumerate(self.artifact.feature_names) if name in categorical]

    def _predict_proba_batch(self, feature_rows: list[dict[str, float]]) -> list[float]:
        frame = pd.DataFrame(
            [{name: row.get(name, 0.0) for name in self.artifact.feature_names} for row in feature_rows]
        )
        model = self.artifact.model
        model_type = self.artifact.model_type

        if model_type == "cat" and self._cat_indices:
            from catboost import Pool

            probabilities = model.predict_proba(Pool(frame, cat_features=self._cat_indices))[:, 1]
        else:
            probabilities = model.predict_proba(frame)[:, 1]
        return [float(value) for value in probabilities]

    def predict(self, ohlcv: pd.DataFrame, cmc_snapshot: dict) -> RegimePrediction:
        try:
            row = FeaturePipeline.build_row("TOKEN", ohlcv, cmc_snapshot)
            confidence = self._predict_proba_batch([row])[0]
            regime: RegimeLabel = "momentum" if confidence >= self.threshold else "chop"
            return RegimePrediction(
                regime=regime,
                confidence=confidence,
                feature_vector=row,
                model_version=self.artifact.version,
            )
        except Exception as exc:
            LOGGER.warning("RegimePredictor.predict failed: %s", exc)
            return RegimePrediction(
                regime="chop",
                confidence=0.0,
                feature_vector={},
                model_version=self.artifact.version,
            )

    def predict_batch(
        self,
        rows: list[tuple[str, pd.DataFrame, dict]],
        universe_context=None,
    ) -> dict[str, RegimePrediction]:
        if not rows:
            return {}
        try:
            if universe_context is None:
                ohlcv_by_symbol = {symbol: frame for symbol, frame, _ in rows}
                cmc_snapshot = {symbol: snap for symbol, _, snap in rows}
                universe_context = FeaturePipeline.build_universe_context(ohlcv_by_symbol, cmc_snapshot)

            raw_features: list[dict[str, float]] = []
            symbols: list[str] = []
            for symbol, frame, snapshot in rows:
                normalized = symbol.upper()
                row = FeaturePipeline.build_row(normalized, frame, snapshot, universe_context)
                raw_features.append(row)
                symbols.append(normalized)

            probabilities = self._predict_proba_batch(raw_features)
            results: dict[str, RegimePrediction] = {}
            for idx, symbol in enumerate(symbols):
                confidence = probabilities[idx]
                regime: RegimeLabel = "momentum" if confidence >= self.threshold else "chop"
                results[symbol] = RegimePrediction(
                    regime=regime,
                    confidence=confidence,
                    feature_vector=raw_features[idx],
                    model_version=self.artifact.version,
                )
            return results
        except Exception as exc:
            LOGGER.warning("RegimePredictor.predict_batch failed: %s", exc)
            return {
                symbol.upper(): RegimePrediction("chop", 0.0, {}, self.artifact.version)
                for symbol, _, _ in rows
            }

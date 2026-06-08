"""Persist and load ML model artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib


@dataclass(frozen=True)
class ModelArtifact:
    model: Any
    feature_names: list[str]
    version: str
    trained_at: str
    metrics: dict[str, float]
    model_type: str = "lgb"
    categorical_features: list[str] | None = None


def _normalize_metrics(raw: dict) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(value, (int, float)):
            metrics[str(key)] = float(value)
    return metrics


def save_artifact(path: str | Path, artifact: ModelArtifact) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": artifact.model,
        "feature_names": artifact.feature_names,
        "version": artifact.version,
        "trained_at": artifact.trained_at,
        "metrics": artifact.metrics,
        "model_type": artifact.model_type,
        "categorical_features": artifact.categorical_features or [],
    }
    joblib.dump(payload, target)
    meta_path = target.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "version": artifact.version,
                "trained_at": artifact.trained_at,
                "metrics": artifact.metrics,
                "feature_count": len(artifact.feature_names),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_artifact(path: str | Path) -> ModelArtifact:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"model artifact not found: {target}")
    payload = joblib.load(target)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"invalid model artifact: {target}")
    feature_names = payload.get("feature_names")
    if not isinstance(feature_names, list) or not feature_names:
        raise ValueError("model artifact missing feature_names")
    cat_features = payload.get("categorical_features") or []
    return ModelArtifact(
        model=payload["model"],
        feature_names=[str(name) for name in feature_names],
        version=str(payload.get("version", "unknown")),
        trained_at=str(payload.get("trained_at", "")),
        metrics=_normalize_metrics(payload.get("metrics") or {}),
        model_type=str(payload.get("model_type", "lgb")),
        categorical_features=[str(name) for name in cat_features] if cat_features else None,
    )

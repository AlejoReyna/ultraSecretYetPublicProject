#!/usr/bin/env python3
"""Train regime classifier with purged CV and multi-model comparison."""

from __future__ import annotations

import json
import shutil
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config.settings import load_settings
from src.ml.cv import purged_cv_split
from src.ml.features.pipeline import FeaturePipeline
from src.ml.model_store import ModelArtifact, save_artifact

MIN_AUC_TARGET = 0.65
MODEL_VERSION = "regime_v2"
PURGE_GAP = 24
N_SPLITS = 5


def _load_frame(matrix_path: Path) -> pd.DataFrame:
    if matrix_path.exists():
        return pd.read_parquet(matrix_path)
    if "--synthetic" in sys.argv:
        return _synthetic_training_frame(rows=3000)
    print(f"Feature matrix not found: {matrix_path}")
    print("Run scripts/build_feature_matrix.py first (or use --synthetic).")
    raise SystemExit(1)


def _synthetic_training_frame(rows: int = 3000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    feature_names = FeaturePipeline.feature_names()
    data = {name: rng.normal(0, 1, rows) for name in feature_names}
    data["symbol"] = ["CAKE"] * rows
    data["timestamp"] = pd.date_range("2024-01-01", periods=rows, freq="15min", tz="UTC")
    score = data["ret_16"] + 0.5 * data["volume_zscore_24"] + rng.normal(0, 0.5, rows)
    data["label"] = (score > score.mean()).astype(int)
    return pd.DataFrame(data)


def _cat_feature_indices(feature_names: list[str]) -> list[int]:
    categorical = set(FeaturePipeline.categorical_feature_names())
    return [idx for idx, name in enumerate(feature_names) if name in categorical]


def _build_models(y: pd.Series, cat_indices: list[int]) -> dict[str, object]:
    from lightgbm import LGBMClassifier

    scale_pos = 2.0 if y.mean() < 0.4 else 1.0
    models: dict[str, object] = {
        "lgb": LGBMClassifier(
            objective="binary",
            n_estimators=500,
            num_leaves=63,
            learning_rate=0.03,
            max_depth=8,
            min_child_samples=50,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            scale_pos_weight=scale_pos,
            n_jobs=2,
            verbose=-1,
            random_state=42,
        ),
    }
    try:
        import xgboost as xgb

        models["xgb"] = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            eval_metric="auc",
            n_jobs=2,
            random_state=42,
        )
    except ImportError:
        print("xgboost not installed; skipping XGBoost")

    try:
        from catboost import CatBoostClassifier

        models["cat"] = CatBoostClassifier(
            iterations=400,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=42,
            verbose=False,
            thread_count=2,
            cat_features=cat_indices if cat_indices else None,
        )
    except ImportError:
        print("catboost not installed; skipping CatBoost")

    return models


def _fit_model(model: object, model_type: str, x_train: pd.DataFrame, y_train: pd.Series, cat_indices: list[int]) -> object:
    if model_type == "lgb" and cat_indices:
        fitted = deepcopy(model)
        fitted.fit(x_train, y_train, categorical_feature=cat_indices)
        return fitted
    fitted = deepcopy(model)
    fitted.fit(x_train, y_train)
    return fitted


def _predict_proba(model: object, model_type: str, x_test: pd.DataFrame, cat_indices: list[int]) -> np.ndarray:
    if model_type == "cat" and cat_indices:
        from catboost import Pool

        pool = Pool(x_test, cat_features=cat_indices)
        return model.predict_proba(pool)[:, 1]
    return model.predict_proba(x_test)[:, 1]


def _evaluate_model(
    model_template: object,
    model_type: str,
    x: pd.DataFrame,
    y: pd.Series,
    cat_indices: list[int],
) -> dict[str, float]:
    fold_aucs: list[float] = []
    for train_idx, test_idx in purged_cv_split(len(x), n_splits=N_SPLITS, purge_gap=PURGE_GAP):
        fitted = _fit_model(model_template, model_type, x.iloc[train_idx], y.iloc[train_idx], cat_indices)
        preds = _predict_proba(fitted, model_type, x.iloc[test_idx], cat_indices)
        test_y = y.iloc[test_idx]
        if test_y.nunique() < 2:
            continue
        fold_aucs.append(float(roc_auc_score(test_y, preds)))

    if not fold_aucs:
        return {"mean_auc": 0.0, "std_auc": 0.0, "worst_auc": 0.0, "fold_aucs": []}

    return {
        "mean_auc": float(np.mean(fold_aucs)),
        "std_auc": float(np.std(fold_aucs)),
        "worst_auc": float(np.min(fold_aucs)),
        "fold_aucs": fold_aucs,
    }


def _feature_importance(model: object, model_type: str, feature_names: list[str]) -> list[tuple[str, float]]:
    if model_type == "lgb" and hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif model_type == "xgb" and hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif model_type == "cat" and hasattr(model, "get_feature_importance"):
        values = model.get_feature_importance()
    else:
        return []
    pairs = sorted(zip(feature_names, values), key=lambda item: item[1], reverse=True)
    return [(name, float(score)) for name, score in pairs[:10]]


def _write_report(
    path: Path,
    *,
    results: dict[str, dict],
    best_name: str,
    positive_rate: float,
    feature_count: int,
    recommendation: str,
) -> None:
    lines = [
        "# Model Quality Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Summary",
        f"- Positive class rate: **{positive_rate:.1%}**",
        f"- Feature count: **{feature_count}**",
        f"- Best model: **{best_name}**",
        f"- Recommendation: **{recommendation}**",
        "",
        "## Per-model purged CV (5 folds, 24-candle purge gap)",
        "",
        "| Model | Mean AUC | Std | Worst-fold AUC | Folds |",
        "|-------|----------|-----|----------------|-------|",
    ]
    for name, metrics in results.items():
        folds = ", ".join(f"{auc:.3f}" for auc in metrics.get("fold_aucs", []))
        lines.append(
            f"| {name} | {metrics['mean_auc']:.4f} | {metrics['std_auc']:.4f} | "
            f"{metrics['worst_auc']:.4f} | {folds} |"
        )

    best = results[best_name]
    lines.extend(
        [
            "",
            "## Best model feature importance (top 10)",
            "",
        ]
    )
    for name, score in best.get("top_features", []):
        lines.append(f"- `{name}`: {score:.4f}")

    lines.extend(
        [
            "",
            "## Shadow mode recommendation",
            "",
            recommendation,
            "",
            "Set `ML_SHADOW_MODE=false` only after worst-fold AUC >= 0.65 and 48h shadow paper validation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    settings = load_settings()
    matrix_path = Path("data/historical/feature_matrix.parquet")
    frame = _load_frame(matrix_path)

    feature_names = FeaturePipeline.feature_names()
    frame = frame[frame["label"] >= 0].dropna(subset=["label"])
    if frame.empty:
        print("No labeled rows available for training.")
        return 1

    frame = frame.sort_values("timestamp").reset_index(drop=True)
    positive_rate = float(frame["label"].mean())
    print(f"Positive class rate: {positive_rate:.1%} ({int(frame['label'].sum())}/{len(frame)})")

    x = frame[feature_names]
    y = frame["label"].astype(int)
    cat_indices = _cat_feature_indices(feature_names)
    cat_names = FeaturePipeline.categorical_feature_names()

    models = _build_models(y, cat_indices)
    results: dict[str, dict] = {}
    best_name = ""
    best_worst = -1.0

    for name, template in models.items():
        metrics = _evaluate_model(template, name, x, y, cat_indices)
        results[name] = metrics
        print(
            f"{name}: mean AUC={metrics['mean_auc']:.4f} "
            f"(±{metrics['std_auc']:.4f}), worst-fold={metrics['worst_auc']:.4f}"
        )
        if metrics["worst_auc"] > best_worst:
            best_worst = metrics["worst_auc"]
            best_name = name

    if not best_name:
        print("No models trained successfully.")
        return 1

    best_template = models[best_name]
    final_model = _fit_model(best_template, best_name, x, y, cat_indices)
    top_features = _feature_importance(final_model, best_name, feature_names)
    results[best_name]["top_features"] = top_features

    worst_auc = results[best_name]["worst_auc"]
    mean_auc = results[best_name]["mean_auc"]
    ranking_active = worst_auc >= MIN_AUC_TARGET
    recommendation = (
        "ACTIVATE ML ranking (`ML_SHADOW_MODE=false`) — worst-fold AUC meets 0.65 gate."
        if ranking_active
        else "KEEP shadow/regime-only fallback — worst-fold AUC below 0.65; ML ranking disabled."
    )

    metrics = {
        "mean_auc": mean_auc,
        "worst_auc": worst_auc,
        "std_auc": results[best_name]["std_auc"],
        "positive_rate": positive_rate,
        "fold_aucs": results[best_name]["fold_aucs"],
    }

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    v2_path = models_dir / f"regime_{best_name}_v2.pkl"
    artifact = ModelArtifact(
        model=final_model,
        feature_names=feature_names,
        version=MODEL_VERSION,
        trained_at=datetime.now(timezone.utc).isoformat(),
        metrics=metrics,
        model_type=best_name,
        categorical_features=cat_names,
    )
    save_artifact(v2_path, artifact)

    prod_path = Path(settings.ml_model_path)
    shutil.copy2(v2_path, prod_path)
    print(f"Promoted {best_name} to {prod_path}")

    validation_path = models_dir / "validation_auc_v2.txt"
    validation_path.write_text(f"{worst_auc:.6f}\n", encoding="utf-8")

    _write_report(
        ROOT / "MODEL_QUALITY_REPORT.md",
        results=results,
        best_name=best_name,
        positive_rate=positive_rate,
        feature_count=len(feature_names),
        recommendation=recommendation,
    )

    meta_path = models_dir / "training_summary_v2.json"
    meta_path.write_text(
        json.dumps(
            {
                "best_model": best_name,
                "metrics": metrics,
                "results": {k: {kk: vv for kk, vv in v.items() if kk != "top_features"} for k, v in results.items()},
                "ranking_active": ranking_active,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote MODEL_QUALITY_REPORT.md (best={best_name}, worst-fold AUC={worst_auc:.4f})")
    if not ranking_active and "--allow-low-auc" not in sys.argv:
        print(f"Worst-fold AUC {worst_auc:.4f} below target {MIN_AUC_TARGET}; regime-only fallback recommended.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

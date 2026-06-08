#!/usr/bin/env python3
"""Optuna hyperparameter search over breakout thresholds using decision logs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import optuna

from src.config.settings import load_settings
from src.ml.hyperopt_replay import HyperoptReplayEngine


def main() -> int:
    settings = load_settings()
    engine = HyperoptReplayEngine(
        settings.decision_log_path,
        settings.execution_log_path,
        feature_matrix="data/historical/feature_matrix.parquet",
    )

    def objective(trial: optuna.Trial) -> float:
        params = {
            "breakout_buffer": trial.suggest_float("breakout_buffer", 0.001, 0.005),
            "bnb_regime_threshold": trial.suggest_float("bnb_regime_threshold", -0.02, 0.0),
            "token_regime_1h_min": trial.suggest_float("token_regime_1h_min", 0.0, 0.01),
            "token_regime_24h_min": trial.suggest_float("token_regime_24h_min", -0.12, -0.04),
            "ml_volume_breakout_multiplier": trial.suggest_float("ml_volume_breakout_multiplier", 1.5, 3.0),
            "ml_volume_cache_multiplier": trial.suggest_float("ml_volume_cache_multiplier", 1.0, 1.5),
            "ml_regime_threshold": trial.suggest_float("ml_regime_threshold", 0.45, 0.65),
            "min_entry_factors": 4,
            "max_slippage_pct": settings.max_slippage_pct,
        }
        metrics = engine.replay(params)
        trial.set_user_attr("metrics", metrics)
        return metrics["objective"]

    study = optuna.create_study(direction="maximize")
    trials = 50
    if "--quick" in sys.argv:
        trials = 5
    study.optimize(objective, n_trials=trials)

    best = {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "metrics": study.best_trial.user_attrs.get("metrics", {}),
    }
    out_path = Path("models/hyperopt_best_params.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

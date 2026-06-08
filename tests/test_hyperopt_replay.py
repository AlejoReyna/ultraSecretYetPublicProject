"""Tests for hyperopt replay engine."""

from __future__ import annotations

import json
from pathlib import Path

from src.ml.hyperopt_replay import HyperoptReplayEngine


def test_hyperopt_replay_returns_deterministic_metrics(tmp_path: Path) -> None:
    decision_log = tmp_path / "decision_log.jsonl"
    execution_log = tmp_path / "execution_log.jsonl"
    decision_log.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "action": "ENTER",
                        "symbol": "CAKE",
                        "position_size_usdc": 100.0,
                        "factor_scores": {
                            "volume_breakout": True,
                            "six_hour_high_break": True,
                            "regime_not_risk_off": True,
                            "slippage_under_cap": True,
                        },
                        "estimated_slippage_pct": 0.005,
                    }
                ),
                json.dumps({"action": "WAIT", "symbol": None}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    execution_log.write_text(
        json.dumps(
            {
                "action": "exit",
                "from_symbol": "CAKE",
                "to_symbol": "USDC",
                "amount_in": 50.0,
                "expected_amount_out": 110.0,
                "result": {"amount_out": 110.0},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    engine = HyperoptReplayEngine(decision_log, execution_log)
    metrics = engine.replay({"min_entry_factors": 4, "max_slippage_pct": 0.01})
    assert metrics["trades"] == 1.0
    assert metrics["total_pnl"] == 10.0
    assert metrics["win_rate"] == 1.0

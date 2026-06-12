"""Offline replay engine for Optuna hyperparameter search."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


class HyperoptReplayEngine:
    """Replay decision/execution logs with alternate threshold parameters."""

    def __init__(
        self,
        decision_log: str | Path,
        execution_log: str | Path,
        feature_matrix: str | Path | None = None,
    ) -> None:
        self.decision_log = Path(decision_log)
        self.execution_log = Path(execution_log)
        self.feature_matrix = Path(feature_matrix) if feature_matrix else None
        self.decisions = read_jsonl(self.decision_log)
        self.executions = read_jsonl(self.execution_log)

    def replay(self, params: dict[str, Any]) -> dict[str, float]:
        """Return counterfactual metrics for a parameter set."""

        total_pnl = 0.0
        wins = 0
        trades = 0
        peak = 0.0
        equity = 0.0
        max_drawdown = 0.0

        exits_by_symbol: dict[str, list[dict[str, Any]]] = {}
        for record in self.executions:
            if str(record.get("action", "")).lower() != "exit":
                continue
            # An exit swap sells the position token INTO the stable, so the
            # position symbol is from_symbol (to_symbol is the stablecoin).
            symbol = str(record.get("from_symbol") or record.get("to_symbol") or "").upper()
            exits_by_symbol.setdefault(symbol, []).append(record)

        for decision in self.decisions:
            if str(decision.get("action", "")).upper() != "ENTER":
                continue
            symbol = str(decision.get("symbol") or "").upper()
            if not symbol:
                continue
            if not self._would_enter(decision, params):
                continue

            trades += 1
            pnl = self._estimate_pnl(symbol, decision, exits_by_symbol)
            total_pnl += pnl
            if pnl > 0:
                wins += 1
            equity += pnl
            peak = max(peak, equity)
            if peak > 0:
                max_drawdown = max(max_drawdown, (peak - equity) / peak)

        win_rate = wins / trades if trades else 0.0
        objective = total_pnl - 2.0 * max_drawdown
        return {
            "total_pnl": total_pnl,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "trades": float(trades),
            "objective": objective,
        }

    def _would_enter(self, decision: dict[str, Any], params: dict[str, Any]) -> bool:
        factor_scores = decision.get("factor_scores") or {}
        if not isinstance(factor_scores, dict):
            return False

        core_names = (
            "volume_breakout",
            "six_hour_high_break",
            "regime_not_risk_off",
            "slippage_under_cap",
        )
        core_pass = sum(1 for name in core_names if factor_scores.get(name))
        min_core = int(params.get("min_entry_factors", 4))
        if core_pass < min_core:
            return False
        if not factor_scores.get("slippage_under_cap"):
            return False

        slippage = decision.get("estimated_slippage_pct")
        max_slippage = float(params.get("max_slippage_pct", 0.01))
        if slippage is not None and float(slippage) > max_slippage:
            return False
        return True

    def _estimate_pnl(
        self,
        symbol: str,
        decision: dict[str, Any],
        exits_by_symbol: dict[str, list[dict[str, Any]]],
    ) -> float:
        entry_size = float(decision.get("position_size_usdc") or 0.0)
        if entry_size <= 0:
            return 0.0
        exits = exits_by_symbol.get(symbol, [])
        if not exits:
            return 0.0
        exit_record = exits[0]
        amount_in = float(exit_record.get("amount_in") or 0.0)
        result = exit_record.get("result") or {}
        amount_out = float(result.get("amount_out") or exit_record.get("expected_amount_out") or 0.0)
        if amount_in <= 0:
            return amount_out - entry_size
        return amount_out - entry_size

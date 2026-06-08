"""Token-level historical win rates from execution logs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_WIN_RATE = 0.5
MIN_TRADES_FOR_RATE = 5
MAX_TRADES_LOOKBACK = 20


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


def load_token_win_rates(
    execution_log_path: str | Path,
    *,
    min_trades: int = MIN_TRADES_FOR_RATE,
    lookback: int = MAX_TRADES_LOOKBACK,
) -> dict[str, float]:
    """Rolling win rate per token from exit records in execution_log.jsonl."""

    exits_by_symbol: dict[str, list[bool]] = {}
    for record in read_jsonl(execution_log_path):
        if str(record.get("action", "")).lower() != "exit":
            continue
        symbol = str(record.get("to_symbol") or record.get("from_symbol") or "").upper()
        if not symbol:
            continue
        pnl = _estimate_exit_pnl(record)
        if pnl is None:
            continue
        exits_by_symbol.setdefault(symbol, []).append(pnl > 0)

    rates: dict[str, float] = {}
    for symbol, outcomes in exits_by_symbol.items():
        recent = outcomes[-lookback:]
        if len(recent) < min_trades:
            rates[symbol] = DEFAULT_WIN_RATE
        else:
            rates[symbol] = sum(recent) / len(recent)
    return rates


def _estimate_exit_pnl(record: dict[str, Any]) -> float | None:
    amount_in = record.get("amount_in")
    result = record.get("result") or {}
    amount_out = result.get("amount_out") if isinstance(result, dict) else None
    if amount_out is None:
        amount_out = record.get("expected_amount_out")
    try:
        entry = float(amount_in) if amount_in is not None else None
        exit_val = float(amount_out) if amount_out is not None else None
    except (TypeError, ValueError):
        return None
    if entry is None or exit_val is None:
        return None
    return exit_val - entry

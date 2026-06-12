"""Append-only execution audit log for swap results."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config.settings import Settings


class ExecutionLogger:
    """Write swap execution records as JSON Lines."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def log(
        self,
        *,
        action: str,
        from_symbol: str,
        to_symbol: str,
        amount_in: float,
        max_slippage_pct: float,
        expected_amount_out: float | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Append one execution record and return it."""

        result_payload = result or {}
        record: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "from_symbol": from_symbol.upper(),
            "to_symbol": to_symbol.upper(),
            "amount_in": amount_in,
            "max_slippage_pct": max_slippage_pct,
            "result": result_payload,
        }
        if expected_amount_out is not None:
            record["expected_amount_out"] = expected_amount_out
        tx_hash = _first_present(result_payload, ("tx_hash", "hash", "transaction_hash"))
        if tx_hash is not None:
            record["tx_hash"] = tx_hash
        approval_hash = _first_present(result_payload, ("approval_hash",))
        if approval_hash is not None:
            record["approval_hash"] = approval_hash
        if error is not None:
            record["error"] = error
        if reason is not None:
            record["reason"] = reason

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            json.dump(record, handle, sort_keys=True)
            handle.write("\n")
        return record


def log_execution(
    settings: Settings,
    *,
    action: str,
    from_symbol: str,
    to_symbol: str,
    amount_in: float,
    max_slippage_pct: float,
    expected_amount_out: float | None = None,
    result: dict[str, Any] | None = None,
    error: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    """Append an execution record using the configured settings path."""

    return ExecutionLogger(settings.execution_log_path).log(
        action=action,
        from_symbol=from_symbol,
        to_symbol=to_symbol,
        amount_in=amount_in,
        max_slippage_pct=max_slippage_pct,
        expected_amount_out=expected_amount_out,
        result=result,
        error=error,
        reason=reason,
    )


def _first_present(payload: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = payload.get(key)
        if value:
            return str(value)
    return None

"""Startup position and pending-swap reconciliation."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

PENDING_COOLDOWN_HOURS = 1.0


def _extract_balance(balance_response: dict[str, Any], symbol: str) -> float:
    normalized = symbol.upper()
    for key in ("amount", "balance", "free", "total"):
        try:
            value = balance_response.get(key)
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            continue
    balances = balance_response.get("balances")
    if isinstance(balances, dict):
        try:
            return float(balances.get(normalized, 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def reconcile_positions_on_startup(
    position_manager: Any,
    toolkit: Any,
    *,
    dust_threshold: float = 1e-8,
) -> list[str]:
    """Remove local positions with zero on-chain balance."""

    removed: list[str] = []
    for position in list(position_manager.list_open_positions()):
        try:
            balance = _extract_balance(toolkit.get_balance(position.symbol), position.symbol)
        except Exception as exc:
            LOGGER.warning("Balance check failed for %s: %s", position.symbol, exc)
            continue
        if balance <= dust_threshold:
            LOGGER.warning(
                "Reconciliation: removing %s from positions.json (on-chain balance=%s)",
                position.symbol,
                balance,
            )
            position_manager.close_position(position.symbol)
            removed.append(position.symbol)
    return removed


def load_pending_swap_cooldowns(
    execution_log_path: str | Path,
    *,
    cooldown_hours: float = PENDING_COOLDOWN_HOURS,
) -> set[str]:
    """Symbols with recent unconfirmed swap attempts."""

    path = Path(execution_log_path)
    if not path.exists():
        return set()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=cooldown_hours)
    cooled: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(record.get("action", "")).lower() != "enter":
            continue
        ts_raw = record.get("timestamp")
        if not ts_raw:
            continue
        try:
            ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if ts < cutoff:
            continue
        has_tx = bool(record.get("tx_hash"))
        has_error = bool(record.get("error"))
        if has_error or not has_tx:
            symbol = str(record.get("to_symbol") or record.get("from_symbol") or "").upper()
            if symbol:
                cooled.add(symbol)
                LOGGER.warning("Pending/unconfirmed swap cooldown active for %s", symbol)
    return cooled

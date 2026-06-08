"""Deployment runtime helpers for live trading."""

from __future__ import annotations

import logging
from typing import Any

from src.config.eligible_tokens import assert_tradable_subset_of_eligible
from src.config.settings import Settings
from src.deployment.alerts import check_disk_guard
from src.deployment.health_server import start_health_server
from src.deployment.health_state import HealthState
from src.deployment.reconciliation import load_pending_swap_cooldowns, reconcile_positions_on_startup
from src.deployment.twak_unlock import verify_twak_unlock

LOGGER = logging.getLogger(__name__)


def deployment_startup(
    settings: Settings,
    *,
    position_manager: Any,
    toolkit: Any,
    ml_bundle: Any | None,
) -> tuple[HealthState | None, Any, set[str]]:
    """
    Run live deployment checks and services.

    Returns (health_state, health_server, pending_swap_cooldown).
    """

    pending_cooldowns: set[str] = set()
    if not settings.paper_trade:
        assert_tradable_subset_of_eligible()
        unlock = verify_twak_unlock()
        if not unlock["ok"]:
            raise RuntimeError(f"TWAK unlock failed: {unlock['detail']}")
        LOGGER.info("TWAK wallet unlocked: %s", unlock.get("address"))
        removed = reconcile_positions_on_startup(position_manager, toolkit)
        if removed:
            LOGGER.warning("Startup reconciliation removed positions: %s", removed)
        pending_cooldowns = load_pending_swap_cooldowns(settings.execution_log_path)

    health_state: HealthState | None = None
    health_server = None
    port = int(getattr(settings, "health_check_port", 0) or 0)
    if port > 0:
        health_state = HealthState()
        ml_mode = _ml_mode_label(ml_bundle)
        health_state.update(status="ok", ml_mode=ml_mode, ml_active=_ml_active(ml_bundle))
        health_server = start_health_server(
            health_state,
            port=port,
            decision_log_path=settings.decision_log_path,
        )
    return health_state, health_server, pending_cooldowns


def _ml_mode_label(ml_bundle: Any | None) -> str:
    if ml_bundle is None:
        return "disabled"
    if getattr(ml_bundle, "is_ranking_active", False):
        return "ranking_active"
    if getattr(ml_bundle, "is_regime_only_fallback", False):
        return "regime_fallback"
    return "shadow"


def _ml_active(ml_bundle: Any | None) -> bool:
    if ml_bundle is None:
        return False
    return bool(getattr(ml_bundle, "is_ranking_active", False))


def disk_allows_entries(settings: Settings) -> bool:
    return check_disk_guard(
        min_free_bytes=int(getattr(settings, "disk_guard_min_free_bytes", 500_000_000)),
        telegram_token=getattr(settings, "telegram_bot_token", None),
        telegram_chat_id=getattr(settings, "telegram_chat_id", None),
    )


def update_health_snapshot(
    health_state: HealthState | None,
    *,
    guardrails: Any,
    portfolio_value: float,
    position_manager: Any,
    ml_bundle: Any | None,
) -> None:
    if health_state is None:
        return
    ath = float(getattr(guardrails, "portfolio_ath", portfolio_value) or portfolio_value)
    drawdown = 0.0
    if ath > 0:
        drawdown = max(0.0, (ath - portfolio_value) / ath * 100.0)
    health_state.update(
        last_cycle_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        positions=len(position_manager.list_open_positions()),
        daily_trades=int(getattr(guardrails, "daily_trade_count", 0)),
        drawdown_pct=drawdown,
        ml_mode=_ml_mode_label(ml_bundle),
        ml_active=_ml_active(ml_bundle),
        status="ok",
    )

"""Hard budget governor for CMC x402 micropayments.

Every paid call must pass through ``allow_call()`` first. The governor
enforces a daily and a total (competition-window) budget, applies a cooldown
after failed paid calls so a broken endpoint cannot re-bill every loop cycle,
and persists a spend ledger to disk so restarts do not reset the budget.

Degradation is always graceful: when the governor refuses, callers fall back
to the free keyless REST layer, which carries every field the strategy's
entry gates actually require.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

LOGGER = logging.getLogger(__name__)


class X402SpendGovernor:
    """Enforce daily/total x402 spend caps with failure cooldown."""

    def __init__(
        self,
        daily_budget_usdc: float,
        total_budget_usdc: float,
        cost_per_call_usdc: float,
        failure_cooldown_seconds: int = 900,
        ledger_path: str | Path = "logs/x402_spend.json",
    ) -> None:
        self.daily_budget_usdc = max(0.0, float(daily_budget_usdc))
        self.total_budget_usdc = max(0.0, float(total_budget_usdc))
        self.cost_per_call_usdc = max(0.0, float(cost_per_call_usdc))
        self.failure_cooldown_seconds = max(0, int(failure_cooldown_seconds))
        self.ledger_path = Path(ledger_path)
        self._day = self._today()
        self._daily_spend = 0.0
        self._total_spend = 0.0
        self._last_failure_monotonic: float | None = None
        self._load()

    # -- public API ---------------------------------------------------------

    def allow_call(self, calls: int = 1) -> bool:
        """Return whether ``calls`` paid requests fit the remaining budget."""

        self._roll_day_if_needed()
        cost = self.cost_per_call_usdc * max(1, calls)
        if self._in_failure_cooldown():
            LOGGER.info(
                "x402 governor: failure cooldown active (%.0fs left); using keyless fallback",
                self._cooldown_remaining(),
            )
            return False
        if self.daily_budget_usdc > 0 and self._daily_spend + cost > self.daily_budget_usdc:
            LOGGER.warning(
                "x402 governor: daily budget reached ($%.2f/$%.2f); keyless only until UTC midnight",
                self._daily_spend,
                self.daily_budget_usdc,
            )
            return False
        if self.total_budget_usdc > 0 and self._total_spend + cost > self.total_budget_usdc:
            LOGGER.warning(
                "x402 governor: total budget reached ($%.2f/$%.2f); keyless only",
                self._total_spend,
                self.total_budget_usdc,
            )
            return False
        return True

    def record_spend(self, amount_usdc: float | None = None) -> None:
        """Record a successful paid call (defaults to the per-call cap)."""

        self._roll_day_if_needed()
        spent = self.cost_per_call_usdc if amount_usdc is None else max(0.0, float(amount_usdc))
        self._daily_spend += spent
        self._total_spend += spent
        self._last_failure_monotonic = None
        self._save()

    def record_failure(self, assume_charged: bool = True) -> None:
        """Record a failed paid call and start the retry cooldown.

        ``assume_charged`` budgets conservatively: a call that failed after the
        402 payment settled still spent money, so count it unless the failure
        is known to have happened before payment.
        """

        self._roll_day_if_needed()
        if assume_charged:
            self._daily_spend += self.cost_per_call_usdc
            self._total_spend += self.cost_per_call_usdc
        self._last_failure_monotonic = time.monotonic()
        self._save()

    def snapshot(self) -> dict[str, float | str | bool]:
        """Telemetry payload for logs and the health endpoint."""

        self._roll_day_if_needed()
        return {
            "day": self._day,
            "daily_spend_usdc": round(self._daily_spend, 4),
            "daily_budget_usdc": self.daily_budget_usdc,
            "total_spend_usdc": round(self._total_spend, 4),
            "total_budget_usdc": self.total_budget_usdc,
            "failure_cooldown_active": self._in_failure_cooldown(),
        }

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).date().isoformat()

    def _roll_day_if_needed(self) -> None:
        today = self._today()
        if today != self._day:
            self._day = today
            self._daily_spend = 0.0
            self._save()

    def _in_failure_cooldown(self) -> bool:
        return self._cooldown_remaining() > 0

    def _cooldown_remaining(self) -> float:
        if self._last_failure_monotonic is None or self.failure_cooldown_seconds <= 0:
            return 0.0
        elapsed = time.monotonic() - self._last_failure_monotonic
        return max(0.0, self.failure_cooldown_seconds - elapsed)

    def _load(self) -> None:
        if not self.ledger_path.exists():
            return
        try:
            payload = json.loads(self.ledger_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            LOGGER.warning("x402 governor: could not read ledger %s: %s", self.ledger_path, exc)
            return
        if not isinstance(payload, dict):
            return
        self._total_spend = float(payload.get("total_spend_usdc", 0.0))
        if str(payload.get("day", "")) == self._day:
            self._daily_spend = float(payload.get("daily_spend_usdc", 0.0))

    def _save(self) -> None:
        try:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            self.ledger_path.write_text(
                json.dumps(
                    {
                        "day": self._day,
                        "daily_spend_usdc": round(self._daily_spend, 6),
                        "total_spend_usdc": round(self._total_spend, 6),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            LOGGER.warning("x402 governor: could not persist ledger: %s", exc)

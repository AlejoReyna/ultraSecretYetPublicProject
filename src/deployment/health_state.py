"""Shared runtime state for the health check HTTP server."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class HealthState:
    """Thread-safe snapshot updated by the trading loop."""

    last_cycle_at: datetime | None = None
    positions: int = 0
    ml_mode: str = "disabled"
    daily_trades: int = 0
    drawdown_pct: float = 0.0
    ml_active: bool = False
    status: str = "starting"
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, **kwargs: Any) -> None:
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            last_cycle = self.last_cycle_at.isoformat() if self.last_cycle_at else None
            return {
                "status": self.status,
                "last_cycle": last_cycle,
                "positions": self.positions,
                "ml_mode": self.ml_mode,
                "daily_trades": self.daily_trades,
                "drawdown_pct": round(self.drawdown_pct, 4),
                "ml_active": self.ml_active,
            }

    def is_stalled(self, stall_minutes: float = 15.0) -> bool:
        with self._lock:
            if self.last_cycle_at is None:
                return False
            age = (datetime.now(timezone.utc) - self.last_cycle_at).total_seconds() / 60.0
            return age > stall_minutes

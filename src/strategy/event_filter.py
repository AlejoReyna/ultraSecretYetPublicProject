"""RWEAL Phase 1 -- Real-World Event Awareness Layer (static event gate).

This is an **entry-only** pre-flight filter. It deliberately does NOT touch the
audited exit path (``calculate_exit_levels``, time-stop, window-flatten) and does
NOT change position sizing. Phase 1 has two mechanisms:

1. **Manual halt** -- a control *file* on disk (default ``TRADING_HALT``). While
   the file exists, ALL entries are suppressed *and* the daily-minimum compliance
   trade is suppressed (an operator full-stop -- accepts the competition's
   one-trade-per-day disqualification risk). A file flag is used instead of an
   env var on purpose: a running process never re-reads a changed ``.env``, and
   the file can be ``stat()``-ed cheaply every cycle for near-instant response.

2. **Event blackout** -- a static ``events.json`` calendar. Within a configurable
   horizon before/around a scheduled adverse event, *discretionary* entries are
   blocked. Symbol-specific events block that symbol; ``GLOBAL`` events (e.g. a
   macro CPI/FOMC release) block all discretionary entries. Unlike a manual halt,
   an event blackout leaves the tiny daily-minimum compliance trade running so the
   agent stays qualified.

Failure model:
* Enabling RWEAL with a **malformed** ``events.json`` is a hard startup error
  (``RwealConfigError``) -- fail fast while the operator is present to fix it.
* A **missing** ``events.json`` is allowed (you can run with only the manual kill
  switch); it is treated as an empty calendar with a warning.
* If the file is edited to an invalid state **while running**, the reload keeps
  the last-known-good calendar and logs an error rather than crashing the loop.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.config.settings import Settings

LOGGER = logging.getLogger(__name__)

GLOBAL_SYMBOL = "GLOBAL"
_VALID_SEVERITY = range(1, 6)
_REQUIRED_KEYS = ("symbol", "event_type", "scheduled_time", "severity")


class RwealConfigError(ValueError):
    """Raised when an events.json file is present but invalid at load time."""


@dataclass(frozen=True)
class Event:
    """A normalized, scheduled real-world event (Phase 1 subset)."""

    symbol: str
    event_type: str
    scheduled_time: datetime
    severity: int
    direction_bias: str = "unknown"
    blackout_minutes: int | None = None
    description: str = ""

    @property
    def is_global(self) -> bool:
        return self.symbol == GLOBAL_SYMBOL


def _parse_timestamp(raw: object, where: str) -> datetime:
    if not isinstance(raw, str) or not raw.strip():
        raise RwealConfigError(f"{where}: scheduled_time must be an ISO-8601 string")
    text = raw.strip()
    # Accept a trailing 'Z' (UTC) which datetime.fromisoformat rejects < 3.11.
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise RwealConfigError(f"{where}: invalid scheduled_time {raw!r}: {exc}") from exc
    # Treat naive timestamps as UTC (the rest of the bot is UTC-native).
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_event(raw: object, index: int) -> Event:
    where = f"events[{index}]"
    if not isinstance(raw, dict):
        raise RwealConfigError(f"{where}: each event must be an object")
    missing = [k for k in _REQUIRED_KEYS if k not in raw]
    if missing:
        raise RwealConfigError(f"{where}: missing required keys: {', '.join(missing)}")

    symbol = str(raw["symbol"]).strip().upper()
    if not symbol:
        raise RwealConfigError(f"{where}: symbol must be non-empty")

    event_type = str(raw["event_type"]).strip().lower()
    if not event_type:
        raise RwealConfigError(f"{where}: event_type must be non-empty")

    scheduled = _parse_timestamp(raw["scheduled_time"], where)

    try:
        severity = int(raw["severity"])
    except (TypeError, ValueError) as exc:
        raise RwealConfigError(f"{where}: severity must be an integer") from exc
    if severity not in _VALID_SEVERITY:
        raise RwealConfigError(f"{where}: severity {severity} out of range 1-5")

    blackout_minutes = raw.get("blackout_minutes")
    if blackout_minutes is not None:
        try:
            blackout_minutes = int(blackout_minutes)
        except (TypeError, ValueError) as exc:
            raise RwealConfigError(f"{where}: blackout_minutes must be an integer") from exc
        if blackout_minutes < 0:
            raise RwealConfigError(f"{where}: blackout_minutes must be >= 0")

    return Event(
        symbol=symbol,
        event_type=event_type,
        scheduled_time=scheduled,
        severity=severity,
        direction_bias=str(raw.get("direction_bias", "unknown")).strip().lower() or "unknown",
        blackout_minutes=blackout_minutes,
        description=str(raw.get("description", "")),
    )


def parse_events(payload: object) -> list[Event]:
    """Validate and normalize a parsed events.json payload into Event objects."""

    if isinstance(payload, dict):
        events_raw = payload.get("events", [])
    elif isinstance(payload, list):
        events_raw = payload
    else:
        raise RwealConfigError("events.json root must be an object or a list")
    if not isinstance(events_raw, list):
        raise RwealConfigError("'events' must be a list")
    return [_parse_event(item, i) for i, item in enumerate(events_raw)]


@dataclass
class EventRiskFilter:
    """Static, entry-only event gate. See module docstring."""

    events_path: Path
    control_file: Path
    blackout_horizon_hours: float = 6.0
    post_event_minutes: int = 60
    _events: list[Event] = field(default_factory=list, init=False)
    _mtime: float | None = field(default=None, init=False)

    @classmethod
    def from_settings(cls, settings: "Settings") -> "EventRiskFilter":
        """Build from Settings and load the calendar (raises on malformed file)."""

        flt = cls(
            events_path=Path(settings.rweal_events_path),
            control_file=Path(settings.rweal_control_file),
            blackout_horizon_hours=float(settings.rweal_blackout_horizon_hours),
            post_event_minutes=int(settings.rweal_post_event_minutes),
        )
        flt.load(strict=True)
        return flt

    # -- calendar loading -------------------------------------------------

    def load(self, *, strict: bool) -> bool:
        """Load events from disk.

        ``strict=True`` (startup): a present-but-malformed file raises
        ``RwealConfigError``. ``strict=False`` (runtime reload): on any error the
        previous calendar is kept and the error is logged.
        Returns True if the in-memory calendar was (re)loaded.
        """

        if not self.events_path.exists():
            if self._events:
                # File removed while running: keep last-good rather than blanking.
                LOGGER.warning(
                    "RWEAL events file %s disappeared; keeping last-known-good (%d events)",
                    self.events_path,
                    len(self._events),
                )
                return False
            LOGGER.warning(
                "RWEAL enabled but no events file at %s; running with manual halt only",
                self.events_path,
            )
            self._mtime = None
            return True

        try:
            mtime = self.events_path.stat().st_mtime
            text = self.events_path.read_text(encoding="utf-8")
            payload = json.loads(text)
            events = parse_events(payload)
        except (OSError, json.JSONDecodeError, RwealConfigError) as exc:
            if strict:
                raise RwealConfigError(
                    f"Invalid RWEAL events file {self.events_path}: {exc}"
                ) from exc
            LOGGER.error(
                "RWEAL events reload failed (%s); keeping last-known-good (%d events)",
                exc,
                len(self._events),
            )
            return False

        self._events = events
        self._mtime = mtime
        LOGGER.info("RWEAL loaded %d event(s) from %s", len(events), self.events_path)
        return True

    def _maybe_reload(self) -> None:
        """Hot-reload the calendar if the file's mtime changed; keep last-good."""

        try:
            exists = self.events_path.exists()
        except OSError:
            return
        if not exists:
            return
        try:
            mtime = self.events_path.stat().st_mtime
        except OSError:
            return
        if mtime != self._mtime:
            self.load(strict=False)

    # -- gates ------------------------------------------------------------

    def manual_halt_active(self) -> bool:
        """True when the operator's control file is present (full-stop)."""

        try:
            return self.control_file.exists()
        except OSError:  # pragma: no cover - defensive
            return False

    def _blackout_window(self, event: Event) -> tuple[datetime, datetime]:
        if event.blackout_minutes is not None:
            delta = timedelta(minutes=event.blackout_minutes)
            return event.scheduled_time - delta, event.scheduled_time + delta
        return (
            event.scheduled_time - timedelta(hours=self.blackout_horizon_hours),
            event.scheduled_time + timedelta(minutes=self.post_event_minutes),
        )

    def _active_reason(self, event: Event, now: datetime) -> str | None:
        start, end = self._blackout_window(event)
        if start <= now <= end:
            return (
                f"event_blackout:{event.event_type}:{event.symbol} "
                f"@ {event.scheduled_time.isoformat()} (sev {event.severity})"
            )
        return None

    def global_blackout(self, now: datetime) -> str | None:
        """Reason string if a GLOBAL (e.g. macro) event blackout is active, else None."""

        self._maybe_reload()
        for event in self._events:
            if event.is_global:
                reason = self._active_reason(event, now)
                if reason:
                    return reason
        return None

    def symbol_blackout(self, symbol: str, now: datetime) -> str | None:
        """Reason string if a blackout is active for ``symbol``, else None.

        Includes GLOBAL events, so a per-symbol check alone is sufficient when
        evaluating a specific discretionary candidate.
        """

        self._maybe_reload()
        target = symbol.strip().upper()
        for event in self._events:
            if event.symbol == target or event.is_global:
                reason = self._active_reason(event, now)
                if reason:
                    return reason
        return None

    def active_symbol_blackouts(self, now: datetime) -> set[str]:
        """Set of non-GLOBAL symbols whose blackout window is active right now.

        Used to exclude blacked-out symbols from discretionary candidate
        selection so other valid symbols are still considered (a symbol-specific
        event blocks only that symbol, not the whole universe). GLOBAL events are
        intentionally excluded here -- they are handled by ``global_blackout``.
        """

        self._maybe_reload()
        blocked: set[str] = set()
        for event in self._events:
            if event.is_global:
                continue
            if self._active_reason(event, now):
                blocked.add(event.symbol)
        return blocked

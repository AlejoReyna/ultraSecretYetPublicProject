"""Tests for RWEAL Phase 1 event filter (src/strategy/event_filter.py)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.config.settings import Settings
from src.strategy.event_filter import (
    EventRiskFilter,
    RwealConfigError,
    parse_events,
)

NOW = datetime(2026, 6, 22, 12, 0, 0, tzinfo=timezone.utc)


def _write(path: Path, events: list[dict]) -> Path:
    path.write_text(json.dumps({"version": "1.0", "events": events}), encoding="utf-8")
    return path


def _filter(events_path: Path, control: Path, **kw) -> EventRiskFilter:
    flt = EventRiskFilter(
        events_path=events_path,
        control_file=control,
        blackout_horizon_hours=kw.get("horizon", 6.0),
        post_event_minutes=kw.get("post", 60),
    )
    flt.load(strict=True)
    return flt


# -- blackout timing --------------------------------------------------------


def test_blocks_entry_before_event(tmp_path: Path) -> None:
    # Event 4h in the future, default 6h horizon -> blocked now.
    events = [{
        "symbol": "CAKE", "event_type": "unlock",
        "scheduled_time": (NOW + timedelta(hours=4)).isoformat(),
        "severity": 5,
    }]
    flt = _filter(_write(tmp_path / "e.json", events), tmp_path / "HALT")
    assert flt.symbol_blackout("CAKE", NOW) is not None


def test_allows_entry_outside_horizon(tmp_path: Path) -> None:
    # Event 10h out, 6h horizon -> not yet blocked.
    events = [{
        "symbol": "CAKE", "event_type": "unlock",
        "scheduled_time": (NOW + timedelta(hours=10)).isoformat(),
        "severity": 5,
    }]
    flt = _filter(_write(tmp_path / "e.json", events), tmp_path / "HALT")
    assert flt.symbol_blackout("CAKE", NOW) is None


def test_allows_entry_after_post_window(tmp_path: Path) -> None:
    # Event 2h ago, post window 60m -> reopened.
    events = [{
        "symbol": "CAKE", "event_type": "unlock",
        "scheduled_time": (NOW - timedelta(hours=2)).isoformat(),
        "severity": 5,
    }]
    flt = _filter(_write(tmp_path / "e.json", events), tmp_path / "HALT")
    assert flt.symbol_blackout("CAKE", NOW) is None


def test_other_symbol_unaffected(tmp_path: Path) -> None:
    events = [{
        "symbol": "CAKE", "event_type": "unlock",
        "scheduled_time": (NOW + timedelta(hours=1)).isoformat(),
        "severity": 5,
    }]
    flt = _filter(_write(tmp_path / "e.json", events), tmp_path / "HALT")
    assert flt.symbol_blackout("BNB", NOW) is None


def test_blackout_minutes_symmetric_window(tmp_path: Path) -> None:
    # Macro with explicit +/-30m window. 20m before -> blocked; 40m before -> not.
    events = [{
        "symbol": "GLOBAL", "event_type": "macro",
        "scheduled_time": NOW.isoformat(), "severity": 4,
        "blackout_minutes": 30,
    }]
    flt = _filter(_write(tmp_path / "e.json", events), tmp_path / "HALT")
    assert flt.global_blackout(NOW - timedelta(minutes=20)) is not None
    assert flt.global_blackout(NOW - timedelta(minutes=40)) is None


def test_global_event_blocks_any_symbol_lookup(tmp_path: Path) -> None:
    events = [{
        "symbol": "GLOBAL", "event_type": "macro",
        "scheduled_time": NOW.isoformat(), "severity": 4,
        "blackout_minutes": 30,
    }]
    flt = _filter(_write(tmp_path / "e.json", events), tmp_path / "HALT")
    # symbol_blackout includes GLOBAL events.
    assert flt.symbol_blackout("ANYTOKEN", NOW) is not None
    assert flt.global_blackout(NOW) is not None


# -- active_symbol_blackouts (selection exclusion) --------------------------


def test_active_symbol_blackouts_returns_symbol_events(tmp_path: Path) -> None:
    events = [
        {"symbol": "CAKE", "event_type": "unlock",
         "scheduled_time": (NOW + timedelta(hours=1)).isoformat(), "severity": 5},
        {"symbol": "XVS", "event_type": "unlock",
         "scheduled_time": (NOW + timedelta(hours=48)).isoformat(), "severity": 5},
    ]
    flt = _filter(_write(tmp_path / "e.json", events), tmp_path / "HALT")
    active = flt.active_symbol_blackouts(NOW)
    assert active == {"CAKE"}  # XVS far out, not yet active


def test_active_symbol_blackouts_excludes_global(tmp_path: Path) -> None:
    events = [{
        "symbol": "GLOBAL", "event_type": "macro",
        "scheduled_time": NOW.isoformat(), "severity": 4, "blackout_minutes": 30,
    }]
    flt = _filter(_write(tmp_path / "e.json", events), tmp_path / "HALT")
    # GLOBAL is handled by the cycle-top gate, not symbol exclusion.
    assert flt.active_symbol_blackouts(NOW) == set()


# -- manual halt ------------------------------------------------------------


def test_manual_halt_follows_control_file(tmp_path: Path) -> None:
    control = tmp_path / "TRADING_HALT"
    flt = _filter(_write(tmp_path / "e.json", []), control)
    assert flt.manual_halt_active() is False
    control.write_text("", encoding="utf-8")
    assert flt.manual_halt_active() is True
    control.unlink()
    assert flt.manual_halt_active() is False


# -- loading / validation ---------------------------------------------------


def test_missing_file_is_allowed(tmp_path: Path) -> None:
    flt = EventRiskFilter(events_path=tmp_path / "nope.json", control_file=tmp_path / "HALT")
    flt.load(strict=True)  # must not raise
    assert flt.symbol_blackout("CAKE", NOW) is None


def test_malformed_file_hard_fails_at_load(tmp_path: Path) -> None:
    bad = tmp_path / "e.json"
    bad.write_text("{not valid json", encoding="utf-8")
    flt = EventRiskFilter(events_path=bad, control_file=tmp_path / "HALT")
    with pytest.raises(RwealConfigError):
        flt.load(strict=True)


def test_missing_required_key_raises(tmp_path: Path) -> None:
    with pytest.raises(RwealConfigError):
        parse_events({"events": [{"symbol": "CAKE", "event_type": "unlock"}]})


def test_bad_severity_raises(tmp_path: Path) -> None:
    with pytest.raises(RwealConfigError):
        parse_events({"events": [{
            "symbol": "CAKE", "event_type": "unlock",
            "scheduled_time": NOW.isoformat(), "severity": 9,
        }]})


def test_reload_keeps_last_good_on_bad_edit(tmp_path: Path) -> None:
    path = tmp_path / "e.json"
    events = [{
        "symbol": "CAKE", "event_type": "unlock",
        "scheduled_time": (NOW + timedelta(hours=1)).isoformat(),
        "severity": 5,
    }]
    flt = _filter(_write(path, events), tmp_path / "HALT")
    assert flt.symbol_blackout("CAKE", NOW) is not None
    # Corrupt the file, bump mtime so reload triggers.
    path.write_text("garbage", encoding="utf-8")
    import os
    future = (NOW.timestamp() + 9999)
    os.utime(path, (future, future))
    # Calendar still serves the last-known-good events instead of crashing.
    assert flt.symbol_blackout("CAKE", NOW) is not None


def test_naive_timestamp_treated_as_utc(tmp_path: Path) -> None:
    events = [{
        "symbol": "CAKE", "event_type": "unlock",
        "scheduled_time": "2026-06-22T13:00:00",  # naive, +1h
        "severity": 5,
    }]
    flt = _filter(_write(tmp_path / "e.json", events), tmp_path / "HALT")
    assert flt.symbol_blackout("CAKE", NOW) is not None


def test_trailing_z_timestamp_parsed(tmp_path: Path) -> None:
    events = [{
        "symbol": "CAKE", "event_type": "unlock",
        "scheduled_time": "2026-06-22T13:00:00Z",
        "severity": 5,
    }]
    flt = _filter(_write(tmp_path / "e.json", events), tmp_path / "HALT")
    assert flt.symbol_blackout("CAKE", NOW) is not None


# -- settings integration ---------------------------------------------------


def test_from_settings_builds_disabled_by_default() -> None:
    s = Settings()
    assert s.enable_rweal is False
    assert s.rweal_control_file == "TRADING_HALT"


def test_from_settings_loads_calendar(tmp_path: Path) -> None:
    path = _write(tmp_path / "e.json", [{
        "symbol": "CAKE", "event_type": "unlock",
        "scheduled_time": (NOW + timedelta(hours=1)).isoformat(),
        "severity": 5,
    }])
    s = Settings(
        enable_rweal=True,
        rweal_events_path=str(path),
        rweal_control_file=str(tmp_path / "HALT"),
    )
    flt = EventRiskFilter.from_settings(s)
    assert flt.symbol_blackout("CAKE", NOW) is not None

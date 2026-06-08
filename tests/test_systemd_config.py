"""Tests for systemd service configuration."""

from __future__ import annotations

from pathlib import Path


def test_systemd_service_points_to_main_and_env() -> None:
    service = Path("systemd/planb-plus.service").read_text(encoding="utf-8")
    assert "EnvironmentFile=/home/ubuntu/planb-plus/.env" in service
    assert "python -m src.main --live" in service
    assert "WorkingDirectory=/home/ubuntu/planb-plus" in service
    assert "RestartSec=30" in service
    assert "StartLimitBurst=5" in service
    assert "verify_twak_unlock.py" in service

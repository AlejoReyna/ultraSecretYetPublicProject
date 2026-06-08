"""Tests for health check HTTP server."""

from __future__ import annotations

import json
import urllib.request

from src.deployment.health_state import HealthState
from src.deployment.health_server import start_health_server


def test_health_endpoint_returns_required_keys() -> None:
    state = HealthState()
    state.update(status="ok", positions=2, ml_mode="regime_fallback", daily_trades=1, drawdown_pct=4.2)
    server = start_health_server(state, port=18080, decision_log_path="decision_log.jsonl")
    try:
        with urllib.request.urlopen("http://127.0.0.1:18080/health", timeout=2) as resp:
            assert resp.status == 200
            payload = json.loads(resp.read().decode("utf-8"))
        for key in ("status", "positions", "ml_mode", "daily_trades", "drawdown_pct"):
            assert key in payload
        assert payload["ml_mode"] == "regime_fallback"
    finally:
        server.shutdown()

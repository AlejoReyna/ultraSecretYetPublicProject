"""Tests for cached market snapshot loading in main."""

from __future__ import annotations

from typing import Any

import src.main as main_module
from src.config.settings import Settings
from src.data.market_snapshot_cache import get_dual_market_snapshot_cache, get_market_snapshot_cache


class FakeCMCClient:
    def __init__(self) -> None:
        self.calls = 0

    def fetch_market_snapshot(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        self.calls += 1
        return {"CAKE": {"symbol": "CAKE", "price": float(self.calls)}}


class DualFakeCMCClient:
    def __init__(self) -> None:
        self.x402_calls = 0
        self.keyless_calls = 0

    def fetch_x402_enriched_snapshot(
        self,
        symbols: list[str],
        id_overrides: dict[str, str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        self.x402_calls += 1
        return {
            "CAKE": {
                "symbol": "CAKE",
                "price": 1.0,
                "estimated_slippage_pct": 0.002,
            }
        }

    def fetch_keyless_quotes_snapshot(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        self.keyless_calls += 1
        return {"CAKE": {"symbol": "CAKE", "price": float(self.keyless_calls)}}


def test_fetch_snapshot_reuses_cmc_client_within_ttl() -> None:
    get_market_snapshot_cache().reset()
    settings = Settings(paper_trade=False, cmc_snapshot_ttl_seconds=7200)
    client = FakeCMCClient()

    first = main_module._fetch_snapshot(settings, client)  # type: ignore[arg-type]
    second = main_module._fetch_snapshot(settings, client)  # type: ignore[arg-type]

    assert first == {"CAKE": {"symbol": "CAKE", "price": 1.0}}
    assert second == first
    assert client.calls == 1
    get_market_snapshot_cache().reset()


def test_fetch_snapshot_dual_mode_refreshes_keyless_each_cycle(monkeypatch: object) -> None:
    get_dual_market_snapshot_cache().reset()
    now = {"value": 1000.0}
    monkeypatch.setattr("src.data.market_snapshot_cache.time.monotonic", lambda: now["value"])  # type: ignore[attr-defined]
    settings = Settings(
        paper_trade=False,
        use_dual_market_data=True,
        use_keyless_primary=False,
        cmc_snapshot_ttl_seconds=7200,
        cmc_keyless_snapshot_ttl_seconds=300,
        loop_seconds=300,
    )
    client = DualFakeCMCClient()

    first = main_module._fetch_snapshot(settings, client)  # type: ignore[arg-type]
    now["value"] += 300
    second = main_module._fetch_snapshot(settings, client)  # type: ignore[arg-type]

    assert first["CAKE"]["price"] == 1.0
    assert second["CAKE"]["price"] == 2.0
    assert second["CAKE"]["estimated_slippage_pct"] == 0.002
    assert client.x402_calls == 1
    assert client.keyless_calls == 2
    get_dual_market_snapshot_cache().reset()

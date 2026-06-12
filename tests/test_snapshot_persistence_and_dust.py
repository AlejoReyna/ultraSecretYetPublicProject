"""Tests for x402 snapshot disk persistence, dust threshold, and event-driven enrichment."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import src.main as main_module
from src.config.settings import Settings
from src.data.enrichment_planner import (
    cheap_gate_pass_count,
    hot_candidate_symbols,
    select_enrichment_symbols,
)
from src.data.market_snapshot_cache import DualMarketSnapshotCache, get_dual_market_snapshot_cache


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "paper_trade": False,
        "use_dual_market_data": True,
        "use_keyless_primary": False,
        "cmc_snapshot_ttl_seconds": 14400,
        "cmc_keyless_snapshot_ttl_seconds": 300,
        "loop_seconds": 300,
        "x402_in_position_ttl_seconds": 1800,
        "x402_min_position_value_usdc": 5.0,
        "x402_hot_refresh_age_seconds": 600,
        "x402_enrich_top_n": 50,
    }
    base.update(overrides)
    return Settings(**base)


class DualFakeCMCClient:
    def __init__(self) -> None:
        self.x402_calls = 0
        self.keyless_calls = 0
        self.x402_symbols: list[list[str]] = []
        self.keyless_payload: dict[str, dict[str, Any]] = {
            "CAKE": {"symbol": "CAKE", "price": 1.0, "volume_24h": 1000.0}
        }

    def fetch_x402_enriched_snapshot(
        self,
        symbols: list[str],
        id_overrides: dict[str, str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        self.x402_calls += 1
        self.x402_symbols.append(list(symbols))
        self.last_id_overrides = dict(id_overrides or {})
        return {
            "CAKE": {"symbol": "CAKE", "price": 1.0, "rsi": 60.0, "estimated_slippage_pct": 0.002}
        }

    def fetch_keyless_quotes_snapshot(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        self.keyless_calls += 1
        return {key: dict(value) for key, value in self.keyless_payload.items()}


# -- disk persistence -------------------------------------------------------


def test_persisted_x402_snapshot_survives_restart(tmp_path: Path) -> None:
    persist = tmp_path / "snapshot_cache.json"
    calls = {"x402": 0}

    def x402_fetcher() -> dict[str, dict[str, Any]]:
        calls["x402"] += 1
        return {"CAKE": {"symbol": "CAKE", "price": 1.0, "rsi": 60.0}}

    def keyless_fetcher() -> dict[str, dict[str, Any]]:
        return {"CAKE": {"symbol": "CAKE", "price": 2.0}}

    cache = DualMarketSnapshotCache(persist_path=persist)
    cache.get_merged_snapshot(14400, 300, x402_fetcher, keyless_fetcher)
    assert calls["x402"] == 1
    assert persist.exists()

    # Simulate a process restart: a brand-new cache instance reads the file
    # and must NOT trigger another paid refresh while the TTL is fresh.
    restarted = DualMarketSnapshotCache(persist_path=persist)
    snapshot = restarted.get_merged_snapshot(14400, 300, x402_fetcher, keyless_fetcher)
    assert calls["x402"] == 1
    assert snapshot["CAKE"]["rsi"] == 60.0
    assert snapshot["CAKE"]["price"] == 2.0  # keyless hot field overlays


def test_persisted_snapshot_expired_ttl_triggers_refresh(tmp_path: Path) -> None:
    persist = tmp_path / "snapshot_cache.json"
    stale_payload = {
        "x402_enriched": {"CAKE": {"symbol": "CAKE", "price": 1.0}},
        "x402_fetched_at_epoch": time.time() - 99999,
    }
    persist.write_text(json.dumps(stale_payload), encoding="utf-8")
    calls = {"x402": 0}

    def x402_fetcher() -> dict[str, dict[str, Any]]:
        calls["x402"] += 1
        return {"CAKE": {"symbol": "CAKE", "price": 3.0}}

    cache = DualMarketSnapshotCache(persist_path=persist)
    assert cache.x402_age_seconds() is not None
    assert cache.x402_age_seconds() > 14400
    cache.get_merged_snapshot(14400, 300, x402_fetcher, lambda: {})
    assert calls["x402"] == 1


def test_corrupt_persist_file_is_ignored(tmp_path: Path) -> None:
    persist = tmp_path / "snapshot_cache.json"
    persist.write_text("{not json", encoding="utf-8")
    cache = DualMarketSnapshotCache(persist_path=persist)
    assert cache.x402_age_seconds() is None


def test_reset_removes_persist_file(tmp_path: Path) -> None:
    persist = tmp_path / "snapshot_cache.json"
    cache = DualMarketSnapshotCache(persist_path=persist)
    cache.get_merged_snapshot(
        14400, 300, lambda: {"CAKE": {"symbol": "CAKE", "price": 1.0}}, lambda: {}
    )
    assert persist.exists()
    cache.reset()
    assert not persist.exists()


# -- dust threshold ---------------------------------------------------------


def _run_fetch(settings: Settings, client: DualFakeCMCClient, **kwargs: Any) -> None:
    main_module._fetch_snapshot(settings, client, **kwargs)  # type: ignore[arg-type]


def test_dust_positions_do_not_activate_in_position_ttl(monkeypatch: Any) -> None:
    get_dual_market_snapshot_cache().reset()
    now = {"value": 1000.0}
    monkeypatch.setattr("src.data.market_snapshot_cache.time.monotonic", lambda: now["value"])
    settings = _settings()
    client = DualFakeCMCClient()

    # Dust ($0.50 total) -> flat TTL 14400s applies, not the 1800s TTL.
    _run_fetch(settings, client, open_position_value_usdc=0.5, position_symbols={"SHIB"})
    assert client.x402_calls == 1
    now["value"] += 2000.0  # past in-position TTL, within flat TTL
    _run_fetch(settings, client, open_position_value_usdc=0.5, position_symbols={"SHIB"})
    assert client.x402_calls == 1  # no paid refresh for dust
    get_dual_market_snapshot_cache().reset()


def test_real_position_activates_in_position_ttl(monkeypatch: Any) -> None:
    get_dual_market_snapshot_cache().reset()
    now = {"value": 1000.0}
    monkeypatch.setattr("src.data.market_snapshot_cache.time.monotonic", lambda: now["value"])
    settings = _settings()
    client = DualFakeCMCClient()

    _run_fetch(settings, client, open_position_value_usdc=20.0, position_symbols={"CAKE"})
    assert client.x402_calls == 1
    now["value"] += 2000.0  # past 1800s in-position TTL
    _run_fetch(settings, client, open_position_value_usdc=20.0, position_symbols={"CAKE"})
    assert client.x402_calls == 2
    get_dual_market_snapshot_cache().reset()


# -- event-driven hot-candidate refresh -------------------------------------


def test_hot_candidate_forces_paid_refresh(monkeypatch: Any) -> None:
    get_dual_market_snapshot_cache().reset()
    now = {"value": 1000.0}
    monkeypatch.setattr("src.data.market_snapshot_cache.time.monotonic", lambda: now["value"])
    settings = _settings()
    client = DualFakeCMCClient()

    _run_fetch(settings, client)
    assert client.x402_calls == 1

    # Flat, snapshot 900s old (>600s hot age, <14400s TTL), candidate passes
    # both cheap gates -> forced paid refresh.
    client.keyless_payload = {
        "CAKE": {
            "symbol": "CAKE",
            "price": 110.0,
            "volume_1h": 500.0,
            "rolling_24h_hourly_volume_avg": 100.0,
            "volume_24h": 2400.0,
            "high_3h": 100.0,
        }
    }
    now["value"] += 900.0
    _run_fetch(settings, client)
    assert client.x402_calls == 2
    get_dual_market_snapshot_cache().reset()


def test_no_hot_candidate_keeps_flat_ttl(monkeypatch: Any) -> None:
    get_dual_market_snapshot_cache().reset()
    now = {"value": 1000.0}
    monkeypatch.setattr("src.data.market_snapshot_cache.time.monotonic", lambda: now["value"])
    settings = _settings()
    client = DualFakeCMCClient()

    _run_fetch(settings, client)
    now["value"] += 900.0
    _run_fetch(settings, client)  # quiet market -> no paid refresh
    assert client.x402_calls == 1
    get_dual_market_snapshot_cache().reset()


# -- cheap gates / enrichment scope ------------------------------------------


def test_cheap_gate_pass_count_both_gates() -> None:
    settings = _settings()
    token = {
        "price": 110.0,
        "volume_1h": 500.0,
        "rolling_24h_hourly_volume_avg": 100.0,
        "high_3h": 100.0,
    }
    assert cheap_gate_pass_count(token, settings) == 2
    assert cheap_gate_pass_count({"price": 1.0}, settings) == 0


def test_hot_candidate_symbols_filters_non_tradable() -> None:
    settings = _settings()
    snapshot = {
        "CAKE": {
            "price": 110.0,
            "volume_1h": 500.0,
            "rolling_24h_hourly_volume_avg": 100.0,
            "high_3h": 100.0,
        },
        "NOTATOKEN": {
            "price": 110.0,
            "volume_1h": 500.0,
            "rolling_24h_hourly_volume_avg": 100.0,
            "high_3h": 100.0,
        },
    }
    assert hot_candidate_symbols(snapshot, settings) == ["CAKE"]


def test_select_enrichment_symbols_caps_to_top_n_with_positions() -> None:
    settings = _settings(x402_enrich_top_n=2)
    targets = ["CAKE", "UNI", "LTC", "AAVE", "BNB"]
    snapshot = {
        "CAKE": {"price": 110.0, "volume_1h": 500.0, "rolling_24h_hourly_volume_avg": 100.0, "high_3h": 100.0, "volume_24h": 2400.0},
        "UNI": {"price": 1.0, "volume_24h": 900.0},
        "LTC": {"price": 1.0, "volume_24h": 100.0},
        "AAVE": {"price": 1.0, "volume_24h": 50.0},
    }
    selected = select_enrichment_symbols(snapshot, targets, {"AAVE"}, settings)
    assert selected[0] == "CAKE"  # best cheap rank first
    assert "AAVE" in selected  # open position always enriched
    assert "BNB" in selected  # regime reference always enriched
    assert len(selected) <= 4


def test_select_enrichment_symbols_zero_means_all() -> None:
    settings = _settings(x402_enrich_top_n=0)
    targets = ["CAKE", "UNI", "LTC"]
    assert select_enrichment_symbols({}, targets, set(), settings) == targets

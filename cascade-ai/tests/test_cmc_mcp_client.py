"""Tests for CoinMarketCap MCP request envelopes."""

from __future__ import annotations

from typing import Any

import requests

from src.config.settings import Settings
from src.data.cmc_mcp_client import CMCMCPClient


class FakeX402Client:
    """Capture outgoing MCP request data."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def request_with_x402(
        self,
        method: str,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        self.calls.append({"method": method, "payload": payload, "headers": headers})
        return {"result": {"ok": True}}


class FailingX402Client:
    """Simulate an x402 failure after the payment flow."""

    def request_with_x402(self, method: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        raise RuntimeError("paid retry failed")


class FakeKeylessResponse:
    """Minimal response object for the Keyless API."""

    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self._payload = payload or {
            "data": {
                "CAKE": {
                    "symbol": "CAKE",
                    "quote": {
                        "USD": {
                            "price": 10.0,
                            "volume_24h": 500000.0,
                            "percent_change_1h": 1.2,
                            "market_cap": 2000000.0,
                        }
                    },
                }
            }
        }

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        return None


def test_cmc_mcp_uses_documented_api_key_header() -> None:
    fake_x402 = FakeX402Client()
    settings = Settings(cmc_api_key="secret", use_keyless_primary=False)
    client = CMCMCPClient(settings, x402_client=fake_x402)  # type: ignore[arg-type]

    result = client.get_crypto_quotes_latest(["CAKE"])

    assert result == {"ok": True}
    assert fake_x402.calls[0]["headers"]["X-CMC-MCP-API-KEY"] == "secret"
    assert "X-CMC_PRO_API_KEY" not in fake_x402.calls[0]["headers"]
    arguments = fake_x402.calls[0]["payload"]["params"]["arguments"]
    assert arguments["id"] == "7186"
    assert arguments["symbol"] == "CAKE"


def test_cmc_mcp_quotes_include_bnb_cmc_id() -> None:
    from src.config.tokens import get_cmc_id_for_mcp

    assert get_cmc_id_for_mcp("BNB") == "1839"

    fake_x402 = FakeX402Client()
    client = CMCMCPClient(Settings(use_keyless_primary=False), x402_client=fake_x402)  # type: ignore[arg-type]

    client.get_crypto_quotes_latest(["BNB"])

    arguments = fake_x402.calls[0]["payload"]["params"]["arguments"]
    assert arguments["id"] == "1839"
    assert arguments["symbol"] == "BNB"


def test_market_metrics_uses_documented_tool_name() -> None:
    fake_x402 = FakeX402Client()
    settings = Settings(use_keyless_primary=False)
    client = CMCMCPClient(settings, x402_client=fake_x402)  # type: ignore[arg-type]

    client.get_crypto_market_metrics(["CAKE"])

    params = fake_x402.calls[0]["payload"]["params"]
    assert params["name"] == "get_crypto_market_metrics"


def test_normalize_keyless_list_response() -> None:
    payload = {
        "data": [
            {
                "symbol": "CAKE",
                "quote": {"USD": {"price": 2.5, "volume_24h": 1000.0, "percent_change_1h": 0.5}},
            }
        ]
    }
    normalized = CMCMCPClient._normalize_keyless_quotes_payload(payload)
    assert normalized["data"]["CAKE"]["price"] == 2.5
    assert normalized["data"]["CAKE"]["volume_24h"] == 1000.0


def test_keyless_primary_uses_symbol_query(monkeypatch: Any) -> None:
    calls: list[dict[str, Any]] = []

    def fake_get(*args: Any, **kwargs: Any) -> FakeKeylessResponse:
        calls.append({"url": args[0], **kwargs})
        return FakeKeylessResponse()

    monkeypatch.setattr(requests, "get", fake_get)
    client = CMCMCPClient(Settings(use_keyless_primary=True), x402_client=FakeX402Client())  # type: ignore[arg-type]

    result = client.get_crypto_quotes_latest(["CAKE"])

    assert result["data"]["CAKE"]["price"] == 10.0
    assert calls[0]["url"].endswith("/cryptocurrency/quotes/latest")
    assert calls[0]["params"]["symbol"] == "CAKE"
    assert calls[0]["params"]["convert"] == "USD"


def test_x402_failure_uses_keyless_fallback(monkeypatch: Any) -> None:
    calls: list[dict[str, Any]] = []

    def fake_get(*args: Any, **kwargs: Any) -> FakeKeylessResponse:
        calls.append({"url": args[0], **kwargs})
        return FakeKeylessResponse()

    monkeypatch.setattr(requests, "get", fake_get)
    settings = Settings(use_keyless_primary=False)
    client = CMCMCPClient(settings, x402_client=FailingX402Client())  # type: ignore[arg-type]

    result = client.get_crypto_quotes_latest(["CAKE"])

    assert result["data"]["CAKE"]["symbol"] == "CAKE"
    assert calls[0]["params"]["symbol"] == "CAKE"


def test_technical_analysis_returns_empty_without_crashing() -> None:
    client = CMCMCPClient(Settings(use_keyless_primary=True), x402_client=FakeX402Client())  # type: ignore[arg-type]

    assert client.get_crypto_technical_analysis(["CAKE"]) == {}


def test_market_snapshot_skips_remaining_calls_when_quotes_unavailable(monkeypatch: Any) -> None:
    client = CMCMCPClient(Settings(), x402_client=FakeX402Client())  # type: ignore[arg-type]
    called: list[str] = []

    monkeypatch.setattr(client, "get_crypto_quotes_latest", lambda symbols: {})
    monkeypatch.setattr(client, "get_crypto_technical_analysis", lambda symbols: called.append("technicals"))
    monkeypatch.setattr(client, "get_global_crypto_derivatives_metrics", lambda: called.append("derivatives"))
    monkeypatch.setattr(client, "get_crypto_market_metrics", lambda symbols: called.append("metrics"))

    assert client.fetch_market_snapshot(["CAKE"]) == {}
    assert called == []


def test_market_snapshot_exposes_regime_and_three_hour_breakout_fields(monkeypatch: Any) -> None:
    client = CMCMCPClient(Settings(), x402_client=FakeX402Client())  # type: ignore[arg-type]

    monkeypatch.setattr(
        client,
        "get_crypto_quotes_latest",
        lambda symbols: {
            "data": {
                "CAKE": {
                    "symbol": "CAKE",
                    "quote": {
                        "USD": {
                            "price": 2.0,
                            "volume_24h": 10_000_000.0,
                            "market_cap": 100_000_000.0,
                            "percent_change_1h": 0.4,
                            "percent_change_24h": -2.5,
                        }
                    },
                },
                "WBNB": {
                    "symbol": "WBNB",
                    "quote": {"USD": {"percent_change_1h": -0.5}},
                },
            }
        },
    )
    monkeypatch.setattr(client, "get_crypto_technical_analysis", lambda symbols: {"data": {}})
    monkeypatch.setattr(client, "get_global_crypto_derivatives_metrics", lambda: {})
    monkeypatch.setattr(
        client,
        "get_crypto_market_metrics",
        lambda symbols: {"data": {"CAKE": {"symbol": "CAKE", "high_3h": 2.05}}},
    )

    snapshot = client.fetch_market_snapshot(["CAKE"])

    assert snapshot["CAKE"]["percent_change_1h"] == 0.4
    assert snapshot["CAKE"]["percent_change_24h"] == -2.5
    assert snapshot["CAKE"]["high_3h"] == 2.05
    assert snapshot["CAKE"]["bnb_1h_trend_pct"] == -0.5

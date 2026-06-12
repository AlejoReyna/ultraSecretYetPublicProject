"""Tests for CoinMarketCap MCP request envelopes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import requests

from src.config.settings import Settings
from src.data.cmc_mcp_client import CMCMCPClient, CmcMcpClient


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


@pytest.fixture(autouse=True)
def isolate_x402_spend_ledger(monkeypatch: Any, tmp_path: Path) -> None:
    """Keep unit tests from inheriting a live spend ledger in the repo cwd."""

    monkeypatch.chdir(tmp_path)


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


def test_extract_tool_result_parses_x402_quote_list() -> None:
    payload = {
        "result": {
            "content": [
                {
                    "type": "text",
                    "text": '[{"symbol":"BNB","price":589.0,"volume_24h":1000000.0,"market_cap":79000000000.0}]',
                }
            ]
        }
    }
    parsed = CmcMcpClient._extract_tool_result(payload)
    assert parsed["data"]["BNB"]["price"] == 589.0


def test_x402_primary_call_parses_mcp_content() -> None:
    class PaidX402Client:
        def request_with_x402(self, method: str, envelope: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
            return {
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": '[{"symbol":"CAKE","price":2.5,"volume_24h":5000000.0,"market_cap":800000000.0}]',
                        }
                    ]
                }
            }

    client = CMCMCPClient(Settings(use_keyless_primary=False), x402_client=PaidX402Client())  # type: ignore[arg-type]
    result = client.get_crypto_quotes_latest(["CAKE"])
    assert result["data"]["CAKE"]["price"] == 2.5


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
    assert "X-CMC_PRO_API_KEY" not in calls[0]["headers"]


def test_keyless_primary_sends_cmc_pro_api_key_when_configured(monkeypatch: Any) -> None:
    calls: list[dict[str, Any]] = []

    def fake_get(*args: Any, **kwargs: Any) -> FakeKeylessResponse:
        calls.append({"url": args[0], **kwargs})
        return FakeKeylessResponse()

    monkeypatch.setattr(requests, "get", fake_get)
    client = CMCMCPClient(
        Settings(use_keyless_primary=True, cmc_api_key="secret"),
        x402_client=FakeX402Client(),  # type: ignore[arg-type]
    )

    client.get_crypto_quotes_latest(["CAKE"])

    assert calls[0]["headers"]["X-CMC_PRO_API_KEY"] == "secret"


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
    client = CMCMCPClient(Settings(use_keyless_primary=True), x402_client=FakeX402Client())  # type: ignore[arg-type]

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


def test_fetch_keyless_quotes_snapshot_without_api_key(monkeypatch: Any) -> None:
    calls: list[dict[str, Any]] = []

    def fake_get(*args: Any, **kwargs: Any) -> FakeKeylessResponse:
        calls.append({"url": args[0], **kwargs})
        return FakeKeylessResponse()

    monkeypatch.setattr(requests, "get", fake_get)
    client = CMCMCPClient(Settings(use_keyless_primary=False), x402_client=FakeX402Client())  # type: ignore[arg-type]

    snapshot = client.fetch_keyless_quotes_snapshot(["CAKE"])

    assert snapshot["CAKE"]["price"] == 10.0
    assert snapshot["CAKE"]["estimated_slippage_pct"] == 0.001
    # CAKE has a pinned UCID, so the id-preferred keyless path queries by id
    # (ticker lookups can resolve to knockoff listings).
    assert calls[0]["params"]["id"] == "7186"
    assert "X-CMC_PRO_API_KEY" not in calls[0]["headers"]


def test_fetch_x402_enriched_snapshot_merges_x402_quotes_and_keyless_metrics(monkeypatch: Any) -> None:
    class PaidX402Client:
        def request_with_x402(self, method: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
            return {
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                '{"data":{"CAKE":{"symbol":"CAKE","price":2.5,"volume_24h":5000000.0,'
                                '"market_cap":800000000.0,"percent_change_1h":0.3}}}'
                            ),
                        }
                    ]
                }
            }

    def fake_get(*args: Any, **kwargs: Any) -> FakeKeylessResponse:
        url = str(args[0])
        if url.endswith("/global-metrics/quotes/latest"):
            return FakeKeylessResponse({"data": {"funding_rate_avg": 0.0002}})
        return FakeKeylessResponse(
            {
                "data": {
                    "CAKE": {
                        "symbol": "CAKE",
                        "quote": {
                            "USD": {
                                "price": 2.5,
                                "high_3h": 2.55,
                                "estimated_slippage_pct": 0.0015,
                            }
                        },
                    }
                }
            }
        )

    monkeypatch.setattr(requests, "get", fake_get)
    client = CMCMCPClient(Settings(use_keyless_primary=False), x402_client=PaidX402Client())  # type: ignore[arg-type]

    snapshot = client.fetch_x402_enriched_snapshot(["CAKE"])

    assert snapshot["CAKE"]["price"] == 2.5
    assert snapshot["CAKE"]["high_3h"] == 2.55
    assert snapshot["CAKE"]["estimated_slippage_pct"] == 0.0015
    assert snapshot["CAKE"]["percent_change_1h"] == 0.3


def test_fetch_x402_enriched_snapshot_paid_calls_are_id_only(monkeypatch: Any) -> None:
    """Probed June 12: the paid tool rejects symbol-only requests after
    settling payment, and id-only responses arrive columnar (headers+rows).
    Pinned UCIDs and keyless-harvested id_overrides merge into one id-only
    call; symbols with no known id are skipped on the paid layer."""

    class PaidX402Client:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def request_with_x402(self, method: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
            arguments = payload["params"]["arguments"]
            self.calls.append(arguments)
            assert "symbol" not in arguments, "paid path must never send symbol args"
            return {
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                '{"headers":["id","symbol","price","volume_24h"],'
                                '"rows":[["7186","CAKE",2.5,5000000.0],["12345","AB",1.0,1000000.0]]}'
                            ),
                        }
                    ],
                    "isError": False,
                }
            }

    def fake_get(*args: Any, **kwargs: Any) -> FakeKeylessResponse:
        return FakeKeylessResponse({"data": {}})

    paid = PaidX402Client()
    monkeypatch.setattr(requests, "get", fake_get)
    client = CMCMCPClient(Settings(use_keyless_primary=False), x402_client=paid)  # type: ignore[arg-type]

    # CAKE pinned (7186); AB unpinned but harvested from keyless; ZZZ unknown -> skipped
    snapshot = client.fetch_x402_enriched_snapshot(
        ["CAKE", "AB", "ZZZ"],
        id_overrides={"AB": "12345"},
    )

    assert len(paid.calls) == 1
    requested_ids = set(paid.calls[0]["id"].split(","))
    assert requested_ids == {"7186", "12345"}
    assert snapshot["AB"]["price"] == 1.0
    assert snapshot["CAKE"]["price"] == 2.5
    assert "ZZZ" not in snapshot


def test_fetch_x402_enriched_snapshot_returns_empty_when_budget_blocks_payment(monkeypatch: Any) -> None:
    class PaidX402Client:
        def __init__(self) -> None:
            self.calls = 0

        def request_with_x402(self, method: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
            self.calls += 1
            return {"result": {"ok": True}}

    paid = PaidX402Client()
    client = CMCMCPClient(
        Settings(
            use_keyless_primary=False,
            x402_daily_budget_usdc=0.001,
            x402_total_budget_usdc=0.001,
            cmc_x402_amount=0.01,
        ),
        x402_client=paid,  # type: ignore[arg-type]
    )

    snapshot = client.fetch_x402_enriched_snapshot(["CAKE"])

    assert snapshot == {}
    assert paid.calls == 0


def test_build_enriched_snapshot_skips_zero_market_cap_and_volume() -> None:
    combined = [
        {"market_cap": 0, "volume_24h": 0, "price": 0},
        {"market_cap": 800_000_000.0, "volume_24h": 5_000_000.0, "price": 2.5},
    ]

    market_cap = CMCMCPClient._first_number_from_many(
        combined,
        ("market_cap", "quote.USD.market_cap"),
        skip_zero=True,
    )
    volume_24h = CMCMCPClient._first_number_from_many(
        combined,
        ("volume_24h", "volume_24h_usd"),
        skip_zero=True,
    )
    price = CMCMCPClient._first_number_from_many(
        combined,
        ("price", "last_price", "quote.USD.price"),
        skip_zero=True,
    )

    assert market_cap == 800_000_000.0
    assert volume_24h == 5_000_000.0
    assert price == 2.5


def test_fetch_x402_enriched_snapshot_estimates_slippage_when_missing(monkeypatch: Any) -> None:
    class PaidX402Client:
        def request_with_x402(self, method: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
            return {
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                '{"data":{"CAKE":{"symbol":"CAKE","price":2.5,"volume_24h":5000000.0,'
                                '"market_cap":800000000.0,"percent_change_1h":0.3}}}'
                            ),
                        }
                    ]
                }
            }

    def fake_get(*args: Any, **kwargs: Any) -> FakeKeylessResponse:
        url = str(args[0])
        if url.endswith("/global-metrics/quotes/latest"):
            return FakeKeylessResponse({"data": {"funding_rate_avg": 0.0002}})
        return FakeKeylessResponse(
            {
                "data": {
                    "CAKE": {
                        "symbol": "CAKE",
                        "quote": {"USD": {"price": 2.5, "high_3h": 2.55}},
                    }
                }
            }
        )

    monkeypatch.setattr(requests, "get", fake_get)
    client = CMCMCPClient(Settings(use_keyless_primary=False), x402_client=PaidX402Client())  # type: ignore[arg-type]

    snapshot = client.fetch_x402_enriched_snapshot(["CAKE"])

    assert snapshot["CAKE"]["estimated_slippage_pct"] == 0.005

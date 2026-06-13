"""CoinMarketCap MCP client using the verified CMC tool names."""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any

import httpx
import requests

from src.config.settings import Settings
from src.config.tokens import TARGET_SYMBOL_BY_KEY, resolve_cmc_coin_id
from src.data.x402_client import X402Client
from src.data.x402_payment import write_402_response

LOGGER = logging.getLogger(__name__)
CMC_MCP_DEFAULT_URL = "https://mcp.coinmarketcap.com/x402/mcp"
MCP_PROTOCOL_VERSION = "2025-03-26"


class CmcMcpError(RuntimeError):
    """Raised when the optional CMC MCP adapter cannot return usable data."""


class CmcMcpClient:
    """Async Streamable HTTP MCP client for CoinMarketCap Agent Hub."""

    def __init__(
        self,
        enabled: bool = False,
        shadow_mode: bool = True,
        url: str = CMC_MCP_DEFAULT_URL,
        timeout_s: float = 15.0,
        signer: Any | None = None,
    ) -> None:
        self.enabled = enabled
        self.shadow_mode = shadow_mode
        self.url = url
        self.timeout_s = timeout_s
        if signer is not None:
            LOGGER.warning("CmcMcpClient ignores direct x402 signers; use X402Client for paid requests")
        self._session_id: str | None = None

    async def initialize(self, client: httpx.AsyncClient) -> dict[str, Any]:
        """POST the MCP initialize request."""

        self._ensure_enabled()
        return await self._post_mcp(
            client,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {
                        "name": "bnb-hack-autonomous-trading-agent",
                        "version": "0.1.0",
                    },
                },
            },
        )

    async def list_tools(self, client: httpx.AsyncClient | None = None) -> list[dict[str, Any]]:
        """Initialize the MCP session and return available tools."""

        self._ensure_enabled()
        if client is not None:
            payload = await self._list_tools(client)
            return self._extract_tools(payload)
        async with httpx.AsyncClient(timeout=self.timeout_s) as owned_client:
            await self.initialize(owned_client)
            payload = await self._list_tools(owned_client)
            return self._extract_tools(payload)

    async def get_crypto_quotes_latest(
        self,
        symbols: list[str],
        convert: str = "USD",
    ) -> dict[str, Any]:
        """Call CMC MCP get_crypto_quotes_latest for the requested symbols."""

        self._ensure_enabled()
        symbol_arg = ",".join(symbol.upper() for symbol in symbols if symbol)
        if not symbol_arg:
            raise CmcMcpError("get_crypto_quotes_latest requires at least one symbol")
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            await self.initialize(client)
            payload = await self._post_mcp(
                client,
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {
                        "name": "get_crypto_quotes_latest",
                        "arguments": {
                            "symbol": symbol_arg,
                            "convert": convert,
                        },
                    },
                },
            )
        return self._extract_tool_result(payload)

    async def _list_tools(self, client: httpx.AsyncClient) -> dict[str, Any]:
        return await self._post_mcp(
            client,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
                "params": {},
            },
        )

    async def _post_mcp(self, client: httpx.AsyncClient, payload: dict[str, Any]) -> dict[str, Any]:
        headers = self._headers()
        try:
            response = await client.post(self.url, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            raise CmcMcpError(f"CMC MCP request failed: {exc}") from exc
        self._capture_session_id(response)
        if response.status_code == 402:
            artifact_path = write_402_response(response)
            raise CmcMcpError(
                "CMC MCP requires x402 payment; use X402Client (official SDK + CDP settlement). "
                f"Saved 402 details to {artifact_path}"
            )

        if response.status_code < 200 or response.status_code >= 300:
            raise CmcMcpError(f"CMC MCP HTTP {response.status_code}: {_short_response_text(response)}")
        return self._parse_response(response)

    def _headers(self) -> dict[str, str]:
        headers = {
            "content-type": "application/json",
            "accept": "application/json, text/event-stream",
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        return headers

    def _capture_session_id(self, response: httpx.Response) -> None:
        session_id = response.headers.get("mcp-session-id")
        if session_id:
            self._session_id = session_id

    def _parse_response(self, response: httpx.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError:
            payload = self._parse_sse_json(response.text)
        if not isinstance(payload, dict):
            raise CmcMcpError("CMC MCP returned non-object JSON")
        if "error" in payload:
            raise CmcMcpError(f"CMC MCP JSON-RPC error: {payload['error']}")
        return payload

    def _parse_sse_json(self, text: str) -> dict[str, Any]:
        for block in text.split("\n\n"):
            data_lines = []
            for line in block.splitlines():
                if line.startswith("data:"):
                    data_lines.append(line.removeprefix("data:").strip())
            if not data_lines:
                continue
            data = "\n".join(data_lines)
            if data == "[DONE]":
                continue
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise CmcMcpError("CMC MCP response was neither JSON nor simple SSE JSON") from exc
        if not isinstance(payload, dict):
            raise CmcMcpError("CMC MCP SSE payload was not a JSON object")
        return payload

    @staticmethod
    def _extract_tools(payload: dict[str, Any]) -> list[dict[str, Any]]:
        result = payload.get("result", payload)
        tools = result.get("tools") if isinstance(result, dict) else result
        if not isinstance(tools, list):
            raise CmcMcpError("CMC MCP tools/list response did not include a tools list")
        return [tool for tool in tools if isinstance(tool, dict)]

    @staticmethod
    def _extract_tool_result(payload: dict[str, Any]) -> dict[str, Any]:
        result = payload.get("result", payload)
        if not isinstance(result, dict):
            return {"result": result}
        content = result.get("content")
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                    continue
                try:
                    parsed = json.loads(item["text"])
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    # Paid id-only responses arrive columnar:
                    # {"headers": [...], "rows": [[...], ...]} (probed June 12).
                    headers = parsed.get("headers")
                    rows = parsed.get("rows")
                    if isinstance(headers, list) and isinstance(rows, list):
                        by_symbol: dict[str, Any] = {}
                        for row in rows:
                            if not isinstance(row, list):
                                continue
                            record = dict(zip(headers, row))
                            symbol = str(record.get("symbol") or "").strip().upper()
                            if symbol:
                                by_symbol[symbol] = record
                        if by_symbol:
                            return {"data": by_symbol}
                    return parsed
                if isinstance(parsed, list):
                    by_symbol: dict[str, Any] = {}
                    for item in parsed:
                        if not isinstance(item, dict):
                            continue
                        symbol = str(item.get("symbol") or "").strip().upper()
                        if symbol:
                            by_symbol[symbol] = item
                    if by_symbol:
                        return {"data": by_symbol}
        return result

    def _ensure_enabled(self) -> None:
        if not self.enabled:
            raise CmcMcpError("CMC MCP client is disabled")


def _short_response_text(response: httpx.Response, limit: int = 500) -> str:
    text = response.text.replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


class CMCMCPClient:
    """Small JSON-RPC/MCP client for CoinMarketCap AI Agent Hub data."""

    def __init__(self, settings: Settings, x402_client: X402Client | None = None) -> None:
        self.settings = settings
        self.endpoint = settings.cmc_x402_endpoint
        self.x402_client = x402_client or X402Client(
            endpoint=settings.cmc_x402_endpoint,
            default_amount=str(settings.cmc_x402_amount),
            default_asset=settings.cmc_x402_asset,
            chain_id=settings.cmc_x402_chain_id,
        )
        from src.data.x402_spend_governor import X402SpendGovernor

        self.spend_governor = X402SpendGovernor(
            daily_budget_usdc=getattr(settings, "x402_daily_budget_usdc", 2.0),
            total_budget_usdc=getattr(settings, "x402_total_budget_usdc", 15.0),
            cost_per_call_usdc=settings.cmc_x402_amount,
            failure_cooldown_seconds=getattr(settings, "x402_failure_cooldown_seconds", 900),
        )

    def get_crypto_quotes_latest(self, symbols: list[str]) -> dict[str, Any]:
        """Fetch latest quotes; Keyless API is primary when ``use_keyless_primary`` is set."""

        return self._call_tool(
            "get_crypto_quotes_latest",
            {
                "id": self._symbols_to_id_arg(symbols),
                "symbol": self._symbols_to_symbol_arg(symbols),
            },
        )

    def get_crypto_technical_analysis(self, symbols: list[str]) -> dict[str, Any]:
        """Call CMC MCP get_crypto_technical_analysis."""

        return self._call_tool("get_crypto_technical_analysis", {"id": self._symbols_to_id_arg(symbols)})

    def get_global_crypto_derivatives_metrics(self) -> dict[str, Any]:
        """Call CMC MCP get_global_crypto_derivatives_metrics."""

        return self._call_tool("get_global_crypto_derivatives_metrics", {})

    def get_crypto_market_metrics(self, symbols: list[str]) -> dict[str, Any]:
        """Fetch market metrics; Keyless quotes endpoint when primary mode is on."""

        return self._call_tool(
            "get_crypto_market_metrics",
            {
                "id": self._symbols_to_id_arg(symbols),
                "symbol": self._symbols_to_symbol_arg(symbols),
            },
        )

    def fetch_market_snapshot(self, symbols: list[str]) -> dict[str, Any]:
        """Fetch and normalize the combined market snapshot for strategy evaluation."""

        normalized_symbols = self._normalize_target_symbols(symbols)
        if not normalized_symbols:
            return {}

        quotes = self._fetch_combined_payload(
            normalized_symbols,
            self.get_crypto_quotes_latest,
        )
        if not quotes:
            LOGGER.warning("CMC quotes unavailable; skipping remaining market snapshot calls")
            return {}

        if not self.settings.use_keyless_primary:
            if self.settings.use_dual_market_data:
                return self.fetch_x402_enriched_snapshot(normalized_symbols)
            return self._snapshot_from_quotes(
                normalized_symbols, quotes, source="x402 (paid, keyless on fallback)"
            )

        technicals = self._fetch_combined_payload(
            normalized_symbols,
            self.get_crypto_technical_analysis,
        )
        derivatives = self.get_global_crypto_derivatives_metrics()
        market_metrics = self._fetch_combined_payload(
            normalized_symbols,
            self.get_crypto_market_metrics,
        )
        return self._build_enriched_snapshot(
            normalized_symbols,
            quotes,
            technicals,
            market_metrics,
            derivatives,
        )

    def fetch_keyless_quotes_snapshot(self, symbols: list[str]) -> dict[str, Any]:
        """Fetch trial REST quotes only (no x402 payment, no API key required)."""

        normalized_symbols = self._normalize_target_symbols(symbols)
        if not normalized_symbols:
            return {}

        quotes = self._fetch_combined_payload(
            normalized_symbols,
            self._fetch_keyless_quotes_batch,
        )
        if not quotes:
            LOGGER.warning("Keyless quotes unavailable")
            return {}
        return self._snapshot_from_quotes(normalized_symbols, quotes)

    def fetch_x402_enriched_snapshot(
        self,
        symbols: list[str],
        id_overrides: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Fetch paid x402 quotes plus free keyless derivatives and market metrics.

        ``id_overrides`` maps SYMBOL -> CMC id, typically harvested from the
        fresh keyless snapshot, so unpinned symbols can still be queried by id
        (the paid MCP tool rejects symbol-only requests: "id: Required").
        """

        normalized_symbols = self._normalize_target_symbols(symbols)
        if not normalized_symbols:
            return {}

        quotes = self._fetch_x402_quotes_id_preferred(normalized_symbols, id_overrides)
        if not quotes:
            LOGGER.warning("x402 quotes unavailable; skipping enriched snapshot")
            return {}

        derivatives = self._fetch_keyless("get_global_crypto_derivatives_metrics", {})
        market_metrics = self._fetch_combined_payload(
            normalized_symbols,
            self._fetch_keyless_market_metrics_batch,
        )
        return self._build_enriched_snapshot(
            normalized_symbols,
            quotes,
            {},
            market_metrics,
            derivatives,
        )

    def _fetch_keyless_quotes_batch(self, symbols: list[str]) -> dict[str, Any]:
        return self._fetch_keyless_id_preferred("get_crypto_quotes_latest", symbols)

    def _fetch_keyless_market_metrics_batch(self, symbols: list[str]) -> dict[str, Any]:
        return self._fetch_keyless_id_preferred("get_crypto_market_metrics", symbols)

    def _fetch_keyless_id_preferred(self, tool_name: str, symbols: list[str]) -> dict[str, Any]:
        """Fetch keyless quotes by CMC id when pinned, by ticker otherwise.

        Ticker lookups are ambiguous: CMC resolves shared tickers (DOGE, UNI,
        TWT, ...) to arbitrary listings, sometimes dead knockoffs with null
        quotes. Pinned ids in CMC_IDS_BY_SYMBOL always win over ticker results.
        """

        with_id = [s for s in symbols if resolve_cmc_coin_id(s)]
        without_id = [s for s in symbols if not resolve_cmc_coin_id(s)]
        merged: dict[str, Any] = {}
        if without_id:
            payload = self._fetch_keyless(
                tool_name, {"symbol": self._symbols_to_symbol_arg(without_id)}
            )
            data = payload.get("data") if isinstance(payload, dict) else None
            if isinstance(data, dict):
                merged.update(data)
        if with_id:
            payload = self._fetch_keyless(tool_name, {"id": self._symbols_to_id_arg(with_id)})
            data = payload.get("data") if isinstance(payload, dict) else None
            if isinstance(data, dict):
                merged.update(data)  # id-based results override ticker results
        return {"data": merged}

    def _fetch_x402_quotes_id_preferred(
        self,
        symbols: list[str],
        id_overrides: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Fetch paid x402 quotes, always by CMC id.

        Probed June 12: the paid MCP tool REJECTS symbol-only requests
        ("id: Required parameter is missing") while still settling the
        payment, so ticker-based paid calls are never made. Id resolution
        order: pinned UCID (CMC_IDS_BY_SYMBOL) > id harvested from the fresh
        keyless snapshot (id_overrides). Symbols with no known id are skipped
        on the paid layer — the free keyless layer still covers them.
        """

        overrides = {k.upper(): str(v) for k, v in (id_overrides or {}).items()}
        resolved: dict[str, str] = {}
        skipped: list[str] = []
        for symbol in symbols:
            key = symbol.upper()
            cmc_id = resolve_cmc_coin_id(key) or overrides.get(key)
            if cmc_id:
                resolved[key] = str(cmc_id)
            else:
                skipped.append(key)
        if skipped:
            LOGGER.debug(
                "Paid x402 skipping %d symbols with no known CMC id: %s",
                len(skipped),
                ",".join(sorted(skipped)),
            )
        if not resolved:
            return {}

        def _fetch_batch(batch: list[str]) -> dict[str, Any]:
            ids = list(dict.fromkeys(resolved[s] for s in batch))
            return self._call_tool_x402("get_crypto_quotes_latest", {"id": ",".join(ids)})

        payload = self._fetch_combined_payload(list(resolved), _fetch_batch)
        return self._by_symbol(payload)

    def _build_enriched_snapshot(
        self,
        normalized_symbols: list[str],
        quotes: dict[str, Any],
        technicals: dict[str, Any],
        market_metrics: dict[str, Any],
        derivatives: dict[str, Any],
    ) -> dict[str, Any]:
        quotes_by_symbol = self._by_symbol(quotes)
        technicals_by_symbol = self._by_symbol(technicals)
        metrics_by_symbol = self._by_symbol(market_metrics)
        macro_context = self._macro_context_from_global_metrics(derivatives)
        bnb_trend = self._first_number(
            quotes_by_symbol.get("BNB", quotes_by_symbol.get("WBNB", {})),
            ("bnb_1h_trend_pct", "percent_change_1h", "price_change_percentage_1h", "change_1h"),
        )

        snapshot: dict[str, Any] = {}
        for symbol in normalized_symbols:
            quote_data = quotes_by_symbol.get(symbol, {})
            technical_data = technicals_by_symbol.get(symbol, {})
            metric_data = metrics_by_symbol.get(symbol, {})
            combined = [quote_data, technical_data, metric_data, derivatives]
            volume_24h = self._first_number_from_many(
                combined,
                ("volume_24h", "volume_24h_usd"),
                skip_zero=True,
            )
            market_cap = self._first_number_from_many(
                combined,
                ("market_cap", "quote.USD.market_cap"),
                skip_zero=True,
            )
            snapshot[symbol] = {
                "symbol": symbol,
                "price": self._first_number_from_many(
                    combined,
                    ("price", "last_price", "quote.USD.price"),
                    skip_zero=True,
                ),
                "market_cap": market_cap,
                "volume_1h": self._first_number_from_many(combined, ("volume_1h", "volume_1h_usd")),
                "volume_24h": volume_24h,
                "percent_change_1h": self._first_number_from_many(
                    combined,
                    ("percent_change_1h", "quote.USD.percent_change_1h", "price_change_percentage_1h", "change_1h"),
                ),
                "percent_change_6h": self._first_number_from_many(
                    combined,
                    ("percent_change_6h", "quote.USD.percent_change_6h", "price_change_percentage_6h", "change_6h"),
                ),
                "percent_change_24h": self._first_number_from_many(
                    combined,
                    (
                        "percent_change_24h",
                        "quote.USD.percent_change_24h",
                        "price_change_percentage_24h",
                        "change_24h",
                    ),
                ),
                "rolling_24h_hourly_volume_avg": self._first_number_from_many(
                    combined,
                    ("rolling_24h_hourly_volume_avg", "avg_hourly_volume_24h"),
                    default=volume_24h / 24 if volume_24h else None,
                ),
                "high_3h": self._first_number_from_many(combined, ("high_3h", "high_3h_price")),
                "high_6h": self._first_number_from_many(combined, ("high_6h", "high_6h_price")),
                "high_24h": self._first_number_from_many(combined, ("high_24h", "high_24h_price")),
                "low_24h": self._first_number_from_many(combined, ("low_24h", "low_24h_price")),
                "bnb_1h_trend_pct": bnb_trend,
                "rsi": self._first_number_from_many(combined, ("rsi", "rsi_14", "technical.rsi")),
                "macd": self._first_number_from_many(combined, ("macd", "technical.macd")),
                "estimated_slippage_pct": self._resolve_estimated_slippage_pct(
                    combined=combined,
                    volume_24h=volume_24h,
                    market_cap=market_cap,
                ),
                "funding_rate": self._first_number_from_many(
                    combined,
                    ("funding_rate", "avg_funding_rate", "funding"),
                ),
                "open_interest_change_pct": self._first_number_from_many(
                    combined,
                    ("open_interest_change_pct", "oi_change_pct", "open_interest_24h_change_pct"),
                ),
                **macro_context,
            }
        LOGGER.info("Built enriched x402 snapshot for %d symbols", len(snapshot))
        return snapshot

    def _snapshot_from_quotes(
        self,
        normalized_symbols: list[str],
        quotes: dict[str, Any],
        source: str = "keyless (free, $0.00)",
    ) -> dict[str, Any]:
        """Build a strategy snapshot from quote payloads only (no enrichment)."""

        quotes_by_symbol = self._by_symbol(quotes)
        bnb_trend = self._first_number(
            quotes_by_symbol.get("BNB", quotes_by_symbol.get("WBNB", {})),
            ("bnb_1h_trend_pct", "percent_change_1h", "price_change_percentage_1h", "change_1h"),
        )
        snapshot: dict[str, Any] = {}
        for symbol in normalized_symbols:
            quote_data = quotes_by_symbol.get(symbol, {})
            volume_24h = self._first_number_from_many(
                [quote_data],
                ("volume_24h", "volume_24h_usd"),
                skip_zero=True,
            )
            market_cap = self._first_number_from_many(
                [quote_data],
                ("market_cap", "quote.USD.market_cap"),
                skip_zero=True,
            )
            estimated_slippage_pct = self._resolve_estimated_slippage_pct(
                combined=[quote_data],
                volume_24h=volume_24h,
                market_cap=market_cap,
            )
            snapshot[symbol] = {
                "symbol": symbol,
                "price": self._first_number_from_many(
                    [quote_data],
                    ("price", "last_price", "quote.USD.price"),
                    skip_zero=True,
                ),
                "market_cap": market_cap,
                "volume_1h": self._first_number_from_many([quote_data], ("volume_1h", "volume_1h_usd")),
                "volume_24h": volume_24h,
                "percent_change_1h": self._first_number_from_many(
                    [quote_data],
                    ("percent_change_1h", "quote.USD.percent_change_1h", "price_change_percentage_1h", "change_1h"),
                ),
                "percent_change_6h": self._first_number_from_many(
                    [quote_data],
                    ("percent_change_6h", "quote.USD.percent_change_6h", "price_change_percentage_6h", "change_6h"),
                ),
                "percent_change_24h": self._first_number_from_many(
                    [quote_data],
                    (
                        "percent_change_24h",
                        "quote.USD.percent_change_24h",
                        "price_change_percentage_24h",
                        "change_24h",
                    ),
                ),
                "rolling_24h_hourly_volume_avg": volume_24h / 24 if volume_24h else None,
                "high_3h": self._first_number_from_many([quote_data], ("high_3h", "high_3h_price")),
                "high_6h": self._first_number_from_many([quote_data], ("high_6h", "high_6h_price")),
                "high_24h": self._first_number_from_many([quote_data], ("high_24h", "high_24h_price")),
                "low_24h": self._first_number_from_many([quote_data], ("low_24h", "low_24h_price")),
                "bnb_1h_trend_pct": bnb_trend,
                "estimated_slippage_pct": estimated_slippage_pct,
            }
        LOGGER.info("Built %s quotes-only snapshot for %d symbols", source, len(snapshot))
        return snapshot

    def _call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        envelope = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            },
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": "2024-11-05",
        }
        if self.settings.cmc_api_key:
            headers["X-CMC-MCP-API-KEY"] = self.settings.cmc_api_key

        if self.settings.use_keyless_primary:
            keyless_data = self._fetch_keyless(tool_name, arguments)
            # SHADOW REMOVED (budget fix 2026-06-13): this block fired an extra
            # PAID x402 call per keyless fetch and discarded the result, bypassing
            # CMC_SNAPSHOT_TTL_SECONDS and burning the daily budget on the 5-min
            # keyless cycle. The legitimate paid path is _call_tool_x402 (TTL-gated).
            return keyless_data

        if not self.spend_governor.allow_call():
            return self._fetch_keyless(tool_name, arguments)

        try:
            payload = self.x402_client.request_with_x402(
                "POST",
                envelope,
                headers=headers,
            )
            if payload is None:
                raise RuntimeError("x402 client returned None")
            self.spend_governor.record_spend()
        except Exception as exc:
            LOGGER.warning("CMC MCP x402 call %s failed: %s", tool_name, exc)
            self.spend_governor.record_failure()
            return self._fetch_keyless(tool_name, arguments)

        if not isinstance(payload, dict):
            LOGGER.warning("CMC MCP call %s returned non-dict JSON; using empty fallback", tool_name)
            return {}
        parsed = CmcMcpClient._extract_tool_result(payload)
        if isinstance(parsed, dict):
            if tool_name in {"get_crypto_quotes_latest", "get_crypto_market_metrics"}:
                parsed = self._normalize_keyless_quotes_payload(parsed)
            return parsed
        LOGGER.warning("CMC MCP call %s returned an unexpected result shape; using empty fallback", tool_name)
        return {}

    def _call_tool_x402(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Always route through x402 payment (never keyless-primary shortcut)."""

        if not self.spend_governor.allow_call():
            return {}

        envelope = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            },
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": "2024-11-05",
        }
        if self.settings.cmc_api_key:
            headers["X-CMC-MCP-API-KEY"] = self.settings.cmc_api_key

        try:
            payload = self.x402_client.request_with_x402(
                "POST",
                envelope,
                headers=headers,
            )
            if payload is None:
                raise RuntimeError("x402 client returned None")
            self.spend_governor.record_spend()
        except Exception as exc:
            LOGGER.warning("CMC MCP x402 call %s failed: %s", tool_name, exc)
            self.spend_governor.record_failure()
            return {}

        if not isinstance(payload, dict):
            LOGGER.warning("CMC MCP x402 call %s returned non-dict JSON; using empty fallback", tool_name)
            return {}
        parsed = CmcMcpClient._extract_tool_result(payload)
        if isinstance(parsed, dict):
            if tool_name in {"get_crypto_quotes_latest", "get_crypto_market_metrics"}:
                parsed = self._normalize_keyless_quotes_payload(parsed)
            return parsed
        LOGGER.warning("CMC MCP x402 call %s returned an unexpected result shape; using empty fallback", tool_name)
        return {}

    @staticmethod
    def _x402_shadow_enabled() -> bool:
        return bool(
            os.getenv("CMC_X402_EPHEMERAL_KEY", "").strip()
            or os.getenv("EVM_PRIVATE_KEY", "").strip()
        )

    def _fetch_keyless(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        # Keyless remains the fail-open data fallback when x402 SDK payment is
        # unavailable or CMC rejects the request shape.
        path: str
        params: dict[str, str]
        if tool_name in {"get_crypto_quotes_latest", "get_crypto_market_metrics"}:
            path = "/cryptocurrency/quotes/latest"
            symbol_arg = str(arguments.get("symbol", "")).strip()
            if symbol_arg:
                params = {"symbol": symbol_arg, "convert": "USD"}
            else:
                params = {"id": str(arguments.get("id", "")), "convert": "USD"}
        elif tool_name == "get_global_crypto_derivatives_metrics":
            path = "/global-metrics/quotes/latest"
            params = {"convert": "USD"}
        else:
            LOGGER.debug(
                "Keyless API has no equivalent for MCP tool %s; returning empty (optional factor)",
                tool_name,
            )
            return {}

        try:
            headers = {"Accept": "application/json"}
            if self.settings.cmc_api_key:
                headers["X-CMC_PRO_API_KEY"] = self.settings.cmc_api_key
            response = requests.get(
                f"{self.settings.cmc_keyless_base_url}{path}",
                params=params,
                headers=headers,
                timeout=15,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            LOGGER.warning("CMC Keyless API call %s failed: %s", tool_name, exc)
            return {}
        if not isinstance(payload, dict):
            return {}
        if tool_name in {"get_crypto_quotes_latest", "get_crypto_market_metrics"}:
            payload = self._normalize_keyless_quotes_payload(payload)
            data = payload.get("data")
            quote_count = len(data) if isinstance(data, dict) else 0
            LOGGER.info("Fetched %d quotes from CMC Keyless API", quote_count)
        return payload

    @classmethod
    def _by_symbol(cls, payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
        items = cls._extract_items(payload)
        by_symbol: dict[str, dict[str, Any]] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("symbol") or item.get("base_symbol") or "").upper()
            if symbol:
                by_symbol[symbol] = item
        if "WBNB" in by_symbol and "BNB" not in by_symbol:
            by_symbol["BNB"] = by_symbol["WBNB"]
        if not by_symbol and isinstance(payload, dict):
            for key, value in payload.items():
                if isinstance(value, dict):
                    symbol = str(value.get("symbol") or key).upper()
                    by_symbol[symbol] = value
        if "WBNB" in by_symbol and "BNB" not in by_symbol:
            by_symbol["BNB"] = by_symbol["WBNB"]
        return by_symbol

    @classmethod
    def _extract_items(cls, payload: Any) -> list[Any]:
        if isinstance(payload, list):
            return payload
        if not isinstance(payload, dict):
            return []
        for key in ("data", "items", "results", "tokens", "quotes"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
            if isinstance(value, dict):
                return list(value.values())
        return []

    def _resolve_estimated_slippage_pct(
        self,
        *,
        combined: list[dict[str, Any]],
        volume_24h: float | None,
        market_cap: float | None,
    ) -> float | None:
        slippage = self._first_number_from_many(combined, ("estimated_slippage_pct", "slippage_pct"))
        if slippage is not None:
            return slippage
        return self._estimate_slippage_from_liquidity(volume_24h, market_cap)

    @staticmethod
    def _estimate_slippage_from_liquidity(
        volume_24h: float | None,
        market_cap: float | None,
    ) -> float | None:
        """Heuristic slippage proxy when CMC keyless/quotes payloads omit slippage."""

        if volume_24h is None or volume_24h <= 0:
            return None

        if market_cap is None or market_cap <= 0:
            if volume_24h >= 10_000_000_000:
                return 0.001
            if volume_24h >= 1_000_000_000:
                return 0.002
            if volume_24h >= 100_000_000:
                return 0.003
            return 0.005

        liquidity_score = volume_24h / market_cap
        if liquidity_score > 0.1:
            return 0.001
        if liquidity_score > 0.01:
            return 0.002
        return 0.005

    @classmethod
    def _macro_context_from_global_metrics(cls, payload: dict[str, Any]) -> dict[str, float]:
        data = payload.get("data") if isinstance(payload, dict) else None
        metrics = data if isinstance(data, dict) else payload
        if not isinstance(metrics, dict):
            return {}
        total_market_cap = cls._first_number(
            metrics,
            (
                "quote.USD.total_market_cap",
                "total_market_cap",
                "total_market_cap_usd",
                "total_market_cap_yesterday",
            ),
            skip_zero=True,
        )
        btc_dominance = cls._first_number(
            metrics,
            ("btc_dominance", "btc_dominance_percentage", "bitcoin_dominance"),
        )
        stablecoin_dominance = cls._first_number(
            metrics,
            ("stablecoin_dominance", "stablecoin_dominance_percentage", "stablecoin_market_cap_dominance"),
        )
        stablecoin_market_cap = cls._first_number(
            metrics,
            ("stablecoin_market_cap", "stablecoin_market_cap_usd"),
            skip_zero=True,
        )
        if stablecoin_dominance is None and total_market_cap and stablecoin_market_cap:
            stablecoin_dominance = stablecoin_market_cap / total_market_cap * 100.0

        context: dict[str, float] = {}
        if total_market_cap is not None:
            context["macro_total_market_cap"] = total_market_cap
        if btc_dominance is not None:
            context["macro_btc_dominance"] = btc_dominance
        if stablecoin_dominance is not None:
            context["macro_stablecoin_dominance"] = stablecoin_dominance
        return context

    @classmethod
    def _first_number_from_many(
        cls,
        payloads: list[dict[str, Any]],
        keys: tuple[str, ...],
        default: float | None = None,
        skip_zero: bool = False,
    ) -> float | None:
        for payload in payloads:
            value = cls._first_number(payload, keys, default=None, skip_zero=skip_zero)
            if value is not None:
                return value
        return default

    @classmethod
    def _first_number(
        cls,
        payload: dict[str, Any],
        keys: tuple[str, ...],
        default: float | None = None,
        skip_zero: bool = False,
    ) -> float | None:
        for key in keys:
            found = cls._read_path(payload, key)
            if found is not None:
                try:
                    val = float(found)
                    if skip_zero and val == 0:
                        continue
                    return val
                except (TypeError, ValueError):
                    continue
        for key in keys:
            found = cls._recursive_lookup(payload, key.split(".")[-1])
            if found is not None:
                try:
                    val = float(found)
                    if skip_zero and val == 0:
                        continue
                    return val
                except (TypeError, ValueError):
                    continue
        return default

    @staticmethod
    def _read_path(payload: dict[str, Any], path: str) -> Any:
        current: Any = payload
        for part in path.split("."):
            if not isinstance(current, dict) or part not in current:
                return None
            current = current[part]
        return current

    @classmethod
    def _recursive_lookup(cls, payload: Any, key: str) -> Any:
        if isinstance(payload, dict):
            if key in payload:
                return payload[key]
            for value in payload.values():
                found = cls._recursive_lookup(value, key)
                if found is not None:
                    return found
        if isinstance(payload, list):
            for value in payload:
                found = cls._recursive_lookup(value, key)
                if found is not None:
                    return found
        return None

    _SNAPSHOT_BATCH_SIZE = 50

    @classmethod
    def _normalize_target_symbols(cls, symbols: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for symbol in symbols:
            key = symbol.strip().upper()
            if key not in TARGET_SYMBOL_BY_KEY or key in seen:
                continue
            seen.add(key)
            normalized.append(key)
        return normalized

    @classmethod
    def _fetch_combined_payload(
        cls,
        symbols: list[str],
        fetcher: Any,
    ) -> dict[str, Any]:
        if not symbols:
            return {}
        if len(symbols) <= cls._SNAPSHOT_BATCH_SIZE:
            try:
                return fetcher(symbols)
            except Exception as exc:
                LOGGER.warning("CMC snapshot fetch failed for %s: %s", ",".join(symbols), exc)
                return {}

        combined: dict[str, Any] = {}
        for start in range(0, len(symbols), cls._SNAPSHOT_BATCH_SIZE):
            batch = symbols[start : start + cls._SNAPSHOT_BATCH_SIZE]
            try:
                payload = fetcher(batch)
            except Exception as exc:
                LOGGER.warning("CMC snapshot batch fetch failed for %s: %s", ",".join(batch), exc)
                continue
            if not payload:
                LOGGER.warning("CMC snapshot batch fetch returned empty for %s", ",".join(batch))
                continue
            batch_by_symbol = cls._by_symbol(payload)
            combined.update(batch_by_symbol)
        return combined

    @staticmethod
    def _symbols_to_id_arg(symbols: list[str]) -> str:
        ids = [cmc_id for symbol in symbols if (cmc_id := resolve_cmc_coin_id(symbol))]
        return ",".join(ids)

    @staticmethod
    def _symbols_to_symbol_arg(symbols: list[str]) -> str:
        return ",".join(symbol.upper() for symbol in symbols)

    @classmethod
    def _normalize_keyless_quotes_payload(cls, payload: dict[str, Any]) -> dict[str, Any]:
        """Flatten ``quote.USD`` fields so strategy code can read price/volume directly."""

        data = payload.get("data")
        items: list[Any] = []
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = list(data.values())
        else:
            return payload

        normalized: dict[str, Any] = {}
        flattened: list[Any] = []
        for item in items:
            # Symbol queries can return a list of every listing sharing the
            # ticker; flatten so each candidate competes individually.
            if isinstance(item, list):
                flattened.extend(item)
            else:
                flattened.append(item)
        for item in flattened:
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("symbol") or "").upper()
            if not symbol:
                continue
            quote_usd = cls._read_path(item, "quote.USD")
            if isinstance(quote_usd, dict):
                flat = {**item, **quote_usd, "symbol": symbol}
            else:
                flat = {**item, "symbol": symbol}
            existing = normalized.get(symbol)
            if existing is not None and cls._quote_rank(existing) >= cls._quote_rank(flat):
                continue  # never let a dead knockoff overwrite a live listing
            normalized[symbol] = flat
        return {"data": normalized}

    @staticmethod
    def _quote_rank(row: dict[str, Any]) -> tuple[int, int, float]:
        """Orderable quality of a quote row: priced > active > bigger market cap."""

        try:
            price = float(row.get("price"))
            priced = 1 if price > 0 else 0
        except (TypeError, ValueError):
            priced = 0
        active = 1 if row.get("is_active") in (1, True, None) else 0
        try:
            market_cap = float(row.get("market_cap") or 0.0)
        except (TypeError, ValueError):
            market_cap = 0.0
        return (priced, active, market_cap)

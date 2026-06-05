"""CoinMarketCap MCP client using the verified CMC tool names."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

import httpx
import requests

from src.config.settings import Settings
from src.config.tokens import TARGET_SYMBOL_BY_KEY, get_cmc_id_optional
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
            LOGGER.warning("CmcMcpClient ignores direct x402 signers; use TWAK-native X402Client for paid requests")
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
                "CMC MCP requires x402 payment; use TWAK-native X402Client for paid requests. "
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
                    return parsed
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

        technicals = self._fetch_combined_payload(
            normalized_symbols,
            self.get_crypto_technical_analysis,
        )
        derivatives = self.get_global_crypto_derivatives_metrics()
        market_metrics = self._fetch_combined_payload(
            normalized_symbols,
            self.get_crypto_market_metrics,
        )

        quotes_by_symbol = self._by_symbol(quotes)
        technicals_by_symbol = self._by_symbol(technicals)
        metrics_by_symbol = self._by_symbol(market_metrics)
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
            volume_24h = self._first_number_from_many(combined, ("volume_24h", "volume_24h_usd"))
            snapshot[symbol] = {
                "symbol": symbol,
                "price": self._first_number_from_many(combined, ("price", "last_price", "quote.USD.price")),
                "market_cap": self._first_number_from_many(combined, ("market_cap", "quote.USD.market_cap")),
                "volume_1h": self._first_number_from_many(combined, ("volume_1h", "volume_1h_usd")),
                "volume_24h": volume_24h,
                "percent_change_1h": self._first_number_from_many(
                    combined,
                    ("percent_change_1h", "quote.USD.percent_change_1h", "price_change_percentage_1h", "change_1h"),
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
                "bnb_1h_trend_pct": bnb_trend,
                "rsi": self._first_number_from_many(combined, ("rsi", "rsi_14", "technical.rsi")),
                "macd": self._first_number_from_many(combined, ("macd", "technical.macd")),
                "estimated_slippage_pct": self._first_number_from_many(
                    combined,
                    ("estimated_slippage_pct", "slippage_pct"),
                ),
                "funding_rate": self._first_number_from_many(
                    combined,
                    ("funding_rate", "avg_funding_rate", "funding"),
                ),
                "open_interest_change_pct": self._first_number_from_many(
                    combined,
                    ("open_interest_change_pct", "oi_change_pct", "open_interest_24h_change_pct"),
                ),
            }
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
            try:
                self.x402_client.request_with_x402("POST", envelope, headers=headers)
            except Exception as exc:
                LOGGER.debug("TWAK x402 shadow request failed: %s", exc)
            return keyless_data

        try:
            payload = self.x402_client.request_with_x402(
                "POST",
                envelope,
                headers=headers,
            )
            if payload is None:
                raise RuntimeError("x402 client returned None")
        except Exception as exc:
            LOGGER.warning(
                "CMC MCP x402 call %s failed; using Keyless trial API: %s",
                tool_name,
                exc,
            )
            payload = self._fetch_keyless(tool_name, arguments)

        if not isinstance(payload, dict):
            LOGGER.warning("CMC MCP call %s returned non-dict JSON; using empty fallback", tool_name)
            return {}
        result = payload.get("result", payload)
        if isinstance(result, dict):
            return result
        LOGGER.warning("CMC MCP call %s returned an unexpected result shape; using empty fallback", tool_name)
        return {}

    def _fetch_keyless(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        # Keyless remains the fail-open data fallback when TWAK-native x402 is
        # unavailable or CMC rejects the generic TWAK request shape.
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
            response = requests.get(
                f"{self.settings.cmc_keyless_base_url}{path}",
                params=params,
                headers={"Accept": "application/json"},
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
        if not by_symbol and isinstance(payload, dict):
            for key, value in payload.items():
                if isinstance(value, dict):
                    symbol = str(value.get("symbol") or key).upper()
                    by_symbol[symbol] = value
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

    @classmethod
    def _first_number_from_many(
        cls,
        payloads: list[dict[str, Any]],
        keys: tuple[str, ...],
        default: float | None = None,
    ) -> float | None:
        for payload in payloads:
            value = cls._first_number(payload, keys, default=None)
            if value is not None:
                return value
        return default

    @classmethod
    def _first_number(
        cls,
        payload: dict[str, Any],
        keys: tuple[str, ...],
        default: float | None = None,
    ) -> float | None:
        for key in keys:
            found = cls._read_path(payload, key)
            if found is not None:
                try:
                    return float(found)
                except (TypeError, ValueError):
                    continue
        for key in keys:
            found = cls._recursive_lookup(payload, key.split(".")[-1])
            if found is not None:
                try:
                    return float(found)
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
            return fetcher(symbols)

        combined: dict[str, Any] = {}
        for start in range(0, len(symbols), cls._SNAPSHOT_BATCH_SIZE):
            batch = symbols[start : start + cls._SNAPSHOT_BATCH_SIZE]
            payload = fetcher(batch)
            if not payload:
                continue
            batch_by_symbol = cls._by_symbol(payload)
            combined.update(batch_by_symbol)
        return combined

    @staticmethod
    def _symbols_to_id_arg(symbols: list[str]) -> str:
        ids = [cmc_id for symbol in symbols if (cmc_id := get_cmc_id_optional(symbol))]
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
        for item in items:
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
            normalized[symbol] = flat
        return {"data": normalized}

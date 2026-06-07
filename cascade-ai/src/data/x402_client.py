"""CoinMarketCap x402 client using the official x402 Python SDK."""

from __future__ import annotations

import json
import logging
import os
from decimal import Decimal, InvalidOperation
from typing import Any

from eth_account import Account
from x402 import max_amount, prefer_network, x402ClientSync
from x402.http.clients import x402_requests
from x402.http.clients.requests import PaymentError
from x402.mechanisms.evm import EthAccountSigner
from x402.mechanisms.evm.exact.register import register_exact_evm_client

LOGGER = logging.getLogger(__name__)

CMC_X402_ENDPOINT = "https://mcp.coinmarketcap.com/x402/mcp"
DEFAULT_PAYMENT_ASSET = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
DEFAULT_PAYMENT_CHAIN = "base"
DEFAULT_PAYMENT_METHOD = "eip3009"
DEFAULT_MAX_PAYMENT_USDC = "0.01"
DEFAULT_CHAIN_ID = 8453
CDP_FACILITATOR_URL = "https://api.cdp.coinbase.com/platform/v2/x402"
MCP_PROTOCOL_VERSION = "2025-03-26"
MCP_HTTP_PROTOCOL_VERSION = "2024-11-05"
MCP_METHODS_REQUIRING_SESSION = frozenset({"tools/call", "tools/list"})


class X402Client:
    """Run CMC x402 requests through the official x402 SDK (sync requests client).

    TWAK 0.17.0 routes paid HTTP through the x402.org facilitator, which rejects
    Base mainnet (``eip155:8453``). This client signs locally with the official SDK
    and preserves CMC MCP headers on retry; CMC settles via CDP on the server side.

    Payment signing uses ``CMC_X402_EPHEMERAL_KEY`` or ``EVM_PRIVATE_KEY`` from the
    environment only (never loaded into ``Settings``). Trading keys remain in TWAK.
    """

    def __init__(
        self,
        endpoint: str = CMC_X402_ENDPOINT,
        timeout_seconds: float = 15.0,
        default_amount: str | None = None,
        default_asset: str = DEFAULT_PAYMENT_ASSET,
        default_chain: str = DEFAULT_PAYMENT_CHAIN,
        default_method: str = DEFAULT_PAYMENT_METHOD,
        chain_id: int = DEFAULT_CHAIN_ID,
        payment_private_key: str | None = None,
        sdk_client: x402ClientSync | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self.default_amount = default_amount or DEFAULT_MAX_PAYMENT_USDC
        self.default_asset = default_asset
        self.default_chain = default_chain
        self.default_method = default_method
        self.chain_id = chain_id
        self._payment_private_key = payment_private_key
        self._sdk_client = sdk_client
        self._mcp_session_id: str | None = None

    def request_with_x402(self, method: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any] | None:
        """Pay and fetch an x402-gated CMC MCP request via the official SDK."""

        try:
            if method.upper() != "POST":
                raise ValueError("CMC MCP x402 client only supports POST requests")
            merged_headers = _merge_mcp_headers(headers)
            rpc_method = str(payload.get("method") or "")
            if rpc_method in MCP_METHODS_REQUIRING_SESSION and not self._mcp_session_id:
                if not self._initialize_mcp_session(merged_headers):
                    return None
            if self._mcp_session_id:
                merged_headers["Mcp-Session-Id"] = self._mcp_session_id

            response = self._post_with_sdk(payload, merged_headers)
            if response is None:
                return None
            self._capture_mcp_session_id(response.headers, response.text)
            if response.status_code < 200 or response.status_code >= 300:
                LOGGER.warning(
                    "x402 SDK request to %s returned HTTP %s: %s",
                    self.endpoint,
                    response.status_code,
                    _short_text(response.text),
                )
                return None
            parsed = _parse_mcp_response(response.text)
            if parsed is None:
                LOGGER.warning("x402 SDK response was not parseable MCP JSON")
                return None
            return parsed
        except PaymentError as exc:
            LOGGER.warning("x402 SDK payment flow failed: %s", exc)
            return None
        except Exception as exc:
            LOGGER.warning("x402 SDK request failed: %s", exc)
            return None

    def _initialize_mcp_session(self, headers: dict[str, str]) -> bool:
        init_payload = {
            "jsonrpc": "2.0",
            "id": "initialize",
            "method": "initialize",
            "params": {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {
                    "name": "cascade-ai-trading-agent",
                    "version": "0.1.0",
                },
            },
        }
        response = self._post_with_sdk(init_payload, dict(headers))
        if response is None:
            return False
        self._capture_mcp_session_id(response.headers, response.text)
        if response.status_code < 200 or response.status_code >= 300:
            LOGGER.warning(
                "CMC MCP initialize returned HTTP %s: %s",
                response.status_code,
                _short_text(response.text),
            )
            return False
        if not self._mcp_session_id:
            LOGGER.warning(
                "CMC MCP initialize succeeded but no Mcp-Session-Id was returned; body=%s",
                _short_text(response.text),
            )
            return False
        if not self._send_initialized_notification(headers):
            LOGGER.warning("CMC MCP notifications/initialized failed")
            return False
        LOGGER.debug("CMC MCP session established")
        return True

    def _send_initialized_notification(self, headers: dict[str, str]) -> bool:
        if not self._mcp_session_id:
            return False
        notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {},
        }
        notify_headers = dict(headers)
        notify_headers["Mcp-Session-Id"] = self._mcp_session_id
        response = self._post_with_sdk(notification, notify_headers)
        if response is None:
            return False
        if response.status_code in {200, 201, 202, 204}:
            return True
        LOGGER.warning(
            "CMC MCP notifications/initialized returned HTTP %s: %s",
            response.status_code,
            _short_text(response.text),
        )
        return False

    def _post_with_sdk(self, payload: dict[str, Any], headers: dict[str, str]) -> Any | None:
        client = self._sdk_client or self._build_sdk_client()
        with x402_requests(client) as session:
            return session.post(
                self.endpoint,
                json=payload,
                headers=headers,
                timeout=self.timeout_seconds,
            )

    def _capture_mcp_session_id(self, headers: Any, body_text: str | None = None) -> None:
        if headers is not None and hasattr(headers, "items"):
            for key, value in headers.items():
                if str(key).lower() == "mcp-session-id" and value:
                    self._mcp_session_id = str(value)
                    return
        if headers is not None:
            getter = getattr(headers, "get", None)
            if callable(getter):
                session_id = getter("mcp-session-id") or getter("Mcp-Session-Id")
                if session_id:
                    self._mcp_session_id = str(session_id)
                    return
        if body_text:
            session_id = _extract_session_id_from_body(body_text)
            if session_id:
                self._mcp_session_id = session_id

    def _build_sdk_client(self) -> x402ClientSync:
        private_key = self._resolve_payment_private_key()
        account = Account.from_key(private_key)
        client = x402ClientSync()
        register_exact_evm_client(client, EthAccountSigner(account))
        network = _network_caip2(self.chain_id, self.default_chain)
        client.register_policy(prefer_network(network))
        client.register_policy(max_amount(int(self._max_payment_atomic())))
        LOGGER.debug(
            "x402 SDK client ready for %s signer=%s facilitator=%s",
            network,
            account.address,
            CDP_FACILITATOR_URL,
        )
        return client

    def _resolve_payment_private_key(self) -> str:
        if self._payment_private_key and self._payment_private_key.strip():
            return _normalize_private_key(self._payment_private_key.strip())
        for env_name in ("CMC_X402_EPHEMERAL_KEY", "EVM_PRIVATE_KEY"):
            value = os.getenv(env_name)
            if value and value.strip():
                return _normalize_private_key(value.strip())
        raise ValueError(
            "x402 payment key missing: set CMC_X402_EPHEMERAL_KEY or EVM_PRIVATE_KEY "
            "(Base mainnet via CDP; TWAK wallet cannot sign EIP-712 for MCP headers)"
        )

    def _max_payment_atomic(self) -> str:
        """Return the max payment cap in atomic token units."""

        amount_text = str(self.default_amount or DEFAULT_MAX_PAYMENT_USDC).strip()
        if amount_text.isdigit():
            return amount_text
        try:
            amount = Decimal(amount_text)
        except (InvalidOperation, ValueError) as exc:
            raise ValueError(f"invalid CMC x402 payment amount: {self.default_amount!r}") from exc
        if amount <= 0:
            raise ValueError("CMC x402 payment amount must be greater than zero")
        decimals = 6 if _is_six_decimal_asset(self.default_asset) else 18
        return str(int(amount * (Decimal(10) ** decimals)))


def _normalize_private_key(key: str) -> str:
    normalized = key.strip()
    if not normalized.startswith("0x"):
        normalized = f"0x{normalized}"
    return normalized


def _merge_mcp_headers(headers: dict[str, str]) -> dict[str, str]:
    merged = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "MCP-Protocol-Version": MCP_HTTP_PROTOCOL_VERSION,
    }
    merged.update(headers)
    api_key = os.getenv("CMC_API_KEY", "").strip()
    if api_key and "X-CMC-MCP-API-KEY" not in merged and "x-cmc-mcp-api-key" not in {k.lower() for k in merged}:
        merged["X-CMC-MCP-API-KEY"] = api_key
    return merged


def _extract_session_id_from_body(body_text: str) -> str | None:
    payload = _parse_mcp_response(body_text)
    if not isinstance(payload, dict):
        return None
    result = payload.get("result")
    if not isinstance(result, dict):
        return None
    for key in ("sessionId", "session_id", "mcpSessionId", "mcp_session_id"):
        value = result.get(key)
        if value:
            return str(value)
    return None


def _network_caip2(chain_id: int, chain_name: str) -> str:
    if chain_name.strip().lower().startswith("eip155:"):
        return chain_name.strip().lower()
    return f"eip155:{chain_id}"


def _is_six_decimal_asset(asset: str) -> bool:
    normalized = asset.strip().lower()
    return normalized in {
        "usdc",
        "usdt",
        DEFAULT_PAYMENT_ASSET.lower(),
    }


def _short_text(text: str, limit: int = 500) -> str:
    compact = text.replace("\n", " ").strip()
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}..."


def _parse_mcp_response(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        payload = _parse_sse_json(stripped)
    return payload if isinstance(payload, dict) else None


def _parse_sse_json(text: str) -> dict[str, Any] | None:
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
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None

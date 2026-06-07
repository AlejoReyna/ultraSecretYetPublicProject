#!/usr/bin/env python3
"""Smoke test a paid CMC MCP quote request through the official x402 SDK."""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.x402_client import CMC_X402_ENDPOINT, DEFAULT_PAYMENT_ASSET, X402Client  # noqa: E402


def _payment_key_configured() -> bool:
    return bool(
        os.getenv("CMC_X402_EPHEMERAL_KEY", "").strip()
        or os.getenv("EVM_PRIVATE_KEY", "").strip()
    )


def run() -> int:
    os.chdir(ROOT)
    load_dotenv(ROOT / ".env")

    if not _payment_key_configured():
        print(
            "x402_paid_quote_skipped=true reason=missing CMC_X402_EPHEMERAL_KEY or EVM_PRIVATE_KEY",
            file=sys.stderr,
        )
        return 2

    cmc_api_key = os.getenv("CMC_API_KEY", "").strip()
    if cmc_api_key:
        print("cmc_api_key=optional configured", file=sys.stderr)
    else:
        print("cmc_api_key=optional not set; x402 USDC payment is the auth", file=sys.stderr)

    client = X402Client(
        endpoint=os.getenv("CMC_MCP_URL", os.getenv("CMC_X402_ENDPOINT", CMC_X402_ENDPOINT)),
        default_amount=os.getenv("CMC_X402_AMOUNT", "0.01"),
        default_asset=os.getenv("CMC_X402_ASSET", DEFAULT_PAYMENT_ASSET),
        chain_id=int(os.getenv("CMC_X402_CHAIN_ID", "8453")),
    )
    envelope = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "tools/call",
        "params": {
            "name": "get_crypto_quotes_latest",
            "arguments": {"symbol": "BNB", "convert": "USD"},
        },
    }
    headers = {"MCP-Protocol-Version": "2024-11-05"}
    if cmc_api_key:
        headers["X-CMC-MCP-API-KEY"] = cmc_api_key

    result = client.request_with_x402("POST", envelope, headers=headers)
    if result is None:
        print("x402_paid_quote_failed=true", file=sys.stderr)
        return 1

    output_path = ROOT / "artifacts" / "x402_sdk_paid_quote_success.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(f"x402_paid_quote_success={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())

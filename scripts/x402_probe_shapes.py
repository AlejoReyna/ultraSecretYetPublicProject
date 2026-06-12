"""Probe paid CMC MCP request shapes: id-only vs symbol-only (2 x $0.01).

Dumps raw responses to artifacts/ so we can see exactly what each shape
returns. Run:

    cd ~/cascade-ai && PYTHONPATH=. .venv/bin/python scripts/x402_probe_shapes.py
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
load_dotenv(ROOT / ".env")

from src.data.x402_client import CMC_X402_ENDPOINT, DEFAULT_PAYMENT_ASSET, X402Client  # noqa: E402


def probe(client: X402Client, label: str, arguments: dict) -> None:
    envelope = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "tools/call",
        "params": {"name": "get_crypto_quotes_latest", "arguments": arguments},
    }
    headers = {"MCP-Protocol-Version": "2024-11-05"}
    result = client.request_with_x402("POST", envelope, headers=headers)
    out = ROOT / "artifacts" / f"x402_probe_{label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8")
    # quick verdict
    verdict = "NO RESPONSE"
    if isinstance(result, dict):
        content = (result.get("result") or {}).get("content")
        if isinstance(content, list) and content and isinstance(content[0], dict):
            text = str(content[0].get("text", ""))
            is_err = bool((result.get("result") or {}).get("isError"))
            verdict = f"isError={is_err} text[:160]={text[:160]!r}"
    print(f"[{label}] {verdict}\n  -> {out}")


def main() -> int:
    client = X402Client(
        endpoint=os.getenv("CMC_MCP_URL", os.getenv("CMC_X402_ENDPOINT", CMC_X402_ENDPOINT)),
        default_amount=os.getenv("CMC_X402_AMOUNT", "0.01"),
        default_asset=os.getenv("CMC_X402_ASSET", DEFAULT_PAYMENT_ASSET),
        chain_id=int(os.getenv("CMC_X402_CHAIN_ID", "8453")),
    )
    # exactly the two shapes the new paid path sends
    probe(client, "id_only", {"id": "1839,1975"})
    probe(client, "symbol_only", {"symbol": "NIGHT,COAI"})
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Update demo_artifacts/ON_CHAIN_PROOF.md from wallet and execution logs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.settings import load_settings
from src.execution.bnb_toolkit_wrapper import BnbToolkitWrapper
from src.main import _extract_symbol_balance, _portfolio_value_usdc
from src.strategy.position_manager import PositionManager


def _read_swaps(path: Path, limit: int = 5) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(row.get("action", "")).lower() in {"enter", "exit", "swap"} and row.get("tx_hash"):
            rows.append(row)
    return rows[-limit:]


def main() -> int:
    settings = load_settings()
    wallet = settings.wallet_address or "not configured"
    swaps = _read_swaps(Path(settings.execution_log_path))

    reg_tx = ""
    reg_path = Path("data/compete_registered.json")
    if reg_path.exists():
        try:
            reg_tx = str(json.loads(reg_path.read_text(encoding="utf-8")).get("tx_hash", ""))
        except json.JSONDecodeError:
            reg_tx = ""

    portfolio = 0.0
    try:
        toolkit = BnbToolkitWrapper(settings)
        pm = PositionManager(settings)
        pm.load_positions()
        portfolio = _portfolio_value_usdc(toolkit, settings, {}, pm)
    except Exception:
        portfolio = 0.0

    lines = [
        "# On-Chain Proof Package",
        "",
        f"**Agent wallet:** `{wallet}`",
        "",
        "## Registration",
        f"- Tx: {reg_tx or 'pending — run `twak compete register`'}",
        "",
        "## Portfolio",
        f"- Estimated value (USDC): **${portfolio:,.2f}**",
        "",
        "## Recent swap tx hashes",
        "",
    ]
    if not swaps:
        lines.append("_No swap tx hashes in execution log yet._")
    for row in swaps:
        tx = row.get("tx_hash", "")
        lines.append(f"- [{tx}](https://bscscan.com/tx/{tx}) — {row.get('from_symbol')}→{row.get('to_symbol')} ({row.get('timestamp')})")

    out = Path("demo_artifacts/ON_CHAIN_PROOF.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Updated {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Pre-live funding and registration checklist."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.eligible_tokens import assert_tradable_subset_of_eligible
from src.config.settings import load_settings
from src.config.tokens import TRADABLE_TARGET_SYMBOLS
from src.deployment.reconciliation import _extract_balance
from src.deployment.twak_unlock import verify_twak_unlock
from src.execution.bnb_toolkit_wrapper import BnbToolkitWrapper


def _addresses_equal(left: str, right: str) -> bool:
    return left.strip().lower() == right.strip().lower()


def _balance(toolkit: BnbToolkitWrapper, symbol: str) -> float:
    return _extract_balance(toolkit.get_balance(symbol), symbol)


def main() -> int:
    settings = load_settings()
    checks: list[tuple[str, bool, str]] = []

    try:
        assert_tradable_subset_of_eligible()
        checks.append(("token allowlist ⊆ eligible 149", True, f"{len(TRADABLE_TARGET_SYMBOLS)} tradable"))
    except Exception as exc:
        checks.append(("token allowlist ⊆ eligible 149", False, str(exc)))

    configured = (settings.wallet_address or "").strip()
    unlock = verify_twak_unlock()
    wallet_ok = bool(configured and unlock["ok"] and unlock["address"] and _addresses_equal(configured, unlock["address"]))
    checks.append(("TWAK wallet matches AGENT_WALLET_ADDRESS", wallet_ok, unlock.get("detail", "")))

    min_bnb = getattr(settings, "min_bnb_gas", 0.05)
    min_usdc = getattr(settings, "min_usdc_balance", 50.0)
    try:
        toolkit = BnbToolkitWrapper(settings)
        bnb = _balance(toolkit, "BNB")
        usdc = _balance(toolkit, "USDC")
        usdt = _balance(toolkit, "USDT")
        stable = max(usdc, usdt)
        checks.append((f"BNB gas ≥ {min_bnb}", bnb >= min_bnb, f"{bnb:.6f} BNB"))
        checks.append((f"stable balance ≥ ${min_usdc}", stable >= min_usdc, f"USDC={usdc:.2f} USDT={usdt:.2f}"))
    except Exception as exc:
        checks.append(("wallet balances", False, str(exc)))

    flag_path = Path("data/compete_registered.json")
    registered = False
    if flag_path.exists():
        try:
            payload = json.loads(flag_path.read_text(encoding="utf-8"))
            registered = bool(payload.get("registered"))
            checks.append(("compete register flag", registered, str(payload.get("tx_hash", "local flag"))))
        except json.JSONDecodeError:
            checks.append(("compete register flag", False, "invalid json"))
    else:
        checks.append(("compete register flag", False, "run `twak compete register` and save data/compete_registered.json"))

    print("Pre-live checklist")
    ok = True
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        suffix = f" — {detail}" if detail else ""
        print(f"{status} {name}{suffix}")
        ok = ok and passed
    print(f"Result: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

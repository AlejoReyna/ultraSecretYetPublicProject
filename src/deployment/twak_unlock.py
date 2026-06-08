"""TWAK unattended unlock verification."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from typing import Any

LOGGER = logging.getLogger(__name__)

_ADDRESS_RE = re.compile(r"0x[a-fA-F0-9]{40}")
_PASSWORD_PROMPT = re.compile(r"(?i)(password|passphrase|unlock)")


def _twak_env() -> dict[str, str]:
    env = os.environ.copy()
    password = os.getenv("TWAK_WALLET_PASSWORD", "").strip()
    if password:
        env["TWAK_WALLET_PASSWORD"] = password
    return env


def verify_twak_unlock(*, chain: str = "bsc", timeout_s: int = 30) -> dict[str, Any]:
    """
    Run `twak wallet address --chain bsc --json`.

    Returns dict with keys: ok (bool), address (str|None), detail (str).
    Fails fast on password prompts or non-zero exit.
    """

    command = ["twak", "wallet", "address", "--chain", chain.strip().lower(), "--json"]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=_twak_env(),
            check=False,
        )
    except FileNotFoundError:
        return {"ok": False, "address": None, "detail": "TWAK CLI not found on PATH"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "address": None, "detail": "TWAK wallet address timed out (password prompt?)"}

    combined = f"{completed.stdout}\n{completed.stderr}"
    if _PASSWORD_PROMPT.search(combined) and not os.getenv("TWAK_WALLET_PASSWORD"):
        return {
            "ok": False,
            "address": None,
            "detail": "TWAK requested password; set TWAK_WALLET_PASSWORD or run `twak wallet keychain save`",
        }
    if completed.returncode != 0:
        return {
            "ok": False,
            "address": None,
            "detail": (completed.stderr or completed.stdout or "non-zero exit").strip()[:200],
        }

    address = _extract_address(completed.stdout) or _extract_address(completed.stderr)
    if not address:
        return {"ok": False, "address": None, "detail": "no wallet address in TWAK response"}
    return {"ok": True, "address": address, "detail": "unlocked"}


def _extract_address(text: str) -> str | None:
    text = text.strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            for key in ("address", "wallet_address", "walletAddress"):
                value = payload.get(key)
                if isinstance(value, str) and _ADDRESS_RE.fullmatch(value):
                    return value
    except json.JSONDecodeError:
        pass
    match = _ADDRESS_RE.search(text)
    return match.group(0) if match else None

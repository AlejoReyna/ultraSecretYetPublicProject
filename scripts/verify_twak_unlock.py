#!/usr/bin/env python3
"""Verify TWAK wallet unlock works without a TTY."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.deployment.twak_unlock import verify_twak_unlock


def main() -> int:
    result = verify_twak_unlock()
    if result["ok"]:
        print(f"TWAK unlock OK: {result['address']}")
        return 0
    print(f"TWAK unlock FAILED: {result['detail']}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Deployment alerts (disk, Telegram)."""

from __future__ import annotations

import json
import logging
import shutil
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def disk_free_bytes(path: str | Path = ".") -> int:
    return shutil.disk_usage(path).free


def disk_used_pct(path: str | Path = ".") -> float:
    usage = shutil.disk_usage(path)
    return (usage.used / usage.total) * 100.0 if usage.total else 0.0


def write_alert(message: str, alert_path: str | Path = "logs/ALERT.log") -> None:
    target = Path(alert_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now(timezone.utc).isoformat()} {message}\n"
    with target.open("a", encoding="utf-8") as handle:
        handle.write(line)
    LOGGER.critical(message)


def send_telegram_alert(
    message: str,
    *,
    bot_token: str | None,
    chat_id: str | None = None,
) -> None:
    if not bot_token:
        return
    chat = chat_id or ""
    if not chat:
        return
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = urllib.parse.urlencode({"chat_id": chat, "text": message[:4000]}).encode("utf-8")
    try:
        urllib.request.urlopen(url, data=payload, timeout=10)  # nosec B310
    except Exception as exc:
        LOGGER.warning("Telegram alert failed: %s", exc)


def check_disk_guard(
    *,
    min_free_bytes: int = 500_000_000,
    alert_pct: float = 80.0,
    telegram_token: str | None = None,
    telegram_chat_id: str | None = None,
    path: str | Path = ".",
) -> bool:
    """Return False when disk is critically low."""

    free = disk_free_bytes(path)
    used_pct = disk_used_pct(path)
    if used_pct >= alert_pct:
        msg = f"Disk usage {used_pct:.1f}% exceeds {alert_pct}% threshold"
        write_alert(msg)
        send_telegram_alert(msg, bot_token=telegram_token, chat_id=telegram_chat_id)
    if free < min_free_bytes:
        msg = f"CRITICAL: disk free {free} bytes below {min_free_bytes} byte guard"
        write_alert(msg)
        send_telegram_alert(msg, bot_token=telegram_token, chat_id=telegram_chat_id)
        return False
    return True

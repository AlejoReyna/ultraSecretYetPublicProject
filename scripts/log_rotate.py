#!/usr/bin/env python3
"""Rotate large JSONL logs and prune old archives."""

from __future__ import annotations

import gzip
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.settings import load_settings
from src.deployment.alerts import check_disk_guard, disk_used_pct, write_alert

DEFAULT_LOGS = ("decision_log.jsonl", "execution_log.jsonl")


def rotate_file(path: Path, max_mb: float, archive_dir: Path) -> bool:
    if not path.exists():
        return False
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb < max_mb:
        return False
    archive_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H")
    archive_path = archive_dir / f"{stamp}-{path.name}.gz"
    with path.open("rb") as src, gzip.open(archive_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    path.write_text("", encoding="utf-8")
    print(f"Rotated {path} -> {archive_path}")
    return True


def prune_archives(archive_dir: Path, keep_days: int = 7) -> int:
    if not archive_dir.exists():
        return 0
    cutoff = datetime.now(timezone.utc).timestamp() - keep_days * 86400
    removed = 0
    for item in archive_dir.glob("*.gz"):
        if item.stat().st_mtime < cutoff:
            item.unlink()
            removed += 1
    return removed


def main() -> int:
    settings = load_settings()
    max_mb = float(getattr(settings, "log_rotate_max_mb", 50) or 50)
    archive_dir = Path("logs/archive")
    rotated = 0
    for name in DEFAULT_LOGS:
        if rotate_file(Path(name), max_mb, archive_dir):
            rotated += 1
    pruned = prune_archives(archive_dir)
    used_pct = disk_used_pct(".")
    if used_pct >= 80.0:
        write_alert(f"Disk usage {used_pct:.1f}% exceeds 80%")
    check_disk_guard(
        telegram_token=getattr(settings, "telegram_bot_token", None),
        telegram_chat_id=getattr(settings, "telegram_chat_id", None),
    )
    print(f"Rotated {rotated} file(s), pruned {pruned} archive(s), disk {used_pct:.1f}% used")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Tests for log rotation."""

from __future__ import annotations

import gzip
from pathlib import Path

from scripts.log_rotate import rotate_file


def test_rotate_creates_gzip_archive(tmp_path, monkeypatch) -> None:
    log_path = tmp_path / "decision_log.jsonl"
    log_path.write_text("x" * (60 * 1024 * 1024), encoding="utf-8")
    archive_dir = tmp_path / "archive"
    assert rotate_file(log_path, max_mb=50, archive_dir=archive_dir) is True
    archives = list(archive_dir.glob("*.gz"))
    assert len(archives) == 1
    with gzip.open(archives[0], "rt", encoding="utf-8") as handle:
        content = handle.read()
    assert len(content) > 0
    assert log_path.read_text(encoding="utf-8") == ""

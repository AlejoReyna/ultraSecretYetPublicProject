"""Tests for pre-live funding checklist."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from scripts import pre_live_check


def _mock_settings(monkeypatch) -> MagicMock:
    settings = MagicMock()
    settings.wallet_address = "0x7CE28f5d2D1B2eFd8f87FF0a7fdC7D2EaB465c9c"
    settings.min_bnb_gas = 0.05
    settings.min_usdc_balance = 50.0
    monkeypatch.setattr(pre_live_check, "load_settings", lambda: settings)
    monkeypatch.setattr(pre_live_check, "assert_tradable_subset_of_eligible", lambda: None)
    monkeypatch.setattr(
        pre_live_check,
        "verify_twak_unlock",
        lambda: {"ok": True, "address": "0x7CE28f5d2D1B2eFd8f87FF0a7fdC7D2EaB465c9c", "detail": "ok"},
    )
    return settings


def test_pre_live_check_fails_when_underfunded(monkeypatch) -> None:
    _mock_settings(monkeypatch)
    toolkit = MagicMock()
    toolkit.get_balance.side_effect = lambda sym: {"amount": 0.01 if sym == "BNB" else 0.0}
    monkeypatch.setattr(pre_live_check, "BnbToolkitWrapper", lambda _: toolkit)
    monkeypatch.setattr(pre_live_check.Path, "exists", lambda self: False)

    assert pre_live_check.main() == 1


def test_pre_live_check_passes_when_funded(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    _mock_settings(monkeypatch)
    toolkit = MagicMock()
    toolkit.get_balance.side_effect = lambda sym: {"amount": 1.0 if sym == "BNB" else 100.0}
    monkeypatch.setattr(pre_live_check, "BnbToolkitWrapper", lambda _: toolkit)

    flag_dir = tmp_path / "data"
    flag_dir.mkdir()
    (flag_dir / "compete_registered.json").write_text(
        json.dumps({"registered": True, "tx_hash": "0xabc"}),
        encoding="utf-8",
    )

    assert pre_live_check.main() == 0

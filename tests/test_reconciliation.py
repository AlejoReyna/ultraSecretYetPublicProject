"""Tests for startup position reconciliation."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.deployment.reconciliation import reconcile_positions_on_startup


def test_removes_position_with_zero_on_chain_balance() -> None:
    position = MagicMock()
    position.symbol = "CAKE"
    position_manager = MagicMock()
    position_manager.list_open_positions.return_value = [position]

    toolkit = MagicMock()
    toolkit.get_balance.return_value = {"amount": 0.0}

    removed = reconcile_positions_on_startup(position_manager, toolkit)
    assert removed == ["CAKE"]
    position_manager.close_position.assert_called_once_with("CAKE")

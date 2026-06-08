"""Tests for hackathon eligible token allowlist."""

from __future__ import annotations

from src.config.eligible_tokens import ELIGIBLE_149, assert_tradable_subset_of_eligible
from src.config.tokens import TRADABLE_TARGET_SYMBOLS


def test_all_tradable_tokens_in_eligible_149() -> None:
    assert len(ELIGIBLE_149) == 149
    assert_tradable_subset_of_eligible()
    tradable = {symbol.upper() for symbol in TRADABLE_TARGET_SYMBOLS}
    eligible = {symbol.upper() for symbol in ELIGIBLE_149}
    assert tradable.issubset(eligible)

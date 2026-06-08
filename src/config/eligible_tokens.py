"""BNB Hack eligible token allowlist verification."""

from __future__ import annotations

from src.config.tokens import ELIGIBLE_149_SYMBOLS, TRADABLE_TARGET_SYMBOLS

ELIGIBLE_149 = ELIGIBLE_149_SYMBOLS


def assert_tradable_subset_of_eligible() -> None:
    """Refuse startup when any tradable symbol is outside the hackathon eligible list."""

    eligible = {symbol.upper() for symbol in ELIGIBLE_149}
    tradable = {symbol.upper() for symbol in TRADABLE_TARGET_SYMBOLS}
    invalid = sorted(tradable - eligible)
    if invalid:
        raise RuntimeError(f"Non-eligible tradable tokens: {invalid}")

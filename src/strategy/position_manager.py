"""In-memory position management for Plan B+."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config.settings import Settings
from src.config.tokens import assert_tradable_symbol


@dataclass
class Position:
    """Open spot position with exit levels."""

    symbol: str
    amount_tokens: float
    entry_price: float
    entry_value_usdc: float
    highest_price: float
    trailing_stop_price: float
    take_profit_price: float
    opened_at: datetime
    # True when the row was rebuilt from on-chain state with synthetic entry
    # data. Exit levels are re-anchored from live price on the first update
    # and exits are deferred one cycle (additive field; dashboard Zod ignores it).
    reconstructed: bool = False


class PositionManager:
    """Track open positions and update stop/take-profit state."""

    def __init__(self, settings: Settings, state_path: str | Path | None = None) -> None:
        self.settings = settings
        self.state_path = Path(state_path or settings.position_state_path)
        self._positions: dict[str, Position] = {}

    def open_position(
        self,
        symbol: str,
        amount_tokens: float,
        entry_price: float,
        entry_value_usdc: float,
    ) -> Position:
        """Open and store a new position."""

        normalized = symbol.upper()
        assert_tradable_symbol(normalized)
        if normalized in self._positions:
            raise ValueError(f"{normalized} position is already open")
        position = Position(
            symbol=normalized,
            amount_tokens=amount_tokens,
            entry_price=entry_price,
            entry_value_usdc=entry_value_usdc,
            highest_price=entry_price,
            trailing_stop_price=entry_price * (1 - self.settings.trailing_stop_pct),
            take_profit_price=entry_price * (1 + self.settings.take_profit_pct),
            opened_at=datetime.now(timezone.utc),
        )
        self._positions[normalized] = position
        self.persist_positions()
        return position

    def restore_position(self, position: Position) -> None:
        """Restore a position from trusted persisted or reconstructed state."""

        assert_tradable_symbol(position.symbol)
        self._positions[position.symbol.upper()] = position
        self.persist_positions()

    def update_price(self, symbol: str, current_price: float) -> str | None:
        """Update trailing stop state and return an exit reason when triggered."""

        normalized = symbol.upper()
        position = self._positions.get(normalized)
        if position is None:
            return None
        if position.reconstructed:
            # First live price for a reconstructed row: re-anchor stops from
            # the observed price (synthetic entries can be 0, which would
            # otherwise leave take_profit_price=0 and fire an instant exit),
            # then defer exit evaluation one cycle.
            position.highest_price = max(position.highest_price, current_price)
            position.trailing_stop_price = current_price * (1 - self.settings.trailing_stop_pct)
            if position.take_profit_price <= 0:
                position.take_profit_price = current_price * (1 + self.settings.take_profit_pct)
            position.reconstructed = False
            self.persist_positions()
            return None
        if current_price > position.highest_price:
            position.highest_price = current_price
            raised_stop = current_price * (1 - self.settings.trailing_stop_pct)
            position.trailing_stop_price = max(position.trailing_stop_price, raised_stop)
            self.persist_positions()
        if current_price >= position.take_profit_price:
            return "take_profit"
        if current_price <= position.trailing_stop_price:
            return "trailing_stop"
        return None

    def close_position(self, symbol: str) -> Position | None:
        """Remove and return an open position if present."""

        position = self._positions.pop(symbol.upper(), None)
        if position is not None:
            self.persist_positions()
        return position

    def list_open_positions(self) -> list[Position]:
        """Return all currently open positions."""

        return list(self._positions.values())

    def get_position(self, symbol: str) -> Position | None:
        """Return an open position by symbol."""

        return self._positions.get(symbol.upper())

    def load_positions(self) -> bool:
        """Load persisted positions from disk and return whether a state file existed."""

        if not self.state_path.exists():
            self.persist_positions()
            return False
        with self.state_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        raw_positions = payload.get("positions", payload) if isinstance(payload, dict) else payload
        if not isinstance(raw_positions, list):
            raise ValueError(f"Invalid position state file: {self.state_path}")

        loaded: dict[str, Position] = {}
        for raw_position in raw_positions:
            if not isinstance(raw_position, dict):
                raise ValueError(f"Invalid position entry in {self.state_path}")
            position = self._position_from_dict(raw_position)
            assert_tradable_symbol(position.symbol)
            loaded[position.symbol] = position
        self._positions = loaded
        return True

    def persist_positions(self) -> None:
        """Persist open positions to the configured JSON state file."""

        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "positions": [self._position_to_dict(position) for position in self.list_open_positions()]
        }
        with self.state_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    @staticmethod
    def _position_to_dict(position: Position) -> dict[str, Any]:
        return {
            "symbol": position.symbol,
            "amount_tokens": position.amount_tokens,
            "entry_price": position.entry_price,
            "entry_value_usdc": position.entry_value_usdc,
            "highest_price": position.highest_price,
            "trailing_stop_price": position.trailing_stop_price,
            "take_profit_price": position.take_profit_price,
            "opened_at": position.opened_at.isoformat(),
            "reconstructed": position.reconstructed,
        }

    @staticmethod
    def _position_from_dict(payload: dict[str, Any]) -> Position:
        opened_at_raw = str(payload["opened_at"])
        opened_at = datetime.fromisoformat(opened_at_raw.replace("Z", "+00:00"))
        if opened_at.tzinfo is None:
            opened_at = opened_at.replace(tzinfo=timezone.utc)
        return Position(
            symbol=str(payload["symbol"]).upper(),
            amount_tokens=float(payload["amount_tokens"]),
            entry_price=float(payload["entry_price"]),
            entry_value_usdc=float(payload["entry_value_usdc"]),
            highest_price=float(payload["highest_price"]),
            trailing_stop_price=float(payload["trailing_stop_price"]),
            take_profit_price=float(payload["take_profit_price"]),
            opened_at=opened_at,
            reconstructed=bool(payload.get("reconstructed", False)),
        )


def calculate_position_pct(
    equity_usd: float,
    atr_pct: float | None,
    regime_multiplier: float,
    risk_state_multiplier: float,
    loss_streak: int,
    max_position_pct: float = 0.05,
    base_risk_per_trade_pct: float = 0.0035,
) -> float:
    """Calculate volatility-scaled position size as a decimal percentage."""

    if equity_usd <= 0 or max_position_pct <= 0:
        return 0.0
    regime_mult = max(0.0, float(regime_multiplier))
    risk_mult = max(0.0, float(risk_state_multiplier))
    if atr_pct is None or atr_pct <= 0:
        fallback = min(0.01, max_position_pct) * regime_mult * risk_mult
        return max(0.0, min(fallback, max_position_pct))

    stop_distance_pct = max(0.015, min(0.08, float(atr_pct) * 2.0))
    raw_position_pct = base_risk_per_trade_pct / stop_distance_pct
    position_pct = min(max_position_pct, raw_position_pct)
    if loss_streak >= 2:
        position_pct *= 0.5
    position_pct *= regime_mult * risk_mult
    return max(0.0, min(position_pct, max_position_pct))


def calculate_exit_levels(
    entry_price: float,
    atr_pct: float | None,
    regime: object,
) -> tuple[float, float]:
    """Return trailing-stop and take-profit percentages for a regime."""

    regime_value = getattr(regime, "value", str(regime))
    if regime_value == "risk_off":
        return 0.025, 0.05
    if atr_pct is None or atr_pct <= 0:
        return 0.035, 0.08

    trailing_stop_pct = max(0.035, min(0.10, float(atr_pct) * 1.5))
    take_profit_pct = max(0.08, min(0.20, float(atr_pct) * 3.0))
    if regime_value == "trending_up":
        return trailing_stop_pct, take_profit_pct
    return min(trailing_stop_pct, 0.06), min(take_profit_pct, 0.12)

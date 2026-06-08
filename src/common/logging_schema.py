"""Versioned JSONL schemas for live and shadow audit logs.

Example:
    log = LiveDecisionLog(run_id="run-1", cycle_id=7, action="WAIT")
    append_to_file("logs/decision_live.jsonl", log)

Interface contract:
    Imports: standard-library dataclasses, datetime, json, pathlib.
    Exports: log dataclasses, to_jsonl(), append_to_file().
    Does not touch execution, wallets, keys, or strategy decisions.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class LogEntry:
    """Base fields shared by every audit log record."""

    schema_version: str = "1.0"
    run_id: str = ""
    mode: Literal["paper", "live"] = "paper"
    path: Literal["live", "shadow"] = "live"
    timestamp: str = field(default_factory=_utc_timestamp)
    cycle_id: int = 0

    def to_jsonl(self) -> str:
        """Return this record as one JSON object line without a trailing newline."""

        return to_jsonl(self)


@dataclass(frozen=True)
class LiveDecisionLog(LogEntry):
    """Decision made by the deterministic live path."""

    path: Literal["live", "shadow"] = "live"
    action: str = "WAIT"
    symbol: str | None = None
    size_pct: float = 0.0
    reasons: list[str] = field(default_factory=list)
    source: str = "LIVE"
    regime: str = "unknown"
    regime_score: float = 0.0
    ema_72: float | None = None
    ema_144: float | None = None
    ema_288: float | None = None
    atr_pct: float | None = None
    position_pct: float = 0.0
    slippage_quote: float | None = None
    risk_state: str = "normal"
    sentiment_delta: float | None = None
    sentiment_fragility: str | None = None
    strategy_mode: str | None = None
    entry_score: float | None = None
    hold_time_seconds: int | None = None
    exit_reason: str | None = None
    gas_cost_usd: float | None = None
    expected_breakeven_pct: float | None = None
    ml_regime: str | None = None
    ml_confidence: float | None = None
    ml_ranking: dict[str, Any] | None = None


@dataclass(frozen=True)
class ShadowDecisionLog(LogEntry):
    """Hypothetical shadow decision, physically separated from live logs."""

    path: Literal["live", "shadow"] = "shadow"
    variant: str = ""
    hypothetical_action: str = "WAIT"
    hypothetical_symbol: str | None = None
    reasons: list[str] = field(default_factory=list)
    source: str = "SHADOW"
    confidence: float | None = None


@dataclass(frozen=True)
class SentimentLiveLog(LogEntry):
    """Raw TIER 1 sentiment context observed by the live path."""

    path: Literal["live", "shadow"] = "live"
    fear_greed_index: int | None = None
    fear_greed_classification: str | None = None
    funding_rate_btc: float | None = None
    open_interest_btc: float | None = None
    gas_price_gwei: float | None = None
    gas_avg_24h_gwei: float | None = None
    sentiment_delta: float = 0.0
    regime_fragility: str = "NONE"


@dataclass(frozen=True)
class SentimentShadowLog(LogEntry):
    """Shadow-only sentiment metric for offline validation."""

    path: Literal["live", "shadow"] = "shadow"
    source: str = "SHADOW"
    metric: str = ""
    value: float = 0.0
    sentiment_score: float = 0.0
    live_decision_id: str | None = None
    shadow_recommendation: str = "OBSERVE"
    validated: bool = False


@dataclass(frozen=True)
class RiskEventLog(LogEntry):
    """Risk-state transition or guardrail event."""

    event_type: str = ""
    severity: str = "info"
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PortfolioSnapshotLog(LogEntry):
    """Portfolio value and drawdown snapshot for one cycle."""

    portfolio_value_usdc: float = 0.0
    all_time_high: float = 0.0
    drawdown_pct: float = 0.0
    open_positions: list[dict[str, Any]] = field(default_factory=list)


def to_jsonl(log_entry: LogEntry) -> str:
    """Serialize a log entry as stable one-line JSON."""

    return json.dumps(asdict(log_entry), sort_keys=True, separators=(",", ":"))


def append_to_file(path: str | Path, log_entry: LogEntry) -> None:
    """Append one log entry to a physical JSONL file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(to_jsonl(log_entry))
        handle.write("\n")

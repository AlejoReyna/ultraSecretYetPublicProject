"""TTL cache for CMC market snapshots independent of the trading loop."""

from __future__ import annotations

import copy
import logging
import time
from typing import Any, Callable

LOGGER = logging.getLogger(__name__)

HOT_SNAPSHOT_FIELDS = (
    "price",
    "market_cap",
    "volume_1h",
    "volume_24h",
    "percent_change_1h",
    "percent_change_6h",
    "percent_change_24h",
    "high_3h",
    "high_6h",
    "high_24h",
    "low_24h",
    "rolling_24h_hourly_volume_avg",
    "bnb_1h_trend_pct",
    "estimated_slippage_pct",
)


def merge_market_snapshots(
    base: dict[str, dict[str, Any]],
    overlay: dict[str, dict[str, Any]],
    *,
    hot_fields: tuple[str, ...] = HOT_SNAPSHOT_FIELDS,
) -> dict[str, dict[str, Any]]:
    """Merge keyless hot fields over an x402-enriched base snapshot."""

    merged = copy.deepcopy(base) if base else {}
    for symbol, overlay_data in overlay.items():
        if not isinstance(overlay_data, dict):
            continue
        normalized = str(symbol).upper()
        existing = merged.get(normalized, {})
        if existing:
            combined = copy.deepcopy(existing)
            for field in hot_fields:
                value = overlay_data.get(field)
                if value is not None and value != 0 and value != "":
                    combined[field] = value
            combined["symbol"] = normalized
        else:
            combined = copy.deepcopy(overlay_data)
            combined["symbol"] = normalized
        merged[normalized] = combined
    return merged


class MarketSnapshotCache:
    """Reuse the last CMC snapshot until its TTL expires."""

    def __init__(self) -> None:
        self._snapshot: dict[str, dict[str, Any]] = {}
        self._fetched_at: float = 0.0

    def get_or_fetch(
        self,
        ttl_seconds: int,
        fetcher: Callable[[], dict[str, dict[str, Any]]],
        *,
        force_refresh: bool = False,
    ) -> dict[str, dict[str, Any]]:
        if ttl_seconds <= 0:
            return fetcher()

        now = time.monotonic()
        age = now - self._fetched_at
        if not force_refresh and self._snapshot and age < ttl_seconds:
            LOGGER.debug(
                "Reusing CMC market snapshot (age=%.0fs ttl=%ss symbols=%s)",
                age,
                ttl_seconds,
                len(self._snapshot),
            )
            return copy.deepcopy(self._snapshot)

        snapshot = fetcher()
        self._snapshot = copy.deepcopy(snapshot)
        self._fetched_at = now
        LOGGER.info(
            "Refreshed CMC market snapshot (ttl=%ss symbols=%s)",
            ttl_seconds,
            len(snapshot),
        )
        return copy.deepcopy(snapshot)

    def reset(self) -> None:
        self._snapshot = {}
        self._fetched_at = 0.0


class DualMarketSnapshotCache:
    """Independent TTL layers for x402-enriched and keyless quote snapshots."""

    def __init__(self) -> None:
        self._x402_enriched: dict[str, dict[str, Any]] = {}
        self._x402_fetched_at: float = 0.0
        self._keyless_quotes: dict[str, dict[str, Any]] = {}
        self._keyless_fetched_at: float = 0.0

    def get_merged_snapshot(
        self,
        x402_ttl_seconds: int,
        keyless_ttl_seconds: int,
        x402_fetcher: Callable[[], dict[str, dict[str, Any]]],
        keyless_fetcher: Callable[[], dict[str, dict[str, Any]]],
    ) -> dict[str, dict[str, Any]]:
        now = time.monotonic()
        self._maybe_refresh_layer(
            layer_name="x402",
            snapshot_attr="_x402_enriched",
            fetched_at_attr="_x402_fetched_at",
            ttl_seconds=x402_ttl_seconds,
            fetcher=x402_fetcher,
            now=now,
        )
        self._maybe_refresh_layer(
            layer_name="keyless",
            snapshot_attr="_keyless_quotes",
            fetched_at_attr="_keyless_fetched_at",
            ttl_seconds=keyless_ttl_seconds,
            fetcher=keyless_fetcher,
            now=now,
        )
        merged = merge_market_snapshots(self._x402_enriched, self._keyless_quotes)
        if not merged:
            LOGGER.warning("Dual market snapshot merge produced no symbols")
        return copy.deepcopy(merged)

    def _maybe_refresh_layer(
        self,
        *,
        layer_name: str,
        snapshot_attr: str,
        fetched_at_attr: str,
        ttl_seconds: int,
        fetcher: Callable[[], dict[str, dict[str, Any]]],
        now: float,
    ) -> None:
        snapshot: dict[str, dict[str, Any]] = getattr(self, snapshot_attr)
        fetched_at: float = getattr(self, fetched_at_attr)
        age = now - fetched_at
        if ttl_seconds > 0 and snapshot and age < ttl_seconds:
            LOGGER.debug(
                "Reusing %s market snapshot (age=%.0fs ttl=%ss symbols=%s)",
                layer_name,
                age,
                ttl_seconds,
                len(snapshot),
            )
            return

        try:
            fresh = fetcher()
        except Exception as exc:
            LOGGER.warning("%s market snapshot fetch failed: %s", layer_name, exc)
            fresh = {}

        if fresh:
            setattr(self, snapshot_attr, copy.deepcopy(fresh))
            setattr(self, fetched_at_attr, now)
            LOGGER.info(
                "Refreshed %s market snapshot (ttl=%ss symbols=%s)",
                layer_name,
                ttl_seconds,
                len(fresh),
            )
        elif not snapshot:
            LOGGER.warning("%s market snapshot fetch returned empty and no cache exists", layer_name)
        else:
            LOGGER.warning(
                "%s market snapshot fetch returned empty; reusing stale cache (%s symbols)",
                layer_name,
                len(snapshot),
            )

    def reset(self) -> None:
        self._x402_enriched = {}
        self._x402_fetched_at = 0.0
        self._keyless_quotes = {}
        self._keyless_fetched_at = 0.0


_DEFAULT_CACHE = MarketSnapshotCache()
_DEFAULT_DUAL_CACHE = DualMarketSnapshotCache()


def get_market_snapshot_cache() -> MarketSnapshotCache:
    return _DEFAULT_CACHE


def get_dual_market_snapshot_cache() -> DualMarketSnapshotCache:
    return _DEFAULT_DUAL_CACHE

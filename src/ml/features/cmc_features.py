"""CMC premium snapshot features for ML regime detection."""

from __future__ import annotations

import math
from typing import Any

CMC_FEATURE_NAMES: list[str] = [
    "funding_rate",
    "funding_rate_zscore_7d",
    "fear_greed_index",
    "fear_greed_delta_1d",
    "open_interest_delta_pct",
    "social_dominance",
    "social_volume_delta",
    "market_cap_rank_pctile",
    "cmc_volume_1h_ratio",
]


def _number(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def compute_cmc_features(
    cmc_snapshot: dict[str, Any],
    *,
    fear_greed_prior: float | None = None,
    funding_history: list[float] | None = None,
    rank_pctile: float = 0.5,
) -> dict[str, float]:
    """Extract normalized CMC premium features from a token snapshot dict."""

    funding_rate = _number(cmc_snapshot.get("funding_rate"))
    fear_greed = _number(cmc_snapshot.get("fear_greed_index"), default=50.0)
    if fear_greed > 1.0:
        fear_greed_norm = fear_greed / 100.0
    else:
        fear_greed_norm = fear_greed

    oi_delta = _number(cmc_snapshot.get("open_interest_change_pct"))
    if abs(oi_delta) > 1.0:
        oi_delta = oi_delta / 100.0

    volume_1h = _number(cmc_snapshot.get("volume_1h"))
    rolling_hourly = _number(cmc_snapshot.get("rolling_24h_hourly_volume_avg"))
    if rolling_hourly <= 0 and volume_1h > 0:
        volume_24h = _number(cmc_snapshot.get("volume_24h"))
        rolling_hourly = volume_24h / 24.0 if volume_24h > 0 else 0.0

    social_dom = _number(cmc_snapshot.get("social_dominance") or cmc_snapshot.get("social_score"))
    social_vol_delta = _number(cmc_snapshot.get("social_volume_change_24h"))

    funding_z = 0.0
    if funding_history:
        mean = sum(funding_history) / len(funding_history)
        var = sum((item - mean) ** 2 for item in funding_history) / max(len(funding_history), 1)
        std = math.sqrt(var) if var > 0 else 0.0
        funding_z = (funding_rate - mean) / std if std > 0 else 0.0

    fear_delta = 0.0
    if fear_greed_prior is not None:
        fear_delta = fear_greed_norm - fear_greed_prior

    return {
        "funding_rate": funding_rate,
        "funding_rate_zscore_7d": funding_z,
        "fear_greed_index": fear_greed_norm,
        "fear_greed_delta_1d": fear_delta,
        "open_interest_delta_pct": oi_delta,
        "social_dominance": social_dom,
        "social_volume_delta": math.log1p(max(social_vol_delta, 0.0)),
        "market_cap_rank_pctile": rank_pctile,
        "cmc_volume_1h_ratio": volume_1h / rolling_hourly if rolling_hourly > 0 else 0.0,
    }

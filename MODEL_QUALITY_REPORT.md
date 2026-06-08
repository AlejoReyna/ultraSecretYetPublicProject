# Model Quality Report

Generated: 2026-06-08T16:49:25.868446+00:00

## Summary
- Positive class rate: **25.3%**
- Feature count: **41**
- Best model: **lgb**
- Recommendation: **KEEP shadow/regime-only fallback — worst-fold AUC below 0.65; ML ranking disabled.**

## Per-model purged CV (5 folds, 24-candle purge gap)

| Model | Mean AUC | Std | Worst-fold AUC | Folds |
|-------|----------|-----|----------------|-------|
| lgb | 0.5564 | 0.0504 | 0.4788 | 0.605, 0.587, 0.514, 0.479, 0.597 |

## Best model feature importance (top 10)

- `hour_of_day`: 3631.0000
- `volatility_48`: 2647.0000
- `day_of_week`: 2109.0000
- `range_compression_6h`: 1836.0000
- `volume_price_divergence`: 1646.0000
- `ema_8_21_spread`: 1622.0000
- `atr_pct_14`: 1533.0000
- `volatility_16`: 1519.0000
- `volume_skew_3h_6h`: 1484.0000
- `rsi_14`: 1143.0000

## Shadow mode recommendation

KEEP shadow/regime-only fallback — worst-fold AUC below 0.65; ML ranking disabled.

Set `ML_SHADOW_MODE=false` only after worst-fold AUC >= 0.65 and 48h shadow paper validation.

# Demo Video Script — Plan B+ (3 minutes)

## 0:00–0:30 — Architecture overview

- Show repo structure: `src/main.py`, `src/execution/twak_interface.py`, `src/ml/`
- Emphasize: **no private key in Python** — all signing via TWAK CLI subprocess
- CMC data: Keyless primary + x402 premium collector (`scripts/cmc_feature_collector.py`)
- ML: local LightGBM, CPU inference <5ms, offline training only

## 0:30–1:00 — AUC gate blocks bad model

- Open `MODEL_QUALITY_REPORT.md`
- Point to worst-fold AUC (e.g. 0.48) vs gate (0.65)
- Show `ML_SHADOW_MODE=true` in `.env`
- Explain: ML ranking disabled; regime-only fallback active (0.3× chop sizing)

## 1:00–1:30 — Shadow mode logs

- `tail decision_log.jsonl` — show `ml_active: false`, `ml_selected_symbol` vs `executed_symbol`
- `curl localhost:8080/logs` — mobile-friendly last 50 decisions
- Open `dashboard.html` — timeline with ML scores overlaid but rule-based execution

## 1:30–2:00 — Health check + real swap

- `curl localhost:8080/health` — status, positions, drawdown, ml_mode=regime_fallback
- Show BSCScan link from `demo_artifacts/ON_CHAIN_PROOF.md`
- Mention registration tx from `twak compete register`

## 2:00–2:30 — Guardrails demo

- Decision log entry with `action: BLOCKED` or daily limit
- Slippage block from TWAK quote-only
- Kill switch / drawdown pause in `risk_events.jsonl`

## 2:30–3:00 — Regime fallback in action

- Log line showing `ml_regime: chop` and reduced `position_size_usdc`
- Compare momentum vs chop multiplier (1.0× vs 0.3×)
- Close: 4 core factors still mandatory; ML additive only

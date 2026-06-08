# Plan B+ Deployment Guide

Live window: **June 22–28, 2026**. ML runs in **regime-only fallback** (`ML_SHADOW_MODE=true`) unless `MODEL_QUALITY_REPORT.md` shows worst-fold AUC ≥ 0.65.

## EC2 setup (c7i-flex.large)

```bash
# 1. Clone and venv
cd /home/ubuntu
git clone <repo-url> planb-plus
cd planb-plus
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-ml.txt
brew install libomp  # macOS only; on Ubuntu use apt if LightGBM fails

# 2. Configure environment
cp .env.example .env
# Set: AGENT_WALLET_ADDRESS, BSC_PROVIDER_URL, TWAK_WALLET_PASSWORD (or keychain)
# ML_ENABLED=true, ML_SHADOW_MODE=true, STRATEGY_MODE=breakout

# 3. TWAK unattended unlock
twak wallet keychain save   # OR set TWAK_WALLET_PASSWORD in .env
python scripts/verify_twak_unlock.py

# 4. Pre-live checks
python scripts/pre_live_check.py
twak compete register
echo '{"registered": true, "tx_hash": "<hash>"}' > data/compete_registered.json

# 5. Train model (regime fallback works even with low AUC)
python scripts/build_feature_matrix.py
python scripts/train_regime_model.py --allow-low-auc

# 6. Systemd user service
mkdir -p ~/.config/systemd/user
cp systemd/planb-plus.service ~/.config/systemd/user/
# Edit paths if not using /home/ubuntu/planb-plus
systemctl --user daemon-reload
systemctl --user enable planb-plus
systemctl --user start planb-plus
journalctl --user -u planb-plus -f
```

## Cron jobs

```cron
# Log rotation every 6 hours
0 */6 * * * cd /home/ubuntu/planb-plus && .venv/bin/python scripts/log_rotate.py

# CMC premium collector (parallel experiment)
*/15 * * * * cd /home/ubuntu/planb-plus && .venv/bin/python scripts/cmc_feature_collector.py

# Dashboard refresh during live window
*/30 * * * * cd /home/ubuntu/planb-plus && .venv/bin/python scripts/generate_dashboard.py

# On-chain proof package
0 */4 * * * cd /home/ubuntu/planb-plus && .venv/bin/python scripts/update_on_chain_proof.py
```

## Health check

```bash
curl -s http://localhost:8080/health | jq .
curl -s http://localhost:8080/logs | jq .
open http://localhost:8080/dashboard
```

## Firewall

```bash
sudo ufw allow 8080/tcp   # health dashboard (restrict to your IP in production)
```

## ML experiment schedule

| Date | Action |
|------|--------|
| Jun 8–14 | `cmc_feature_collector.py` runs via cron |
| Jun 14 | `python scripts/rebuild_with_cmc.py` — check `MODEL_QUALITY_REPORT_V3.md` |
| Jun 15–21 | Retrain only if worst-fold AUC ≥ 0.65; else keep regime fallback |
| Jun 22 | `ML_SHADOW_MODE=true` for live unless AUC gate passes |

## Disk guard

- `scripts/log_rotate.py` archives logs > 50 MB
- Trading loop halts new entries when free disk < 500 MB
- Alerts written to `logs/ALERT.log`; optional Telegram via `TELEGRAM_BOT_TOKEN`

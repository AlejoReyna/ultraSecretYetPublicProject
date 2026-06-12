# BNB HACK Track 1 — Competition Plan

Target: top-5 total return over June 22–28 without ever approaching the ~30% drawdown disqualification gate. 292 registered hackers, 5 paid slots, ranked on raw PnL with a drawdown cap, minimum 1 trade per UTC day (7 total), simulated transaction costs applied.

## Why the dev defaults could not win

The agent as configured was built to survive, not to rank. Scalping mode risks 1% of the portfolio per trade chasing +1.5% moves on 5-minute-cached CMC data; a perfect day nets ~+0.15% before the ~0.6–1.2% round-trip cost floor (2× 0.25% PancakeSwap fee + slippage + gas), which the take-profit barely clears. Breakout mode capped positions at 5% with 0.35% risk per trade — at most ~1–2% portfolio upside in a perfect week. Meanwhile the rules hand every participant a 30% risk budget. Using 5% of it is how you finish 150th safely.

## The competition strategy: asymmetric barbell

The scoring function is a one-week return race with a hard floor. The optimal shape is maximum convexity above the floor:

1. **Risk-on tape** (BNB trend up, breakouts confirming): deploy heavily — up to 30% per position, ~3 concurrent positions (~90% of book) into the strongest 4-factor momentum breakouts among the 149 eligible tokens. Wide trailing stops (ATR-scaled, 6–10%) so winners run; the 0.2%-buffered 3h-high breakout entry plus volume confirmation keeps entries out of chop.
2. **Risk-off tape**: the regime gate fails closed, the book sits in USDC — which is itself an eligible token, so parked capital is fully rule-compliant. A flat week loses ~0–2% while a large share of 292 momentum-chasing agents bleed or blow through the drawdown gate. Survival is itself ranking-positive.
3. **Hard floor**: portfolio-level kill switch at **20%** drawdown from all-time-high (10-point buffer under the 30% DQ line to absorb intra-hour gaps and liquidation slippage), soft de-risking (half size) from 12%, 8% realized daily-loss pause, loss-streak size reduction.

Expected distribution: most weeks finish between -5% and +10%; an up-week in BSC memecoins/majors with this sizing can print +20–50%, which is the region historical week-long PnL contests are won in. The plan does not guarantee a win — nothing does in a 292-entrant return race — but it maximizes the probability mass in the winning region while keeping P(DQ) near zero for spot-only, no-leverage exposure.

## Rule-compliance engineering (changes made)

| Risk | Fix |
| --- | --- |
| Kill switch **halted the loop** — a halted agent stops trading and silently fails the 1-trade/day minimum, a second disqualification after surviving the first | Kill switch now liquidates and stays alive in capital-preservation mode; only the compliance backstop trades |
| Daily pause / loss-streak pause blocked **all** trades, including the daily minimum | New last-resort backstop: if no trade by 22:00 UTC, execute a $5 USDC→USDT swap (both legs eligible tokens, negligible market risk), counted via `record_compliance_trade()` |
| Kill switch was **not persisted** — a process restart silently re-armed full trading at 20% drawdown | `kill_switch` now saved/restored in guardrail state |
| Entry bookkeeping called `record_trade_result(0.0)` which **reset the loss streak on every entry** — streak pause could never fire while the agent kept entering | Zero PnL no longer resets the streak; only profitable exits do |
| Sub-$1 portfolio hours score 0% per the rules | Non-issue at sane sizing; never fully drain the wallet — backstop swap is $5, keep ≥$20 in stables at all times |

## Drawdown math (spot-only, no leverage)

Worst realistic cycle: 3 positions × 30% deployed, simultaneous -15% gap on all three between 5-minute polls → -13.5% portfolio, still 6.5 points above the kill switch and 16.5 above DQ. Kill switch at -20% liquidates into stables; even with 3% aggregate liquidation slippage the mark lands ~-23%, seven points under the DQ line measured hourly. Reaching -30% would require a >50% simultaneous instant gap across three uncorrelated-entry positions with zero stop executions — the trailing stops (6–10%) fire far earlier in any continuous move.

## x402 budget

Observed burn (~$1/hour) had two causes found in code: each enriched snapshot pays 3 batch calls (148 symbols / 50 per batch) plus MCP session-init overhead, and — the real leak — a paid fetch that failed *after* settlement never updated the cache timestamp, so the agent re-paid every 5-minute cycle until the endpoint recovered.

Fixes now in place: a spend governor (`src/data/x402_spend_governor.py`) gates every paid call with a daily cap, a total cap, and a 15-minute failure cooldown, persisting a ledger to `logs/x402_spend.json` across restarts; paid refresh cadence is adaptive — every 2h while flat, every 30 min while holding positions — and the free keyless REST layer (which carries every field the entry gates actually use: price, volume, 1h/24h change) refreshes every 5 minutes regardless.

Cost model at $0.01/call, 3 calls per enrichment:

| Profile | Cadence | Cost/day | Week |
| --- | --- | --- | --- |
| Floor | flat 2h cadence only | ~$0.36 | ~$2.50 |
| Default (configured) | 2h flat / 30 min in position, ~12h/day deployed | ~$1.60 | ~$11 |
| Aggressive | 30 min always + headroom | ~$3.60 | ~$25 |

Recommendation: fund the ephemeral x402 key with **$20** and keep the configured caps ($2/day, $15 total). The marginal trading edge of paid quotes over the free keyless layer is modest — the paid layer's real value is redundancy when the trial REST endpoint throttles mid-window. Past ~$25/week, more x402 spend buys nothing the strategy reads. If the keyless endpoint proves unreliable in the paper soak, raise `X402_DAILY_BUDGET_USDC` to 4.00 rather than tightening TTLs.

## Runbook

**Before June 21 (build deadline):**
1. `cp .env.competition .env`, fill RPC, wallet, CMC key, x402 key.
2. `pytest` — confirm the suite passes after the guardrail changes (some tests may assert the old loss-streak reset/halt behavior; update those assertions, they encoded the bugs).
3. 48h paper soak: `python -m src.main --paper-trade --demo-mode`, verify ENTER cycles occur and the 22:00 UTC backstop fires on a no-trade day (set clock or stub).
4. Micro-live proof: fund with ~$50, run `--live --once` until one full entry→exit cycle reconciles on-chain (README lists this as still unproven — it is the single biggest remaining risk, not the strategy).
5. Register: `twak compete register` (before June 22) + submit agent address & strategy writeup on DoraHacks.

**June 22, 00:00 UTC:** fund the wallet (size to what you can afford to lose — this is real capital at real risk), confirm non-zero in-scope balance at window start, `--live --preflight`, then start the loop under a process supervisor (auto-restart; guardrail state now survives restarts).

**During the window:** check `logs/portfolio_snapshots.jsonl` drawdown_pct and `risk_events.jsonl` daily; verify ≥1 trade before 22:00 UTC each day; never manually override the kill switch.

## Honest caveats

- Returns are measured by the organizers hour-by-hour on-chain; their drawdown definition ("for example 30%") is not fully specified. The 20% internal stop is sized for that ambiguity. If they clarify in Telegram, re-tune — anything from 18–22% is reasonable per your risk choice.
- Simulated transaction costs in scoring penalize churn: 6 trades/day max, no scalping.
- This plan trades real funds. Nothing here is financial advice; the competition can lose the entire deployed amount regardless of guardrails (contract risk, token rug within the eligible list, RPC outage during a crash).

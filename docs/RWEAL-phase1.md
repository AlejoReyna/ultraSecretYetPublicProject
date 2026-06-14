# RWEAL Phase 1 — Real-World Event Awareness Layer (static entry gate)

Phase 1 of the proposition doc (Section 10). It is a **pre-flight, entry-only**
risk filter. It does **not** modify the audited exit path
(`calculate_exit_levels`, time-stop, window-flatten) and does **not** change
position sizing. When disabled it is a zero-behaviour-change no-op.

## What it does

Two independent mechanisms, both reading from disk so a running process picks up
operator changes without a restart:

1. **Manual halt** — a control *file* (default `TRADING_HALT` in the working
   directory). While it exists, **all** entries are suppressed *and* the
   daily-minimum compliance trade is suppressed. This is a deliberate full stop:
   it accepts the competition's one-trade-per-UTC-day disqualification risk.
   - A file is used instead of an env var because a running process never
     re-reads a changed `.env`. The file is `stat()`-ed inside the main loop's
     1-second wake cycle, so a halt takes effect within seconds, not up to one
     `LOOP_SECONDS` (default 300s).

2. **Event blackout** — a static `events.json` calendar. Within a configurable
   window around a scheduled adverse event, **discretionary** entries are
   blocked. A `GLOBAL` event (e.g. a macro CPI/FOMC release) blocks all
   discretionary entries; a symbol-specific event blocks only that symbol —
   blacked-out symbols are removed from candidate *selection* so the next-best
   unaffected symbol is still considered that cycle (rather than skipping the
   cycle entirely). Unlike a manual halt, an event blackout **leaves the
   daily-minimum compliance trade running** so the agent stays qualified; the
   compliance trade itself is steered away from a blacked-out symbol (it falls
   back to the fixed stable→token swap).

### Halt timing guarantees

The manual halt is enforced at three points so a `TRADING_HALT` that appears
mid-cycle cannot still trade: (a) the cycle-top gate, (b) a live re-check
immediately before `_attempt_entry_v25`, and (c) a live re-check immediately
before each `_ensure_daily_minimum_trade` compliance call. The 1-second wake
loop additionally shortens the wait before the next cycle begins.

## Blackout window

For each event the active window is:

- If the event sets `blackout_minutes = M`: `[scheduled - M, scheduled + M]`
  (symmetric — intended for point-in-time macro releases).
- Otherwise: `[scheduled - RWEAL_BLACKOUT_HORIZON_HOURS, scheduled + RWEAL_POST_EVENT_MINUTES]`.

## Configuration

| Env var | Default | Meaning |
|---|---|---|
| `ENABLE_RWEAL` | `false` | Master switch. `false` = no-op. |
| `RWEAL_EVENTS_PATH` | `events.json` | Path to the calendar. |
| `RWEAL_CONTROL_FILE` | `TRADING_HALT` | Manual full-stop flag file (presence = halt). |
| `RWEAL_BLACKOUT_HORIZON_HOURS` | `6` | Hours before an event to block entries. |
| `RWEAL_POST_EVENT_MINUTES` | `60` | Minutes after an event to keep blocking. |

## events.json format

Root is an object with an `events` array (a bare array is also accepted). Each
event requires `symbol`, `event_type`, `scheduled_time` (ISO-8601, `Z` or
offset; naive timestamps treated as UTC), and `severity` (integer 1–5). Optional:
`direction_bias`, `blackout_minutes`, `description`. Use `symbol: "GLOBAL"` for
market-wide (macro) events. See `events.example.json`.

## Failure model

- **Enabled + malformed `events.json`** → hard startup error (`RwealConfigError`).
  Fail fast while the operator is present.
- **Enabled + missing `events.json`** → allowed; runs with the manual halt only,
  logs a warning. (You can use the kill switch without a calendar.)
- **File edited to an invalid state while running** → the reload keeps the
  last-known-good calendar and logs an error rather than crashing the loop.

## Operating during competition

- Populate `events.json` ≥24h before judging week with high-confidence events
  only (macro blackouts, large unlocks, manual blackouts). Do **not** add
  speculative Tier 2/3 intelligence in Phase 1.
- To halt instantly: `touch TRADING_HALT`. To resume: `rm TRADING_HALT`.
- Both `events.json` and `TRADING_HALT` are git-ignored (operator-local).

## Explicitly out of scope for Phase 1

No automated feeds/APIs, no sizing reduction (`caution` regime), no exit-level
changes (`event_regime` is **not** added to `calculate_exit_levels`), no on-chain
or social signals. Those are Phase 2/3 and gated on a shadow-mode dry run.

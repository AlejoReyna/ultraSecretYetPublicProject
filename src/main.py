"""CLI entrypoint for the Plan B+ trading agent."""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import re
import signal
import sys
import time
import types
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import src.strategy as strategy_package
from src.common.logging_schema import (
    LiveDecisionLog,
    PortfolioSnapshotLog,
    RiskEventLog,
    SentimentLiveLog,
    append_to_file,
)
from src.config.settings import Settings, load_settings
from src.deployment.runtime import deployment_startup, disk_allows_entries, update_health_snapshot
from src.config.tokens import (
    TARGET_SYMBOLS,
    TRADABLE_TARGET_SYMBOLS,
    has_verified_bsc_contract,
    is_liquid,
    is_momentum_candidate_symbol,
)
from src.data.cmc_mcp_client import CMCMCPClient
from src.data.enrichment_planner import hot_candidate_symbols, select_enrichment_symbols
from src.data.market_snapshot_cache import get_dual_market_snapshot_cache, get_market_snapshot_cache
from src.execution import liquidity_analyzer as liquidity_analyzer_module
from src.execution.bnb_toolkit_wrapper import BnbToolkitWrapper
from src.execution.decision_log import DecisionAction, log_decision
from src.execution.execution_log import log_execution
from src.execution.execution_reconciler import ExecutionReconciler, ReconciliationResult
from src.execution.swap_router import PancakeSwapRouter
from src.execution.twak_interface import TWAKInterface
from src.strategy.breakout_engine import BreakoutDecision, BreakoutEngine
import importlib

from src.strategy.candidate_adapter import (
    breakout_decision_to_candidate,
    coerce_entry_candidate,
    decimal_div as _decimal_div,
)
from src.strategy.entry_types import EntryCandidate
from src.strategy.event_filter import EventRiskFilter
from src.strategy.factory import create_strategy_bundle, fallback_evaluate_universe

_fallback_scorer = importlib.import_module("src.strategy.6falgorithm.fallback_scorer")
fallback_best_near_miss = _fallback_scorer.fallback_best_near_miss
from src.strategy.scalping_guardrails import ScalpingGuardrails
from src.strategy.scalping_position_manager import ScalpingPositionManager
from src.strategy.guardrails import Guardrails, RiskDecision, RiskState, TradeRecord
from src.strategy.position_manager import Position, PositionManager, calculate_position_pct
from src.strategy.regime_detector import MarketRegime, RegimeDetector, RegimeResult
from src.strategy.sentiment_tier1 import SentimentResult, SentimentTier1
from src.strategy.volatility import PriceCache

LOGGER = logging.getLogger(__name__)
LIVE_WINDOW_MONTH = 6
LIVE_WINDOW_START_DAY = 22
LIVE_WINDOW_END_DAY = 28
PREFLIGHT_QUOTE_AMOUNT_USDC = 0.5
COMPLIANCE_TRADE_USDC = 0.5
COMPLIANCE_TRIGGER_HOUR_UTC = 22
# Portfolio floor: never let the daily compliance trade spend the balance
# below this retained USDC amount (preserves a floor on a near-liquidated book).
MIN_PORTFOLIO_RETAINED_USDC = 2.0
COMPLIANCE_TO_SYMBOL = "TWT"
SCHEMA_VERSION = "2.6.0"


try:
    from src.strategy import scoring as scoring
except ImportError:
    scoring = types.ModuleType("src.strategy.scoring")
    sys.modules["src.strategy.scoring"] = scoring
    setattr(strategy_package, "scoring", scoring)


if hasattr(liquidity_analyzer_module, "LiquidityAnalyzer"):
    LiquidityAnalyzer = liquidity_analyzer_module.LiquidityAnalyzer
else:

    class LiquidityAnalyzer:
        """Compatibility adapter for the function-only liquidity module."""

        def analyze_liquidity(
            self,
            symbol: str,
            position_usd: float,
            twak_quote_small: float | None,
            twak_quote_normal: float | None,
            max_slippage_pct: float,
        ) -> liquidity_analyzer_module.LiquidityResult:
            return liquidity_analyzer_module.analyze_liquidity(
                symbol=symbol,
                position_usd=position_usd,
                twak_quote_small=twak_quote_small,
                twak_quote_normal=twak_quote_normal,
                max_slippage_pct=max_slippage_pct,
            )

    liquidity_analyzer_module.LiquidityAnalyzer = LiquidityAnalyzer


@dataclass(frozen=True)
class PreflightCheck:
    """Single live-readiness check result."""

    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class MinimumTradeDecision:
    """Daily minimum-trade compliance request."""

    symbol: str | None
    size_pct: float
    reason: str


@dataclass(frozen=True)
class EntryAttempt:
    """Result of a reconciled entry attempt."""

    entered: bool
    reason: str
    position_pct: float
    liquidity: Any | None
    reconcile_result: ReconciliationResult | None = None


def emergency_liquidate(
    position_manager: PositionManager,
    router: PancakeSwapRouter,
    guardrails: Guardrails,
) -> None:
    """Market-sell all process-local open positions back to USDC."""

    stable_symbol = guardrails.settings.default_stable_symbol
    for position in position_manager.list_open_positions():
        if position.symbol == stable_symbol:
            continue
        LOGGER.warning("Emergency liquidating %s", position.symbol)
        execution_slippage = _require_execution_slippage(guardrails.settings.max_slippage_pct)
        result = _execute_logged_swap(
            guardrails.settings,
            router,
            "emergency_liquidation",
            position.symbol,
            stable_symbol,
            position.amount_tokens,
            execution_slippage,
        )
        if not _execution_has_tx_hash(result):
            LOGGER.error(
                "Emergency liquidation for %s returned no tx hash; local position remains open",
                position.symbol,
            )
            continue
        position_manager.close_position(position.symbol)


def _maybe_flatten_for_window(
    settings: Settings,
    position_manager: PositionManager,
    router: PancakeSwapRouter,
    guardrails: Guardrails,
    now: datetime,
) -> bool:
    """Liquidate the whole book to USDC shortly before the competition deadline.

    Returns True when the flatten window is active (caller should also block new
    entries for the rest of the run). No-op when ``competition_end_utc`` is unset
    or unparseable, so default behaviour is unchanged.
    """

    end_iso = (getattr(settings, "competition_end_utc", "") or "").strip()
    if not end_iso:
        return False
    try:
        end_dt = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))
    except ValueError:
        LOGGER.warning("Invalid COMPETITION_END_UTC=%r; window flatten disabled", end_iso)
        return False
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)
    flatten_minutes = float(getattr(settings, "flatten_before_end_minutes", 30) or 0.0)
    if now < end_dt - timedelta(minutes=flatten_minutes):
        return False
    open_positions = position_manager.list_open_positions()
    if open_positions:
        LOGGER.warning(
            "Competition window flatten: liquidating %s open positions before deadline %s",
            len(open_positions),
            end_dt.isoformat(),
        )
        emergency_liquidate(position_manager, router, guardrails)
    return True


def print_balances(toolkit: BnbToolkitWrapper, settings: Settings) -> None:
    """Print the operator's key balances for preflight checks."""

    print(f"Trading wallet (BSC){_wallet_suffix(settings.wallet_address)}")
    symbols = ["BNB", settings.default_stable_symbol.upper(), "USDT"]
    seen: set[str] = set()
    for symbol in symbols:
        if symbol in seen:
            continue
        seen.add(symbol)
        balance = toolkit.get_balance(symbol)
        amount = balance.get("balance", balance.get("amount"))
        print(f"  {symbol}: {_number(amount):.8f}")
    _print_x402_wallet_section(settings)


def _wallet_suffix(address: str | None) -> str:
    value = (address or "").strip()
    return f" {_mask_address(value)}" if value else ""


def _print_x402_wallet_section(settings: Settings) -> None:
    """Print the x402 data-payment wallet (Base) balance and spend ledger."""

    try:
        from src.data.x402_wallet_view import fetch_x402_wallet_view
    except ImportError as exc:
        print(f"x402 data wallet (Base): unavailable ({exc})")
        return

    view = fetch_x402_wallet_view(base_rpc_url=settings.base_rpc_url)
    if view.address is None:
        print("x402 data wallet (Base): not configured (no payment key in env)")
        return
    print(f"x402 data wallet (Base) {_mask_address(view.address)}")
    if view.usdc_balance is not None:
        print(f"  USDC: {view.usdc_balance:.6f}")
    else:
        print(f"  USDC: read failed ({view.error or 'unknown error'})")

    try:
        from src.data.x402_spend_governor import X402SpendGovernor

        ledger = X402SpendGovernor(
            daily_budget_usdc=getattr(settings, "x402_daily_budget_usdc", 2.0),
            total_budget_usdc=getattr(settings, "x402_total_budget_usdc", 15.0),
            cost_per_call_usdc=settings.cmc_x402_amount,
            failure_cooldown_seconds=getattr(settings, "x402_failure_cooldown_seconds", 900),
        ).snapshot()
        print(
            "  spend today: ${daily:.2f}/${daily_cap:.2f} | window total: ${total:.2f}/${total_cap:.2f}".format(
                daily=float(ledger["daily_spend_usdc"]),
                daily_cap=float(ledger["daily_budget_usdc"]),
                total=float(ledger["total_spend_usdc"]),
                total_cap=float(ledger["total_budget_usdc"]),
            )
        )
    except Exception as exc:
        LOGGER.debug("x402 spend ledger unavailable: %s", exc)


def run_live_preflight(settings: Settings) -> bool:
    """Run live readiness checks without broadcasting transactions."""

    checks: list[PreflightCheck] = []

    def record(name: str, passed: bool, detail: str = "") -> None:
        checks.append(PreflightCheck(name=name, passed=passed, detail=detail))

    record("settings loaded", True, "ok")
    record(
        "settings live mode",
        settings.paper_trade is False,
        "PAPER_TRADE=false" if settings.paper_trade is False else "PAPER_TRADE=true",
    )

    configured_wallet = (settings.wallet_address or "").strip()
    record(
        "wallet address configured",
        bool(configured_wallet),
        _mask_address(configured_wallet) if configured_wallet else "missing",
    )

    twak_interface = _twak_interface_from_settings(settings, paper_trade=False)
    try:
        wallet_payload = twak_interface.wallet_address("bsc")
        twak_wallet = _extract_wallet_address(wallet_payload)
        wallet_matches = bool(configured_wallet and twak_wallet and _addresses_equal(configured_wallet, twak_wallet))
        if wallet_matches:
            wallet_detail = _mask_address(twak_wallet or "")
        elif twak_wallet:
            wallet_detail = f"returned {_mask_address(twak_wallet)}; expected {_mask_address(configured_wallet)}"
        else:
            wallet_detail = "no address returned"
        record("TWAK wallet unlock", wallet_matches, wallet_detail)
    except Exception as exc:  # pragma: no cover - exercised by CLI tests with fakes
        record("TWAK wallet unlock", False, _safe_error(exc))

    balances: dict[str, float] = {}
    try:
        toolkit = BnbToolkitWrapper(settings)
        for symbol in ("BNB", "USDC", "USDT"):
            balances[symbol] = _extract_symbol_balance(toolkit.get_balance(symbol), symbol)
        record("BSC balance read", True, "BNB, USDC, USDT")
    except Exception as exc:  # pragma: no cover - exercised by CLI tests with fakes
        record("BSC balance read", False, _safe_error(exc))

    record("BNB balance > 0", balances.get("BNB", 0.0) > 0, _balance_check_detail("BNB", balances))
    record("USDC balance > 0", balances.get("USDC", 0.0) > 0, _balance_check_detail("USDC", balances))

    try:
        quote = twak_interface.quote_swap("USDC", "BNB", PREFLIGHT_QUOTE_AMOUNT_USDC, 0.01)
        record("TWAK quote-only", bool(quote), _quote_check_detail(quote))
    except Exception as exc:  # pragma: no cover - exercised by CLI tests with fakes
        record("TWAK quote-only", False, _safe_error(exc))

    snapshot: dict[str, Any] = {}
    try:
        cmc_client = CMCMCPClient(settings)
        fetched_snapshot = cmc_client.fetch_market_snapshot(TARGET_SYMBOLS)
        if isinstance(fetched_snapshot, dict):
            snapshot = fetched_snapshot
            record("CMC x402 market snapshot", bool(snapshot), f"{len(snapshot)} item(s)")
        else:
            record("CMC x402 market snapshot", False, "non-dict snapshot")
    except Exception as exc:  # pragma: no cover - exercised by CLI tests with fakes
        record("CMC x402 market snapshot", False, _safe_error(exc))

    priced_targets = _priced_target_symbols(snapshot)
    record(
        "snapshot target price",
        bool(priced_targets),
        f"{len(priced_targets)} priced target(s)" if priced_targets else "none",
    )

    _print_preflight_report(checks)
    return all(check.passed for check in checks)


def withdraw_funds(
    toolkit: BnbToolkitWrapper,
    symbol: str,
    to_address: str,
    amount: float,
) -> None:
    """Transfer funds out of the configured agent wallet."""

    if amount <= 0:
        raise ValueError("withdraw amount must be greater than zero")
    if not _is_evm_address(to_address):
        raise ValueError("withdraw address must be a 0x-prefixed EVM address")

    result = toolkit.transfer(to_address, symbol, amount)
    tx_hash = result.get("tx_hash") or result.get("transaction_hash") or result.get("hash")
    if tx_hash:
        print(f"withdraw_tx_hash={tx_hash}")
    else:
        print(result)


def _is_evm_address(value: str) -> bool:
    return bool(re.fullmatch(r"0x[a-fA-F0-9]{40}", value.strip()))


def _addresses_equal(left: str, right: str) -> bool:
    return left.strip().lower() == right.strip().lower()


def _mask_address(address: str) -> str:
    value = (address or "").strip()
    if not value:
        return "missing"
    if len(value) <= 10:
        return value
    return f"{value[:6]}...{value[-4:]}"


def _extract_wallet_address(payload: Any) -> str | None:
    if isinstance(payload, str):
        match = re.search(r"0x[a-fA-F0-9]{40}", payload)
        return match.group(0) if match else None
    if isinstance(payload, dict):
        for key in ("address", "wallet_address", "walletAddress", "account", "account_address"):
            value = payload.get(key)
            if isinstance(value, str) and _is_evm_address(value):
                return value
        for value in payload.values():
            found = _extract_wallet_address(value)
            if found:
                return found
    if isinstance(payload, list):
        for value in payload:
            found = _extract_wallet_address(value)
            if found:
                return found
    return None


def _balance_check_detail(symbol: str, balances: dict[str, float]) -> str:
    if symbol not in balances:
        return "not read"
    return "available" if balances[symbol] > 0 else "zero"


def _quote_check_detail(quote: dict[str, Any]) -> str:
    if not quote:
        return "empty quote"
    if "--quote-only" in quote.get("command", []):
        return "quote-only command parsed"
    return "quote parsed"


def _priced_target_symbols(snapshot: dict[str, Any]) -> list[str]:
    priced: list[str] = []
    for key, value in snapshot.items():
        if not isinstance(value, dict):
            continue
        symbol = str(value.get("symbol") or key).upper()
        if symbol in {item.upper() for item in TARGET_SYMBOLS} and _maybe_number(value.get("price")) is not None:
            priced.append(symbol)
    return priced


def _safe_error(exc: Exception) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    message = re.sub(
        r"(?i)(password|secret|api[_-]?key|access[_-]?secret|token)=([^,\s]+)",
        r"\1=<redacted>",
        message,
    )
    return message[:180]


def _print_preflight_report(checks: list[PreflightCheck]) -> None:
    passed = all(check.passed for check in checks)
    print("Live preflight")
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        suffix = f" - {check.detail}" if check.detail else ""
        print(f"{status} {check.name}{suffix}")
    print(f"Preflight result: {'PASS' if passed else 'FAIL'}")


def _twak_interface_from_settings(settings: Settings, paper_trade: bool) -> Any:
    """Build TWAK interface and apply live swap retry settings."""

    twak_interface = TWAKInterface(paper_trade=paper_trade)
    try:
        twak_interface.approval_retry_max = settings.swap_approval_retry_max
        twak_interface.approval_retry_delay_seconds = settings.swap_approval_retry_delay_seconds
    except AttributeError:
        pass
    return twak_interface


def run_agent(settings: Settings, max_cycles: int | None = None) -> None:
    """Run the v2.5 live/paper trading loop."""

    run_id = f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    Path("logs").mkdir(parents=True, exist_ok=True)
    cmc_client = CMCMCPClient(settings)
    toolkit = BnbToolkitWrapper(settings)
    twak_interface = _twak_interface_from_settings(settings, paper_trade=settings.paper_trade)
    router = PancakeSwapRouter(twak_interface)
    price_cache = PriceCache(maxlen=getattr(settings, "price_cache_maxlen", 2880) or 2880)
    sentiment = SentimentTier1(
        cmc_keyless_base=settings.cmc_keyless_base_url,
        bsc_rpc_url=settings.bsc_rpc_url or "",
        cache_ttl_seconds=_sentiment_cache_ttl(settings),
    )
    regime_detector = RegimeDetector(price_cache, sentiment, settings)
    liquidity_analyzer = LiquidityAnalyzer()
    execution_reconciler = ExecutionReconciler(toolkit)
    strategy_bundle = create_strategy_bundle(settings, price_cache, twak_interface)
    position_manager = strategy_bundle.position_manager
    guardrails = strategy_bundle.guardrails
    scoring.evaluate_universe = strategy_bundle.evaluate_universe
    ml_bundle: Any | None = None
    if settings.ml_enabled:
        try:
            from src.ml.bundle import MLBundle

            ml_bundle = MLBundle.from_settings(settings)
            LOGGER.info("ML bundle loaded from %s", settings.ml_model_path)
        except Exception as exc:
            LOGGER.warning("ML bundle disabled due to load failure: %s", exc)
            ml_bundle = None
    shadow_logger = _build_shadow_logger(price_cache, settings)
    positions_loaded = position_manager.load_positions()
    needs_balance_reconstruction = not positions_loaded and not settings.paper_trade
    if positions_loaded:
        LOGGER.info("Loaded %s persisted open positions", len(position_manager.list_open_positions()))

    # Re-log the snapshot-cache restore here: the singleton loads at import
    # time, before logging is configured, so its own INFO line is dropped.
    if settings.use_dual_market_data and not settings.use_keyless_primary:
        restored_age = get_dual_market_snapshot_cache().x402_age_seconds()
        if restored_age is not None:
            LOGGER.info(
                "Restored persisted x402 snapshot at startup (age=%.0fs); no paid refresh until TTL expires",
                restored_age,
            )

    health_state, _health_server, pending_swap_cooldowns = deployment_startup(
        settings,
        position_manager=position_manager,
        toolkit=toolkit,
        ml_bundle=ml_bundle,
    )
    if pending_swap_cooldowns:
        LOGGER.warning("Pending swap cooldown symbols: %s", sorted(pending_swap_cooldowns))

    # RWEAL Phase 1: static, entry-only event gate. Built once; disabled by
    # default. from_settings() raises on a present-but-malformed events file so a
    # bad calendar fails fast at startup rather than silently going blind.
    event_filter: EventRiskFilter | None = None
    if settings.enable_rweal:
        event_filter = EventRiskFilter.from_settings(settings)
        LOGGER.info("RWEAL enabled (entry gate + manual halt file: %s)", settings.rweal_control_file)

    running = True
    cycles_completed = 0
    previous_risk_state: RiskState | None = None
    breakout_near_miss_cooldowns: dict[str, int] = {}
    # Rising-edge tracker so a mid-sleep manual halt re-evaluates promptly
    # without busy-looping the (expensive) main cycle while halted.
    _rweal_halt_was_active = False

    def _stop(_signum: int, _frame: Any) -> None:
        nonlocal running
        running = False

    def _interruptible_sleep() -> None:
        """Sleep one loop interval, waking every 1s. If the RWEAL manual halt
        file appears mid-sleep, break early so the next cycle blocks entries
        within seconds (default LOOP_SECONDS would otherwise lag up to 300s)."""

        nonlocal _rweal_halt_was_active
        sleep_until = time.monotonic() + settings.loop_seconds
        while running and time.monotonic() < sleep_until:
            if event_filter is not None and event_filter.manual_halt_active():
                if not _rweal_halt_was_active:
                    _rweal_halt_was_active = True
                    LOGGER.warning("RWEAL manual halt detected mid-sleep; re-evaluating now")
                    break
            else:
                _rweal_halt_was_active = False
            time.sleep(min(1.0, sleep_until - time.monotonic()))

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    while running:
        cycle_number = cycles_completed + 1
        breakout_near_miss_cooldowns = {
            symbol: until_cycle
            for symbol, until_cycle in breakout_near_miss_cooldowns.items()
            if until_cycle >= cycle_number
        }
        recent_near_miss_excludes = {
            symbol
            for symbol, until_cycle in breakout_near_miss_cooldowns.items()
            if until_cycle >= cycle_number
        }
        recent_near_miss_excludes.update(_breakout_recent_analysis_excludes_from_log(settings))
        now_utc = datetime.now(timezone.utc)
        open_positions = position_manager.list_open_positions()
        market_snapshot = _fetch_snapshot(
            settings,
            cmc_client,
            open_position_value_usdc=sum(
                float(getattr(position, "entry_value_usdc", 0.0) or 0.0)
                for position in open_positions
            ),
            position_symbols={position.symbol.upper() for position in open_positions},
        )
        _update_price_cache(price_cache, market_snapshot, now_utc)
        window_flatten_active = _maybe_flatten_for_window(
            settings, position_manager, router, guardrails, now_utc
        )
        if needs_balance_reconstruction:
            reconstructed = _reconstruct_positions_from_balances(
                position_manager,
                toolkit,
                settings,
                market_snapshot,
            )
            LOGGER.info("Reconstructed %s open positions from wallet balances", reconstructed)
            needs_balance_reconstruction = False
        portfolio_value = _portfolio_value_usdc(toolkit, settings, market_snapshot, position_manager)
        regime_result, sentiment_result = _detect_regime_with_sentiment_fallback(
            regime_detector,
            sentiment,
            market_snapshot,
            settings,
        )
        risk_decision = guardrails.evaluate(portfolio_value, regime_result)
        risk_state_changed = previous_risk_state != risk_decision.state
        previous_risk_state = risk_decision.state

        candidate: EntryCandidate | None = None
        liquidity: Any | None = None
        action = "WAIT"
        entry_position_pct = 0.0
        entries_allowed = _risk_allows_new_entries(guardrails, risk_decision, portfolio_value, settings)
        if window_flatten_active:
            entries_allowed = False
        entries_blocked_reason = None if entries_allowed else _entries_blocked_reason(
            guardrails,
            risk_decision,
            portfolio_value,
            settings,
        )
        if window_flatten_active:
            entries_blocked_reason = "competition_window_flatten"
        decision_reasons_pre: list[str] = []
        if entries_allowed and not disk_allows_entries(settings):
            entries_allowed = False
            entries_blocked_reason = "disk_guard_free_space_below_threshold"
            decision_reasons_pre.append("disk guard: free space below threshold")
        # RWEAL Phase 1 global gate. Manual halt = full stop (also suppresses the
        # daily-minimum compliance trade, below). Global event blackout blocks
        # discretionary entries but leaves the compliance backstop running.
        rweal_manual_halt = False
        if event_filter is not None:
            rweal_manual_halt = event_filter.manual_halt_active()
            if rweal_manual_halt:
                entries_allowed = False
                entries_blocked_reason = "rweal_manual_halt"
                decision_reasons_pre.append("RWEAL: manual trading halt active")
            else:
                _rweal_global = event_filter.global_blackout(now_utc)
                if _rweal_global:
                    entries_allowed = False
                    entries_blocked_reason = "rweal_event_blackout_global"
                    decision_reasons_pre.append(f"RWEAL: {_rweal_global}")
        decision_reasons = list(risk_decision.reasons) + decision_reasons_pre
        cycle_status = "ok"

        if risk_decision.state == RiskState.KILL_SWITCH:
            LOGGER.critical("Kill switch active. Liquidating.")
            action = "HALT"
            cycle_status = "kill switch"
            decision_reasons = decision_reasons or ["drawdown_kill_switch"]
            _write_v25_cycle_logs(
                settings,
                run_id,
                cycle_number,
                action,
                market_snapshot,
                portfolio_value,
                price_cache,
                regime_result,
                sentiment_result,
                risk_decision,
                position_manager,
                guardrails,
                candidate,
                liquidity,
                entry_position_pct,
                decision_reasons,
                risk_state_changed,
            )
            _log_legacy_cycle_from_v25(
                settings,
                cycle_number,
                market_snapshot,
                portfolio_value,
                candidate,
                entries_allowed=False,
                action="HALT",
                reason="drawdown kill switch",
                position_pct=entry_position_pct,
                liquidity=liquidity,
                position_count=len(position_manager.list_open_positions()),
                entries_blocked_reason="risk_state:kill_switch",
            )
            if position_manager.list_open_positions():
                emergency_liquidate(position_manager, router, guardrails)
            # Stay alive in capital-preservation mode instead of halting: the
            # competition requires at least one trade per UTC day, so a halted
            # agent would be disqualified on trade count even after surviving
            # the drawdown gate. Only the tiny compliance-swap backstop runs here
            # -- unless the operator has set the RWEAL manual halt, which is a
            # deliberate full stop that overrides the compliance backstop. Use a
            # live re-check so a halt set mid-cycle is honoured immediately.
            if not (
                rweal_manual_halt
                or (event_filter is not None and event_filter.manual_halt_active())
            ):
                _ensure_daily_minimum_trade(
                    settings,
                    router,
                    guardrails,
                    datetime.now(timezone.utc),
                    portfolio_value,
                    twak_interface=twak_interface,
                    liquidity_analyzer=liquidity_analyzer,
                    event_filter=event_filter,
                )
            if settings.demo_mode:
                _print_demo_cycle_summary(
                    cycle_number,
                    market_snapshot,
                    portfolio_value,
                    decision=None,
                    entries_allowed=False,
                    position_count=len(position_manager.list_open_positions()),
                    status=cycle_status,
                    settings=settings,
                )
            cycles_completed += 1
            if max_cycles is not None and cycles_completed >= max_cycles:
                LOGGER.info("Completed %s cycle(s); exiting", cycles_completed)
                break
            _interruptible_sleep()
            continue

        _process_position_exits(position_manager, router, guardrails, market_snapshot, portfolio_value, price_cache)
        _monitor_position_exits_if_needed(
            position_manager,
            router,
            guardrails,
            market_snapshot,
            portfolio_value,
            settings,
            price_cache,
        )

        if not entries_allowed:
            LOGGER.info("Risk state currently blocks new entries: %s", risk_decision.state.value)
            if risk_decision.allow_new_entries:
                decision_reasons = decision_reasons or ["daily trade limit reached"]
            else:
                decision_reasons = decision_reasons or [f"Risk state: {risk_decision.state.value}"]
        else:
            skip_entries = (
                settings.strategy_mode == "scalping"
                and len(position_manager.list_open_positions()) >= 1
            )
            if skip_entries:
                decision_reasons.append("Scalping mode: monitoring open position")
            else:
                if ml_bundle is not None:
                    try:
                        ml_bundle.refresh_ohlcv_if_stale()
                    except Exception as exc:
                        LOGGER.warning("ML OHLCV refresh failed: %s", exc)
                exclude_symbols = {position.symbol for position in position_manager.list_open_positions()}
                exclude_symbols.update(pending_swap_cooldowns)
                if settings.strategy_mode == "breakout":
                    exclude_symbols.update(recent_near_miss_excludes)
                # RWEAL Phase 1: exclude symbols in an active event blackout from
                # selection so a blacked-out top pick does not suppress otherwise
                # valid alternatives (symbol-specific events block only that
                # symbol, not the whole universe). GLOBAL/macro blackouts are
                # handled at the cycle-top gate, not here.
                rweal_blacked_out: set[str] = set()
                if event_filter is not None:
                    rweal_blacked_out = event_filter.active_symbol_blackouts(now_utc)
                    if rweal_blacked_out:
                        exclude_symbols.update(rweal_blacked_out)
                candidate = _evaluate_universe_v25(
                    market_snapshot,
                    portfolio_value,
                    regime_result,
                    risk_decision,
                    settings,
                    twak_interface,
                    exclude_symbols=exclude_symbols,
                    sentiment_result=sentiment_result,
                    ml_bundle=ml_bundle,
                )
                # Defensive backstop: drop any discretionary candidate that still
                # carries an active blackout (e.g. a path that bypassed excludes).
                if candidate is not None and event_filter is not None:
                    _rweal_symbol = event_filter.symbol_blackout(candidate.symbol, now_utc)
                    if _rweal_symbol:
                        LOGGER.warning("Entry blocked by RWEAL: %s", _rweal_symbol)
                        decision_reasons.append(f"RWEAL: {_rweal_symbol}")
                        candidate = None
                if candidate is None and settings.strategy_mode != "scalping":
                    minimum_trade = check_daily_minimum_compliance(
                        guardrails, regime_result, cycle_number, now_utc, settings
                    )
                    if minimum_trade is not None:
                        candidate = _minimum_trade_candidate(
                            minimum_trade,
                            market_snapshot,
                            portfolio_value,
                            settings,
                            risk_decision,
                        )
                        # A compliance trade should not be routed into a symbol
                        # facing a scheduled event; fall through to the fixed
                        # stable->token compliance swap instead.
                        if (
                            candidate is not None
                            and event_filter is not None
                            and event_filter.symbol_blackout(candidate.symbol, now_utc)
                        ):
                            LOGGER.warning(
                                "RWEAL: compliance candidate %s is blacked out; "
                                "falling back to fixed compliance swap",
                                candidate.symbol,
                            )
                            decision_reasons.append("RWEAL: compliance symbol blacked out")
                            candidate = None

            # RWEAL Phase 1: final, instant halt guard. Re-check the control file
            # immediately before execution so a TRADING_HALT that appears mid-cycle
            # (after the cycle-top gate) cannot still open a position this cycle.
            if (
                candidate is not None
                and event_filter is not None
                and event_filter.manual_halt_active()
            ):
                rweal_manual_halt = True
                LOGGER.warning("RWEAL manual halt detected pre-entry; skipping execution")
                decision_reasons.append("RWEAL: manual halt (pre-execution)")
                candidate = None
            if not skip_entries:
                if candidate is None:
                    decision_reasons.append("No candidate passed gates")
                else:
                    attempt = _attempt_entry_v25(
                        settings,
                        toolkit,
                        router,
                        execution_reconciler,
                        liquidity_analyzer,
                        position_manager,
                        guardrails,
                        price_cache,
                        regime_result,
                        risk_decision,
                        candidate,
                        portfolio_value,
                    )
                    liquidity = attempt.liquidity
                    entry_position_pct = attempt.position_pct
                    decision_reasons.extend([candidate.reason, attempt.reason])
                    if attempt.entered:
                        action = "ENTER"

        # Live halt re-check (not the cycle-top cache): this backstop runs at the
        # very end of the cycle, after all data work, so a halt set mid-cycle
        # must still suppress the compliance swap.
        rweal_halt_now = rweal_manual_halt or (
            event_filter is not None and event_filter.manual_halt_active()
        )
        if action != "ENTER" and not rweal_halt_now and _ensure_daily_minimum_trade(
            settings,
            router,
            guardrails,
            datetime.now(timezone.utc),
            portfolio_value,
            twak_interface=twak_interface,
            liquidity_analyzer=liquidity_analyzer,
            event_filter=event_filter,
        ):
            action = "ENTER"
            decision_reasons.append("compliance: daily minimum trade")

        if settings.demo_mode:
            demo_decision = _breakout_decision_from_candidate(
                candidate,
                action == "ENTER",
                entry_position_pct * portfolio_value,
                liquidity,
                decision_reasons[-1] if decision_reasons else "ok",
            )
            _print_demo_cycle_summary(
                cycle_number,
                market_snapshot,
                portfolio_value,
                demo_decision,
                entries_allowed,
                len(position_manager.list_open_positions()),
                status=cycle_status,
                settings=settings,
                entry_score=candidate.entry_score if candidate is not None else None,
            )

        _write_v25_cycle_logs(
            settings,
            run_id,
            cycle_number,
            action,
            market_snapshot,
            portfolio_value,
            price_cache,
            regime_result,
            sentiment_result,
            risk_decision,
            position_manager,
            guardrails,
            candidate,
            liquidity,
            entry_position_pct,
            decision_reasons,
            risk_state_changed,
        )
        open_symbols = {position.symbol for position in position_manager.list_open_positions()}
        telemetry_exclude_symbols = set(open_symbols)
        telemetry_exclude_symbols.update(pending_swap_cooldowns)
        if settings.strategy_mode == "breakout":
            telemetry_exclude_symbols.update(recent_near_miss_excludes)
        telemetry_candidate = _telemetry_candidate_for_log(
            settings,
            strategy_bundle,
            market_snapshot,
            portfolio_value,
            regime_result,
            risk_decision,
            twak_interface,
            telemetry_exclude_symbols,
            sentiment_result,
            candidate,
        )
        legacy_reason = decision_reasons[-1] if decision_reasons else "ok"
        if candidate is None and telemetry_candidate is not None:
            legacy_reason = telemetry_candidate.reason
        _log_legacy_cycle_from_v25(
            settings,
            cycle_number,
            market_snapshot,
            portfolio_value,
            telemetry_candidate,
            entries_allowed=entries_allowed,
            action="ENTER" if action == "ENTER" else ("WAIT" if entries_allowed else "BLOCKED"),
            reason=legacy_reason,
            position_pct=entry_position_pct,
            liquidity=liquidity,
            position_count=len(position_manager.list_open_positions()),
            entries_blocked_reason=entries_blocked_reason,
        )
        _update_breakout_near_miss_cooldowns(
            settings,
            cycle_number,
            action,
            telemetry_candidate,
            breakout_near_miss_cooldowns,
        )
        if shadow_logger is not None:
            try:
                shadow_logger.log_all_variants(cycle_number, market_snapshot, regime_result)
            except Exception as exc:
                LOGGER.warning("Shadow logging failed: %s", exc)

        _log_live_window_warning(guardrails)
        update_health_snapshot(
            health_state,
            guardrails=guardrails,
            portfolio_value=portfolio_value,
            position_manager=position_manager,
            ml_bundle=ml_bundle,
        )
        cycles_completed += 1
        if max_cycles is not None and cycles_completed >= max_cycles:
            LOGGER.info("Completed %s cycle(s); exiting", cycles_completed)
            break

        _interruptible_sleep()


def _sentiment_cache_ttl(settings: Settings) -> int:
    return int(
        getattr(
            settings,
            "sentiment_cache_ttl",
            getattr(settings, "sentiment_cache_ttl_seconds", 300),
        )
        or 300
    )


def _build_shadow_logger(price_cache: PriceCache, settings: Settings) -> Any | None:
    try:
        from src.research.shadow_decisions import ShadowDecisionsLogger
        from src.strategy.jump_model_detector import JumpModelDetector

        return ShadowDecisionsLogger(
            jump_model=JumpModelDetector(price_cache),
            settings=settings,
            decision_log_path="logs/decision_shadow.jsonl",
        )
    except ImportError:
        return None


def _detect_regime_with_sentiment_fallback(
    regime_detector: RegimeDetector,
    sentiment: SentimentTier1,
    snapshot: dict[str, dict[str, Any]],
    settings: Settings,
) -> tuple[RegimeResult, SentimentResult]:
    try:
        regime_result = regime_detector.detect(snapshot)
    except Exception as exc:
        LOGGER.warning("Regime detection failed; using neutral fallback: %s", exc)
        sentiment_result = _neutral_sentiment_result()
        return _fallback_regime_result(snapshot, settings, sentiment_result), sentiment_result
    try:
        sentiment_result = sentiment.compute_sentiment()
    except Exception as exc:
        LOGGER.warning("Sentiment logging failed; using neutral fallback: %s", exc)
        sentiment_result = SentimentResult(
            fear_greed_index=None,
            fear_greed_classification=None,
            funding_rate_btc=None,
            open_interest_btc=None,
            gas_price_gwei=None,
            gas_avg_24h_gwei=None,
            sentiment_delta=regime_result.sentiment_delta,
            regime_fragility=regime_result.sentiment_fragility,
        )
    return regime_result, sentiment_result


def _neutral_sentiment_result() -> SentimentResult:
    return SentimentResult(
        fear_greed_index=None,
        fear_greed_classification=None,
        funding_rate_btc=None,
        open_interest_btc=None,
        gas_price_gwei=None,
        gas_avg_24h_gwei=None,
        sentiment_delta=0.0,
        regime_fragility="NONE",
    )


def _fallback_regime_result(
    snapshot: dict[str, dict[str, Any]],
    settings: Settings,
    sentiment_result: SentimentResult,
) -> RegimeResult:
    bnb = snapshot.get("BNB", {})
    positive_count = sum(
        1
        for key in ("percent_change_1h", "percent_change_6h", "percent_change_24h")
        if _number(bnb.get(key), 0.0) > 0
    )
    if positive_count >= 2:
        regime = MarketRegime.RANGING
        score = 1.0
        position_multiplier = 0.5
        max_slippage = min(settings.max_slippage_pct, 0.0075)
    else:
        regime = MarketRegime.RISK_OFF
        score = 0.0
        position_multiplier = 0.1
        max_slippage = min(settings.max_slippage_pct, 0.005)
    return RegimeResult(
        regime=regime,
        score=score,
        reasons=["regime_detection_fallback"],
        position_multiplier=position_multiplier,
        min_entry_factors=5,
        max_slippage_pct=max_slippage,
        sentiment_delta=sentiment_result.sentiment_delta,
        sentiment_fragility=sentiment_result.regime_fragility,
    )


def _update_price_cache(
    price_cache: PriceCache,
    snapshot: dict[str, dict[str, Any]],
    timestamp: datetime,
) -> None:
    for symbol, data in snapshot.items():
        if not isinstance(data, dict):
            continue
        price = _maybe_number(data.get("price"))
        if price is None:
            continue
        high = _first_market_number(data, ("high_24h", "high_6h", "high_3h"), price)
        low = _first_market_number(data, ("low_24h", "low_6h", "low_3h"), price)
        open_price = _first_market_number(data, ("open_24h", "open", "open_price"), price)
        volume = _first_market_number(data, ("volume_24h", "volume"), 0.0)
        price_cache.add_ohlcv(
            symbol=symbol,
            open_price=open_price,
            high=high,
            low=low,
            close=price,
            volume=volume,
            timestamp=timestamp,
        )


def _risk_allows_new_entries(
    guardrails: Guardrails,
    risk_decision: RiskDecision,
    portfolio_value: float,
    settings: Settings,
) -> bool:
    if not risk_decision.allow_new_entries:
        return False
    if settings.strategy_mode == "scalping" and isinstance(guardrails, ScalpingGuardrails):
        return guardrails.scalping_entries_allowed(portfolio_value)
    daily_count = int(getattr(guardrails, "_daily_trade_count", 0))
    return daily_count < risk_decision.max_daily_trades


def _entries_blocked_reason(
    guardrails: Guardrails,
    risk_decision: RiskDecision,
    portfolio_value: float,
    settings: Settings,
) -> str | None:
    """Return a stable reason code when new entries are globally blocked."""

    if not risk_decision.allow_new_entries:
        return f"risk_state:{risk_decision.state.value}"
    if settings.strategy_mode == "scalping" and isinstance(guardrails, ScalpingGuardrails):
        if not guardrails.scalping_entries_allowed(portfolio_value):
            return "scalping_guardrails"
    daily_count = int(getattr(guardrails, "_daily_trade_count", 0))
    if daily_count >= risk_decision.max_daily_trades:
        if risk_decision.state == RiskState.REDUCED_RISK:
            return "reduced_risk_daily_trade_limit"
        return "daily_trade_limit"
    return None


def _breakout_recent_analysis_excludes_from_log(settings: Settings) -> set[str]:
    """Symbols from recent non-entry breakout decisions, persisted across restarts."""

    if settings.strategy_mode != "breakout":
        return set()

    cooldown_cycles = max(0, int(getattr(settings, "breakout_near_miss_cooldown_cycles", 1) or 0))
    if cooldown_cycles <= 0:
        return set()

    path = Path(settings.decision_log_path)
    if not path.exists():
        return set()

    lines: deque[str] = deque(maxlen=cooldown_cycles)
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    lines.append(line)
    except OSError as exc:
        LOGGER.warning("Could not read breakout recent-analysis cooldown log: %s", exc)
        return set()

    excludes: set[str] = set()
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        symbol = _breakout_cooldown_symbol_from_record(record)
        if symbol is not None:
            excludes.add(symbol)
    return excludes


def _breakout_cooldown_symbol_from_record(record: dict[str, Any]) -> str | None:
    strategy_mode = str(record.get("strategy_mode") or "breakout").lower()
    if strategy_mode != "breakout":
        return None

    action = str(record.get("action") or "").upper()
    if action == "ENTER":
        return None

    symbol = str(record.get("symbol") or "").upper()
    return symbol or None


def _update_breakout_near_miss_cooldowns(
    settings: Settings,
    cycle_number: int,
    action: str,
    telemetry_candidate: EntryCandidate | None,
    cooldowns: dict[str, int],
) -> None:
    """Temporarily rotate away from non-entry breakout telemetry symbols."""

    if settings.strategy_mode != "breakout" or telemetry_candidate is None:
        return

    symbol = (telemetry_candidate.symbol or "").upper()
    if not symbol:
        return

    if action == "ENTER":
        cooldowns.pop(symbol, None)
        return

    cooldown_cycles = max(0, int(getattr(settings, "breakout_near_miss_cooldown_cycles", 1) or 0))
    if cooldown_cycles <= 0:
        cooldowns.pop(symbol, None)
        return

    cooldowns[symbol] = cycle_number + cooldown_cycles


def _evaluate_universe_v25(
    snapshot: dict[str, dict[str, Any]],
    portfolio_value: float,
    regime_result: RegimeResult,
    risk_decision: RiskDecision,
    settings: Settings,
    twak_interface: TWAKInterface | None = None,
    exclude_symbols: set[str] | None = None,
    sentiment_result: SentimentResult | None = None,
    ml_bundle: Any | None = None,
) -> EntryCandidate | None:
    evaluate = getattr(scoring, "evaluate_universe", None)
    if evaluate is not None and evaluate is not fallback_evaluate_universe:
        try:
            candidate = evaluate(
                snapshot,
                portfolio_value,
                regime_result,
                risk_decision,
                settings=settings,
                twak_interface=twak_interface,
                exclude_symbols=exclude_symbols or set(),
                sentiment_result=sentiment_result,
                ml_bundle=ml_bundle,
            )
        except TypeError:
            try:
                candidate = evaluate(
                    snapshot,
                    portfolio_value,
                    regime_result,
                    risk_decision,
                    settings=settings,
                    twak_interface=twak_interface,
                    exclude_symbols=exclude_symbols or set(),
                    ml_bundle=ml_bundle,
                )
            except TypeError:
                try:
                    candidate = evaluate(
                        snapshot,
                        portfolio_value,
                        regime_result,
                        risk_decision,
                        settings=settings,
                        twak_interface=twak_interface,
                        exclude_symbols=exclude_symbols or set(),
                    )
                except TypeError:
                    candidate = evaluate(snapshot, portfolio_value, regime_result, risk_decision)
        return coerce_entry_candidate(candidate, portfolio_value, settings, risk_decision)
    return fallback_evaluate_universe(
        snapshot,
        portfolio_value,
        regime_result,
        risk_decision,
        settings=settings,
        twak_interface=twak_interface,
        exclude_symbols=exclude_symbols or set(),
    )


def check_daily_minimum_compliance(
    guardrails: Guardrails,
    regime_result: RegimeResult,
    cycle_id: int,
    now_utc: datetime,
    settings: Settings,
) -> MinimumTradeDecision | None:
    """Return a small forced-entry request near UTC day-end when no trade happened."""

    del cycle_id
    if int(getattr(guardrails, "_daily_trade_count", 0)) >= 1:
        return None
    if now_utc.hour < COMPLIANCE_TRIGGER_HOUR_UTC:
        return None
    if regime_result.regime == MarketRegime.RISK_OFF:
        return MinimumTradeDecision(
            symbol=None,
            size_pct=min(0.005, settings.max_position_pct),
            reason="daily_minimum_compliance_risk_off",
        )
    return MinimumTradeDecision(
        symbol=None,
        size_pct=min(0.01, settings.max_position_pct),
        reason="daily_minimum_compliance",
    )


def _minimum_trade_candidate(
    decision: MinimumTradeDecision,
    snapshot: dict[str, dict[str, Any]],
    portfolio_value: float,
    settings: Settings,
    risk_decision: RiskDecision,
) -> EntryCandidate | None:
    ranked_symbols: list[tuple[float, str, dict[str, Any]]] = []
    for symbol, data in snapshot.items():
        normalized = symbol.upper()
        if decision.symbol is not None and normalized != decision.symbol.upper():
            continue
        payload = {"symbol": normalized, **data}
        if not is_momentum_candidate_symbol(normalized) or not has_verified_bsc_contract(normalized) or not is_liquid(payload):
            continue
        price = _maybe_number(payload.get("price"))
        if price is None or price <= 0:
            continue
        ranked_symbols.append((_first_market_number(payload, ("volume_24h", "market_cap"), 0.0), normalized, payload))
    if not ranked_symbols:
        return None
    ranked_symbols.sort(reverse=True)
    _, symbol, data = ranked_symbols[0]
    price = float(data["price"])
    position_usd = portfolio_value * decision.size_pct * max(0.0, risk_decision.position_multiplier)
    return EntryCandidate(
        symbol=symbol,
        price=price,
        position_size_usdc=position_usd,
        expected_amount_out=_decimal_div(position_usd, price),
        slippage_small=_maybe_number(data.get("estimated_slippage_small_pct")),
        slippage_normal=_maybe_number(data.get("estimated_slippage_pct")),
        reason=decision.reason,
        factor_scores={"daily_minimum": True},
        true_factor_count=1,
        source="daily_minimum",
    )


def _attempt_entry_v25(
    settings: Settings,
    toolkit: BnbToolkitWrapper,
    router: PancakeSwapRouter,
    execution_reconciler: ExecutionReconciler,
    liquidity_analyzer: LiquidityAnalyzer,
    position_manager: PositionManager,
    guardrails: Guardrails,
    price_cache: PriceCache,
    regime_result: RegimeResult,
    risk_decision: RiskDecision,
    candidate: EntryCandidate,
    portfolio_value: float,
) -> EntryAttempt:
    if position_manager.get_position(candidate.symbol) is not None:
        return EntryAttempt(False, "position already open", 0.0, None)

    liquidity = liquidity_analyzer.analyze_liquidity(
        symbol=candidate.symbol,
        position_usd=candidate.position_size_usdc,
        twak_quote_small=candidate.slippage_small,
        twak_quote_normal=candidate.slippage_normal,
        max_slippage_pct=risk_decision.max_slippage_pct,
    )
    if getattr(liquidity, "recommendation", "") == "REJECT":
        return EntryAttempt(False, f"Liquidity: {liquidity.recommendation}", 0.0, liquidity)

    atr_pct = price_cache.get_atr_pct(candidate.symbol, 14)
    if settings.strategy_mode == "scalping":
        position_usd = candidate.position_size_usdc
        position_pct = position_usd / portfolio_value if portfolio_value > 0 else 0.0
    else:
        ml_multiplier = 1.0
        if getattr(candidate, "ml_context", None) is not None:
            ml_multiplier = float(getattr(candidate.ml_context, "position_size_multiplier", 1.0))
        position_pct = calculate_position_pct(
            equity_usd=portfolio_value,
            atr_pct=atr_pct,
            regime_multiplier=regime_result.position_multiplier * ml_multiplier,
            risk_state_multiplier=risk_decision.position_multiplier,
            loss_streak=int(getattr(guardrails, "_loss_streak", 0)),
            max_position_pct=settings.max_position_pct,
            base_risk_per_trade_pct=settings.base_risk_per_trade_pct,
        )
        if getattr(liquidity, "recommendation", "") == "REDUCE_SIZE":
            position_pct *= 0.5
        position_pct *= max(0.0, float(getattr(candidate, "position_size_multiplier", 1.0) or 1.0))
        position_usd = portfolio_value * position_pct
    if candidate.factor_scores.get("regime_not_risk_off") is False:
        position_pct *= 0.5
        position_usd = portfolio_value * position_pct
    capped_position_usd = _cap_spend_to_portfolio_floor(position_usd, portfolio_value)
    if capped_position_usd < position_usd:
        LOGGER.warning(
            "Reducing %s entry from $%.2f to $%.2f to preserve $%.2f portfolio floor",
            candidate.symbol,
            position_usd,
            capped_position_usd,
            MIN_PORTFOLIO_RETAINED_USDC,
        )
        position_usd = capped_position_usd
        position_pct = position_usd / portfolio_value if portfolio_value > 0 else 0.0
    if position_usd <= 0:
        return EntryAttempt(False, "portfolio floor prevents spend", position_pct, liquidity)

    expected_amount_out = _decimal_div(position_usd, candidate.price)
    balance_before = _balance_before_for_reconciliation(toolkit, candidate.symbol)
    try:
        swap_result = _execute_logged_swap(
            settings,
            router,
            "entry",
            settings.default_stable_symbol,
            candidate.symbol,
            position_usd,
            risk_decision.max_slippage_pct,
            expected_amount_out=float(expected_amount_out),
        )
    except Exception as exc:
        return EntryAttempt(False, f"swap failed: {exc}", position_pct, liquidity)

    reconciled_tx = _tx_for_reconciliation(
        swap_result,
        candidate.symbol,
        expected_amount_out,
        balance_before,
        settings.paper_trade,
    )
    reconcile_result = execution_reconciler.reconcile(
        tx_result=reconciled_tx,
        expected_amount_out=expected_amount_out,
        slippage_tolerance=Decimal(str(risk_decision.max_slippage_pct)),
        balance_before=balance_before,
    )
    if reconcile_result.status != "SUCCESS":
        LOGGER.error("Execution failed for %s: %s", candidate.symbol, reconcile_result.status)
        return EntryAttempt(False, f"Execution failed: {reconcile_result.status}", position_pct, liquidity, reconcile_result)

    amount_out = float(reconcile_result.amount_out_actual)
    entry_price = candidate.price
    if amount_out > 0:
        entry_price = position_usd / amount_out

    _open_local_position_v25(
        position_manager,
        candidate.symbol,
        amount_out,
        entry_price,
        position_usd,
        atr_pct,
        regime_result.regime,
    )
    guardrails.record_trade(
        TradeRecord(
            symbol=candidate.symbol,
            side="buy",
            value_usdc=position_usd,
            realized_pnl_usdc=0.0,
            timestamp=datetime.now(timezone.utc),
        ),
        portfolio_value,
    )
    guardrails.record_trade_result(realized_pnl_pct=0.0)
    return EntryAttempt(True, "reconcile success", position_pct, liquidity, reconcile_result)


def _open_local_position_v25(
    position_manager: PositionManager,
    symbol: str,
    amount_tokens: float,
    entry_price: float,
    position_usd: float,
    atr_pct: float | None,
    regime: MarketRegime,
) -> None:
    # Use the volatility-aware signature when the manager supports it; fall back
    # to the legacy 4-arg form only for an older PositionManager. Checking the
    # signature explicitly (instead of catching TypeError) avoids masking a
    # genuine TypeError raised inside open_position.
    open_params = inspect.signature(position_manager.open_position).parameters
    if "atr_pct" in open_params and "regime" in open_params:
        position_manager.open_position(
            symbol=symbol,
            amount_tokens=amount_tokens,
            entry_price=entry_price,
            position_usd=position_usd,
            atr_pct=atr_pct,
            regime=regime,
        )
    else:
        position_manager.open_position(symbol, amount_tokens, entry_price, position_usd)


def _monitor_position_exits_if_needed(
    position_manager: PositionManager,
    router: PancakeSwapRouter,
    guardrails: Guardrails,
    market_snapshot: dict[str, dict[str, Any]],
    portfolio_value: float,
    settings: Settings,
    price_cache: PriceCache | None = None,
) -> None:
    if not position_manager.list_open_positions():
        return
    last_exit_check = float(getattr(run_agent, "_last_exit_check", 0.0))
    if time.time() - last_exit_check > getattr(settings, "position_monitor_seconds", 60):
        _process_position_exits(position_manager, router, guardrails, market_snapshot, portfolio_value, price_cache)
        setattr(run_agent, "_last_exit_check", time.time())


def _compute_expected_breakeven_pct(
    estimated_slippage_pct: float | None,
    gas_price_gwei: float | None,
    bnb_price_usd: float | None,
    position_size_usd: float,
    swap_fee_pct: float = 0.0025,
) -> float | None:
    """Estimate round-trip cost floor: slippage + gas (as pct of size) + swap fee."""

    try:
        total = swap_fee_pct
        if estimated_slippage_pct is not None:
            total += estimated_slippage_pct
        if (
            gas_price_gwei is not None
            and bnb_price_usd is not None
            and position_size_usd > 0
        ):
            gas_cost_usd = gas_price_gwei * 21000 * 1e-9 * bnb_price_usd
            total += gas_cost_usd / position_size_usd
        return total if total > 0 else None
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _ml_fields_from_candidate(candidate: EntryCandidate | None) -> dict[str, Any]:
    if candidate is None:
        return {}
    fields: dict[str, Any] = {}
    ctx = getattr(candidate, "ml_context", None)
    if ctx is not None:
        fields["ml_regime"] = getattr(ctx, "regime", None)
        fields["ml_confidence"] = getattr(ctx, "confidence", None)
    ml_ranking = getattr(candidate, "ml_ranking", None)
    if ml_ranking is not None:
        fields["ml_ranking"] = ml_ranking
        if isinstance(ml_ranking, dict):
            if "ml_active" in ml_ranking:
                fields["ml_active"] = ml_ranking.get("ml_active")
            if "ml_selected_symbol" in ml_ranking:
                fields["ml_selected_symbol"] = ml_ranking.get("ml_selected_symbol")
            if "executed_symbol" in ml_ranking:
                fields["executed_symbol"] = ml_ranking.get("executed_symbol")
            if "ml_scores" in ml_ranking:
                fields["ml_scores"] = ml_ranking.get("ml_scores")
    return fields


def _write_v25_cycle_logs(
    settings: Settings,
    run_id: str,
    cycle_id: int,
    action: str,
    snapshot: dict[str, dict[str, Any]],
    portfolio_value: float,
    price_cache: PriceCache,
    regime_result: RegimeResult,
    sentiment_result: SentimentResult,
    risk_decision: RiskDecision,
    position_manager: PositionManager,
    guardrails: Guardrails,
    candidate: EntryCandidate | None,
    liquidity: Any | None,
    position_pct: float,
    reasons: list[str],
    risk_state_changed: bool,
) -> None:
    mode = "paper" if settings.paper_trade else "live"
    symbol = candidate.symbol if candidate else None
    exit_meta = getattr(_execute_position_exit, "_last_exit_meta", None)
    estimated_slippage_pct = (
        getattr(liquidity, "slippage_normal", None)
        if liquidity is not None
        else (candidate.slippage_normal if candidate is not None else None)
    )
    position_size_usd = position_pct * portfolio_value
    bnb_price_usd = _maybe_number(snapshot.get("BNB", {}).get("price"))
    expected_breakeven_pct = _compute_expected_breakeven_pct(
        estimated_slippage_pct=estimated_slippage_pct,
        gas_price_gwei=sentiment_result.gas_price_gwei,
        bnb_price_usd=bnb_price_usd,
        position_size_usd=position_size_usd if position_size_usd > 0 else portfolio_value,
    )
    ml_fields = _ml_fields_from_candidate(candidate)
    append_to_file(
        "logs/decision_live.jsonl",
        LiveDecisionLog(
            schema_version=SCHEMA_VERSION,
            run_id=run_id,
            mode=mode,
            path="live",
            timestamp=datetime.now(timezone.utc).isoformat(),
            cycle_id=cycle_id,
            action=action,
            symbol=symbol,
            size_pct=position_pct,
            reasons=[reason for reason in reasons if reason],
            regime=regime_result.regime.value,
            regime_score=regime_result.score,
            ema_72=price_cache.get_ema("BNB", 72),
            ema_144=price_cache.get_ema("BNB", 144),
            ema_288=price_cache.get_ema("BNB", 288),
            atr_pct=price_cache.get_atr_pct(symbol, 14) if symbol else None,
            position_pct=position_pct,
            slippage_quote=getattr(liquidity, "slippage_normal", None) if liquidity is not None else None,
            risk_state=risk_decision.state.value,
            sentiment_delta=regime_result.sentiment_delta,
            sentiment_fragility=regime_result.sentiment_fragility,
            strategy_mode=settings.strategy_mode,
            entry_score=candidate.entry_score if candidate else None,
            hold_time_seconds=exit_meta.get("hold_time_seconds") if exit_meta else None,
            exit_reason=exit_meta.get("exit_reason") if exit_meta else None,
            expected_breakeven_pct=expected_breakeven_pct,
            ml_regime=ml_fields.get("ml_regime"),
            ml_confidence=ml_fields.get("ml_confidence"),
            ml_ranking=ml_fields.get("ml_ranking"),
        ),
    )
    if exit_meta is not None:
        setattr(_execute_position_exit, "_last_exit_meta", None)
    append_to_file(
        "logs/sentiment_live.jsonl",
        SentimentLiveLog(
            schema_version=SCHEMA_VERSION,
            run_id=run_id,
            mode=mode,
            path="live",
            timestamp=datetime.now(timezone.utc).isoformat(),
            cycle_id=cycle_id,
            fear_greed_index=sentiment_result.fear_greed_index,
            fear_greed_classification=sentiment_result.fear_greed_classification,
            funding_rate_btc=sentiment_result.funding_rate_btc,
            open_interest_btc=sentiment_result.open_interest_btc,
            gas_price_gwei=sentiment_result.gas_price_gwei,
            gas_avg_24h_gwei=sentiment_result.gas_avg_24h_gwei,
            sentiment_delta=sentiment_result.sentiment_delta,
            regime_fragility=sentiment_result.regime_fragility,
        ),
    )
    if risk_state_changed:
        append_to_file(
            "logs/risk_events.jsonl",
            RiskEventLog(
                schema_version=SCHEMA_VERSION,
                run_id=run_id,
                mode=mode,
                path="live",
                timestamp=datetime.now(timezone.utc).isoformat(),
                cycle_id=cycle_id,
                event_type=risk_decision.state.value,
                severity="CRITICAL" if risk_decision.state == RiskState.KILL_SWITCH else "WARNING",
                details={"reasons": risk_decision.reasons, "portfolio_value": portfolio_value},
            ),
        )
    all_time_high = _guardrail_all_time_high(guardrails)
    append_to_file(
        "logs/portfolio_snapshots.jsonl",
        PortfolioSnapshotLog(
            schema_version=SCHEMA_VERSION,
            run_id=run_id,
            mode=mode,
            path="live",
            timestamp=datetime.now(timezone.utc).isoformat(),
            cycle_id=cycle_id,
            portfolio_value_usdc=portfolio_value,
            all_time_high=all_time_high,
            drawdown_pct=(all_time_high - portfolio_value) / all_time_high if all_time_high > 0 else 0.0,
            open_positions=_open_positions_payload(position_manager),
        ),
    )


def _guardrail_all_time_high(guardrails: Guardrails) -> float:
    return float(getattr(guardrails, "_all_time_high_usdc", getattr(guardrails, "_all_time_high", 0.0)))


def _telemetry_candidate_for_log(
    settings: Settings,
    strategy_bundle: Any,
    market_snapshot: dict[str, dict[str, Any]],
    portfolio_value: float,
    regime_result: RegimeResult,
    risk_decision: RiskDecision,
    twak_interface: TWAKInterface,
    exclude_symbols: set[str],
    sentiment_result: SentimentResult | None,
    selected: EntryCandidate | None,
) -> EntryCandidate | None:
    """Return the best evaluated symbol for dashboard telemetry when no entry triggers."""

    if selected is not None:
        return selected

    if settings.strategy_mode == "scalping" and strategy_bundle.scalping_engine is not None:
        cooldown_checker = getattr(strategy_bundle.position_manager, "is_symbol_on_cooldown", None)
        near_miss = strategy_bundle.scalping_engine.best_near_miss(
            market_snapshot,
            portfolio_value,
            regime_result,
            risk_decision,
            sentiment_result=sentiment_result,
            exclude_symbols=exclude_symbols,
            cooldown_checker=cooldown_checker,
        )
        if near_miss is not None:
            return near_miss
        return _telemetry_candidate_from_priced_targets(
            settings,
            market_snapshot,
            portfolio_value,
            regime_result,
            risk_decision,
            sentiment_result,
            strategy_bundle,
        )

    engine = BreakoutEngine(settings, twak_interface)
    filtered_snapshot = {
        symbol: data
        for symbol, data in market_snapshot.items()
        if symbol.upper() not in {item.upper() for item in exclude_symbols}
    }
    decision = engine.evaluate_universe(filtered_snapshot, portfolio_value)
    telemetry = breakout_decision_to_candidate(
        decision,
        market_snapshot,
        portfolio_value,
        settings,
        risk_decision,
        for_telemetry=True,
    )
    if telemetry is not None:
        return telemetry

    return fallback_best_near_miss(
        market_snapshot,
        portfolio_value,
        regime_result,
        risk_decision,
        settings=settings,
        exclude_symbols=exclude_symbols,
    )


def _telemetry_candidate_from_priced_targets(
    settings: Settings,
    market_snapshot: dict[str, dict[str, Any]],
    portfolio_value: float,
    regime_result: RegimeResult,
    risk_decision: RiskDecision,
    sentiment_result: SentimentResult | None,
    strategy_bundle: Any,
) -> EntryCandidate | None:
    """Last-resort telemetry: score the highest-volume priced tradable symbol."""

    if settings.strategy_mode != "scalping" or strategy_bundle.scalping_engine is None:
        return None

    ranked: list[tuple[float, EntryCandidate]] = []
    for symbol in _priced_target_symbols(market_snapshot):
        if not is_momentum_candidate_symbol(symbol):
            continue
        data = market_snapshot.get(symbol, {})
        if not isinstance(data, dict):
            continue
        candidate = strategy_bundle.scalping_engine._score_symbol_for_telemetry(
            symbol,
            {"symbol": symbol, **data},
            portfolio_value,
            regime_result,
            risk_decision,
            sentiment_result,
        )
        if candidate is None:
            continue
        volume = _first_market_number(data, ("volume_24h", "market_cap"), 0.0)
        ranked.append((volume, candidate))

    if not ranked:
        return None

    ranked.sort(key=lambda item: item[0], reverse=True)
    best = ranked[0][1]
    score = best.entry_score or 0.0
    minimum = settings.scalping_entry_score_min
    if score >= minimum:
        return best
    return EntryCandidate(
        symbol=best.symbol,
        price=best.price,
        position_size_usdc=0.0,
        expected_amount_out=best.expected_amount_out,
        slippage_small=best.slippage_small,
        slippage_normal=best.slippage_normal,
        reason=f"best {best.symbol} scalping score {score:.0f}/100 < {minimum:.0f}",
        factor_scores=best.factor_scores,
        true_factor_count=best.true_factor_count,
        source=best.source,
        entry_score=score,
        strategy_mode=best.strategy_mode,
    )


def _log_legacy_cycle_from_v25(
    settings: Settings,
    cycle_number: int,
    market_snapshot: dict[str, dict[str, Any]],
    portfolio_value: float,
    candidate: EntryCandidate | None,
    entries_allowed: bool,
    action: DecisionAction,
    reason: str,
    position_pct: float,
    liquidity: Any | None,
    position_count: int,
    entries_blocked_reason: str | None = None,
) -> None:
    decision = _breakout_decision_from_candidate(
        candidate,
        action == "ENTER",
        portfolio_value * position_pct,
        liquidity,
        reason,
    )
    exit_meta = getattr(_execute_position_exit, "_last_exit_meta", None)
    _log_cycle_decision(
        settings,
        cycle_number,
        market_snapshot,
        portfolio_value,
        decision,
        entries_allowed,
        position_count,
        action=action,
        reason=reason,
        strategy_mode=settings.strategy_mode,
        entry_score=candidate.entry_score if candidate else None,
        entries_blocked_reason=entries_blocked_reason,
        exit_reason=exit_meta.get("exit_reason") if exit_meta else None,
        hold_time_seconds=exit_meta.get("hold_time_seconds") if exit_meta else None,
        ml_regime=_ml_fields_from_candidate(candidate).get("ml_regime"),
        ml_confidence=_ml_fields_from_candidate(candidate).get("ml_confidence"),
        ml_ranking=_ml_fields_from_candidate(candidate).get("ml_ranking"),
        ml_active=_ml_fields_from_candidate(candidate).get("ml_active"),
        ml_selected_symbol=_ml_fields_from_candidate(candidate).get("ml_selected_symbol"),
        executed_symbol=_ml_fields_from_candidate(candidate).get("executed_symbol"),
        ml_scores=_ml_fields_from_candidate(candidate).get("ml_scores"),
    )


def _breakout_decision_from_candidate(
    candidate: EntryCandidate | None,
    should_enter: bool,
    position_size_usdc: float,
    liquidity: Any | None,
    reason: str,
) -> BreakoutDecision | None:
    if candidate is None:
        return None
    return BreakoutDecision(
        should_enter=should_enter,
        symbol=candidate.symbol,
        position_size_usdc=position_size_usdc if should_enter else 0.0,
        factor_scores=candidate.factor_scores,
        true_factor_count=candidate.true_factor_count,
        reason=reason or candidate.reason,
        estimated_slippage_pct=getattr(liquidity, "slippage_normal", candidate.slippage_normal),
        entry_score=candidate.entry_score,
        position_size_multiplier=candidate.position_size_multiplier,
    )


def _balance_before_for_reconciliation(toolkit: BnbToolkitWrapper, token_out: str) -> dict[str, Decimal]:
    normalized = token_out.upper()
    if hasattr(toolkit, "get_balances"):
        try:
            payload = toolkit.get_balances()
            balances = _decimal_balances_from_payload(payload)
            if balances:
                return balances
        except Exception:
            LOGGER.debug("get_balances failed; falling back to get_balance(%s)", normalized, exc_info=True)
    payload = toolkit.get_balance(normalized)
    balances = _decimal_balances_from_payload(payload)
    if normalized not in balances:
        balances[normalized] = Decimal(str(_extract_symbol_balance(payload, normalized)))
    return balances


def _decimal_balances_from_payload(payload: Any) -> dict[str, Decimal]:
    if not isinstance(payload, dict):
        return {}
    balances = payload.get("balances")
    if isinstance(balances, dict):
        return {str(key).upper(): Decimal(str(value)) for key, value in balances.items()}
    symbol = payload.get("symbol")
    amount = payload.get("amount", payload.get("balance"))
    if symbol is not None and amount is not None and not isinstance(amount, dict):
        return {str(symbol).upper(): Decimal(str(amount))}
    return {}


def _tx_for_reconciliation(
    tx_result: dict[str, Any],
    token_out: str,
    expected_amount_out: Decimal,
    balance_before: dict[str, Decimal],
    paper_trade: bool,
) -> dict[str, Any]:
    normalized = token_out.upper()
    tx = dict(tx_result or {})
    tx["token_out"] = normalized
    tx["to_symbol"] = normalized
    if paper_trade:
        tx.setdefault("status", 1)
        tx.setdefault("receipt", {"status": 1, "gasUsed": 0, "blockNumber": 0})
        after = dict(balance_before)
        after[normalized] = after.get(normalized, Decimal("0")) + expected_amount_out
        tx.setdefault("balance_after", {key: str(value) for key, value in after.items()})
    return tx


def _open_positions_payload(position_manager: PositionManager) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for position in position_manager.list_open_positions():
        payload.append(
            {
                "symbol": position.symbol,
                "amount_tokens": position.amount_tokens,
                "entry_price": position.entry_price,
                "entry_value_usdc": position.entry_value_usdc,
                "highest_price": position.highest_price,
                "trailing_stop_price": position.trailing_stop_price,
                "take_profit_price": position.take_profit_price,
                "opened_at": position.opened_at.isoformat(),
            }
        )
    return payload


if not hasattr(scoring, "evaluate_universe"):
    scoring.evaluate_universe = fallback_evaluate_universe


def _fetch_snapshot(
    settings: Settings,
    cmc_client: CMCMCPClient,
    open_position_value_usdc: float = 0.0,
    position_symbols: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    if settings.paper_trade:
        return _paper_market_snapshot()

    if settings.use_dual_market_data and not settings.use_keyless_primary:
        keyless_ttl = settings.cmc_keyless_snapshot_ttl_seconds or settings.loop_seconds
        # Event-driven paid enrichment: the short in-position TTL only applies
        # when open positions are above the dust threshold; otherwise the flat
        # heartbeat TTL governs and hot candidates force ad-hoc refreshes.
        dust_threshold = float(getattr(settings, "x402_min_position_value_usdc", 5.0))
        real_position = open_position_value_usdc >= dust_threshold
        if real_position:
            in_position_ttl = getattr(settings, "x402_in_position_ttl_seconds", 1800) or 1800
            x402_ttl = min(settings.cmc_snapshot_ttl_seconds, in_position_ttl)
        else:
            x402_ttl = settings.cmc_snapshot_ttl_seconds
            if open_position_value_usdc > 0:
                LOGGER.debug(
                    "Open positions worth $%.2f are below dust threshold $%.2f; using flat x402 TTL %ss",
                    open_position_value_usdc,
                    dust_threshold,
                    x402_ttl,
                )

        cache = get_dual_market_snapshot_cache()

        def _fetch_keyless() -> dict[str, dict[str, Any]]:
            return cmc_client.fetch_keyless_quotes_snapshot(TARGET_SYMBOLS)

        # Refresh the FREE keyless layer first so hot-candidate detection and
        # enrichment scoping run on current prices before any paid call.
        keyless_snapshot = cache.refresh_keyless(keyless_ttl, _fetch_keyless)

        force_x402 = False
        x402_age = cache.x402_age_seconds()
        hot_age = getattr(settings, "x402_hot_refresh_age_seconds", 600)
        if x402_age is not None and x402_age > hot_age and x402_age < x402_ttl:
            hot_symbols = hot_candidate_symbols(keyless_snapshot, settings)
            if hot_symbols:
                force_x402 = True
                LOGGER.info(
                    "Hot candidates %s passed both cheap core gates; forcing paid x402 refresh (age=%.0fs > %ss)",
                    hot_symbols,
                    x402_age,
                    hot_age,
                )

        enrich_symbols = select_enrichment_symbols(
            keyless_snapshot,
            list(TARGET_SYMBOLS),
            position_symbols or set(),
            settings,
        )

        # The paid MCP tool requires CMC ids (symbol-only requests are
        # rejected after settling payment). Harvest ids for unpinned symbols
        # from the fresh keyless rows so the paid layer can cover them.
        id_overrides: dict[str, str] = {}
        for sym, row in keyless_snapshot.items():
            if isinstance(row, dict) and row.get("id") is not None:
                id_overrides[str(sym).upper()] = str(row["id"])

        snapshot = cache.get_merged_snapshot(
            x402_ttl,
            keyless_ttl,
            lambda: cmc_client.fetch_x402_enriched_snapshot(enrich_symbols, id_overrides),
            _fetch_keyless,
            force_x402_refresh=force_x402,
        )
        _ensure_bnb_reference(snapshot, cmc_client)
        return snapshot

    def _load() -> dict[str, dict[str, Any]]:
        snapshot = cmc_client.fetch_market_snapshot(TARGET_SYMBOLS)
        _ensure_bnb_reference(snapshot, cmc_client)
        return snapshot

    return get_market_snapshot_cache().get_or_fetch(settings.cmc_snapshot_ttl_seconds, _load)


def _ensure_bnb_reference(snapshot: dict[str, dict[str, Any]], cmc_client: CMCMCPClient) -> None:
    if "BNB" in snapshot:
        return
    if "WBNB" in snapshot:
        snapshot["BNB"] = {"symbol": "BNB", **snapshot["WBNB"]}
        return
    try:
        # Keyless on purpose: this runs every cycle and BNB is only a regime
        # reference. get_crypto_quotes_latest would route through PAID x402
        # when keyless-primary is off ($0.01/cycle leak, found June 12).
        payload = cmc_client._fetch_keyless(
            "get_crypto_quotes_latest", {"id": "1839"}  # id-only: ticker lookups can hit knockoffs
        )
        by_symbol = cmc_client._by_symbol(payload)
        bnb = by_symbol.get("BNB")
        if isinstance(bnb, dict):
            volume_24h = _maybe_number(bnb.get("volume_24h"))
            snapshot["BNB"] = {
                "symbol": "BNB",
                "price": _maybe_number(bnb.get("price")),
                "market_cap": _maybe_number(bnb.get("market_cap")),
                "volume_24h": volume_24h,
                "rolling_24h_hourly_volume_avg": volume_24h / 24 if volume_24h else None,
                "percent_change_1h": _maybe_number(bnb.get("percent_change_1h")),
                "percent_change_6h": _maybe_number(bnb.get("percent_change_6h")),
                "percent_change_24h": _maybe_number(bnb.get("percent_change_24h")),
                "high_24h": _maybe_number(bnb.get("high_24h")),
                "low_24h": _maybe_number(bnb.get("low_24h")),
            }
    except Exception as exc:
        LOGGER.warning("Could not fetch BNB reference snapshot: %s", exc)


def _load_positions_or_reconstruct(
    position_manager: PositionManager,
    toolkit: BnbToolkitWrapper,
    settings: Settings,
    market_snapshot: dict[str, dict[str, Any]] | None = None,
) -> int:
    """Load persisted positions or reconstruct live positions from wallet balances."""

    if position_manager.load_positions():
        return len(position_manager.list_open_positions())
    if settings.paper_trade:
        return 0
    return _reconstruct_positions_from_balances(
        position_manager,
        toolkit,
        settings,
        market_snapshot or {},
    )


def _reconstruct_positions_from_balances(
    position_manager: PositionManager,
    toolkit: BnbToolkitWrapper,
    settings: Settings,
    market_snapshot: dict[str, dict[str, Any]],
) -> int:
    """Reconstruct target-token positions from wallet balances when no state exists."""

    reconstructed = 0
    for symbol in TRADABLE_TARGET_SYMBOLS:
        if not has_verified_bsc_contract(symbol):
            continue
        try:
            balance_response = toolkit.get_balance(symbol)
        except Exception as exc:
            # One bad contract address or RPC hiccup must never kill startup;
            # skip the symbol and keep reconstructing the rest of the wallet.
            LOGGER.warning("Balance read failed for %s during reconstruction; skipping: %s", symbol, exc)
            continue
        amount_tokens = _extract_symbol_balance(balance_response, symbol)
        if amount_tokens <= 0:
            continue
        price = _number(market_snapshot.get(symbol, {}).get("price"), 1.0)
        if price <= 0:
            price = 1.0
        now = datetime.now(timezone.utc)
        position = Position(
            symbol=symbol,
            amount_tokens=amount_tokens,
            entry_price=price,
            entry_value_usdc=amount_tokens * price,
            highest_price=price,
            trailing_stop_price=price * (1 - settings.trailing_stop_pct),
            take_profit_price=price * (1 + settings.take_profit_pct),
            opened_at=now,
            current_price=price,
            current_price_at=now,
        )
        position_manager.restore_position(position)
        reconstructed += 1
    return reconstructed


def _extract_symbol_balance(balance_response: dict[str, Any], symbol: str) -> float:
    """Parse common bnb-chain-agentkit balance response shapes."""

    normalized = symbol.upper()
    for key in ("amount", "balance", "free", "total"):
        amount = _maybe_number(balance_response.get(key))
        if amount is not None:
            return amount

    balances = balance_response.get("balances")
    if isinstance(balances, dict):
        for balance_symbol, value in balances.items():
            if str(balance_symbol).upper() == normalized:
                amount = _maybe_number(value)
                return amount or 0.0
    if isinstance(balances, list):
        amount = _extract_from_balance_items(balances, normalized)
        if amount is not None:
            return amount

    data = balance_response.get("data")
    if isinstance(data, list):
        amount = _extract_from_balance_items(data, normalized)
        if amount is not None:
            return amount
    if isinstance(data, dict):
        nested = _extract_symbol_balance(data, symbol)
        if nested > 0:
            return nested
    return 0.0


def _extract_from_balance_items(items: list[Any], symbol: str) -> float | None:
    for item in items:
        if not isinstance(item, dict):
            continue
        item_symbol = str(item.get("symbol") or item.get("token") or item.get("asset") or "").upper()
        if item_symbol != symbol:
            continue
        for key in ("amount", "balance", "free", "total"):
            amount = _maybe_number(item.get(key))
            if amount is not None:
                return amount
    return None


def _process_position_exits(
    position_manager: PositionManager,
    router: PancakeSwapRouter,
    guardrails: Guardrails,
    market_snapshot: dict[str, dict[str, Any]],
    portfolio_value: float,
    price_cache: PriceCache | None = None,
) -> None:
    check_exits = getattr(position_manager, "check_exits", None)
    if callable(check_exits):
        try:
            check_exits(market_snapshot, price_cache)
        except TypeError:
            check_exits(market_snapshot)
        if isinstance(position_manager, ScalpingPositionManager):
            while True:
                signal = position_manager.pop_pending_exit()
                if signal is None:
                    break
                _execute_position_exit(
                    position_manager,
                    router,
                    guardrails,
                    signal.symbol,
                    signal.current_price,
                    portfolio_value,
                    exit_reason=signal.reason,
                )
        return

    for position in list(position_manager.list_open_positions()):
        token_data = market_snapshot.get(position.symbol, {})
        current_price = _number(token_data.get("price"), position.entry_price)
        exit_reason = position_manager.update_price(position.symbol, current_price)
        if exit_reason is None:
            continue
        _execute_position_exit(
            position_manager,
            router,
            guardrails,
            position.symbol,
            current_price,
            portfolio_value,
            exit_reason=exit_reason,
        )


def _execute_position_exit(
    position_manager: PositionManager,
    router: PancakeSwapRouter,
    guardrails: Guardrails,
    symbol: str,
    current_price: float,
    portfolio_value: float,
    *,
    exit_reason: str,
) -> None:
    LOGGER.info("Exiting %s because %s was hit", symbol, exit_reason)
    position = position_manager.get_position(symbol)
    if position is None:
        return
    execution_slippage = _require_execution_slippage(guardrails.settings.max_slippage_pct)
    expected_amount_out = current_price * position.amount_tokens
    try:
        result = _execute_logged_swap(
            guardrails.settings,
            router,
            "exit",
            symbol,
            guardrails.settings.default_stable_symbol,
            position.amount_tokens,
            execution_slippage,
            expected_amount_out=expected_amount_out,
        )
    except Exception as exc:
        # A failed exit swap (e.g. an on-chain revert when trying to sell an
        # illiquid or dust position) must NOT crash the agent. Log it, leave the
        # position open, and let the next cycle retry. Without this guard a
        # single reverting swap takes the whole process down and systemd
        # crash-loops it.
        LOGGER.error(
            "Exit swap for %s (%s) failed: %s; position left open, will retry next cycle",
            symbol,
            exit_reason,
            exc,
        )
        return
    if not _execution_has_tx_hash(result):
        LOGGER.error("Exit swap for %s returned no tx hash; local position remains open", symbol)
        return
    hold_time_seconds = None
    if isinstance(position_manager, ScalpingPositionManager):
        hold_time_seconds = position_manager.hold_time_seconds(symbol)
    closed = position_manager.close_position(symbol)
    if closed is not None:
        realized_pnl = (current_price - closed.entry_price) * closed.amount_tokens
        trade = TradeRecord(
            symbol=closed.symbol,
            side="sell",
            value_usdc=current_price * closed.amount_tokens,
            realized_pnl_usdc=realized_pnl,
            timestamp=datetime.now().astimezone(),
        )
        if isinstance(guardrails, ScalpingGuardrails):
            guardrails.record_scalping_trade(
                trade,
                portfolio_value,
                exit_reason=exit_reason,
            )
        else:
            guardrails.record_trade(trade, portfolio_value)
        if hold_time_seconds is not None:
            setattr(_execute_position_exit, "_last_exit_meta", {
                "symbol": closed.symbol,
                "exit_reason": exit_reason,
                "hold_time_seconds": hold_time_seconds,
            })


def _maybe_enter_position(
    decision: BreakoutDecision,
    position_manager: PositionManager,
    router: PancakeSwapRouter,
    guardrails: Guardrails,
    market_snapshot: dict[str, dict[str, Any]],
    portfolio_value: float,
    twak_interface: TWAKInterface,
) -> None:
    if not decision.should_enter or decision.symbol is None:
        LOGGER.info("No entry: %s", decision.reason)
        return
    if position_manager.get_position(decision.symbol) is not None:
        LOGGER.info("Signal ignored for %s because a position is already open", decision.symbol)
        return

    token_data = market_snapshot[decision.symbol]
    slippage = decision.estimated_slippage_pct
    if slippage is None:
        slippage = _maybe_number(token_data.get("estimated_slippage_pct"))
    if slippage is None or slippage < 0:
        slippage = twak_interface.estimate_slippage_pct(
            amount=decision.position_size_usdc,
            from_token=guardrails.settings.default_stable_symbol,
            to_token=decision.symbol,
        )
    if slippage is None or slippage < 0:
        LOGGER.warning("Signal ignored for %s because slippage is missing", decision.symbol)
        return
    capped_size = _cap_spend_to_portfolio_floor(decision.position_size_usdc, portfolio_value)
    if capped_size < decision.position_size_usdc:
        LOGGER.warning(
            "Reducing %s entry from $%.2f to $%.2f to preserve $%.2f portfolio floor",
            decision.symbol,
            decision.position_size_usdc,
            capped_size,
            MIN_PORTFOLIO_RETAINED_USDC,
        )
        decision = BreakoutDecision(
            should_enter=decision.should_enter,
            symbol=decision.symbol,
            position_size_usdc=capped_size,
            factor_scores=decision.factor_scores,
            true_factor_count=decision.true_factor_count,
            reason=decision.reason,
            estimated_slippage_pct=decision.estimated_slippage_pct,
            ml_context=decision.ml_context,
        )
    if decision.position_size_usdc <= 0:
        LOGGER.warning("Signal ignored for %s because portfolio floor prevents spend", decision.symbol)
        return
    guardrails.validate_new_trade(
        decision.symbol,
        decision.position_size_usdc,
        portfolio_value,
        slippage,
    )
    price = _number(token_data.get("price"))
    if price <= 0:
        raise RuntimeError(f"Cannot enter {decision.symbol}: normalized price is missing")

    LOGGER.info("Entering %s with %s", decision.symbol, decision.reason)
    execution_slippage = _require_execution_slippage(guardrails.settings.max_slippage_pct)
    expected_amount_out = decision.position_size_usdc / price
    result = _execute_logged_swap(
        guardrails.settings,
        router,
        "entry",
        guardrails.settings.default_stable_symbol,
        decision.symbol,
        decision.position_size_usdc,
        execution_slippage,
        expected_amount_out=expected_amount_out,
    )
    if not _execution_has_tx_hash(result):
        LOGGER.error("Entry swap for %s returned no tx hash; local position not opened", decision.symbol)
        return
    amount_tokens = expected_amount_out
    position_manager.open_position(decision.symbol, amount_tokens, price, decision.position_size_usdc)
    guardrails.record_trade(
        TradeRecord(
            symbol=decision.symbol,
            side="buy",
            value_usdc=decision.position_size_usdc,
            realized_pnl_usdc=0.0,
            timestamp=datetime.now().astimezone(),
        ),
        portfolio_value,
    )


def _portfolio_value_usdc(
    toolkit: BnbToolkitWrapper,
    settings: Settings,
    market_snapshot: dict[str, dict[str, Any]] | None = None,
    position_manager: PositionManager | None = None,
) -> float:
    balance = toolkit.get_balance(settings.default_stable_symbol)
    for key in ("portfolio_value_usdc", "total_usdc", "value_usdc"):
        value = balance.get(key)
        if value is not None:
            return _number(value, 10000.0)

    stable_symbol = settings.default_stable_symbol.upper()
    stable_value = _extract_symbol_balance(balance, stable_symbol)
    position_value = 0.0
    if position_manager is not None and market_snapshot is not None:
        for position in position_manager.list_open_positions():
            token_data = market_snapshot.get(position.symbol, {})
            price = _number(token_data.get("price"), position.entry_price)
            position_value += position.amount_tokens * price
    if stable_value > 0 or position_value > 0:
        return stable_value + position_value

    balances = balance.get("balances")
    if isinstance(balances, dict):
        total = sum(_number(value) for value in balances.values())
        if total > 0:
            return total
    LOGGER.warning("Could not parse portfolio value from balance response; using paper fallback")
    return 10000.0


def _paper_market_snapshot() -> dict[str, dict[str, Any]]:
    baseline: dict[str, dict[str, Any]] = {}
    baseline["BNB"] = {
        "symbol": "BNB",
        "price": 600.0,
        "open_24h": 594.0,
        "high_24h": 606.0,
        "low_24h": 588.0,
        "volume_1h": 50_000_000.0,
        "rolling_24h_hourly_volume_avg": 45_000_000.0,
        "volume_24h": 1_080_000_000.0,
        "market_cap": 90_000_000_000.0,
        "percent_change_1h": 0.004,
        "percent_change_6h": 0.011,
        "percent_change_24h": 0.018,
        "estimated_slippage_pct": 0.001,
    }
    for symbol in TARGET_SYMBOLS:
        baseline[symbol] = {
            "symbol": symbol,
            "price": 1.0,
            "open_24h": 0.99,
            "high_24h": 1.02,
            "low_24h": 0.98,
            "volume_1h": 100.0,
            "rolling_24h_hourly_volume_avg": 100.0,
            "volume_24h": 10_000_000.0,
            "market_cap": 100_000_000.0,
            "high_6h": 1.1,
            "high_3h": 1.1,
            "bnb_1h_trend_pct": 0.1,
            "percent_change_1h": 0.003,
            "percent_change_6h": 0.01,
            "percent_change_24h": 0.02,
            "token_percent_change_1h": 0.003,
            "token_percent_change_24h": 0.02,
            "rsi": 50.0,
            "macd": 0.0,
            "estimated_slippage_pct": 0.002,
            "funding_rate": 0.0001,
            "open_interest_change_pct": 0.0,
        }
    baseline["CAKE"] = {
        **baseline["CAKE"],
        "price": 2.16,
        "open_24h": 2.05,
        "high_24h": 2.18,
        "low_24h": 2.01,
        "volume_1h": 2600.0,
        "rolling_24h_hourly_volume_avg": 1000.0,
        "high_6h": 2.10,
        "high_3h": 2.10,
        "percent_change_1h": 0.006,
        "percent_change_6h": 0.018,
        "percent_change_24h": 0.04,
        "rsi": 62.0,
    }
    return baseline


def _print_demo_cycle_summary(
    cycle_number: int,
    market_snapshot: dict[str, dict[str, Any]],
    portfolio_value: float,
    decision: BreakoutDecision | None,
    entries_allowed: bool,
    position_count: int,
    status: str = "ok",
    settings: Settings | None = None,
    entry_score: float | None = None,
) -> None:
    """Print one compact operator-facing cycle summary for demos."""

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    priced_targets = len(_priced_target_symbols(market_snapshot))
    action = "WAIT"
    symbol = "-"
    factors = "-"
    slippage = "-"
    if decision is not None:
        action = "ENTER" if decision.should_enter else "WAIT"
        symbol = decision.symbol or "-"
        factors = f"{decision.true_factor_count}/6"
        slippage = _format_fraction_pct(decision.estimated_slippage_pct)

    if decision is None:
        reason = "guardrails blocked new entries" if not entries_allowed else "no signal evaluated"
    else:
        reason = decision.reason

    print(f"Cycle {cycle_number} summary ({timestamp})")
    print(f"  Status: {status}")
    print(f"  Portfolio: ${portfolio_value:,.2f}")
    print(f"  Market: {priced_targets} priced target(s)")
    if settings is not None and settings.strategy_mode == "scalping":
        score_label = f"{int(entry_score)}/100" if entry_score is not None else "-/100"
        tp_pct = settings.scalping_take_profit_pct * 100
        sl_pct = settings.scalping_stop_loss_pct * 100
        max_hold = settings.scalping_max_hold_minutes
        print(f"  [SCALP] Score: {score_label} | Symbol: {symbol} | Action: {action}")
        print(f"  [SCALP] TP: +{tp_pct:.1f}% | SL: -{sl_pct:.1f}% | Max Hold: {max_hold}min")
    else:
        print(f"  Signal: {action} {symbol} factors={factors} slippage={slippage}")
    print(f"  Positions: {position_count} open")
    print(f"  Reason: {reason}")


def _log_cycle_decision(
    settings: Settings,
    cycle_number: int,
    market_snapshot: dict[str, dict[str, Any]],
    portfolio_value: float,
    decision: BreakoutDecision | None,
    entries_allowed: bool,
    position_count: int,
    action: DecisionAction | None = None,
    reason: str | None = None,
    strategy_mode: str | None = None,
    entry_score: float | None = None,
    entries_blocked_reason: str | None = None,
    exit_reason: str | None = None,
    hold_time_seconds: int | None = None,
    ml_regime: str | None = None,
    ml_confidence: float | None = None,
    ml_ranking: dict[str, Any] | None = None,
    ml_active: bool | None = None,
    ml_selected_symbol: str | None = None,
    executed_symbol: str | None = None,
    ml_scores: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Persist and print the operator-facing decision for one cycle."""

    if action is not None:
        resolved_action = action
    elif decision is not None and decision.should_enter:
        resolved_action = "ENTER"
    else:
        resolved_action = "WAIT"

    if reason is not None:
        resolved_reason = reason
    elif decision is not None:
        resolved_reason = decision.reason
    elif entries_allowed:
        resolved_reason = "no signal evaluated"
    else:
        resolved_reason = "guardrails blocked new entries"

    symbol = decision.symbol if decision is not None else None
    estimated_slippage = decision.estimated_slippage_pct if decision is not None else None
    true_factor_count = decision.true_factor_count if decision is not None else 0
    factor_scores = dict(decision.factor_scores) if decision is not None else {}
    position_size_usdc = decision.position_size_usdc if decision is not None else 0.0
    priced_target_count = len(_priced_target_symbols(market_snapshot))

    record = log_decision(
        settings,
        cycle_number=cycle_number,
        portfolio_value_usdc=portfolio_value,
        position_count=position_count,
        entries_allowed=entries_allowed,
        action=resolved_action,
        reason=resolved_reason,
        priced_target_count=priced_target_count,
        symbol=symbol,
        position_size_usdc=position_size_usdc,
        factor_scores=factor_scores,
        true_factor_count=true_factor_count,
        estimated_slippage_pct=estimated_slippage,
        strategy_mode=strategy_mode,
        entry_score=entry_score,
        entries_blocked_reason=entries_blocked_reason,
        exit_reason=exit_reason,
        hold_time_seconds=hold_time_seconds,
        ml_regime=ml_regime,
        ml_confidence=ml_confidence,
        ml_ranking=ml_ranking,
        ml_active=ml_active,
        ml_selected_symbol=ml_selected_symbol,
        executed_symbol=executed_symbol,
        ml_scores=ml_scores,
    )

    factors = f"{true_factor_count}/6" if decision is not None else "-"
    if strategy_mode == "scalping" and entry_score is not None:
        factors = f"{int(entry_score)}/100"
    LOGGER.info(
        'Decision cycle=%s action=%s symbol=%s factors=%s slippage=%s reason="%s"',
        cycle_number,
        resolved_action,
        symbol or "-",
        factors,
        _format_fraction_pct(estimated_slippage),
        resolved_reason,
    )
    return record


def _format_fraction_pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.2f}%"


def _ensure_daily_minimum_trade(
    settings: Settings,
    router: PancakeSwapRouter,
    guardrails: Guardrails,
    now_utc: datetime,
    portfolio_value_usdc: float,
    *,
    twak_interface: TWAKInterface | None = None,
    liquidity_analyzer: LiquidityAnalyzer | None = None,
    event_filter: EventRiskFilter | None = None,
) -> bool:
    """Fail-safe for the competition's one-trade-per-UTC-day minimum.

    If no trade has been recorded today and fewer than two hours remain in the
    UTC day, execute a tiny allowlisted stable-to-token swap through TWAK.
    This keeps the agent qualified even when risk states (daily pause,
    loss-streak pause, kill switch) block directional entries, at minimal size.
    The richer momentum-ranked minimum-trade path still runs first when entries
    are allowed; this is the last resort.
    """

    if int(getattr(guardrails, "_daily_trade_count", 0)) >= 1:
        return False
    if guardrails.compliance_trade_recorded_today(now_utc):
        return False
    if now_utc.hour < COMPLIANCE_TRIGGER_HOUR_UTC:
        return False
    amount_in = _cap_spend_to_portfolio_floor(COMPLIANCE_TRADE_USDC, portfolio_value_usdc)
    if amount_in < COMPLIANCE_TRADE_USDC:
        LOGGER.warning(
            "Skipping compliance minimum trade: $%.2f portfolio cannot preserve $%.2f floor",
            portfolio_value_usdc,
            MIN_PORTFOLIO_RETAINED_USDC,
        )
        return False
    stable = settings.default_stable_symbol.upper()
    if stable == "BNB":
        stable = "USDC"
    counter = COMPLIANCE_TO_SYMBOL
    if counter == stable:
        counter = "USDC" if stable != "USDC" else "USDT"
    if "BNB" in {stable, counter}:
        LOGGER.error("Compliance minimum trade refused because BNB would be used as a leg")
        return False
    # RWEAL: never route the fixed compliance swap into a token facing a
    # SYMBOL-SPECIFIC scheduled event. COMPLIANCE_TO_SYMBOL is hardcoded, so
    # without this guard a blacked-out counter (e.g. TWT) would be bought
    # directly into the event. Use active_symbol_blackouts (which EXCLUDES
    # GLOBAL/macro) so the differentiate policy holds: a GLOBAL macro blackout
    # still lets the tiny compliance swap fire (avoid DQ); only a per-symbol
    # event on the counter blocks it. Manual halt is handled by the callers.
    if event_filter is not None and counter in event_filter.active_symbol_blackouts(now_utc):
        LOGGER.warning(
            "Skipping fixed compliance swap: counter %s in a symbol-specific event blackout",
            counter,
        )
        return False
    if twak_interface is not None and liquidity_analyzer is not None:
        try:
            slippage_normal = twak_interface.estimate_slippage_pct(amount_in, stable, counter)
            slippage_small = twak_interface.estimate_slippage_pct(amount_in / 2, stable, counter)
            liquidity = liquidity_analyzer.analyze_liquidity(
                symbol=counter,
                position_usd=amount_in,
                twak_quote_small=slippage_small,
                twak_quote_normal=slippage_normal,
                max_slippage_pct=_require_execution_slippage(settings.max_slippage_pct),
            )
        except Exception as exc:
            LOGGER.error("Compliance minimum trade liquidity check failed; will retry next cycle: %s", exc)
            return False
        if getattr(liquidity, "recommendation", "") == "REJECT":
            LOGGER.warning(
                "Skipping compliance minimum trade: %s route liquidity recommendation is REJECT",
                counter,
            )
            return False
    try:
        result = _execute_logged_swap(
            settings,
            router,
            "compliance_min_trade",
            stable,
            counter,
            amount_in,
            _require_execution_slippage(settings.max_slippage_pct),
            reason="compliance: daily minimum trade",
        )
    except Exception as exc:
        LOGGER.error("Compliance minimum trade failed; will retry next cycle: %s", exc)
        return False
    if not _execution_has_tx_hash(result):
        LOGGER.error("Compliance minimum trade returned no tx hash; will retry next cycle")
        return False
    guardrails.record_compliance_trade(now_utc)
    LOGGER.warning(
        "Compliance minimum trade executed: %s -> %s $%.2f",
        stable,
        counter,
        amount_in,
    )
    return True


def _log_live_window_warning(guardrails: Guardrails) -> None:
    now = datetime.now().astimezone()
    in_window = (
        now.month == LIVE_WINDOW_MONTH
        and LIVE_WINDOW_START_DAY <= now.day <= LIVE_WINDOW_END_DAY
    )
    if not in_window:
        return
    bought_today = any(record.side == "buy" and record.timestamp.date() == now.date() for record in guardrails.trade_records)
    if not bought_today:
        LOGGER.warning("Live-window target: no trade has been generated today; guardrails will not be overridden")


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _maybe_number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_market_number(
    payload: dict[str, Any],
    keys: tuple[str, ...],
    default: float,
) -> float:
    for key in keys:
        value = _maybe_number(payload.get(key))
        if value is not None:
            return value
    return default


def _require_execution_slippage(slippage_pct: float | None) -> float:
    if slippage_pct is None or slippage_pct <= 0:
        raise RuntimeError("execution slippage must be configured before calling swap_router")
    return slippage_pct


def _cap_spend_to_portfolio_floor(amount_usdc: float, portfolio_value_usdc: float) -> float:
    max_spend = max(0.0, portfolio_value_usdc - MIN_PORTFOLIO_RETAINED_USDC)
    return max(0.0, min(amount_usdc, max_spend))


def _execute_logged_swap(
    settings: Settings,
    router: PancakeSwapRouter,
    action: str,
    from_symbol: str,
    to_symbol: str,
    amount_in: float,
    max_slippage_pct: float,
    expected_amount_out: float | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    try:
        result = router.swap_exact_in(
            from_symbol,
            to_symbol,
            amount_in,
            max_slippage_pct,
            expected_amount_out=expected_amount_out,
        )
    except Exception as exc:
        log_execution(
            settings,
            action=action,
            from_symbol=from_symbol,
            to_symbol=to_symbol,
            amount_in=amount_in,
            max_slippage_pct=max_slippage_pct,
            expected_amount_out=expected_amount_out,
            error=str(exc),
            reason=reason,
        )
        raise

    log_execution(
        settings,
        action=action,
        from_symbol=from_symbol,
        to_symbol=to_symbol,
        amount_in=amount_in,
        max_slippage_pct=max_slippage_pct,
        expected_amount_out=expected_amount_out,
        result=result,
        reason=reason,
    )
    return result


def _execution_has_tx_hash(result: dict[str, Any]) -> bool:
    return bool(result.get("tx_hash") or result.get("hash") or result.get("transaction_hash"))


def _settings_with_updates(settings: Settings, updates: dict[str, Any]) -> Settings:
    if hasattr(settings, "model_copy"):
        return settings.model_copy(update=updates)
    return settings.copy(update=updates)


def _settings_with_mode(settings: Settings, paper_trade: bool) -> Settings:
    return _settings_with_updates(settings, {"paper_trade": paper_trade})


def _configure_logging(settings: Settings) -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if not settings.demo_mode:
        return
    for logger_name in (
        "urllib3",
        "urllib3.connectionpool",
        "web3",
        "web3.providers",
        "web3.RequestManager",
    ):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Plan B+ BSC Momentum Breakout Scalper")
    parser.add_argument("--paper-trade", action="store_true", help="Run with deterministic paper execution")
    parser.add_argument("--live", action="store_true", help="Run live TWAK swap execution")
    parser.add_argument("--emergency-liquidate", action="store_true", help="Sell open positions to USDC")
    parser.add_argument("--balance", action="store_true", help="Print wallet balances and exit")
    parser.add_argument("--preflight", action="store_true", help="Run live readiness checks without broadcasting")
    parser.add_argument("--once", action="store_true", help="Run one trading cycle and exit")
    parser.add_argument("--demo-mode", action="store_true", help="Print compact per-cycle demo summaries")
    parser.add_argument("--withdraw", metavar="SYMBOL", help="Transfer SYMBOL from the agent wallet")
    parser.add_argument("--to", dest="withdraw_to", help="Destination EVM address for --withdraw")
    parser.add_argument("--amount", dest="withdraw_amount", type=float, help="Token amount for --withdraw")
    args = parser.parse_args(argv)
    if args.paper_trade and args.live:
        parser.error("--paper-trade and --live are mutually exclusive")
    if args.preflight and not args.live:
        parser.error("--preflight requires --live")
    if args.preflight and (args.emergency_liquidate or args.balance or args.withdraw or args.once):
        parser.error("--preflight cannot be combined with --emergency-liquidate, --balance, --withdraw, or --once")
    if args.withdraw and not args.live:
        parser.error("--withdraw requires --live")
    if args.withdraw and (not args.withdraw_to or args.withdraw_amount is None):
        parser.error("--withdraw requires --to and --amount")
    if (args.withdraw_to or args.withdraw_amount is not None) and not args.withdraw:
        parser.error("--to and --amount require --withdraw")
    return args


def main(argv: list[str] | None = None) -> int:
    """CLI main function."""

    args = parse_args(argv)
    try:
        settings = load_settings()
    except Exception as exc:
        if args.preflight:
            _print_preflight_report([PreflightCheck("settings loaded", False, _safe_error(exc))])
            return 1
        raise
    if args.emergency_liquidate:
        settings = _settings_with_mode(settings, args.paper_trade)
    elif args.live:
        settings = _settings_with_mode(settings, False)
    elif args.paper_trade or not args.live:
        settings = _settings_with_mode(settings, True)
    if args.demo_mode:
        settings = _settings_with_updates(settings, {"demo_mode": True})

    _configure_logging(settings)

    if args.preflight:
        return 0 if run_live_preflight(settings) else 1

    if args.emergency_liquidate:
        toolkit = BnbToolkitWrapper(settings)
        twak_interface = _twak_interface_from_settings(settings, paper_trade=settings.paper_trade)
        router = PancakeSwapRouter(twak_interface)
        position_manager = PositionManager(settings)
        guardrails = Guardrails(settings)
        _load_positions_or_reconstruct(position_manager, toolkit, settings)
        emergency_liquidate(position_manager, router, guardrails)
        return 0

    if args.balance:
        toolkit = BnbToolkitWrapper(settings)
        print_balances(toolkit, settings)
        return 0

    if args.withdraw:
        toolkit = BnbToolkitWrapper(settings)
        withdraw_funds(toolkit, args.withdraw, args.withdraw_to, args.withdraw_amount)
        return 0

    run_agent(settings, max_cycles=1 if args.once else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""CLI entrypoint for the Plan B+ trading agent."""

from __future__ import annotations

import argparse
import logging
import re
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from src.config.settings import Settings, load_settings
from src.config.tokens import TARGET_SYMBOLS, TRADABLE_TARGET_SYMBOLS, has_bsc_contract
from src.data.cmc_mcp_client import CMCMCPClient
from src.execution.bnb_toolkit_wrapper import BnbToolkitWrapper
from src.execution.decision_log import DecisionAction, log_decision
from src.execution.execution_log import log_execution
from src.execution.swap_router import PancakeSwapRouter
from src.execution.twak_interface import TWAKInterface
from src.strategy.breakout_engine import BreakoutDecision, BreakoutEngine
from src.strategy.guardrails import Guardrails, TradeRecord
from src.strategy.position_manager import Position, PositionManager

LOGGER = logging.getLogger(__name__)
LIVE_WINDOW_MONTH = 6
LIVE_WINDOW_START_DAY = 22
LIVE_WINDOW_END_DAY = 28
PREFLIGHT_QUOTE_AMOUNT_USDC = 0.5


@dataclass(frozen=True)
class PreflightCheck:
    """Single live-readiness check result."""

    name: str
    passed: bool
    detail: str = ""


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


def print_balances(toolkit: BnbToolkitWrapper, settings: Settings) -> None:
    """Print the operator's key balances for preflight checks."""

    symbols = ["BNB", settings.default_stable_symbol.upper(), "USDT"]
    seen: set[str] = set()
    for symbol in symbols:
        if symbol in seen:
            continue
        seen.add(symbol)
        balance = toolkit.get_balance(symbol)
        amount = balance.get("balance", balance.get("amount"))
        print(f"{symbol}: {_number(amount):.8f}")


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

    twak_interface = TWAKInterface(paper_trade=False)
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
            record("CMC/x402 market snapshot", bool(snapshot), f"{len(snapshot)} item(s)")
        else:
            record("CMC/x402 market snapshot", False, "non-dict snapshot")
    except Exception as exc:  # pragma: no cover - exercised by CLI tests with fakes
        record("CMC/x402 market snapshot", False, _safe_error(exc))

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


def run_agent(settings: Settings, max_cycles: int | None = None) -> None:
    """Run the 5-minute trading loop."""

    cmc_client = CMCMCPClient(settings)
    toolkit = BnbToolkitWrapper(settings)
    twak_interface = TWAKInterface(paper_trade=settings.paper_trade)
    router = PancakeSwapRouter(twak_interface)
    engine = BreakoutEngine(settings, twak_interface)
    position_manager = PositionManager(settings)
    guardrails = Guardrails(settings)
    positions_loaded = position_manager.load_positions()
    needs_balance_reconstruction = not positions_loaded and not settings.paper_trade
    if positions_loaded:
        LOGGER.info("Loaded %s persisted open positions", len(position_manager.list_open_positions()))
    running = True
    cycles_completed = 0

    def _stop(_signum: int, _frame: Any) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    while running:
        cycle_number = cycles_completed + 1
        market_snapshot = _fetch_snapshot(settings, cmc_client)
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
        if guardrails.update_portfolio_value(portfolio_value):
            LOGGER.critical("Drawdown kill switch triggered; liquidating open positions")
            _log_cycle_decision(
                settings,
                cycle_number,
                market_snapshot,
                portfolio_value,
                decision=None,
                entries_allowed=False,
                position_count=len(position_manager.list_open_positions()),
                action="HALT",
                reason="drawdown kill switch",
            )
            emergency_liquidate(position_manager, router, guardrails)
            if settings.demo_mode:
                _print_demo_cycle_summary(
                    cycle_number,
                    market_snapshot,
                    portfolio_value,
                    decision=None,
                    entries_allowed=False,
                    position_count=len(position_manager.list_open_positions()),
                    status="drawdown kill switch",
                )
            break

        _process_position_exits(position_manager, router, guardrails, market_snapshot, portfolio_value)
        decision: BreakoutDecision | None = None
        entries_allowed = guardrails.can_open_new_trade()
        if entries_allowed:
            decision = engine.evaluate_universe(market_snapshot, portfolio_value)
            _log_cycle_decision(
                settings,
                cycle_number,
                market_snapshot,
                portfolio_value,
                decision,
                entries_allowed,
                len(position_manager.list_open_positions()),
            )
            _maybe_enter_position(
                decision,
                position_manager,
                router,
                guardrails,
                market_snapshot,
                portfolio_value,
                twak_interface,
            )
        else:
            LOGGER.info("Guardrails currently block new entries")
            _log_cycle_decision(
                settings,
                cycle_number,
                market_snapshot,
                portfolio_value,
                decision=None,
                entries_allowed=False,
                position_count=len(position_manager.list_open_positions()),
                action="BLOCKED",
                reason="guardrails blocked new entries",
            )

        if settings.demo_mode:
            _print_demo_cycle_summary(
                cycle_number,
                market_snapshot,
                portfolio_value,
                decision,
                entries_allowed,
                len(position_manager.list_open_positions()),
            )
        _log_live_window_warning(guardrails)
        cycles_completed += 1
        if max_cycles is not None and cycles_completed >= max_cycles:
            LOGGER.info("Completed %s cycle(s); exiting", cycles_completed)
            break

        sleep_until = time.monotonic() + settings.loop_seconds
        while running and time.monotonic() < sleep_until:
            time.sleep(min(1.0, sleep_until - time.monotonic()))


def _fetch_snapshot(settings: Settings, cmc_client: CMCMCPClient) -> dict[str, dict[str, Any]]:
    if settings.paper_trade:
        return _paper_market_snapshot()
    return cmc_client.fetch_market_snapshot(TARGET_SYMBOLS)


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
        if not has_bsc_contract(symbol):
            continue
        balance_response = toolkit.get_balance(symbol)
        amount_tokens = _extract_symbol_balance(balance_response, symbol)
        if amount_tokens <= 0:
            continue
        price = _number(market_snapshot.get(symbol, {}).get("price"), 1.0)
        if price <= 0:
            price = 1.0
        position = Position(
            symbol=symbol,
            amount_tokens=amount_tokens,
            entry_price=price,
            entry_value_usdc=amount_tokens * price,
            highest_price=price,
            trailing_stop_price=price * (1 - settings.trailing_stop_pct),
            take_profit_price=price * (1 + settings.take_profit_pct),
            opened_at=datetime.now(timezone.utc),
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
) -> None:
    for position in list(position_manager.list_open_positions()):
        token_data = market_snapshot.get(position.symbol, {})
        current_price = _number(token_data.get("price"), position.entry_price)
        exit_reason = position_manager.update_price(position.symbol, current_price)
        if exit_reason is None:
            continue
        LOGGER.info("Exiting %s because %s was hit", position.symbol, exit_reason)
        execution_slippage = _require_execution_slippage(guardrails.settings.max_slippage_pct)
        expected_amount_out = current_price * position.amount_tokens
        result = _execute_logged_swap(
            guardrails.settings,
            router,
            "exit",
            position.symbol,
            guardrails.settings.default_stable_symbol,
            position.amount_tokens,
            execution_slippage,
            expected_amount_out=expected_amount_out,
        )
        if not _execution_has_tx_hash(result):
            LOGGER.error("Exit swap for %s returned no tx hash; local position remains open", position.symbol)
            continue
        closed = position_manager.close_position(position.symbol)
        if closed is not None:
            realized_pnl = (current_price - closed.entry_price) * closed.amount_tokens
            guardrails.record_trade(
                TradeRecord(
                    symbol=closed.symbol,
                    side="sell",
                    value_usdc=current_price * closed.amount_tokens,
                    realized_pnl_usdc=realized_pnl,
                    timestamp=datetime.now().astimezone(),
                ),
                portfolio_value,
            )


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
    for symbol in TARGET_SYMBOLS:
        baseline[symbol] = {
            "symbol": symbol,
            "price": 1.0,
            "volume_1h": 100.0,
            "rolling_24h_hourly_volume_avg": 100.0,
            "volume_24h": 10_000_000.0,
            "market_cap": 100_000_000.0,
            "high_6h": 1.1,
            "high_3h": 1.1,
            "bnb_1h_trend_pct": 0.1,
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
        "volume_1h": 2600.0,
        "rolling_24h_hourly_volume_avg": 1000.0,
        "high_6h": 2.10,
        "high_3h": 2.10,
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
    )

    factors = f"{true_factor_count}/6" if decision is not None else "-"
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


def _require_execution_slippage(slippage_pct: float | None) -> float:
    if slippage_pct is None or slippage_pct <= 0:
        raise RuntimeError("execution slippage must be configured before calling swap_router")
    return slippage_pct


def _execute_logged_swap(
    settings: Settings,
    router: PancakeSwapRouter,
    action: str,
    from_symbol: str,
    to_symbol: str,
    amount_in: float,
    max_slippage_pct: float,
    expected_amount_out: float | None = None,
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
        twak_interface = TWAKInterface(paper_trade=settings.paper_trade)
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

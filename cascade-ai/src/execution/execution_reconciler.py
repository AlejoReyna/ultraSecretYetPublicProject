"""On-chain execution reconciliation before local position opening.

Example:
    reconciler = ExecutionReconciler(toolkit)
    result = reconciler.reconcile(tx, Decimal("100"), Decimal("0.01"), before)

Interface contract:
    Imports: Decimal and standard-library helpers only.
    Exports: ReconciliationResult, ExecutionReconciler.
    Does not broadcast transactions or mutate position state.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

_TWAK_AMOUNT_RE = re.compile(r"^([0-9]+(?:\.[0-9]+)?)")


@dataclass(frozen=True)
class ReconciliationResult:
    """Receipt, balance-delta, and slippage reconciliation outcome."""

    status: str
    tx_hash: str
    token_out: str
    amount_out_expected: Decimal
    amount_out_actual: Decimal
    effective_slippage_pct: Decimal
    gas_used: int
    block_number: int
    receipt_status: int
    balance_delta_confirmed: bool


class ExecutionReconciler:
    """Validate tx receipt success and token balance delta."""

    def __init__(self, bnb_api_wrapper: Any) -> None:
        self.bnb_api = bnb_api_wrapper

    def reconcile(
        self,
        tx_result: dict,
        expected_amount_out: Decimal,
        slippage_tolerance: Decimal,
        balance_before: dict[str, Decimal],
    ) -> ReconciliationResult:
        """Return reconciliation status without mutating live positions."""

        if _is_twak_success(tx_result):
            return self._reconcile_twak_swap(tx_result, expected_amount_out, balance_before)

        receipt = tx_result.get("receipt") if isinstance(tx_result.get("receipt"), dict) else {}
        receipt_status = int(receipt.get("status", tx_result.get("status", -1)))
        tx_hash = str(tx_result.get("hash") or tx_result.get("tx_hash") or "")
        token_out = self._token_out(tx_result, balance_before)
        gas_used = int(receipt.get("gasUsed", receipt.get("gas_used", 0)) or 0)
        block_number = int(receipt.get("blockNumber", receipt.get("block_number", 0)) or 0)
        if receipt_status != 1:
            return self._result(
                "FAILED", tx_hash, token_out, expected_amount_out, Decimal("0"), gas_used, block_number, receipt_status, False
            )

        amount_from_logs = self._amount_from_logs(tx_result, receipt)
        balance_after = self._balance_after(tx_result, token_out)
        before = balance_before.get(token_out, Decimal("0"))
        after = balance_after.get(token_out, Decimal("0"))
        balance_delta = after - before
        balance_delta_confirmed = balance_delta > 0
        if not balance_delta_confirmed and amount_from_logs <= 0:
            return self._result(
                "FAILED", tx_hash, token_out, expected_amount_out, Decimal("0"), gas_used, block_number, receipt_status, False
            )

        actual = max(amount_from_logs, balance_delta)
        slippage = self._effective_slippage(expected_amount_out, actual)
        status = "SLIPPAGE_EXCEEDED" if slippage > slippage_tolerance else "SUCCESS"
        return self._result(
            status,
            tx_hash,
            token_out,
            expected_amount_out,
            actual,
            gas_used,
            block_number,
            receipt_status,
            balance_delta_confirmed,
        )

    def _reconcile_twak_swap(
        self,
        tx_result: dict[str, Any],
        expected_amount_out: Decimal,
        balance_before: dict[str, Decimal],
    ) -> ReconciliationResult:
        tx_hash = str(tx_result.get("hash") or tx_result.get("tx_hash") or "")
        token_out = self._token_out(tx_result, balance_before)
        actual = _twak_amount_out(tx_result)
        if actual is None or actual <= 0:
            return self._result(
                "FAILED",
                tx_hash,
                token_out,
                expected_amount_out,
                Decimal("0"),
                0,
                0,
                -1,
                False,
            )
        return self._result(
            "SUCCESS",
            tx_hash,
            token_out,
            expected_amount_out,
            actual,
            0,
            0,
            1,
            False,
        )

    @staticmethod
    def _result(
        status: str,
        tx_hash: str,
        token_out: str,
        expected: Decimal,
        actual: Decimal,
        gas_used: int,
        block_number: int,
        receipt_status: int,
        balance_delta_confirmed: bool,
    ) -> ReconciliationResult:
        return ReconciliationResult(
            status=status,
            tx_hash=tx_hash,
            token_out=token_out,
            amount_out_expected=expected,
            amount_out_actual=actual,
            effective_slippage_pct=ExecutionReconciler._effective_slippage(expected, actual),
            gas_used=gas_used,
            block_number=block_number,
            receipt_status=receipt_status,
            balance_delta_confirmed=balance_delta_confirmed,
        )

    def _balance_after(self, tx_result: dict, token_out: str) -> dict[str, Decimal]:
        raw_after = tx_result.get("balance_after")
        if isinstance(raw_after, dict):
            return {str(key).upper(): Decimal(str(value)) for key, value in raw_after.items()}
        if hasattr(self.bnb_api, "get_balance"):
            payload = self.bnb_api.get_balance(token_out)
            return self._balances_from_payload(payload)
        return {}

    @staticmethod
    def _balances_from_payload(payload: Any) -> dict[str, Decimal]:
        if not isinstance(payload, dict):
            return {}
        balances = payload.get("balances")
        if isinstance(balances, dict):
            return {str(key).upper(): Decimal(str(value)) for key, value in balances.items()}
        symbol = payload.get("symbol")
        amount = payload.get("amount", payload.get("balance"))
        if symbol is not None and amount is not None:
            return {str(symbol).upper(): Decimal(str(amount))}
        return {}

    @staticmethod
    def _token_out(tx_result: dict, balance_before: dict[str, Decimal]) -> str:
        for key in ("token_out", "to_symbol", "symbol"):
            value = tx_result.get(key)
            if value:
                return str(value).upper()
        if balance_before:
            return next(iter(balance_before)).upper()
        return "UNKNOWN"

    @staticmethod
    def _amount_from_logs(tx_result: dict, receipt: dict) -> Decimal:
        for payload in (tx_result, receipt):
            for key in ("amount_out", "amountOut", "received_amount", "to_amount", "output", "minReceived"):
                if key in payload:
                    parsed = _parse_twak_amount(payload[key])
                    if parsed is not None:
                        return parsed
        logs = receipt.get("logs", tx_result.get("logs", []))
        if isinstance(logs, list):
            for item in logs:
                if isinstance(item, dict):
                    for key in ("amount_out", "amountOut", "amount"):
                        if key in item:
                            return Decimal(str(item[key]))
        return Decimal("0")

    @staticmethod
    def _effective_slippage(expected: Decimal, actual: Decimal) -> Decimal:
        if expected <= 0:
            return Decimal("0")
        return (expected - actual) / expected


def _is_twak_success(tx_result: dict[str, Any]) -> bool:
    if tx_result.get("mode") != "twak" and tx_result.get("tool") != "swap":
        return False
    try:
        if int(tx_result.get("returncode", -1)) != 0:
            return False
    except (TypeError, ValueError):
        return False
    return bool(tx_result.get("hash") or tx_result.get("tx_hash"))


def _parse_twak_amount(value: Any) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return Decimal(str(value))
    text = str(value).strip()
    if not text:
        return None
    match = _TWAK_AMOUNT_RE.match(text)
    if match is None:
        return None
    return Decimal(match.group(1))


def _twak_amount_out(tx_result: dict[str, Any]) -> Decimal | None:
    for key in ("output", "minReceived", "amount_out", "amountOut"):
        parsed = _parse_twak_amount(tx_result.get(key))
        if parsed is not None and parsed > 0:
            return parsed
    return None

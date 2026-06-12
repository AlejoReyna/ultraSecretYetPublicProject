"""Subprocess interface for the verified Trust Wallet Agent Kit CLI."""

from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass
from typing import Any, Optional

from src.config.tokens import TOKEN_CONTRACTS, resolve_twak_token

LOGGER = logging.getLogger(__name__)

_TX_HASH_RE = re.compile(r"0x[a-fA-F0-9]{64}")
_BSCSCAN_TX_RE = re.compile(r"(https?://(?:www\.)?bscscan\.com/tx/(0x[a-fA-F0-9]{64}))")

# Quotes are read-only and retried every cycle; a hung quote should fail fast
# instead of stalling the 5-minute loop. Execution keeps a long timeout because
# it broadcasts on-chain and must wait for confirmation.
QUOTE_TIMEOUT_SECONDS = 15
EXEC_TIMEOUT_SECONDS = 120


@dataclass(frozen=True)
class TWAKResult:
    """Result returned by a TWAK CLI command."""

    command: list[str]
    returncode: int
    stdout: str
    stderr: str


class TWAKInterface:
    """Secure wrapper around documented TWAK commands."""

    def __init__(self, paper_trade: bool = False) -> None:
        self.paper_trade = paper_trade

    def get_quote(
        self,
        amount: float,
        from_token: str,
        to_token: str,
        chain: str = "bsc",
    ) -> dict[str, Any]:
        """Fetch a TWAK quote-only swap JSON payload without broadcasting."""

        from_addr = resolve_twak_token(from_token)
        to_addr = resolve_twak_token(to_token)
        command = [
            "twak",
            "swap",
            str(amount),
            from_addr,
            to_addr,
            "--chain",
            chain.strip().lower(),
            "--quote-only",
            "--json",
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                check=False,
                shell=False,
                text=True,
                timeout=QUOTE_TIMEOUT_SECONDS,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            LOGGER.warning("TWAK quote failed for %s -> %s: %s", from_token, to_token, exc)
            return {"error": str(exc), "errorCode": "QUOTE_SUBPROCESS_FAILED"}

        stdout = completed.stdout.strip()
        if not stdout:
            LOGGER.warning(
                "TWAK quote returned empty stdout for %s -> %s (rc=%s)",
                from_token,
                to_token,
                completed.returncode,
            )
            return {
                "error": completed.stderr.strip() or "empty quote response",
                "errorCode": "QUOTE_EMPTY",
                "returncode": completed.returncode,
            }

        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            decoded, _ = self._decode_swap_stdout(stdout)
            payload = decoded if isinstance(decoded, dict) else {"raw": decoded}

        if not isinstance(payload, dict):
            return {"error": "invalid quote payload", "errorCode": "QUOTE_INVALID", "raw": payload}

        if completed.returncode != 0 and "error" not in payload and "errorCode" not in payload:
            payload.setdefault("error", completed.stderr.strip() or f"exit code {completed.returncode}")
            payload.setdefault("errorCode", "QUOTE_FAILED")
        return payload

    def estimate_slippage_pct(
        self,
        amount: float,
        from_token: str,
        to_token: str,
        chain: str = "bsc",
    ) -> Optional[float]:
        """Estimate slippage as a fraction (0.01 = 1%) from a TWAK quote-only response."""

        try:
            quote = self.get_quote(amount, from_token, to_token, chain=chain)
            if quote.get("error") or quote.get("errorCode"):
                LOGGER.warning(
                    "TWAK quote error for %s -> %s: %s",
                    from_token,
                    to_token,
                    quote.get("error") or quote.get("errorCode"),
                )
                return None

            price_impact_raw = quote.get("priceImpact")
            if price_impact_raw is not None and str(price_impact_raw).strip() != "":
                # DEX-reported impact; "0" means liquid — not TWAK's minReceived safety floor.
                return float(price_impact_raw)

            output = self._parse_amount_field(quote.get("output"))
            min_received = self._parse_amount_field(quote.get("minReceived"))
            if output is None or min_received is None or output <= 0:
                LOGGER.warning(
                    "TWAK quote missing output/minReceived for %s -> %s",
                    from_token,
                    to_token,
                )
                return None

            return (output - min_received) / output
        except Exception as exc:
            LOGGER.warning(
                "Failed to estimate slippage for %s -> %s: %s",
                from_token,
                to_token,
                exc,
            )
            return None

    @staticmethod
    def _parse_amount_field(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip()
        if not text:
            return None
        return float(text.split()[0])

    def wallet_create(self) -> TWAKResult:
        """Run twak wallet create."""

        return self._run(["twak", "wallet", "create"])

    def compete_register(self) -> TWAKResult:
        """Run twak compete register."""

        return self._run(["twak", "compete", "register"])

    def wallet_address(self, chain: str = "bsc") -> dict[str, Any]:
        """Return the configured TWAK wallet address without broadcasting."""

        normalized_chain = chain.strip().lower()
        if not normalized_chain:
            raise ValueError("wallet address chain must be provided")
        result = self._run(["twak", "wallet", "address", "--chain", normalized_chain, "--json"])
        return self._json_payload_from_result(result, "wallet-address")

    def request_x402(
        self,
        url: str,
        method: str = "GET",
        body: dict[str, Any] | None = None,
        max_payment_atomic: str = "10000",
        prefer_network: str = "base",
        prefer_method: str = "eip3009",
        prefer_asset: str = "USDC",
    ) -> dict[str, Any]:
        """Make an x402-gated HTTP request through TWAK native local signing."""

        normalized_method = method.strip().upper()
        if not normalized_method:
            raise ValueError("x402 request method must be provided")
        command = [
            "twak",
            "x402",
            "request",
            url,
            "--method",
            normalized_method,
            "--max-payment",
            str(max_payment_atomic),
            "--prefer-network",
            prefer_network,
            "--prefer-method",
            prefer_method,
            "--prefer-asset",
            prefer_asset,
            "--yes",
            "--json",
        ]
        if body is not None:
            command.extend(["--body", json.dumps(body, separators=(",", ":"), sort_keys=True)])
        result = self._run(command)
        return self._json_payload_from_result(result, "x402-request")

    def swap(
        self,
        from_symbol: str,
        to_symbol: str,
        amount: float,
        slippage_pct: float,
    ) -> dict[str, Any]:
        """Execute a swap through TWAK."""

        if amount <= 0:
            raise ValueError("swap amount must be greater than zero")
        if slippage_pct <= 0:
            raise ValueError("swap slippage must be greater than zero")
        if self.paper_trade:
            return {
                "mode": "paper",
                "tool": "twak-swap",
                "from_symbol": from_symbol,
                "to_symbol": to_symbol,
                "amount_in": amount,
                "estimated_amount_out": amount,
                "slippage_pct": slippage_pct,
                "tx_hash": f"paper-twak-swap-{from_symbol.upper()}-{to_symbol.upper()}",
            }

        result = self._run(
            [
                "twak",
                "swap",
                str(amount),
                resolve_twak_token(from_symbol),
                resolve_twak_token(to_symbol),
                "--slippage",
                self._fraction_to_cli_percent(slippage_pct),
                "--chain",
                "bsc",
                "--json",
            ]
        )
        return self._swap_payload_from_result(result)

    def quote_swap(
        self,
        from_symbol: str,
        to_symbol: str,
        amount: float,
        slippage_pct: float,
    ) -> dict[str, Any]:
        """Fetch a TWAK quote without signing or broadcasting a swap."""

        if amount <= 0:
            raise ValueError("quote amount must be greater than zero")
        if slippage_pct <= 0:
            raise ValueError("quote slippage must be greater than zero")

        result = self._run(
            [
                "twak",
                "swap",
                str(amount),
                resolve_twak_token(from_symbol),
                resolve_twak_token(to_symbol),
                "--slippage",
                self._fraction_to_cli_percent(slippage_pct),
                "--chain",
                "bsc",
                "--quote-only",
                "--json",
            ],
            timeout=QUOTE_TIMEOUT_SECONDS,
        )
        return self._json_payload_from_result(result, "swap-quote")

    def start(self) -> TWAKResult:
        """Run twak start."""

        return self._run(["twak", "start"])

    @staticmethod
    def _run(command: list[str], timeout: int = EXEC_TIMEOUT_SECONDS) -> TWAKResult:
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                check=False,
                shell=False,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            command_name = " ".join(command[:3])
            raise RuntimeError(f"{command_name} timed out") from exc
        except FileNotFoundError as exc:
            raise RuntimeError("TWAK CLI was not found on PATH") from exc

        result = TWAKResult(
            command=command,
            returncode=completed.returncode,
            stdout=completed.stdout.strip(),
            stderr=completed.stderr.strip(),
        )
        if result.returncode != 0:
            command_name = " ".join(command[:3])

            parts = []
            if result.stderr:
                parts.append(f"stderr: {result.stderr}")
            if result.stdout:
                parts.append(f"stdout: {result.stdout}")
            message = " | ".join(parts) if parts else "<no output>"

            raise RuntimeError(f"{command_name} failed with exit code {result.returncode}: {message}")
        return result

    @staticmethod
    def _amount_to_atomic_units(amount: float, asset: str) -> str:
        normalized = asset.strip().lower()
        six_decimal_assets = {
            "usdc",
            "usdt",
            "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913",
        }
        decimals = 6 if normalized in six_decimal_assets else 18
        return str(int(round(amount * (10**decimals))))

    @staticmethod
    def _fraction_to_cli_percent(slippage_pct: float) -> str:
        """Convert internal fraction slippage to TWAK's percent CLI value."""

        return format(slippage_pct * 100, ".12g")

    @staticmethod
    def _json_payload_from_result(result: TWAKResult, tool: str) -> dict[str, Any]:
        decoded, mixed_stdout = TWAKInterface._decode_swap_stdout(result.stdout)
        payload = decoded if isinstance(decoded, dict) else {"raw": decoded}
        if mixed_stdout:
            payload["raw"] = result.stdout
        payload["mode"] = "twak"
        payload["tool"] = tool
        payload["command"] = result.command
        payload["returncode"] = result.returncode
        if result.stderr:
            payload.setdefault("stderr", result.stderr)
        return payload

    @staticmethod
    def _swap_payload_from_result(result: TWAKResult) -> dict[str, Any]:
        decoded, mixed_stdout = TWAKInterface._decode_swap_stdout(result.stdout)
        payload = decoded if isinstance(decoded, dict) else {"raw": decoded}
        if mixed_stdout:
            payload["raw"] = result.stdout

        text_hashes = TWAKInterface._extract_swap_text_hashes(result.stdout)
        swap_hash = payload.get("tx_hash") or payload.get("hash") or text_hashes.get("tx_hash")
        if swap_hash:
            payload["hash"] = str(swap_hash)
            payload["tx_hash"] = str(swap_hash)
        explorer = payload.get("explorer") or text_hashes.get("explorer")
        if explorer:
            payload["explorer"] = str(explorer)
        if text_hashes.get("approval_hash"):
            payload["approval_hash"] = text_hashes["approval_hash"]
        if text_hashes.get("approval_explorer"):
            payload["approval_explorer"] = text_hashes["approval_explorer"]

        payload["mode"] = "twak"
        payload["tool"] = "swap"
        payload["command"] = result.command
        payload["returncode"] = result.returncode
        if result.stderr:
            payload.setdefault("stderr", result.stderr)
        return payload

    @staticmethod
    def _decode_swap_stdout(stdout: str) -> tuple[Any, bool]:
        if not stdout:
            return {}, False
        try:
            return json.loads(stdout), False
        except json.JSONDecodeError:
            pass

        decoder = json.JSONDecoder()
        for index, char in enumerate(stdout):
            if char != "{":
                continue
            try:
                decoded, consumed = decoder.raw_decode(stdout[index:])
            except json.JSONDecodeError:
                continue
            if stdout[index + consumed :].strip() == "":
                return decoded, True
        return {"raw": stdout}, False

    @staticmethod
    def _extract_swap_text_hashes(stdout: str) -> dict[str, str]:
        hashes: dict[str, str] = {}
        for line in stdout.splitlines():
            lower_line = line.lower()
            url_match = _BSCSCAN_TX_RE.search(line)
            tx_hash = url_match.group(2) if url_match else None
            if tx_hash is None:
                hash_match = _TX_HASH_RE.search(line)
                tx_hash = hash_match.group(0) if hash_match else None
            if tx_hash is None:
                continue

            if "approval tx" in lower_line:
                hashes["approval_hash"] = tx_hash
                if url_match:
                    hashes["approval_explorer"] = url_match.group(1)
            elif "swap tx" in lower_line:
                hashes["tx_hash"] = tx_hash
                if url_match:
                    hashes["explorer"] = url_match.group(1)
        return hashes

    @staticmethod
    def _tx_hash_from_stdout(stdout: str) -> str | None:
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            payload = None
        if payload is not None:
            found = TWAKInterface._tx_hash_from_json(payload)
            if found is not None:
                return found

        match = _TX_HASH_RE.search(stdout)
        return match.group(0) if match else None

    @staticmethod
    def _tx_hash_from_json(payload: Any) -> str | None:
        if isinstance(payload, dict):
            for key in ("tx_hash", "txHash", "transaction_hash", "transactionHash", "hash"):
                value = payload.get(key)
                if isinstance(value, str) and _TX_HASH_RE.fullmatch(value.strip()):
                    return value.strip()
            for value in payload.values():
                found = TWAKInterface._tx_hash_from_json(value)
                if found is not None:
                    return found
        if isinstance(payload, list):
            for value in payload:
                found = TWAKInterface._tx_hash_from_json(value)
                if found is not None:
                    return found
        return None

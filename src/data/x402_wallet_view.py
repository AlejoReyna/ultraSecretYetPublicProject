"""Read-only view of the x402 data-payment wallet (Base chain).

Strictly diagnostic: derives the public address from the env-only payment key
and reads the USDC balance on Base plus the local spend ledger. Never signs,
never transfers, never prints key material.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from decimal import Decimal

LOGGER = logging.getLogger(__name__)

DEFAULT_BASE_RPC_URL = "https://mainnet.base.org"
BASE_USDC_TOKEN = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
USDC_DECIMALS = Decimal(10) ** 6
_BALANCE_OF_SELECTOR = "0x70a08231"


@dataclass(frozen=True)
class X402WalletView:
    """Snapshot of the x402 payment wallet for operator display."""

    address: str | None
    usdc_balance: Decimal | None
    rpc_url: str | None
    error: str | None = None


def x402_wallet_address() -> str | None:
    """Derive the x402 payment address from the env-only key, or None."""

    for env_name in ("CMC_X402_EPHEMERAL_KEY", "EVM_PRIVATE_KEY"):
        key = os.getenv(env_name, "").strip()
        if not key:
            continue
        try:
            from eth_account import Account

            if not key.startswith("0x"):
                key = f"0x{key}"
            return Account.from_key(key).address
        except Exception as exc:
            LOGGER.warning("Could not derive x402 wallet address from %s: %s", env_name, exc)
            return None
    return None


def fetch_x402_wallet_view(
    base_rpc_url: str | None = None,
    usdc_token: str = BASE_USDC_TOKEN,
) -> X402WalletView:
    """Return the x402 wallet address and its USDC balance on Base."""

    address = x402_wallet_address()
    if address is None:
        return X402WalletView(
            address=None,
            usdc_balance=None,
            rpc_url=None,
            error="no CMC_X402_EPHEMERAL_KEY/EVM_PRIVATE_KEY configured",
        )

    rpc_url = (base_rpc_url or "").strip() or os.getenv("BASE_RPC_URL", "").strip() or DEFAULT_BASE_RPC_URL
    try:
        balance = _erc20_balance(rpc_url, usdc_token, address)
    except Exception as exc:
        LOGGER.warning("x402 wallet balance read failed via %s: %s", rpc_url, exc)
        return X402WalletView(address=address, usdc_balance=None, rpc_url=rpc_url, error=str(exc)[:120])
    return X402WalletView(address=address, usdc_balance=balance, rpc_url=rpc_url)


def _erc20_balance(rpc_url: str, token: str, holder: str) -> Decimal:
    """Read an ERC-20 balance with a raw eth_call (no ABI dependency)."""

    from web3 import Web3

    w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 10}))
    data = _BALANCE_OF_SELECTOR + holder.lower().removeprefix("0x").rjust(64, "0")
    raw = w3.eth.call(
        {
            "to": Web3.to_checksum_address(token),
            "data": data,
        }
    )
    return Decimal(int.from_bytes(raw, byteorder="big")) / USDC_DECIMALS

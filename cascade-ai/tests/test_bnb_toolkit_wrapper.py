"""Tests for bnb-chain-agentkit payload adaptation."""

from __future__ import annotations

from typing import Any

import pytest

from src.config.settings import Settings
from src.config.tokens import (
    STABLE_TARGET_SYMBOLS,
    TARGET_SYMBOLS,
    TOKEN_CONTRACTS_BSC,
    TRADABLE_TARGET_SYMBOLS,
    has_bsc_contract,
)
from src.execution import bnb_toolkit_wrapper as wrapper_module
from src.execution.bnb_toolkit_wrapper import BnbToolkitWrapper


class FakeTool:
    """Capture payloads passed through LangChain tool invocation."""

    def __init__(self, name: str, result: object = "ok") -> None:
        self.name = name
        self.result = result
        self.payloads: list[dict[str, Any]] = []

    def invoke(self, payload: dict[str, Any]) -> object:
        self.payloads.append(payload)
        return self.result


class FakeCall:
    """Fake Web3 contract call result."""

    def __init__(self, value: int) -> None:
        self.value = value

    def call(self) -> int:
        return self.value


class FakeContractFunctions:
    """Fake ERC-20 functions surface."""

    def __init__(self, raw_balance: int, decimals: int) -> None:
        self.raw_balance = raw_balance
        self.decimals_value = decimals
        self.balance_accounts: list[str] = []

    def decimals(self) -> FakeCall:
        return FakeCall(self.decimals_value)

    def balanceOf(self, account: str) -> FakeCall:
        self.balance_accounts.append(account)
        return FakeCall(self.raw_balance)


class FakeContract:
    """Fake ERC-20 contract."""

    def __init__(self, raw_balance: int, decimals: int) -> None:
        self.functions = FakeContractFunctions(raw_balance, decimals)


class FakeEth:
    """Fake Web3 eth namespace."""

    def __init__(self, native_balance: int, token_balance: int, token_decimals: int) -> None:
        self.native_balance = native_balance
        self.token_balance = token_balance
        self.token_decimals = token_decimals
        self.contract_calls: list[dict[str, Any]] = []

    def get_balance(self, account: str) -> int:
        return self.native_balance

    def contract(self, address: str, abi: list[dict[str, Any]]) -> FakeContract:
        self.contract_calls.append({"address": address, "abi": abi})
        return FakeContract(self.token_balance, self.token_decimals)


class FakeWeb3:
    """Fake Web3 client for local unit tests."""

    def __init__(self, native_balance: int = 10**18, token_balance: int = 2_500_000, token_decimals: int = 6) -> None:
        self.eth = FakeEth(native_balance, token_balance, token_decimals)

    def to_checksum_address(self, value: str) -> str:
        return value

    def from_wei(self, value: int, unit: str) -> float:
        assert unit == "ether"
        return value / 10**18


def _wrapper_with_tools(*tools: FakeTool) -> BnbToolkitWrapper:
    wrapper = object.__new__(BnbToolkitWrapper)
    wrapper.settings = Settings(paper_trade=False, wallet_address="0x1111111111111111111111111111111111111111")
    wrapper.paper_trade = False
    wrapper.api_wrapper = None
    wrapper.toolkit = None
    wrapper.tools = list(tools)
    wrapper.w3 = None
    return wrapper


def _wrapper_with_web3(fake_web3: FakeWeb3) -> BnbToolkitWrapper:
    wrapper = object.__new__(BnbToolkitWrapper)
    wrapper.settings = Settings(
        paper_trade=False,
        wallet_address="0x1111111111111111111111111111111111111111",
        bsc_rpc_url="https://bsc.example",
    )
    wrapper.paper_trade = False
    wrapper.api_wrapper = None
    wrapper.toolkit = None
    wrapper.tools = []
    wrapper.w3 = fake_web3
    return wrapper


def test_configured_contracts_belong_to_target_universe() -> None:
    target_keys = {symbol.upper() for symbol in TARGET_SYMBOLS}
    configured_in_target = {symbol for symbol in TOKEN_CONTRACTS_BSC if symbol in target_keys}
    assert configured_in_target
    assert configured_in_target <= target_keys


def test_hackathon_tradable_symbols_count_as_bsc_even_without_static_address() -> None:
    assert has_bsc_contract("PENGU") is True
    assert has_bsc_contract("CAKE") is True
    assert has_bsc_contract("USDC") is False


def test_stables_remain_available_but_not_directionally_tradable() -> None:
    tradable_keys = {symbol.upper() for symbol in TRADABLE_TARGET_SYMBOLS}
    assert STABLE_TARGET_SYMBOLS.isdisjoint(tradable_keys)
    assert {"USDT", "USDC"}.issubset(STABLE_TARGET_SYMBOLS)


def test_wrapper_defaults_to_loaded_settings(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        wrapper_module,
        "load_settings",
        lambda: Settings(paper_trade=True, default_stable_symbol="USDC"),
    )

    wrapper = BnbToolkitWrapper()

    assert wrapper.paper_trade is True
    assert wrapper.get_balance()["symbol"] == "USDC"


def test_live_constructor_does_not_initialize_agentkit(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        BnbToolkitWrapper,
        "_initialize_live_tools",
        lambda self: (_ for _ in ()).throw(AssertionError("should be lazy")),
    )

    wrapper = BnbToolkitWrapper(
        Settings(
            paper_trade=False,
            wallet_address="0x1111111111111111111111111111111111111111",
            bsc_rpc_url="https://bsc.example",
        )
    )

    assert wrapper.tools == []
    assert wrapper.w3 is None


def test_live_get_balance_reads_native_bnb_with_web3() -> None:
    wrapper = _wrapper_with_web3(FakeWeb3(native_balance=2 * 10**18))

    result = wrapper.get_balance("BNB")

    assert result["mode"] == "live"
    assert result["symbol"] == "BNB"
    assert result["balance"] == 2.0
    assert result["raw_balance"] == 2 * 10**18


def test_live_get_balance_accepts_explicit_account_and_contract_token() -> None:
    fake_web3 = FakeWeb3(token_balance=2_500_000, token_decimals=6)
    wrapper = _wrapper_with_web3(fake_web3)
    account = "0x2222222222222222222222222222222222222222"

    result = wrapper.get_balance(account, TOKEN_CONTRACTS_BSC["USDC"])

    assert fake_web3.eth.contract_calls[0]["address"] == TOKEN_CONTRACTS_BSC["USDC"]
    assert result["mode"] == "live"
    assert result["account"] == account
    assert result["token"] == TOKEN_CONTRACTS_BSC["USDC"]
    assert result["balance"] == 2.5
    assert result["balances"] == {"USDC": 2.5}


def test_agentkit_swap_path_is_disabled() -> None:
    tool = FakeTool("swap", {"amount_out": 99.0})
    wrapper = _wrapper_with_tools(tool)

    with pytest.raises(RuntimeError, match="TWAKInterface.swap"):
        wrapper.swap("USDC", "CAKE", 25.5, 0.01)

    assert tool.payloads == []


def test_live_transfer_uses_recipient_token_amount_schema() -> None:
    tool = FakeTool("transfer", "ok")
    wrapper = _wrapper_with_tools(tool)

    wrapper.transfer("0x2222222222222222222222222222222222222222", "USDC", 1.25)

    assert tool.payloads == [
        {
            "recipient": "0x2222222222222222222222222222222222222222",
            "token": TOKEN_CONTRACTS_BSC["USDC"],
            "amount": "1.25",
        }
    ]

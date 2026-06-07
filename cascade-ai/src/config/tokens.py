"""Token allowlists for Plan B+ trading decisions."""

from __future__ import annotations

# Official BNB Hack eligible BEP-20 list supplied from the rules page. The
# rules list includes SLX twice; the duplicate is preserved intentionally.
ELIGIBLE_149_SYMBOLS: list[str] = [
    "ETH",
    "USDT",
    "USDC",
    "XRP",
    "TRX",
    "DOGE",
    "ZEC",
    "ADA",
    "LINK",
    "BCH",
    "DAI",
    "TON",
    "USD1",
    "USDe",
    "M",
    "LTC",
    "AVAX",
    "SHIB",
    "XAUt",
    "WLFI",
    "H",
    "DOT",
    "UNI",
    "ASTER",
    "DEXE",
    "USDD",
    "ETC",
    "AAVE",
    "ATOM",
    "U",
    "STABLE",
    "FIL",
    "INJ",
    "币安人生",
    "NIGHT",
    "FET",
    "TUSD",
    "BONK",
    "PENGU",
    "CAKE",
    "SIREN",
    "LUNC",
    "ZRO",
    "KITE",
    "FDUSD",
    "BEAT",
    "PIEVERSE",
    "BTT",
    "NFT",
    "EDGE",
    "FLOKI",
    "LDO",
    "B",
    "FF",
    "PENDLE",
    "NEX",
    "STG",
    "AXS",
    "TWT",
    "HOME",
    "RAY",
    "COMP",
    "GWEI",
    "XCN",
    "GENIUS",
    "XPL",
    "BAT",
    "SKYAI",
    "APE",
    "IP",
    "SFP",
    "TAG",
    "NXPC",
    "AB",
    "SAHARA",
    "1INCH",
    "CHEEMS",
    "BANANAS31",
    "RIVER",
    "MYX",
    "RAVE",
    "SNX",
    "FORM",
    "LAB",
    "HTX",
    "USDf",
    "CTM",
    "BDX",
    "SLX",
    "UB",
    "DUCKY",
    "FRAX",
    "BILL",
    "WFI",
    "KOGE",
    "ALE",
    "FRXUSD",
    "USDF",
    "GOMINING",
    "VCNT",
    "GUA",
    "DUSD",
    "SMILEK",
    "0G",
    "BEAM",
    "MY",
    "SLX",
    "SOON",
    "REAL",
    "Q",
    "AIOZ",
    "ZIG",
    "YFI",
    "TAC",
    "lisUSD",
    "CYS",
    "ZAMA",
    "TRIA",
    "HUMA",
    "PLUME",
    "ZIL",
    "XPR",
    "ZETA",
    "BabyDoge",
    "NILA",
    "ROSE",
    "VELO",
    "UAI",
    "BRETT",
    "OPEN",
    "BSB",
    "TOSHI",
    "BAS",
    "ACH",
    "AXL",
    "LUR",
    "ELF",
    "KAVA",
    "APR",
    "IRYS",
    "EURI",
    "XUSD",
    "BARD",
    "DUSK",
    "SUSHI",
    "PEAQ",
    "COAI",
    "BDCA",
    "XAUM",
]

# Deduplicated operational universe (148 unique symbols; SLX appears twice in rules).
TARGET_SYMBOLS: list[str] = list(dict.fromkeys(ELIGIBLE_149_SYMBOLS))

TARGET_SYMBOL_BY_KEY: dict[str, str] = {symbol.upper(): symbol for symbol in TARGET_SYMBOLS}

STABLE_TARGET_SYMBOLS: set[str] = {
    "USDT",
    "USDC",
    "DAI",
    "USD1",
    "USDe",
    "USDD",
    "TUSD",
    "FDUSD",
    "USDf",
    "FRXUSD",
    "USDF",
    "DUSD",
    "lisUSD",
    "XUSD",
    "EURI",
    "FRAX",
}

TRADABLE_TARGET_SYMBOLS: list[str] = [
    symbol for symbol in TARGET_SYMBOLS if symbol.upper() not in STABLE_TARGET_SYMBOLS
]

TOKEN_CONTRACTS_BSC: dict[str, str] = {
    "ETH": "0x2170Ed0880ac9A755fd29B2688956BD959F933F8",
    "USDT": "0x55d398326f99059fF775485246999027B3197955",
    "USDC": "0x8AC76a51cc950d9822D68b83fE1Ad97B32Cd580d",
    "CAKE": "0x0E09FaBB73Bd3Ade0a17ECC321fD13a19e81cE82",
    "LINK": "0xF8A0BF9cF54Bb92F17374d9e9A321E6a111a51bD",
    "AAVE": "0xfb6115445Bff7b52FeB98650C87f44907E58f802",
    "UNI": "0xBf5140A22578168FD562DCcF235E5D43A02ce9B1",
    "INJ": "0xa2B726B1145A4773F68593CF171187d8EBe4d495",
    "SHIB": "0x2859e4544C4bB03966803b044A93563Bd2D0DD4D",
    "DOGE": "0xbA2aE424d960c26247Dd6c32edC70B295c744C43",
    "BONK": "0xa697e272a73744b343528c3bc4702f2565b2f422",
    "FLOKI": "0xfb5B838b6cfEEdC2873aB27866079AC55363D37E",
    "WBNB": "0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c",
    "BTCB": "0x7130d2A12B9BCbFAe4f2634d864A1Ee1Ce3Ead9c",
    "ADA": "0x3EE2200Efb3400fAbB9AacF31297cBdD1d435D47",
    "XRP": "0x1D2F0dA169ceB9fC7B3144628dB156f3F6c60dBE",
    "DOT": "0x7083609fCE4d1d8Dc0C979AAb8c869Ea2C873402",
    "LTC": "0x4338665CBB7B2485A8855A139b75D5e34AB0DB94",
    "ATOM": "0x0Eb3a705fc54725037CC9e008bDede697f62F335",
    "FIL": "0x0D8Ce2A99Bb6e3B7Db580ED848240e4a0F9aE153",
}

# TWAK/LiquidMesh token identifiers (symbols resolved to BSC contract addresses).
TOKEN_CONTRACTS: dict[str, str] = {
    "BNB": "BNB",
    **TOKEN_CONTRACTS_BSC,
}

CMC_IDS_BY_SYMBOL: dict[str, str] = {
    "ETH": "1027",
    "USDT": "825",
    "USDC": "3408",
    "CAKE": "7186",
    "LINK": "1975",
    "AAVE": "7278",
    "UNI": "7083",
    "INJ": "7226",
    "SHIB": "5994",
    "DOGE": "74",
    "BONK": "23095",
    "FLOKI": "10804",
    "BNB": "1839",
    "WBNB": "7192",
    "BTCB": "4023",
    "ADA": "2010",
    "XRP": "52",
    "DOT": "6636",
    "LTC": "2",
    "ATOM": "3794",
    "FIL": "2280",
}

LIQUIDITY_BLACKLIST: set[str] = {
    "lisUSD",
    "ALE",
    "DUCKY",
    "SMILEK",
    "BDCA",
    "NILA",
    "LUR",
}

# One-week competition window: these hard floors avoid opening into tokens where
# slippage and thin books can dominate expected PnL without improving drawdown.
MIN_VOLUME_24H_USD = 5_000_000
MIN_MARKET_CAP_USD = 50_000_000


def is_target_symbol(symbol: str) -> bool:
    """Return whether a symbol is in the BNB Hack eligible target universe."""

    return symbol.strip().upper() in TARGET_SYMBOL_BY_KEY


def is_tradable_symbol(symbol: str) -> bool:
    """Return whether a symbol may be opened as a directional trade."""

    key = symbol.strip().upper()
    if key not in TARGET_SYMBOL_BY_KEY:
        return False
    return key not in STABLE_TARGET_SYMBOLS


def is_liquid(token_data: dict[str, object]) -> bool:
    """Return whether a token clears hard blacklist and minimum liquidity floors."""

    symbol = str(token_data.get("symbol", "")).strip().upper()
    blacklisted = {blocked.upper() for blocked in LIQUIDITY_BLACKLIST}
    if symbol in blacklisted:
        return False
    volume_24h = _number(token_data.get("volume_24h"), 0.0)
    market_cap = _number(token_data.get("market_cap"), 0.0)
    return volume_24h >= MIN_VOLUME_24H_USD and market_cap >= MIN_MARKET_CAP_USD


def assert_target_symbol(symbol: str) -> None:
    """Raise when a token is outside the BNB Hack eligible target universe."""

    normalized = symbol.strip().upper()
    if normalized not in TARGET_SYMBOL_BY_KEY:
        raise ValueError(f"{normalized} is not in the TARGET_SYMBOLS allowlist")


def assert_tradable_symbol(symbol: str) -> None:
    """Raise when a token should not be opened as a directional trade."""

    normalized = symbol.strip().upper()
    if not is_tradable_symbol(normalized):
        raise ValueError(f"{normalized} is not in the tradable target allowlist")


def has_bsc_contract(symbol: str) -> bool:
    """Return whether a symbol is tradable as BEP-20 on BSC for this hackathon.

    The eligible universe is BSC-native; verified addresses are listed in
    ``TOKEN_CONTRACTS_BSC`` when known. TWAK resolves remaining hack symbols
    by ticker on BSC (see ``resolve_twak_token``).
    """

    normalized = symbol.strip().upper()
    if normalized in TOKEN_CONTRACTS_BSC or normalized == "BNB":
        return True
    return is_tradable_symbol(normalized)


def resolve_twak_token(symbol: str) -> str:
    """Return the TWAK CLI token argument for a symbol or pass through addresses."""

    normalized = symbol.strip().upper()
    if normalized.startswith("0X") and len(normalized) == 42:
        return symbol.strip()
    return TOKEN_CONTRACTS.get(normalized, symbol.strip())


def get_bsc_token_address(symbol: str) -> str:
    """Return the verified BSC token identifier used by bnb-chain-agentkit."""

    normalized = symbol.strip().upper()
    assert_target_symbol(normalized)
    try:
        return TOKEN_CONTRACTS_BSC[normalized]
    except KeyError as exc:
        raise ValueError(
            f"No BSC contract configured for {normalized}; TWAK may still resolve the symbol directly"
        ) from exc


def resolve_cmc_coin_id(symbol: str) -> str | None:
    """Return a configured CoinMarketCap ID without TARGET_SYMBOLS gating."""

    return CMC_IDS_BY_SYMBOL.get(symbol.strip().upper())


def get_cmc_id_optional(symbol: str) -> str | None:
    """Return the CoinMarketCap ID when configured, otherwise None."""

    normalized = symbol.strip().upper()
    if normalized not in TARGET_SYMBOL_BY_KEY:
        return None
    return CMC_IDS_BY_SYMBOL.get(normalized)


def get_cmc_id_for_mcp(symbol: str) -> str:
    """Return the CoinMarketCap ID for MCP/x402 tool calls."""

    normalized = symbol.strip().upper()
    cmc_id = resolve_cmc_coin_id(normalized)
    if cmc_id is None:
        raise ValueError(f"No CoinMarketCap ID configured for MCP lookup: {normalized}")
    return cmc_id


def get_cmc_id(symbol: str) -> str:
    """Return the CoinMarketCap cryptocurrency ID for a target symbol."""

    normalized = symbol.strip().upper()
    assert_target_symbol(normalized)
    cmc_id = get_cmc_id_optional(normalized)
    if cmc_id is None:
        raise ValueError(f"No CoinMarketCap ID configured for {normalized}")
    return cmc_id


def _number(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

"""Application settings loaded from environment variables."""

from __future__ import annotations

import os
from typing import Any, Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.ml.constants import ML_DEFAULT_20


class Settings(BaseModel):
    """Runtime settings for the trading agent."""

    cmc_api_key: Optional[str] = None
    bsc_rpc_url: Optional[str] = None
    base_rpc_url: Optional[str] = None
    opbnb_provider_url: Optional[str] = "https://opbnb-mainnet-rpc.bnbchain.org"
    wallet_address: Optional[str] = None
    usdc_token_address: Optional[str] = None
    default_stable_symbol: str = "USDC"
    cmc_x402_endpoint: str = "https://mcp.coinmarketcap.com/x402/mcp"
    cmc_x402_amount: float = 0.01
    cmc_x402_asset: str = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
    cmc_mcp_enabled: bool = False
    cmc_mcp_shadow_mode: bool = True
    cmc_mcp_url: str = "https://mcp.coinmarketcap.com/x402/mcp"
    # Do not add CMC_X402_EPHEMERAL_KEY here. Settings is passed into execution
    # objects, and x402 data micropayment keys must stay isolated in src.data.
    cmc_x402_chain_id: int = 8453
    cmc_x402_max_usdc_per_call: float = 0.01
    x402_daily_budget_usdc: float = 2.0
    x402_total_budget_usdc: float = 15.0
    x402_failure_cooldown_seconds: int = 900
    x402_in_position_ttl_seconds: int = 1800
    use_keyless_primary: bool = False
    use_dual_market_data: bool = False
    cmc_keyless_base_url: str = "https://pro-api.coinmarketcap.com/trial-pro-api/v3"
    cmc_snapshot_ttl_seconds: int = 7200
    cmc_keyless_snapshot_ttl_seconds: int = 300
    paper_trade: bool = True
    loop_seconds: int = 300
    price_cache_maxlen: int = 2880
    max_position_pct: float = 0.05
    max_daily_trades: int = 3
    max_daily_loss_pct: float = 0.03
    max_slippage_pct: float = 0.01
    drawdown_soft_stop_pct: float = 0.10
    drawdown_kill_switch_pct: float = 0.18
    trailing_stop_pct: float = 0.035
    take_profit_pct: float = 0.08
    base_risk_per_trade_pct: float = 0.0035
    risk_off_max_slippage_pct: float = 0.005
    loss_streak_reduce_size: int = 2
    loss_streak_pause: int = 3
    position_monitor_seconds: int = 60
    sentiment_cache_ttl: int = 300
    sentiment_cache_ttl_seconds: int = 300
    sentiment_fgi_extreme_greed_threshold: int = 75
    sentiment_fgi_extreme_fear_threshold: int = 20
    sentiment_funding_crowded_long_threshold: float = 0.001
    sentiment_funding_crowded_short_threshold: float = -0.0005
    sentiment_gas_elevated_gwei: float = 0.3
    sentiment_max_negative_delta: float = -2.5
    sentiment_max_positive_delta: float = 1.0
    # One-week competition window: require BNB to be only mildly weak at worst
    # so entries are not opened into broad-market rollovers that increase drawdown.
    bnb_regime_threshold: float = -0.01
    # One-week competition window: token must already be flat-to-positive on 1h
    # while avoiding severe 24h downtrends; these are fail-closed drawdown guards.
    token_regime_1h_min: float = 0.0025
    token_regime_24h_min: float = -0.08
    # One-week competition window: 3h highs catch breakouts earlier, while the
    # 0.2% buffer avoids chasing tiny noisy ticks that can inflate drawdown.
    breakout_lookback_hours: int = 3
    breakout_buffer: float = 0.002
    # Minimum passing count across the three actionable core entry factors;
    # regime is logged separately and applied as a size modifier.
    min_entry_factors: int = Field(default=3, ge=1, le=4)
    # Size multiplier applied when regime is risk-off. 0.5 halves size
    # (v2 default); 1.0 makes regime informational only.
    regime_size_multiplier: float = 0.5
    log_level: str = "INFO"
    demo_mode: bool = False
    position_state_path: str = "positions.json"
    guardrail_state_path: str = "guardrail_state.json"
    execution_log_path: str = "execution_log.jsonl"
    decision_log_path: str = "decision_log.jsonl"
    strategy_mode: Literal["breakout", "scalping"] = "breakout"
    scalping_entry_score_min: float = 60.0
    scalping_position_pct: float = 0.01
    scalping_take_profit_pct: float = 0.015
    scalping_stop_loss_pct: float = 0.008
    scalping_max_hold_minutes: int = 30
    scalping_time_stop_minutes: int = 20
    scalping_symbol_cooldown_minutes: int = 15
    scalping_daily_loss_cap_pct: float = 0.02
    scalping_max_daily_trades: int = 10
    scalping_max_gas_gwei: float = 5.0
    scalping_max_slippage_pct: float = 0.005
    scalping_pump_filter_15m_pct: float = 0.05
    scalping_min_market_cap_usd: float = 1_000_000.0
    scalping_consecutive_loss_limit: int = 3
    scalping_consecutive_loss_cooldown_hours: float = 1.0
    ml_enabled: bool = False
    ml_model_path: str = "models/regime_lgbm_v1.pkl"
    ml_regime_threshold: float = 0.55
    ml_momentum_size_multiplier: float = 1.0
    ml_chop_size_multiplier: float = 0.5
    ml_universe_symbols: list[str] = Field(default_factory=lambda: list(ML_DEFAULT_20))
    ml_ohlcv_cache_db: str = "data/ml_ohlcv_cache.sqlite"
    ml_ohlcv_lookback_candles: int = 96
    ml_volume_breakout_multiplier: float = 2.0
    ml_volume_cache_multiplier: float = 1.2
    ml_min_auc: float = 0.65
    ml_shadow_mode: bool = True
    ml_regime_only_chop_multiplier: float = 0.3
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    health_check_port: int = 8080
    log_rotate_max_mb: float = 50.0
    cmc_collector_enabled: bool = True
    cmc_collector_interval_min: int = 15
    min_bnb_gas: float = 0.05
    min_usdc_balance: float = 50.0
    disk_guard_min_free_bytes: int = 500_000_000


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _none_if_blank(value: str | None) -> str | None:
    if value is None or value.strip() == "":
        return None
    return value


def _get_symbol_list(name: str, default: list[str]) -> list[str]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return list(default)
    return [item.strip().upper() for item in value.split(",") if item.strip()]


def load_settings(dotenv_path: str | None = None) -> Settings:
    """Load settings from .env and the current process environment."""

    load_dotenv(dotenv_path=dotenv_path)
    loop_seconds = _get_int("LOOP_SECONDS", 300)
    use_keyless_primary = _get_bool("USE_KEYLESS_PRIMARY", False)
    has_x402_signer = bool(
        os.getenv("CMC_X402_EPHEMERAL_KEY", "").strip()
        or os.getenv("EVM_PRIVATE_KEY", "").strip()
    )
    use_dual_env = os.getenv("USE_DUAL_MARKET_DATA")
    if use_dual_env is not None:
        use_dual_market_data = _get_bool("USE_DUAL_MARKET_DATA", False)
    else:
        use_dual_market_data = has_x402_signer and not use_keyless_primary
    values: dict[str, Any] = {
        "cmc_api_key": _none_if_blank(os.getenv("CMC_API_KEY")),
        "bsc_rpc_url": _none_if_blank(os.getenv("BSC_RPC_URL") or os.getenv("BSC_PROVIDER_URL")),
        "base_rpc_url": _none_if_blank(os.getenv("BASE_RPC_URL")),
        "opbnb_provider_url": _none_if_blank(os.getenv("OPBNB_PROVIDER_URL"))
        or "https://opbnb-mainnet-rpc.bnbchain.org",
        "wallet_address": _none_if_blank(os.getenv("WALLET_ADDRESS") or os.getenv("AGENT_WALLET_ADDRESS")),
        "usdc_token_address": _none_if_blank(os.getenv("USDC_TOKEN_ADDRESS")),
        "default_stable_symbol": os.getenv("DEFAULT_STABLE_SYMBOL", "USDC"),
        "cmc_x402_endpoint": os.getenv(
            "CMC_X402_ENDPOINT",
            "https://mcp.coinmarketcap.com/x402/mcp",
        ),
        "cmc_x402_amount": _get_float("CMC_X402_AMOUNT", 0.01),
        "cmc_x402_asset": os.getenv("CMC_X402_ASSET", "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"),
        "cmc_mcp_enabled": _get_bool("CMC_MCP_ENABLED", False),
        "cmc_mcp_shadow_mode": _get_bool("CMC_MCP_SHADOW_MODE", True),
        "cmc_mcp_url": os.getenv(
            "CMC_MCP_URL",
            os.getenv("CMC_X402_ENDPOINT", "https://mcp.coinmarketcap.com/x402/mcp"),
        ),
        "cmc_x402_chain_id": _get_int("CMC_X402_CHAIN_ID", 8453),
        "cmc_x402_max_usdc_per_call": _get_float("CMC_X402_MAX_USDC_PER_CALL", 0.01),
        "x402_daily_budget_usdc": _get_float("X402_DAILY_BUDGET_USDC", 2.0),
        "x402_total_budget_usdc": _get_float("X402_TOTAL_BUDGET_USDC", 15.0),
        "x402_failure_cooldown_seconds": _get_int("X402_FAILURE_COOLDOWN_SECONDS", 900),
        "x402_in_position_ttl_seconds": _get_int("X402_IN_POSITION_TTL_SECONDS", 1800),
        "use_keyless_primary": use_keyless_primary,
        "use_dual_market_data": use_dual_market_data,
        "cmc_keyless_base_url": os.getenv(
            "CMC_KEYLESS_BASE_URL",
            "https://pro-api.coinmarketcap.com/trial-pro-api/v3",
        ),
        "cmc_snapshot_ttl_seconds": _get_int("CMC_SNAPSHOT_TTL_SECONDS", 7200),
        "cmc_keyless_snapshot_ttl_seconds": _get_int(
            "CMC_KEYLESS_SNAPSHOT_TTL_SECONDS",
            loop_seconds,
        ),
        "paper_trade": _get_bool("PAPER_TRADE", True),
        "loop_seconds": loop_seconds,
        "price_cache_maxlen": _get_int("PRICE_CACHE_MAXLEN", 2880),
        "max_position_pct": _get_float("MAX_POSITION_PCT", 0.05),
        "max_daily_trades": _get_int("MAX_DAILY_TRADES", 3),
        "max_daily_loss_pct": _get_float("MAX_DAILY_LOSS_PCT", 0.03),
        "max_slippage_pct": _get_float("MAX_SLIPPAGE_PCT", 0.01),
        "drawdown_soft_stop_pct": _get_float("DRAWDOWN_SOFT_STOP_PCT", 0.10),
        "drawdown_kill_switch_pct": _get_float("DRAWDOWN_KILL_SWITCH_PCT", 0.18),
        "trailing_stop_pct": _get_float("TRAILING_STOP_PCT", 0.035),
        "take_profit_pct": _get_float("TAKE_PROFIT_PCT", 0.08),
        "base_risk_per_trade_pct": _get_float("BASE_RISK_PER_TRADE_PCT", 0.0035),
        "risk_off_max_slippage_pct": _get_float("RISK_OFF_MAX_SLIPPAGE_PCT", 0.005),
        "loss_streak_reduce_size": _get_int("LOSS_STREAK_REDUCE_SIZE", 2),
        "loss_streak_pause": _get_int("LOSS_STREAK_PAUSE", 3),
        "position_monitor_seconds": _get_int("POSITION_MONITOR_SECONDS", 60),
        "sentiment_cache_ttl": _get_int(
            "SENTIMENT_CACHE_TTL",
            _get_int("SENTIMENT_CACHE_TTL_SECONDS", 300),
        ),
        "sentiment_cache_ttl_seconds": _get_int(
            "SENTIMENT_CACHE_TTL_SECONDS",
            _get_int("SENTIMENT_CACHE_TTL", 300),
        ),
        "sentiment_fgi_extreme_greed_threshold": _get_int("SENTIMENT_FGI_EXTREME_GREED_THRESHOLD", 75),
        "sentiment_fgi_extreme_fear_threshold": _get_int("SENTIMENT_FGI_EXTREME_FEAR_THRESHOLD", 20),
        "sentiment_funding_crowded_long_threshold": _get_float(
            "SENTIMENT_FUNDING_CROWDED_LONG_THRESHOLD", 0.001
        ),
        "sentiment_funding_crowded_short_threshold": _get_float(
            "SENTIMENT_FUNDING_CROWDED_SHORT_THRESHOLD", -0.0005
        ),
        "sentiment_gas_elevated_gwei": _get_float("SENTIMENT_GAS_ELEVATED_GWEI", 0.3),
        "sentiment_max_negative_delta": _get_float("SENTIMENT_MAX_NEGATIVE_DELTA", -2.5),
        "sentiment_max_positive_delta": _get_float("SENTIMENT_MAX_POSITIVE_DELTA", 1.0),
        "bnb_regime_threshold": _get_float("BNB_REGIME_THRESHOLD", -0.01),
        "token_regime_1h_min": _get_float("TOKEN_REGIME_1H_MIN", 0.0025),
        "token_regime_24h_min": _get_float("TOKEN_REGIME_24H_MIN", -0.08),
        "breakout_lookback_hours": _get_int("BREAKOUT_LOOKBACK_HOURS", 3),
        "breakout_buffer": _get_float("BREAKOUT_BUFFER", 0.002),
        "min_entry_factors": _get_int("MIN_ENTRY_FACTORS", 4),
        "regime_size_multiplier": _get_float("REGIME_SIZE_MULTIPLIER", 0.5),
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        "demo_mode": _get_bool("DEMO_MODE", False),
        "position_state_path": os.getenv("POSITION_STATE_PATH", "positions.json"),
        "guardrail_state_path": os.getenv("GUARDRAIL_STATE_PATH", "guardrail_state.json"),
        "execution_log_path": os.getenv("EXECUTION_LOG_PATH", "execution_log.jsonl"),
        "decision_log_path": os.getenv("DECISION_LOG_PATH", "decision_log.jsonl"),
        "strategy_mode": os.getenv("STRATEGY_MODE", "breakout"),
        "scalping_entry_score_min": _get_float("SCALPING_ENTRY_SCORE_MIN", 60.0),
        "scalping_position_pct": _get_float("SCALPING_POSITION_PCT", 0.01),
        "scalping_take_profit_pct": _get_float("SCALPING_TAKE_PROFIT_PCT", 0.015),
        "scalping_stop_loss_pct": _get_float("SCALPING_STOP_LOSS_PCT", 0.008),
        "scalping_max_hold_minutes": _get_int("SCALPING_MAX_HOLD_MINUTES", 30),
        "scalping_time_stop_minutes": _get_int("SCALPING_TIME_STOP_MINUTES", 20),
        "scalping_symbol_cooldown_minutes": _get_int("SCALPING_SYMBOL_COOLDOWN_MINUTES", 15),
        "scalping_daily_loss_cap_pct": _get_float("SCALPING_DAILY_LOSS_CAP_PCT", 0.02),
        "scalping_max_daily_trades": _get_int("SCALPING_MAX_DAILY_TRADES", 10),
        "scalping_max_gas_gwei": _get_float("SCALPING_MAX_GAS_GWEI", 5.0),
        "scalping_max_slippage_pct": _get_float("SCALPING_MAX_SLIPPAGE_PCT", 0.005),
        "scalping_pump_filter_15m_pct": _get_float("SCALPING_PUMP_FILTER_15M_PCT", 0.05),
        "scalping_min_market_cap_usd": _get_float("SCALPING_MIN_MARKET_CAP_USD", 1_000_000.0),
        "scalping_consecutive_loss_limit": _get_int("SCALPING_CONSECUTIVE_LOSS_LIMIT", 3),
        "scalping_consecutive_loss_cooldown_hours": _get_float("SCALPING_CONSECUTIVE_LOSS_COOLDOWN_HOURS", 1.0),
        "ml_enabled": _get_bool("ML_ENABLED", False),
        "ml_model_path": os.getenv("ML_MODEL_PATH", "models/regime_lgbm_v1.pkl"),
        "ml_regime_threshold": _get_float("ML_REGIME_THRESHOLD", 0.55),
        "ml_momentum_size_multiplier": _get_float("ML_MOMENTUM_SIZE_MULTIPLIER", 1.0),
        "ml_chop_size_multiplier": _get_float("ML_CHOP_SIZE_MULTIPLIER", 0.5),
        "ml_universe_symbols": _get_symbol_list("ML_UNIVERSE_SYMBOLS", ML_DEFAULT_20),
        "ml_ohlcv_cache_db": os.getenv("ML_OHLCV_CACHE_DB", "data/ml_ohlcv_cache.sqlite"),
        "ml_ohlcv_lookback_candles": _get_int("ML_OHLCV_LOOKBACK_CANDLES", 96),
        "ml_volume_breakout_multiplier": _get_float("ML_VOLUME_BREAKOUT_MULTIPLIER", 2.0),
        "ml_volume_cache_multiplier": _get_float("ML_VOLUME_CACHE_MULTIPLIER", 1.2),
        "ml_min_auc": _get_float("ML_MIN_AUC", 0.65),
        "ml_shadow_mode": _get_bool("ML_SHADOW_MODE", True),
        "ml_regime_only_chop_multiplier": _get_float("ML_REGIME_ONLY_CHOP_MULTIPLIER", 0.3),
        "telegram_bot_token": _none_if_blank(os.getenv("TELEGRAM_BOT_TOKEN")),
        "telegram_chat_id": _none_if_blank(os.getenv("TELEGRAM_CHAT_ID")),
        "health_check_port": _get_int("HEALTH_CHECK_PORT", 8080),
        "log_rotate_max_mb": _get_float("LOG_ROTATE_MAX_MB", 50.0),
        "cmc_collector_enabled": _get_bool("CMC_COLLECTOR_ENABLED", True),
        "cmc_collector_interval_min": _get_int("CMC_COLLECTOR_INTERVAL_MIN", 15),
        "min_bnb_gas": _get_float("MIN_BNB_GAS", 0.05),
        "min_usdc_balance": _get_float("MIN_USDC_BALANCE", 50.0),
        "disk_guard_min_free_bytes": _get_int("DISK_GUARD_MIN_FREE_BYTES", 500_000_000),
    }
    mode = str(values.get("strategy_mode", "breakout")).strip().lower()
    if mode not in {"breakout", "scalping"}:
        mode = "breakout"
    values["strategy_mode"] = mode
    return Settings(**values)

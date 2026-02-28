"""
Unified configuration for the trading system.
Uses Pydantic for validation and environment variable support.
"""
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class RiskConfig:
    initial_capital: float = 10000.0
    risk_per_trade: float = 0.02
    max_drawdown: float = 0.20
    max_position_size: float = 0.25
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    max_leverage: float = 1.0
    position_limit: int = 5

    def __post_init__(self):
        if not 0 < self.risk_per_trade < 1:
            raise ValueError(f"risk_per_trade must be between 0 and 1, got {self.risk_per_trade}")
        if not 0 < self.max_drawdown < 1:
            raise ValueError(f"max_drawdown must be between 0 and 1, got {self.max_drawdown}")
        if self.initial_capital <= 0:
            raise ValueError(f"initial_capital must be positive, got {self.initial_capital}")
        if not 0 < self.max_position_size <= 1:
            raise ValueError(f"max_position_size must be between 0 and 1, got {self.max_position_size}")


@dataclass
class DataConfig:
    symbol: str = "BTCUSDT"
    timeframe: str = "15m"
    max_lookback_bars: int = 3000
    binance_base_url: str = "https://api.binance.com/api/v3/klines"
    binance_api_key: str = ""
    binance_api_secret: str = ""
    cache_dir: str = "data/cache"

    VALID_TIMEFRAMES = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"}

    def __post_init__(self):
        self.binance_api_key = os.environ.get("BINANCE_API_KEY", self.binance_api_key)
        self.binance_api_secret = os.environ.get("BINANCE_API_SECRET", self.binance_api_secret)
        if self.timeframe not in self.VALID_TIMEFRAMES:
            raise ValueError(f"Invalid timeframe '{self.timeframe}', must be one of {self.VALID_TIMEFRAMES}")
        if self.max_lookback_bars < 1:
            raise ValueError(f"max_lookback_bars must be >= 1, got {self.max_lookback_bars}")
        env = os.environ.get("ENVIRONMENT", "development")
        if env == "production":
            if not self.binance_api_key:
                raise ValueError("BINANCE_API_KEY must be set in production environment")
            if not self.binance_api_secret:
                raise ValueError("BINANCE_API_SECRET must be set in production environment")


@dataclass
class APIConfig:
    max_retries: int = 3
    timeout: int = 10
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    def __post_init__(self):
        self.telegram_bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", self.telegram_bot_token)
        self.telegram_chat_id = os.environ.get("TELEGRAM_CHAT_ID", self.telegram_chat_id)
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")
        if self.timeout < 1:
            raise ValueError(f"timeout must be >= 1, got {self.timeout}")
        if self.telegram_bot_token and ":" not in self.telegram_bot_token:
            raise ValueError("telegram_bot_token must follow the format '<bot_id>:<token_hash>'")


@dataclass
class LogConfig:
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = "logs/trading_system.log"


@dataclass
class TradingConfig:
    """Unified trading configuration with validation."""
    risk: RiskConfig = field(default_factory=RiskConfig)
    data: DataConfig = field(default_factory=DataConfig)
    api: APIConfig = field(default_factory=APIConfig)
    log: LogConfig = field(default_factory=LogConfig)

    # Legacy compatibility attributes
    INITIAL_CAPITAL: float = 10000.0
    DEFAULT_RISK_PERC: float = 0.02
    MAX_DRAWDOWN: float = 0.20
    MIN_TRADE_SIZE: int = 1
    MAX_POSITION_SIZE: int = 100
    COMMISSION_RATE: float = 0.001
    DEFAULT_TIMEFRAME: str = "15m"
    DEFAULT_SYMBOL: str = "BTCUSDT"
    MAX_LOOKBACK_BARS: int = 3000
    BINANCE_BASE_URL: str = "https://api.binance.com/api/v3/klines"
    MAX_RETRIES: int = 3
    TIMEOUT: int = 10
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    def __post_init__(self):
        self.INITIAL_CAPITAL = self.risk.initial_capital
        self.DEFAULT_RISK_PERC = self.risk.risk_per_trade
        self.MAX_DRAWDOWN = self.risk.max_drawdown
        self.COMMISSION_RATE = self.risk.commission_rate
        self.DEFAULT_TIMEFRAME = self.data.timeframe
        self.DEFAULT_SYMBOL = self.data.symbol
        self.MAX_LOOKBACK_BARS = self.data.max_lookback_bars
        self.BINANCE_BASE_URL = self.data.binance_base_url
        self.MAX_RETRIES = self.api.max_retries
        self.TIMEOUT = self.api.timeout
        self.LOG_LEVEL = self.log.level
        self.LOG_FORMAT = self.log.format

    @classmethod
    def from_yaml(cls, path: str) -> "TradingConfig":
        """Load configuration from YAML file with env var overrides."""
        import yaml
        with open(path, 'r') as f:
            raw = yaml.safe_load(f)

        risk_raw = raw.get('risk_config', {})
        data_raw = raw.get('data_config', {})

        risk = RiskConfig(
            initial_capital=float(os.environ.get(
                "INITIAL_CAPITAL",
                risk_raw.get('initial_capital', 10000.0)
            )),
            risk_per_trade=risk_raw.get('position_sizing', {}).get('base_risk_per_trade', 0.02),
            max_drawdown=risk_raw.get('portfolio', {}).get('max_drawdown', 0.20),
            max_position_size=risk_raw.get('position_sizing', {}).get('max_position_size', 0.25),
            commission_rate=risk_raw.get('commission_rate', 0.001),
            slippage_rate=risk_raw.get('slippage_rate', 0.0005),
            max_leverage=risk_raw.get('portfolio', {}).get('max_leverage', 1.0),
            position_limit=risk_raw.get('portfolio', {}).get('position_limit', 5),
        )

        data = DataConfig(
            symbol=f"{data_raw.get('base_symbol', 'BTC')}{data_raw.get('quote_symbol', 'USDT')}",
            timeframe=data_raw.get('timeframe', '15m'),
        )

        return cls(risk=risk, data=data)

    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert config to dictionary (legacy compat)."""
        return {k: v for k, v in cls.__dict__.items()
                if not k.startswith('_') and k.isupper()}

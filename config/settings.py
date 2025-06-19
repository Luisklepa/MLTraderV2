"""
Global configuration settings for the trading system.
"""
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class TradingConfig:
    # Risk Management
    INITIAL_CAPITAL: float = 10000.0
    DEFAULT_RISK_PERC: float = 0.02
    MAX_DRAWDOWN: float = 0.20
    
    # Trading Parameters
    MIN_TRADE_SIZE: int = 1
    MAX_POSITION_SIZE: int = 100
    COMMISSION_RATE: float = 0.001
    
    # Data Parameters
    DEFAULT_TIMEFRAME: str = '15m'
    DEFAULT_SYMBOL: str = 'BTCUSDT'
    MAX_LOOKBACK_BARS: int = 3000
    
    # API Configuration
    BINANCE_BASE_URL: str = 'https://api.binance.com/api/v3/klines'
    MAX_RETRIES: int = 3
    TIMEOUT: int = 10
    
    # Logging Configuration
    LOG_LEVEL: str = 'INFO'
    LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {k: v for k, v in cls.__dict__.items() 
                if not k.startswith('_') and k.isupper()} 
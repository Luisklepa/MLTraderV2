"""Custom exception hierarchy for MLTraderV2."""


class MLTraderError(Exception):
    """Base exception for the trading system."""


class DataError(MLTraderError):
    """Raised when data is invalid, missing, or corrupted."""


class ModelError(MLTraderError):
    """Raised when model loading, prediction, or training fails."""


class TradingError(MLTraderError):
    """Raised when trade execution or position management fails."""


class ConfigError(MLTraderError):
    """Raised when configuration is invalid or missing."""

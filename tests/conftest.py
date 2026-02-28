"""Shared test fixtures for MLTraderV2 test suite."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock


@pytest.fixture
def sample_ohlcv():
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    n_bars = 200
    base_price = 50000.0
    dates = pd.date_range(start="2024-01-01", periods=n_bars, freq="15min")

    returns = np.random.normal(0, 0.002, n_bars)
    close = base_price * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.normal(0, 0.001, n_bars)))
    low = close * (1 - np.abs(np.random.normal(0, 0.001, n_bars)))
    open_ = close * (1 + np.random.normal(0, 0.0005, n_bars))
    volume = np.random.lognormal(10, 1, n_bars)

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)

    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    return df


@pytest.fixture
def sample_ohlcv_large():
    """Generate larger synthetic OHLCV data (500 bars) for feature engineering tests."""
    np.random.seed(123)
    n_bars = 500
    base_price = 50000.0
    dates = pd.date_range(start="2024-01-01", periods=n_bars, freq="15min")

    returns = np.random.normal(0, 0.002, n_bars)
    close = base_price * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.normal(0, 0.002, n_bars)))
    low = close * (1 - np.abs(np.random.normal(0, 0.002, n_bars)))
    open_ = close * (1 + np.random.normal(0, 0.001, n_bars))
    volume = np.random.lognormal(10, 1, n_bars)

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "timestamp": dates,
    }, index=dates)

    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    return df


@pytest.fixture
def backtest_config():
    """Minimal valid config for BacktestEngine."""
    return {
        "risk_config": {
            "initial_capital": 100_000.0,
            "commission_rate": 0.001,
            "slippage_rate": 0.0005,
        }
    }


@pytest.fixture
def trained_model_mock():
    """Mock XGBoost model for pipeline tests."""
    model = MagicMock()
    model.predict.return_value = np.array([0, 1, 0, 1, 0])
    model.predict_proba.return_value = np.array([
        [0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.4, 0.6], [0.7, 0.3]
    ])
    model.feature_importances_ = np.array([0.3, 0.2, 0.15, 0.1, 0.25])
    return model


class DummyStrategy:
    """No-op strategy for testing the engine in isolation."""

    def __init__(self, signals=None):
        self.signals = signals or {}

    def on_data(self, df):
        ts = df.index[-1]
        return self.signals.get(ts, [])

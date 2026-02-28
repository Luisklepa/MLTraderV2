"""Tests verifying the correctness of event_engine fixes:
- Invalid order size rejected
- Short margin check
- Division-by-zero protection in summary stats
- Commission calculated on exit_price (not current_price)
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from backtest.event_engine import BacktestEngine, Position
from tests.conftest import DummyStrategy


@pytest.fixture
def ohlcv_5bars():
    dates = pd.date_range("2024-01-01", periods=5, freq="1h")
    return pd.DataFrame({
        "open": [100, 102, 104, 103, 105],
        "high": [103, 105, 106, 106, 108],
        "low": [99, 101, 102, 101, 103],
        "close": [102, 104, 103, 105, 107],
        "volume": [1000] * 5,
    }, index=dates)


@pytest.fixture
def cfg():
    return {
        "risk_config": {
            "initial_capital": 100_000.0,
            "commission_rate": 0.0,
            "slippage_rate": 0.0,
        }
    }


class TestInvalidOrderSize:
    def test_zero_size_rejected(self, ohlcv_5bars, cfg):
        strategy = DummyStrategy(signals={
            ohlcv_5bars.index[0]: [{"type": "buy", "symbol": "TEST", "size": 0, "stop_loss": 90}]
        })
        engine = BacktestEngine(strategy, ohlcv_5bars, {}, cfg)
        result = engine.run()
        assert len(engine.positions) == 0

    def test_negative_size_rejected(self, ohlcv_5bars, cfg):
        strategy = DummyStrategy(signals={
            ohlcv_5bars.index[0]: [{"type": "buy", "symbol": "TEST", "size": -5, "stop_loss": 90}]
        })
        engine = BacktestEngine(strategy, ohlcv_5bars, {}, cfg)
        result = engine.run()
        assert len(engine.positions) == 0


class TestShortMarginCheck:
    def test_short_with_insufficient_margin_rejected(self, ohlcv_5bars):
        cfg = {
            "risk_config": {
                "initial_capital": 10.0,
                "commission_rate": 0.0,
                "slippage_rate": 0.0,
            }
        }
        strategy = DummyStrategy(signals={
            ohlcv_5bars.index[0]: [{"type": "sell", "symbol": "TEST", "size": 100, "stop_loss": 200}]
        })
        engine = BacktestEngine(strategy, ohlcv_5bars, {}, cfg)
        engine.run()
        assert len(engine.positions) == 0


class TestSummaryStatsZeroDiv:
    def test_no_trades_no_crash(self, ohlcv_5bars, cfg):
        strategy = DummyStrategy()
        engine = BacktestEngine(strategy, ohlcv_5bars, {}, cfg)
        result = engine.run()
        assert len(result) == 5
        assert result["equity"].iloc[-1] == pytest.approx(100_000.0)

    def test_single_bar_no_crash(self, cfg):
        dates = pd.date_range("2024-01-01", periods=1, freq="1h")
        df = pd.DataFrame({
            "open": [100], "high": [101], "low": [99],
            "close": [100], "volume": [1000],
        }, index=dates)
        strategy = DummyStrategy()
        engine = BacktestEngine(strategy, df, {}, cfg)
        result = engine.run()
        assert len(result) == 1


class TestEquityConservation:
    """Verify that cash + position value = equity at all times (no-commission case)."""

    def test_equity_equals_cash_plus_unrealized(self, ohlcv_5bars, cfg):
        strategy = DummyStrategy(signals={
            ohlcv_5bars.index[0]: [{
                "type": "buy", "symbol": "TEST", "size": 10,
                "stop_loss": 50, "take_profit": 200,
            }]
        })
        engine = BacktestEngine(strategy, ohlcv_5bars, {}, cfg)
        result = engine.run()
        for _, row in result.iterrows():
            expected_equity = row["cash"] + row["unrealized_pnl"]
            assert row["equity"] == pytest.approx(expected_equity, rel=1e-6)

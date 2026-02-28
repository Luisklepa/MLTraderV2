"""Tests for the BacktestEngine in backtest/event_engine.py."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from backtest.event_engine import BacktestEngine, Position
from tests.conftest import DummyStrategy


class TestPosition:
    def test_position_creation(self):
        pos = Position(
            symbol="BTCUSDT",
            type="long",
            size=1.0,
            entry_price=50000.0,
            entry_time=datetime(2024, 1, 1),
            stop_loss=49000.0,
            take_profit=52000.0,
        )
        assert pos.pnl == 0.0
        assert pos.unrealized_pnl == 0.0
        assert pos.type == "long"


class TestSlippageAndCommission:
    def test_buy_slippage_increases_price(self, sample_ohlcv, backtest_config):
        engine = BacktestEngine(DummyStrategy(), sample_ohlcv, {}, backtest_config)
        slipped = engine.calculate_slippage(100.0, 1.0, is_buy=True)
        assert slipped > 100.0

    def test_sell_slippage_decreases_price(self, sample_ohlcv, backtest_config):
        engine = BacktestEngine(DummyStrategy(), sample_ohlcv, {}, backtest_config)
        slipped = engine.calculate_slippage(100.0, 1.0, is_buy=False)
        assert slipped < 100.0

    def test_commission_proportional(self, sample_ohlcv, backtest_config):
        engine = BacktestEngine(DummyStrategy(), sample_ohlcv, {}, backtest_config)
        c1 = engine.calculate_commission(100.0, 1.0)
        c2 = engine.calculate_commission(100.0, 2.0)
        assert abs(c2 - 2 * c1) < 1e-10


class TestCashAccounting:
    """Verify cash conservation across full long and short trade cycles."""

    def _make_engine(self, sample_ohlcv, backtest_config):
        engine = BacktestEngine(DummyStrategy(), sample_ohlcv, {}, backtest_config)
        engine.current_time = sample_ohlcv.index[0]
        engine.slippage_rate = 0.0
        engine.commission_rate = 0.0
        return engine

    def test_long_round_trip_zero_cost(self, sample_ohlcv, backtest_config):
        """Buy and sell at the same price with zero costs should return to initial capital."""
        engine = self._make_engine(sample_ohlcv, backtest_config)
        initial = engine.cash

        order = {"symbol": "BTC", "type": "buy", "size": 1.0}
        pos = engine.execute_order(order, 50000.0)
        assert pos is not None

        cash_after_buy = engine.cash
        assert cash_after_buy == initial - 50000.0

        engine.close_position(pos, 50000.0, "test")
        assert abs(engine.cash - initial) < 1e-6

    def test_long_round_trip_profit(self, sample_ohlcv, backtest_config):
        """Long trade with price increase should yield positive PnL."""
        engine = self._make_engine(sample_ohlcv, backtest_config)
        initial = engine.cash

        order = {"symbol": "BTC", "type": "buy", "size": 1.0}
        pos = engine.execute_order(order, 50000.0)
        engine.close_position(pos, 51000.0, "test")

        assert engine.cash > initial
        expected_profit = 1000.0
        assert abs(engine.cash - initial - expected_profit) < 1e-6

    def test_short_round_trip_zero_cost(self, sample_ohlcv, backtest_config):
        """Short at a price and buy back at same price should return to initial capital."""
        engine = self._make_engine(sample_ohlcv, backtest_config)
        initial = engine.cash

        order = {"symbol": "BTC", "type": "sell", "size": 1.0}
        pos = engine.execute_order(order, 50000.0)
        assert pos is not None
        assert pos.type == "short"

        cash_after_sell = engine.cash
        assert cash_after_sell == initial + 50000.0

        engine.close_position(pos, 50000.0, "test")
        assert abs(engine.cash - initial) < 1e-6

    def test_short_round_trip_profit(self, sample_ohlcv, backtest_config):
        """Short trade with price decrease should yield positive PnL."""
        engine = self._make_engine(sample_ohlcv, backtest_config)
        initial = engine.cash

        order = {"symbol": "BTC", "type": "sell", "size": 1.0}
        pos = engine.execute_order(order, 50000.0)
        engine.close_position(pos, 49000.0, "test")

        assert engine.cash > initial
        expected_profit = 1000.0
        assert abs(engine.cash - initial - expected_profit) < 1e-6

    def test_long_with_commission(self, sample_ohlcv, backtest_config):
        """Commission should reduce final cash."""
        engine = BacktestEngine(DummyStrategy(), sample_ohlcv, {}, backtest_config)
        engine.current_time = sample_ohlcv.index[0]
        engine.slippage_rate = 0.0
        initial = engine.cash

        order = {"symbol": "BTC", "type": "buy", "size": 1.0}
        pos = engine.execute_order(order, 50000.0)
        engine.close_position(pos, 50000.0, "test")

        assert engine.cash < initial

    def test_insufficient_cash_rejected(self, sample_ohlcv, backtest_config):
        """Order exceeding available cash should be rejected."""
        engine = self._make_engine(sample_ohlcv, backtest_config)
        order = {"symbol": "BTC", "type": "buy", "size": 100.0}
        pos = engine.execute_order(order, 50000.0)
        assert pos is None


class TestStopLossAndTakeProfit:
    def test_long_stop_loss_triggers(self, sample_ohlcv, backtest_config):
        engine = BacktestEngine(DummyStrategy(), sample_ohlcv, {}, backtest_config)
        engine.current_time = sample_ohlcv.index[0]

        pos = Position(
            symbol="BTC", type="long", size=1.0,
            entry_price=50000.0, entry_time=engine.current_time,
            stop_loss=49000.0, take_profit=52000.0,
        )
        engine.positions["BTC"] = pos

        bar = pd.Series({"open": 49500, "high": 49800, "low": 48500, "close": 48800})
        engine.update_positions(bar)

        assert "BTC" not in engine.positions
        assert len(engine.trade_stats["pnl"]) == 1

    def test_long_take_profit_triggers(self, sample_ohlcv, backtest_config):
        engine = BacktestEngine(DummyStrategy(), sample_ohlcv, {}, backtest_config)
        engine.current_time = sample_ohlcv.index[0]

        pos = Position(
            symbol="BTC", type="long", size=1.0,
            entry_price=50000.0, entry_time=engine.current_time,
            stop_loss=49000.0, take_profit=52000.0,
        )
        engine.positions["BTC"] = pos

        bar = pd.Series({"open": 51000, "high": 52500, "low": 50800, "close": 52200})
        engine.update_positions(bar)

        assert "BTC" not in engine.positions

    def test_short_stop_loss_triggers(self, sample_ohlcv, backtest_config):
        engine = BacktestEngine(DummyStrategy(), sample_ohlcv, {}, backtest_config)
        engine.current_time = sample_ohlcv.index[0]

        pos = Position(
            symbol="BTC", type="short", size=1.0,
            entry_price=50000.0, entry_time=engine.current_time,
            stop_loss=51000.0, take_profit=48000.0,
        )
        engine.positions["BTC"] = pos

        bar = pd.Series({"open": 50500, "high": 51500, "low": 50200, "close": 51200})
        engine.update_positions(bar)

        assert "BTC" not in engine.positions

    def test_unrealized_pnl_updates(self, sample_ohlcv, backtest_config):
        engine = BacktestEngine(DummyStrategy(), sample_ohlcv, {}, backtest_config)
        engine.current_time = sample_ohlcv.index[0]

        pos = Position(
            symbol="BTC", type="long", size=1.0,
            entry_price=50000.0, entry_time=engine.current_time,
            stop_loss=40000.0, take_profit=60000.0,
        )
        engine.positions["BTC"] = pos

        bar = pd.Series({"open": 50500, "high": 51000, "low": 50200, "close": 50800})
        engine.update_positions(bar)

        assert "BTC" in engine.positions
        assert abs(pos.unrealized_pnl - 800.0) < 1e-6


class TestNextBarExecution:
    """Verify orders are queued and executed on the next bar (no lookahead)."""

    def test_orders_execute_on_next_bar(self, sample_ohlcv, backtest_config):
        ts0 = sample_ohlcv.index[0]
        ts1 = sample_ohlcv.index[1]

        signals = {
            ts0: [{"type": "buy", "size": 0.1, "symbol": "BTC", "stop_loss": 40000, "take_profit": 60000}],
        }
        strategy = DummyStrategy(signals)
        engine = BacktestEngine(strategy, sample_ohlcv.iloc[:5], {}, backtest_config)
        results = engine.run()

        buy_trades = [t for t in engine.trades if t["type"] == "buy"]
        assert len(buy_trades) >= 1
        assert buy_trades[0]["time"] == ts1


class TestRunIntegration:
    def test_run_returns_dataframe(self, sample_ohlcv, backtest_config):
        engine = BacktestEngine(DummyStrategy(), sample_ohlcv.iloc[:20], {}, backtest_config)
        results = engine.run()
        assert isinstance(results, pd.DataFrame)
        assert "equity" in results.columns
        assert "drawdown" in results.columns
        assert len(results) == 20

    def test_no_trades_no_crash(self, sample_ohlcv, backtest_config):
        engine = BacktestEngine(DummyStrategy(), sample_ohlcv.iloc[:10], {}, backtest_config)
        results = engine.run()
        assert len(results) == 10
        assert results["equity"].iloc[-1] == pytest.approx(100_000.0, rel=1e-4)

"""Tests for backtest/engine.py — BacktestResult, BacktestEngine utilities."""
import numpy as np
import pandas as pd
import pytest

from backtest.engine import BacktestResult, BacktestEngine


class TestBacktestResult:
    def test_dataclass_fields(self):
        result = BacktestResult(
            strategy_name="TestStrategy",
            final_value=11000.0,
            return_pct=10.0,
            sharpe_ratio=1.5,
            max_drawdown=5.0,
            total_trades=50,
            win_rate=60.0,
            parameters={"atr": 2.0},
        )
        assert result.strategy_name == "TestStrategy"
        assert result.final_value == 11000.0
        assert result.return_pct == 10.0
        assert result.sharpe_ratio == 1.5
        assert result.max_drawdown == 5.0
        assert result.total_trades == 50
        assert result.win_rate == 60.0
        assert result.parameters == {"atr": 2.0}


class TestCalculateProfitFactor:
    @pytest.fixture
    def engine(self):
        return BacktestEngine(initial_cash=10000)

    def test_normal_case(self, engine):
        ta = {"won": {"pnl": {"total": 500}}, "lost": {"pnl": {"total": -200}}}
        pf = engine.calculate_profit_factor(ta)
        assert pf == pytest.approx(500 / 200)

    def test_zero_loss(self, engine):
        ta = {"won": {"pnl": {"total": 500}}, "lost": {"pnl": {"total": 0}}}
        pf = engine.calculate_profit_factor(ta)
        assert pf == float("inf")

    def test_zero_both(self, engine):
        ta = {"won": {"pnl": {"total": 0}}, "lost": {"pnl": {"total": 0}}}
        pf = engine.calculate_profit_factor(ta)
        assert pf == 1.0

    def test_missing_keys(self, engine):
        pf = engine.calculate_profit_factor({})
        assert pf == 1.0

    def test_flat_pnl_values(self, engine):
        ta = {"won": {"pnl": 300}, "lost": {"pnl": -100}}
        pf = engine.calculate_profit_factor(ta)
        assert pf == pytest.approx(3.0)


class TestCalculateRobustScore:
    @pytest.fixture
    def engine(self):
        return BacktestEngine(initial_cash=10000)

    def test_returns_neg_inf_insufficient_trades(self, engine):
        result = BacktestResult("S", 10500, 5, 1.0, 5.0, 5, 55.0, {})
        score = engine.calculate_robust_score(result, 1.5)
        assert score == -float("inf")

    def test_returns_neg_inf_high_drawdown(self, engine):
        result = BacktestResult("S", 10500, 5, 1.0, 30.0, 50, 55.0, {})
        score = engine.calculate_robust_score(result, 1.5)
        assert score == -float("inf")

    def test_returns_neg_inf_low_sharpe(self, engine):
        result = BacktestResult("S", 10500, 5, 0.3, 5.0, 50, 55.0, {})
        score = engine.calculate_robust_score(result, 1.5)
        assert score == -float("inf")

    def test_returns_neg_inf_low_profit_factor(self, engine):
        result = BacktestResult("S", 10500, 5, 1.0, 5.0, 50, 55.0, {})
        score = engine.calculate_robust_score(result, 1.0)
        assert score == -float("inf")

    def test_positive_score_when_valid(self, engine):
        result = BacktestResult("S", 11000, 10, 1.5, 10.0, 50, 60.0, {})
        score = engine.calculate_robust_score(result, 2.0)
        assert score > 0

    def test_higher_sharpe_yields_higher_score(self, engine):
        r1 = BacktestResult("S", 11000, 10, 1.5, 10.0, 50, 60.0, {})
        r2 = BacktestResult("S", 11000, 10, 2.5, 10.0, 50, 60.0, {})
        s1 = engine.calculate_robust_score(r1, 2.0)
        s2 = engine.calculate_robust_score(r2, 2.0)
        assert s2 > s1


class TestEngineInit:
    def test_default_initial_cash(self):
        engine = BacktestEngine()
        assert engine.initial_cash > 0

    def test_custom_initial_cash(self):
        engine = BacktestEngine(initial_cash=50000)
        assert engine.initial_cash == 50000

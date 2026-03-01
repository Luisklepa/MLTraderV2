"""Tests for the RiskManager in core/risk_management.py."""

import pandas as pd

from core.risk_management import RiskManager


class TestPositionSizing:
    def test_higher_confidence_larger_position(self):
        rm = RiskManager()
        size_high = rm.calculate_position_size(
            prediction_probability=0.9,
            volatility=0.02,
            correlation_risk=0.0,
            current_exposure=0.0,
            account_size=100_000,
        )
        size_low = rm.calculate_position_size(
            prediction_probability=0.5,
            volatility=0.02,
            correlation_risk=0.0,
            current_exposure=0.0,
            account_size=100_000,
        )
        assert size_high > size_low

    def test_higher_volatility_smaller_position(self):
        rm = RiskManager()
        size_low_vol = rm.calculate_position_size(
            prediction_probability=0.8,
            volatility=0.01,
            correlation_risk=0.0,
            current_exposure=0.0,
            account_size=100_000,
        )
        size_high_vol = rm.calculate_position_size(
            prediction_probability=0.8,
            volatility=0.10,
            correlation_risk=0.0,
            current_exposure=0.0,
            account_size=100_000,
        )
        assert size_low_vol > size_high_vol

    def test_position_respects_max_size(self):
        rm = RiskManager({"max_position_size": 0.05})
        size = rm.calculate_position_size(
            prediction_probability=1.0,
            volatility=0.001,
            correlation_risk=0.0,
            current_exposure=0.0,
            account_size=100_000,
        )
        assert size <= 100_000 * 0.05 + 1e-6

    def test_full_exposure_blocks_new_position(self):
        rm = RiskManager()
        size = rm.calculate_position_size(
            prediction_probability=0.9,
            volatility=0.02,
            correlation_risk=0.0,
            current_exposure=1.0,
            account_size=100_000,
        )
        assert size == 0.0


class TestAdaptiveStops:
    def test_long_stop_below_entry(self):
        rm = RiskManager()
        sl, tp = rm.calculate_adaptive_stops(
            entry_price=50000.0,
            atr=500.0,
            volatility=pd.Series([0.02] * 20),
            trend_strength=0.5,
            position_type="long",
        )
        assert sl < 50000.0
        assert tp > 50000.0

    def test_short_stop_above_entry(self):
        rm = RiskManager()
        sl, tp = rm.calculate_adaptive_stops(
            entry_price=50000.0,
            atr=500.0,
            volatility=pd.Series([0.02] * 20),
            trend_strength=0.5,
            position_type="short",
        )
        assert sl > 50000.0
        assert tp < 50000.0

    def test_min_stop_distance_enforced(self):
        rm = RiskManager({"min_stop_distance": 0.02})
        sl, _ = rm.calculate_adaptive_stops(
            entry_price=50000.0,
            atr=1.0,
            volatility=pd.Series([0.001] * 20),
            trend_strength=0.0,
            position_type="long",
        )
        assert (50000.0 - sl) >= 50000.0 * 0.02 - 1e-6


class TestDrawdownCheck:
    def test_no_drawdown_not_breached(self):
        rm = RiskManager()
        equity = pd.Series([100, 101, 102, 103])
        assert not rm.check_drawdown_breach(equity)

    def test_large_drawdown_breached(self):
        rm = RiskManager({"max_drawdown": 0.10})
        equity = pd.Series([100, 105, 110, 90])
        assert rm.check_drawdown_breach(equity)

    def test_boundary_drawdown(self):
        rm = RiskManager({"max_drawdown": 0.19})
        equity = pd.Series([100, 80])
        assert rm.check_drawdown_breach(equity)


class TestPortfolioConstraints:
    def test_max_positions_blocks_new(self):
        rm = RiskManager({"max_open_positions": 2})
        positions = {
            "BTC": {"size": 0.5, "asset": "BTC"},
            "ETH": {"size": 0.3, "asset": "ETH"},
        }
        correlations = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], index=["BTC", "ETH"], columns=["BTC", "ETH"])
        new_pos = {"asset": "SOL", "size": 0.2}
        result = rm.adjust_for_portfolio_constraints(new_pos, positions, correlations)
        assert result is None

    def test_portfolio_risk_zero_for_empty(self):
        rm = RiskManager()
        risk = rm.calculate_portfolio_risk({}, pd.DataFrame())
        assert risk == 0.0

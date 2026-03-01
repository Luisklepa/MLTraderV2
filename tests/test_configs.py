"""Tests for config/robustness_config.py and config/walk_forward_config.py."""

from config.robustness_config import RobustnessConfig
from config.walk_forward_config import WalkForwardConfig


# ---------------------------------------------------------------------------
# RobustnessConfig
# ---------------------------------------------------------------------------
class TestRobustnessConfig:
    def test_default_values(self):
        cfg = RobustnessConfig()
        assert cfg.MIN_TRADES_PER_PERIOD == 10
        assert cfg.MIN_SHARPE_RATIO == 0.5
        assert cfg.SIGNIFICANCE_LEVEL == 0.05
        assert cfg.N_SIMULATIONS == 1000

    def test_score_weights_sum_to_one(self):
        cfg = RobustnessConfig()
        total = sum(cfg.SCORE_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-6

    def test_validate_results_all_pass(self):
        cfg = RobustnessConfig()
        metrics = {
            "trades_mean": 20,
            "return_sharpe": 1.0,
            "win_rate_mean": 0.55,
            "drawdown_mean": -0.10,
            "return_cv": 1.0,
            "win_rate_std": 0.05,
            "regime_consistency": 0.8,
            "mc_positive_prob": 0.7,
            "mc_sharpe": 0.5,
            "mc_var_95": -0.1,
            "returns_p_value": 0.01,
        }
        result = cfg.validate_results(metrics)
        assert result["sufficient_trades"] is True
        assert result["sharpe_ratio"] is True
        assert result["statistical_significance"] is True
        assert result["monte_carlo"] is True
        assert result["regime_consistency"] is True

    def test_validate_results_insufficient_trades(self):
        cfg = RobustnessConfig()
        metrics = {"trades_mean": 3}
        result = cfg.validate_results(metrics)
        assert result["sufficient_trades"] is False

    def test_validate_results_bad_sharpe(self):
        cfg = RobustnessConfig()
        metrics = {"return_sharpe": 0.2}
        result = cfg.validate_results(metrics)
        assert result["sharpe_ratio"] is False

    def test_validate_results_mc_fails(self):
        cfg = RobustnessConfig()
        metrics = {"mc_positive_prob": 0.3, "mc_sharpe": 0.1, "mc_var_95": -0.5}
        result = cfg.validate_results(metrics)
        assert result["monte_carlo"] is False

    def test_calculate_final_score_critical_failure_returns_zero(self):
        cfg = RobustnessConfig()
        metrics = {"trades_mean": 0, "returns_p_value": 0.9}
        score = cfg.calculate_final_score(metrics)
        assert score == 0.0

    def test_calculate_final_score_positive_when_all_pass(self):
        cfg = RobustnessConfig()
        metrics = {
            "trades_mean": 20,
            "return_sharpe": 1.5,
            "win_rate_mean": 0.55,
            "drawdown_mean": -0.05,
            "return_cv": 0.5,
            "win_rate_std": 0.03,
            "regime_consistency": 0.9,
            "mc_positive_prob": 0.8,
            "mc_sharpe": 0.6,
            "mc_var_95": -0.05,
            "returns_p_value": 0.01,
            "trades_cv": 0.5,
            "drawdown_cv": 0.5,
            "long_short_balance": 1.0,
        }
        score = cfg.calculate_final_score(metrics)
        assert 0 < score <= 1.0

    def test_score_bounded_by_one(self):
        cfg = RobustnessConfig()
        metrics = {
            "trades_mean": 100,
            "return_sharpe": 5.0,
            "win_rate_mean": 0.8,
            "drawdown_mean": 0.0,
            "return_cv": 0.0,
            "win_rate_std": 0.0,
            "regime_consistency": 1.0,
            "mc_positive_prob": 1.0,
            "mc_sharpe": 3.0,
            "mc_var_95": 0.0,
            "returns_p_value": 0.001,
            "trades_cv": 0.0,
            "drawdown_cv": 0.0,
            "long_short_balance": 1.0,
        }
        score = cfg.calculate_final_score(metrics)
        assert score <= 1.0


# ---------------------------------------------------------------------------
# WalkForwardConfig
# ---------------------------------------------------------------------------
class TestWalkForwardConfig:
    def test_default_values(self):
        cfg = WalkForwardConfig()
        assert cfg.TRAIN_SIZE == 252
        assert cfg.TEST_SIZE == 63
        assert cfg.GAP_SIZE == 5
        assert cfg.USE_EXPANDING is False

    def test_param_grid_non_empty(self):
        cfg = WalkForwardConfig()
        assert len(cfg.PARAM_GRID) > 0
        assert "atr_multiplier" in cfg.PARAM_GRID
        assert "atr_periods" in cfg.PARAM_GRID

    def test_secondary_metrics(self):
        cfg = WalkForwardConfig()
        assert "total_return_pct" in cfg.SECONDARY_METRICS
        assert "max_drawdown" in cfg.SECONDARY_METRICS
        assert "profit_factor" in cfg.SECONDARY_METRICS
        assert "win_rate" in cfg.SECONDARY_METRICS

    def test_plot_settings(self):
        cfg = WalkForwardConfig()
        assert cfg.PLOT_SETTINGS["plot_width"] > 0
        assert cfg.PLOT_SETTINGS["template"] == "plotly_white"
        assert cfg.PLOT_SETTINGS["show_individual_trades"] is True

    def test_min_total_bars_covers_one_window(self):
        cfg = WalkForwardConfig()
        one_window = cfg.TRAIN_SIZE + cfg.GAP_SIZE + cfg.TEST_SIZE
        assert cfg.MIN_TOTAL_BARS >= one_window

    def test_performance_criteria_defaults(self):
        cfg = WalkForwardConfig()
        assert cfg.MIN_TRADES_PER_WINDOW == 10
        assert cfg.MIN_SHARPE_RATIO == 0.5
        assert cfg.MIN_WIN_RATE == 0.4
        assert cfg.MAX_DRAWDOWN_THRESHOLD == -0.2
        assert cfg.MIN_PROFIT_FACTOR == 1.2

    def test_to_dict(self):
        d = WalkForwardConfig.to_dict()
        assert isinstance(d, dict)

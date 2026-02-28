"""Enhanced tests for ML modules after fixes:
- Importance-aware correlation removal
- PSI log protection
- Configurable drift thresholds
- Robustness metrics finite values
"""
import pytest
import numpy as np
import pandas as pd

from ml.feature_selection import remove_highly_correlated, auto_select_features
from ml.drift_detection import calculate_psi, detect_drift
from backtest.robustness_metrics import calculate_robustness_metrics, calculate_robustness_score


class TestImportanceAwareCorrelation:
    def test_drops_less_important_feature(self):
        np.random.seed(42)
        n = 200
        base = np.random.randn(n)
        df = pd.DataFrame({
            "important": base,
            "redundant": base + np.random.randn(n) * 0.01,
            "independent": np.random.randn(n),
        })
        importance = pd.Series({"important": 0.8, "redundant": 0.1, "independent": 0.5})
        kept = remove_highly_correlated(
            df, ["important", "redundant", "independent"],
            threshold=0.95, importance=importance,
        )
        assert "important" in kept
        assert "redundant" not in kept
        assert "independent" in kept

    def test_without_importance_drops_second(self):
        np.random.seed(42)
        n = 200
        base = np.random.randn(n)
        df = pd.DataFrame({
            "a": base,
            "b": base + np.random.randn(n) * 0.01,
        })
        kept = remove_highly_correlated(df, ["a", "b"], threshold=0.95)
        assert "b" not in kept or "a" not in kept
        assert len(kept) == 1


class TestPSILogProtection:
    def test_zero_bin_no_crash(self):
        expected = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        actual = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        psi = calculate_psi(expected, actual, n_bins=5)
        assert np.isfinite(psi)

    def test_nan_values_handled(self):
        expected = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        actual = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        psi = calculate_psi(expected, actual)
        assert np.isfinite(psi)


class TestDriftConfigurable:
    def test_custom_alert_thresholds(self):
        np.random.seed(42)
        ref = pd.DataFrame({"f1": np.random.randn(100)})
        prod = pd.DataFrame({"f1": np.random.randn(100) + 5})
        result = detect_drift(ref, prod, ["f1"], psi_threshold=0.1, alert_ratio=0.5, alert_count=10)
        assert len(result["drifted_features"]) > 0

    def test_high_alert_count_suppresses_alert(self):
        np.random.seed(42)
        ref = pd.DataFrame({"f1": np.random.randn(100)})
        prod = pd.DataFrame({"f1": np.random.randn(100) + 5})
        result = detect_drift(ref, prod, ["f1"], psi_threshold=0.1, alert_ratio=1.0, alert_count=100)
        assert result["alert"] is False


class TestRobustnessFiniteValues:
    def _make_results(self, n=5):
        return [
            {
                "pnl_total": float(np.random.randn()),
                "win_rate": 0.5 + np.random.randn() * 0.1,
                "total_trades": max(1, int(abs(np.random.randn() * 10))),
                "max_drawdown": -abs(np.random.randn() * 0.1),
                "long_win_rate": 0.5,
                "short_win_rate": 0.5,
            }
            for _ in range(n)
        ]

    def test_no_inf_in_metrics(self):
        np.random.seed(42)
        results = self._make_results()
        metrics = calculate_robustness_metrics(results)
        for key, val in metrics.items():
            assert np.isfinite(val), f"{key} is not finite: {val}"

    def test_zero_mean_returns_bounded_cv(self):
        results = [
            {"pnl_total": 0.0, "win_rate": 0.5, "total_trades": 10,
             "max_drawdown": -0.1, "long_win_rate": 0.5, "short_win_rate": 0.5}
        ] * 3
        metrics = calculate_robustness_metrics(results)
        assert metrics["return_cv"] == 10.0
        assert np.isfinite(metrics["return_cv"])

    def test_robustness_score_bounded(self):
        np.random.seed(42)
        results = self._make_results()
        metrics = calculate_robustness_metrics(results)
        score = calculate_robustness_score(metrics, config=None)
        assert 0.0 <= score <= 1.0

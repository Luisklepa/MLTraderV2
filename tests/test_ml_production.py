"""Tests for FASE 5 production ML utilities."""

import numpy as np
import pandas as pd
import pytest

from core.data_validation import validate_ohlcv
from core.exceptions import DataError
from ml.drift_detection import calculate_psi, detect_drift
from ml.feature_selection import remove_highly_correlated


class TestFeatureSelection:
    def test_removes_correlated_features(self):
        np.random.seed(42)
        n = 100
        x1 = np.random.randn(n)
        x2 = x1 + np.random.randn(n) * 0.01  # near-perfect correlation
        x3 = np.random.randn(n)  # independent

        df = pd.DataFrame({"f1": x1, "f2": x2, "f3": x3})
        kept = remove_highly_correlated(df, ["f1", "f2", "f3"], threshold=0.95)

        assert "f1" in kept
        assert "f3" in kept
        assert len(kept) == 2

    def test_no_removal_when_uncorrelated(self):
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "a": np.random.randn(50),
                "b": np.random.randn(50),
                "c": np.random.randn(50),
            }
        )
        kept = remove_highly_correlated(df, ["a", "b", "c"], threshold=0.95)
        assert len(kept) == 3


class TestPSI:
    def test_identical_distributions_low_psi(self):
        np.random.seed(42)
        data = np.random.randn(1000)
        psi = calculate_psi(data, data)
        assert psi < 0.05

    def test_shifted_distribution_high_psi(self):
        np.random.seed(42)
        reference = np.random.randn(1000)
        shifted = np.random.randn(1000) + 3.0
        psi = calculate_psi(reference, shifted)
        assert psi > 0.25

    def test_empty_arrays_returns_zero(self):
        psi = calculate_psi(np.array([]), np.array([]))
        assert psi == 0.0


class TestDriftDetection:
    def test_no_drift_on_same_data(self):
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "f1": np.random.randn(500),
                "f2": np.random.randn(500),
            }
        )
        result = detect_drift(df, df, ["f1", "f2"])
        assert result["alert"] is False
        assert len(result["drifted_features"]) == 0

    def test_drift_detected_on_shifted_data(self):
        np.random.seed(42)
        ref = pd.DataFrame(
            {
                "f1": np.random.randn(500),
                "f2": np.random.randn(500),
            }
        )
        prod = pd.DataFrame(
            {
                "f1": np.random.randn(500) + 5.0,
                "f2": np.random.randn(500) + 5.0,
            }
        )
        result = detect_drift(ref, prod, ["f1", "f2"])
        assert len(result["drifted_features"]) > 0


class TestDataValidation:
    def test_valid_ohlcv_passes(self, sample_ohlcv):
        result = validate_ohlcv(sample_ohlcv)
        assert len(result) == len(sample_ohlcv)

    def test_empty_dataframe_raises(self):
        with pytest.raises(DataError, match="empty"):
            validate_ohlcv(pd.DataFrame())

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(DataError, match="Missing required"):
            validate_ohlcv(df)

    def test_negative_close_dropped(self):
        df = pd.DataFrame(
            {
                "open": [100, -1, 102],
                "high": [101, 0, 103],
                "low": [99, -2, 101],
                "close": [100, -1, 102],
                "volume": [10, 10, 10],
            }
        )
        result = validate_ohlcv(df)
        assert len(result) == 2

    def test_high_low_swap_fixed(self):
        df = pd.DataFrame(
            {
                "open": [100],
                "high": [99],  # intentionally swapped
                "low": [101],  # intentionally swapped
                "close": [100],
                "volume": [10],
            }
        )
        result = validate_ohlcv(df)
        assert result["high"].iloc[0] >= result["low"].iloc[0]

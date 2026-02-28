"""Tests for ml/train_model.py — the ML training module."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
import tempfile
import os

from ml.train_model import MLTradingTrainer


@pytest.fixture
def sample_dataset(tmp_path):
    """Create a temporary CSV dataset for training tests."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "feature_a": np.random.randn(n),
        "feature_b": np.random.randn(n),
        "feature_c": np.random.randn(n),
        "target_long": np.random.randint(0, 2, n),
        "target_short": np.random.randint(0, 2, n),
        "close": 50000 + np.cumsum(np.random.randn(n) * 100),
        "open": 50000 + np.cumsum(np.random.randn(n) * 100),
        "high": 50100 + np.cumsum(np.random.randn(n) * 100),
        "low": 49900 + np.cumsum(np.random.randn(n) * 100),
        "volume": np.random.lognormal(10, 1, n),
    })
    path = tmp_path / "test_dataset.csv"
    df.to_csv(path, index=False)
    return str(path)


class TestLoadAndPrepareData:
    def test_excludes_non_feature_columns(self, sample_dataset):
        trainer = MLTradingTrainer(
            dataset_path=sample_dataset,
            n_estimators=10,
            n_folds=2,
            balance_strategy=None,
            auto_select=False,
        )
        X, y_long, y_short = trainer.load_and_prepare_data()

        excluded = {"close", "open", "high", "low", "volume",
                     "target_long", "target_short"}
        for col in excluded:
            assert col not in X.columns, f"{col} should be excluded from features"

    def test_returns_correct_shapes(self, sample_dataset):
        trainer = MLTradingTrainer(
            dataset_path=sample_dataset,
            n_estimators=10,
            n_folds=2,
            balance_strategy=None,
            auto_select=False,
        )
        X, y_long, y_short = trainer.load_and_prepare_data()
        assert len(X) == len(y_long) == len(y_short)
        assert X.shape[1] == 3  # feature_a, feature_b, feature_c

    def test_nan_rows_dropped(self, tmp_path):
        df = pd.DataFrame({
            "feature_a": [1.0, np.nan, 3.0],
            "target_long": [0, 1, 0],
            "target_short": [1, 0, 1],
        })
        path = tmp_path / "nan_dataset.csv"
        df.to_csv(path, index=False)

        trainer = MLTradingTrainer(
            dataset_path=str(path),
            n_estimators=10,
            n_folds=2,
            auto_select=False,
        )
        X, y_long, y_short = trainer.load_and_prepare_data()
        assert len(X) == 2


class TestCreateBalancedDataset:
    def test_no_balance(self, sample_dataset):
        trainer = MLTradingTrainer(
            dataset_path=sample_dataset,
            auto_select=False,
        )
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        y = pd.Series([0, 0, 1])
        X_out, y_out = trainer.create_balanced_dataset(X, y, strategy=None)
        assert len(X_out) == 3

    def test_smote_increases_minority(self, sample_dataset):
        trainer = MLTradingTrainer(
            dataset_path=sample_dataset,
            auto_select=False,
        )
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 3))
        y = pd.Series([0] * 90 + [1] * 10)
        X_out, y_out = trainer.create_balanced_dataset(X, y, strategy="smote")
        assert (y_out == 1).sum() > 10


class TestTrainSingleModel:
    def test_smote_edge_case_skipped(self, sample_dataset):
        """When minority class has very few samples, SMOTE k_neighbors is reduced."""
        trainer = MLTradingTrainer(
            dataset_path=sample_dataset,
            n_estimators=10,
            n_folds=2,
            balance_strategy="smote",
            auto_select=False,
        )
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(50, 3))
        y = pd.Series([0] * 44 + [1] * 6)

        model = trainer._train_single_model(X, y, "LONG")
        assert model is not None

    def test_persists_threshold(self, sample_dataset):
        trainer = MLTradingTrainer(
            dataset_path=sample_dataset,
            n_estimators=10,
            n_folds=2,
            balance_strategy=None,
            auto_select=False,
        )
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 3))
        y = pd.Series(np.random.randint(0, 2, 100))

        trainer._train_single_model(X, y, "LONG")
        assert isinstance(trainer.best_threshold_long, float)
        assert 0 < trainer.best_threshold_long < 1


class TestSplitDataTemporal:
    def test_split_sizes(self, sample_dataset):
        trainer = MLTradingTrainer(
            dataset_path=sample_dataset,
            auto_select=False,
        )
        X = pd.DataFrame(np.random.randn(100, 2))
        y = pd.Series(np.random.randint(0, 2, 100))

        X_train, X_test, y_train, y_test = trainer.split_data_temporal(X, y, test_size=0.2)
        assert len(X_train) == 80
        assert len(X_test) == 20

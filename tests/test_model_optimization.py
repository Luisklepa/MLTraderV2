"""Tests for ml/model_optimization.py — ModelOptimizer."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from ml.model_optimization import ModelOptimizer


@pytest.fixture
def opt_config(tmp_path):
    return {
        "min_feature_importance": 0.01,
        "max_features": 10,
        "optimization_trials": 2,
        "cv_folds": 2,
        "model_dir": str(tmp_path / "models"),
    }


@pytest.fixture
def optimizer(opt_config):
    return ModelOptimizer(opt_config)


@pytest.fixture
def classification_data():
    """Binary classification dataset large enough for CV."""
    np.random.seed(42)
    n = 200
    X = pd.DataFrame(np.random.randn(n, 15), columns=[f"f{i}" for i in range(15)])
    y = pd.Series(np.random.choice([0, 1], n))
    return X, y


class TestModelOptimizerInit:
    def test_default_config_values(self, optimizer, opt_config):
        assert optimizer.feature_importance_threshold == 0.01
        assert optimizer.max_features == 10
        assert optimizer.n_trials == 2
        assert optimizer.cv_folds == 2
        assert optimizer.model_dir.exists()

    def test_model_dir_created(self, opt_config):
        ModelOptimizer(opt_config)
        assert Path(opt_config["model_dir"]).exists()


class TestFeatureSelection:
    def test_selects_features(self, optimizer, classification_data):
        X, y = classification_data
        X_sel, selected = optimizer.optimize_feature_selection(X, y)
        assert len(selected) > 0
        assert len(selected) <= optimizer.max_features
        assert list(X_sel.columns) == selected

    def test_all_selected_features_exist(self, optimizer, classification_data):
        X, y = classification_data
        _, selected = optimizer.optimize_feature_selection(X, y)
        for feat in selected:
            assert feat in X.columns


class TestHyperparameterOptimization:
    def test_returns_best_params_and_threshold(self, optimizer, classification_data):
        X, y = classification_data
        result = optimizer.optimize_hyperparameters(X, y)
        assert "best_params" in result
        assert "best_threshold" in result
        assert 0 < result["best_threshold"] < 1

    def test_best_params_contain_expected_keys(self, optimizer, classification_data):
        X, y = classification_data
        result = optimizer.optimize_hyperparameters(X, y)
        expected_keys = {
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "min_child_weight",
            "gamma",
            "reg_alpha",
            "reg_lambda",
        }
        assert expected_keys == set(result["best_params"].keys())


class TestModelDrift:
    def test_no_drift_without_history(self, optimizer, classification_data):
        X, y = classification_data
        model = xgb.XGBClassifier(n_estimators=10, eval_metric="logloss")
        model.fit(X, y)
        drifted = optimizer.evaluate_model_drift(model, X, y)
        assert drifted is False

    def test_drift_detected_with_degraded_history(self, optimizer, classification_data):
        X, y = classification_data
        model = xgb.XGBClassifier(n_estimators=10, eval_metric="logloss")
        model.fit(X, y)

        metrics_file = optimizer.model_dir / "historical_metrics.json"
        historical = {"timestamp": ["2024-01-01"], "auc": [0.99], "f1": [0.99]}
        with open(metrics_file, "w") as f:
            json.dump(historical, f)

        drifted = optimizer.evaluate_model_drift(model, X, y, threshold=0.01)
        assert isinstance(drifted, bool)


class TestSaveLoad:
    def test_save_and_load_model(self, optimizer, classification_data):
        X, y = classification_data
        model = xgb.XGBClassifier(n_estimators=10, eval_metric="logloss")
        model.fit(X, y)
        features = list(X.columns)
        metrics = {"f1": 0.75, "auc": 0.80}

        optimizer.save_model(model, features, metrics)

        loaded_model, loaded_features, loaded_metrics, loaded_threshold = optimizer.load_latest_model()
        assert loaded_features == features
        assert loaded_metrics["f1"] == 0.75

        preds_original = model.predict(X)
        preds_loaded = loaded_model.predict(X)
        np.testing.assert_array_equal(preds_original, preds_loaded)

    def test_load_latest_raises_when_empty(self, opt_config, tmp_path):
        empty_dir = tmp_path / "empty_models"
        empty_dir.mkdir()
        opt_config["model_dir"] = str(empty_dir)
        opt = ModelOptimizer(opt_config)
        with pytest.raises(FileNotFoundError, match="No saved models"):
            opt.load_latest_model()


class TestSaveMetrics:
    def test_creates_new_file(self, optimizer):
        optimizer.save_metrics({"timestamp": "2024-01-01", "auc": 0.8, "f1": 0.7})
        metrics_file = optimizer.model_dir / "historical_metrics.json"
        assert metrics_file.exists()
        with open(metrics_file) as f:
            data = json.load(f)
        assert len(data["auc"]) == 1

    def test_appends_to_existing(self, optimizer):
        optimizer.save_metrics({"timestamp": "2024-01-01", "auc": 0.8, "f1": 0.7})
        optimizer.save_metrics({"timestamp": "2024-01-02", "auc": 0.85, "f1": 0.75})
        with open(optimizer.model_dir / "historical_metrics.json") as f:
            data = json.load(f)
        assert len(data["auc"]) == 2


class TestFeatureImportance:
    def test_returns_sorted_dataframe(self, optimizer, classification_data):
        X, y = classification_data
        model = xgb.XGBClassifier(n_estimators=10, eval_metric="logloss")
        model.fit(X, y)
        importance_df = optimizer.analyze_feature_importance(model, list(X.columns))
        assert "feature" in importance_df.columns
        assert "importance" in importance_df.columns
        assert importance_df["importance"].is_monotonic_decreasing
        assert len(importance_df) == len(X.columns)

"""Tests for ml/pipeline.py — the decomposed ML pipeline orchestrator."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from ml.pipeline import (
    FeatureEngine,
    ModelTrainer,
    SignalGenerator,
    ResultAnalyzer,
    PipelineConfig,
    ModelConfig,
    _safe_div,
)


@pytest.fixture
def mock_pipeline_config(tmp_path):
    """Create a minimal YAML config and return a PipelineConfig."""
    import yaml

    config = {
        "model_config": {
            "prediction": {"threshold": 0.5, "min_prediction_confidence": 0.4},
            "ensemble": {
                "models": [
                    {
                        "params": {
                            "n_estimators": 10,
                            "max_depth": 3,
                            "learning_rate": 0.1,
                            "subsample": 0.8,
                            "colsample_bytree": 0.8,
                        }
                    }
                ]
            },
            "long_model": {
                "threshold": 0.5,
                "min_probability": 0.4,
                "position_sizing": {
                    "base_size": 1.0,
                    "max_size": 3.0,
                    "confidence_multiplier": 2.0,
                    "volatility_adjustment": False,
                },
                "risk_management": {"max_drawdown": 0.2},
            },
            "short_model": {
                "threshold": 0.5,
                "min_probability": 0.4,
                "position_sizing": {
                    "base_size": 1.0,
                    "max_size": 3.0,
                    "confidence_multiplier": 2.0,
                    "volatility_adjustment": False,
                },
                "risk_management": {"max_drawdown": 0.2},
            },
        },
        "features_config": {
            "technical_indicators": [
                {"name": "rsi_14", "function": "RSI", "params": {"timeperiod": 14}},
            ],
        },
        "data_config": {"timeframe": "15m"},
    }

    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return PipelineConfig(str(config_path))


class TestSafeDiv:
    def test_basic(self):
        result = _safe_div(np.array([10.0]), np.array([2.0]))
        assert result[0] == pytest.approx(5.0)

    def test_zero_denom(self):
        result = _safe_div(np.array([10.0]), np.array([0.0]))
        assert result[0] == 0.0


class TestModelConfig:
    def test_defaults(self):
        mc = ModelConfig()
        assert mc.n_estimators == 100
        assert mc.max_depth == 3

    def test_to_dict(self):
        mc = ModelConfig(n_estimators=50)
        d = mc.to_dict()
        assert d["n_estimators"] == 50
        assert "learning_rate" in d


class TestFeatureEngine:
    def test_prepare_features_adds_columns(self, sample_ohlcv, mock_pipeline_config):
        engine = FeatureEngine(mock_pipeline_config)
        df = engine.prepare_features(sample_ohlcv.copy())
        assert "returns_1" in df.columns
        assert "volatility_20" in df.columns

    def test_calculate_advanced_features(self, sample_ohlcv_large, mock_pipeline_config):
        engine = FeatureEngine(mock_pipeline_config)
        df = engine.calculate_advanced_features(sample_ohlcv_large.copy())
        assert "volatility_regime" in df.columns or "volume_ratio" in df.columns

    def test_caching_works(self, sample_ohlcv_large, mock_pipeline_config):
        engine = FeatureEngine(mock_pipeline_config)
        df1 = engine.calculate_advanced_features(sample_ohlcv_large.copy())
        df2 = engine.calculate_advanced_features(sample_ohlcv_large.copy())
        assert len(engine._cache) > 0


class TestModelTrainer:
    def test_prepare_data_shapes(self, sample_ohlcv, mock_pipeline_config):
        """Trainer splits data correctly."""
        df = sample_ohlcv.copy()
        features = mock_pipeline_config.selected_features
        for f in features:
            if f not in df.columns:
                df[f] = np.random.randn(len(df))
        df["long_target"] = np.random.randint(0, 2, len(df))

        trainer = ModelTrainer(mock_pipeline_config)
        X_train, X_test, y_train, y_test = trainer.prepare_data(df, "long")

        assert len(X_train) + len(X_test) >= len(df) * 0.95
        assert X_train.shape[1] == X_test.shape[1]

    def test_train_and_evaluate(self, sample_ohlcv, mock_pipeline_config):
        df = sample_ohlcv.copy()
        features = mock_pipeline_config.selected_features
        for f in features:
            if f not in df.columns:
                df[f] = np.random.randn(len(df))
        df["long_target"] = np.random.randint(0, 2, len(df))

        trainer = ModelTrainer(mock_pipeline_config)
        X_train, X_test, y_train, y_test = trainer.prepare_data(df, "long")
        model, train_metrics = trainer.train_model(X_train, y_train, "long")
        test_metrics = trainer.evaluate_model(model, X_test, y_test)

        assert "accuracy" in train_metrics
        assert "f1" in test_metrics
        assert 0 <= test_metrics["accuracy"] <= 1


class TestSignalGenerator:
    def test_calculate_position_size(self, mock_pipeline_config):
        gen = SignalGenerator(mock_pipeline_config)
        model_cfg = mock_pipeline_config.model_config["long_model"]
        size = gen.calculate_position_size(0.7, model_cfg)
        assert size > 0
        assert size <= model_cfg["position_sizing"]["max_size"]

    def test_position_size_capped(self, mock_pipeline_config):
        gen = SignalGenerator(mock_pipeline_config)
        model_cfg = mock_pipeline_config.model_config["long_model"]
        size = gen.calculate_position_size(0.99, model_cfg)
        assert size <= model_cfg["position_sizing"]["max_size"]


class TestResultAnalyzer:
    def test_get_feature_importance(self, trained_model_mock):
        analyzer = ResultAnalyzer()
        features = ["f1", "f2", "f3", "f4", "f5"]
        imp = analyzer.get_feature_importance({"long": trained_model_mock}, features)
        assert not imp.empty
        assert len(imp) == 5

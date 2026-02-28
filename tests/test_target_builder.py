"""Tests for ml/target_builder.py — dynamic target generation and quality metrics."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from ml.target_builder import MLTargetBuilder, TargetQualityMetrics


@pytest.fixture
def target_builder_config():
    """Minimal config for MLTargetBuilder."""
    filter_config = {
        "filters": {
            "volatility": {"enabled": False, "atr_column": "atr_14",
                           "atr_percentile_min": 0.1, "atr_percentile_max": 0.9},
            "volume": {"enabled": False, "volume_ratio_column": "volume_ratio",
                       "volume_ratio_min": 0.5},
            "trend": {"enabled": False, "ema_fast": 20, "ema_slow": 50,
                      "min_trend_strength": 0.001},
        },
        "atr_multiplier": 1.0,
        "profit_ratio": 1.0,
        "max_overlap_ratio": 0.3,
        "min_samples": 10,
    }
    return filter_config


@pytest.fixture
def target_builder(target_builder_config, tmp_path):
    return MLTargetBuilder(
        horizon=10,
        atr_column="atr_14",
        long_config=target_builder_config,
        short_config=target_builder_config,
        analyze_threshold_range=False,
        experiment_tracking=False,
        output_dir=str(tmp_path),
    )


@pytest.fixture
def df_with_atr(sample_ohlcv_large):
    """OHLCV data enriched with ATR-like column."""
    df = sample_ohlcv_large.copy()
    df["atr_14"] = (df["high"] - df["low"]).rolling(14).mean()
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    df["ema_20"] = df["close"].ewm(span=20).mean()
    df["ema_50"] = df["close"].ewm(span=50).mean()
    df = df.dropna()
    return df


class TestCalculateFutureReturns:
    def test_returns_correct_length(self, target_builder, df_with_atr):
        returns = target_builder._calculate_future_returns(df_with_atr)
        assert len(returns) == len(df_with_atr)
        assert returns.iloc[-10:].isna().all()


class TestGenerateTargets:
    def test_long_targets_binary(self, target_builder, df_with_atr, target_builder_config):
        targets = target_builder._generate_targets(df_with_atr, target_builder_config, "long")
        assert set(targets.dropna().unique()).issubset({0, 1})

    def test_short_targets_values(self, target_builder, df_with_atr, target_builder_config):
        targets = target_builder._generate_targets(df_with_atr, target_builder_config, "short")
        assert set(targets.dropna().unique()).issubset({0, -1})


class TestBuildTargets:
    def test_build_targets_returns_series(self, target_builder, df_with_atr):
        targets = target_builder.build_targets(df_with_atr)
        assert isinstance(targets, pd.Series)
        assert len(targets) == len(df_with_atr)

    def test_build_targets_values_valid(self, target_builder, df_with_atr):
        targets = target_builder.build_targets(df_with_atr)
        assert set(targets.unique()).issubset({-1, 0, 1})


class TestRemoveOverlappingSignals:
    def test_reduces_close_signals(self, target_builder):
        idx = pd.date_range("2024-01-01", periods=20, freq="1min")
        targets = pd.Series(0, index=idx)
        targets.iloc[0] = 1
        targets.iloc[2] = 1  # Within 10-min horizon
        targets.iloc[15] = 1  # Outside horizon

        clean = target_builder._remove_overlapping_signals(targets)
        assert clean.iloc[0] == 1
        assert clean.iloc[2] == 0
        assert clean.iloc[15] == 1


class TestTargetQualityMetrics:
    def test_quality_metrics_computed(self, target_builder, df_with_atr):
        future_returns = target_builder._calculate_future_returns(df_with_atr)
        targets = pd.Series(0, index=df_with_atr.index)
        targets.iloc[10] = 1
        targets.iloc[100] = -1
        targets.iloc[200] = 1

        metrics = target_builder._calculate_target_quality_metrics(targets, df_with_atr)
        assert isinstance(metrics, TargetQualityMetrics)
        assert metrics.signal_count == 3
        assert 0 <= metrics.robustness_score <= 1

    def test_quality_metrics_zero_signals(self, target_builder, df_with_atr):
        targets = pd.Series(0, index=df_with_atr.index)
        metrics = target_builder._calculate_target_quality_metrics(targets, df_with_atr)
        assert metrics.signal_count == 0
        assert metrics.mean_return == 0.0


class TestThresholdSensitivity:
    def test_runs_without_error(self, target_builder, df_with_atr):
        target_builder._analyze_threshold_sensitivity(df_with_atr)

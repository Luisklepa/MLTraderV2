"""Tests for ml/feature_pipeline.py — the central feature engineering module."""

import numpy as np
import pandas as pd
import pytest

from ml.feature_pipeline import (
    _EXCLUDE_COLS,
    MLFeaturePipeline,
    _rolling_pct_rank,
    _rolling_slope,
    _safe_div,
    add_anti_failure_features,
    add_conditional_features,
    add_cross_features,
    add_lag_features,
    add_market_structure_features,
    add_momentum_features,
    add_moving_average_features,
    add_pattern_features,
    add_price_features,
    add_temporal_features,
    add_volatility_features,
    add_volume_features,
)


class TestSafeDiv:
    def test_normal_division(self):
        a = np.array([10.0, 20.0, 30.0])
        b = np.array([2.0, 5.0, 10.0])
        result = _safe_div(a, b)
        np.testing.assert_array_almost_equal(result, [5.0, 4.0, 3.0])

    def test_zero_denominator_returns_fill(self):
        a = np.array([10.0, 20.0])
        b = np.array([0.0, 5.0])
        result = _safe_div(a, b, fill=-1.0)
        assert result[0] == -1.0
        assert result[1] == pytest.approx(4.0)

    def test_near_zero_denominator(self):
        a = np.array([1.0])
        b = np.array([1e-15])
        result = _safe_div(a, b)
        assert result[0] == 0.0

    def test_no_inf_in_output(self):
        a = np.random.randn(100)
        b = np.random.randn(100)
        b[::10] = 0.0
        result = _safe_div(a, b)
        assert np.all(np.isfinite(result))


class TestRollingHelpers:
    def test_rolling_pct_rank_range(self):
        s = pd.Series(np.random.randn(100))
        result = _rolling_pct_rank(s, 20)
        valid = result.dropna()
        assert valid.min() >= 0.0
        assert valid.max() <= 1.0

    def test_rolling_slope_not_all_nan(self):
        s = pd.Series(np.arange(50, dtype=float))
        result = _rolling_slope(s, 10)
        valid = result.dropna()
        assert len(valid) > 0
        assert valid.iloc[-1] > 0


class TestFeatureGenerators:
    """Test each add_*_features function independently."""

    def test_price_features(self, sample_ohlcv):
        df = add_price_features(sample_ohlcv.copy())
        assert "returns" in df.columns
        assert "log_returns" in df.columns
        assert "return_5" in df.columns
        assert not df["returns"].iloc[1:].isna().all()

    def test_volume_features(self, sample_ohlcv):
        df = sample_ohlcv.copy()
        df["returns"] = df["close"].pct_change()
        df = add_volume_features(df)
        assert "vwap" in df.columns
        assert "volume_ratio" in df.columns
        assert np.all(np.isfinite(df["volume_ratio"].dropna()))

    def test_moving_average_features(self, sample_ohlcv):
        df = add_moving_average_features(sample_ohlcv.copy())
        assert "sma_20" in df.columns
        assert "ema_50" in df.columns
        assert "sma_cross_20_50" in df.columns
        assert set(df["sma_cross_20_50"].dropna().unique()).issubset({0, 1})

    def test_momentum_features(self, sample_ohlcv_large):
        df = sample_ohlcv_large.copy()
        df["returns"] = df["close"].pct_change()
        df = add_momentum_features(df)
        assert "rsi_14" in df.columns
        assert "macd" in df.columns
        assert "adx_14" in df.columns
        rsi_valid = df["rsi_14"].dropna()
        assert rsi_valid.min() >= 0
        assert rsi_valid.max() <= 100

    def test_volatility_features(self, sample_ohlcv_large):
        df = sample_ohlcv_large.copy()
        df["returns"] = df["close"].pct_change()
        df = add_volatility_features(df)
        assert "atr_14" in df.columns
        assert "bb_width_20" in df.columns
        assert "donchian_width" in df.columns
        assert np.all(np.isfinite(df["bb_width_20"].dropna()))

    def test_pattern_features(self, sample_ohlcv):
        df = add_pattern_features(sample_ohlcv.copy())
        assert "doji" in df.columns
        assert "body_size" in df.columns
        assert "candle_type" in df.columns

    def test_market_structure_features(self, sample_ohlcv):
        df = add_market_structure_features(sample_ohlcv.copy())
        assert "resistance_distance" in df.columns
        assert "breakout_high" in df.columns
        assert "trend_strength" in df.columns

    def test_temporal_features(self, sample_ohlcv):
        df = sample_ohlcv.copy()
        df["timestamp"] = df.index
        df = add_temporal_features(df)
        assert "hour" in df.columns
        assert "day_of_week" in df.columns

    def test_cross_features_skip_missing(self):
        df = pd.DataFrame({"rsi_14": [50.0], "close": [100.0]})
        df = add_cross_features(df, verbose=True)
        assert "rsi_14_atr_14_ratio" not in df.columns

    def test_anti_failure_features(self, sample_ohlcv_large):
        df = sample_ohlcv_large.copy()
        df["returns"] = df["close"].pct_change()
        df = add_anti_failure_features(df)
        assert "vol_spike_20" in df.columns
        assert "pos_streak" in df.columns

    def test_lag_features(self, sample_ohlcv):
        df = sample_ohlcv.copy()
        df["returns"] = df["close"].pct_change()
        df["rsi_14"] = 50.0
        df = add_lag_features(df)
        assert "returns_lag_1" in df.columns
        assert "returns_lag_5" in df.columns

    def test_conditional_features_long(self, sample_ohlcv):
        df = sample_ohlcv.copy()
        df["macd"] = 1.0
        df["rsi_14"] = 65.0
        df["engulfing"] = 100
        df["low_20"] = df["low"].rolling(20).min()
        df["volatility_20"] = 0.01
        df = add_conditional_features(df, "long")
        assert "bullish_momentum" in df.columns
        assert "rsi_above_60" in df.columns

    def test_conditional_features_short(self, sample_ohlcv):
        df = sample_ohlcv.copy()
        df["macd"] = -1.0
        df["rsi_14"] = 35.0
        df["engulfing"] = -100
        df["high_20"] = df["high"].rolling(20).max()
        df["volatility_20"] = 0.01
        df = add_conditional_features(df, "short")
        assert "bearish_momentum" in df.columns
        assert "rsi_below_40" in df.columns


class TestMLFeaturePipeline:
    def test_feature_engineering_no_crash(self, sample_ohlcv_large):
        pipe = MLFeaturePipeline()
        df = pipe.feature_engineering(sample_ohlcv_large.copy())
        assert len(df.columns) > 20

    def test_feature_engineering_no_inf(self, sample_ohlcv_large):
        pipe = MLFeaturePipeline()
        df = pipe.feature_engineering(sample_ohlcv_large.copy())
        numeric = df.select_dtypes(include=[np.number])
        inf_count = np.isinf(numeric.values).sum()
        assert inf_count == 0, f"Found {inf_count} infinite values in features"

    def test_generate_target_variable(self, sample_ohlcv_large):
        pipe = MLFeaturePipeline()
        df = pipe.generate_target_variable(sample_ohlcv_large.copy(), future_bars=10)
        assert "target" in df.columns
        assert "future_return" in df.columns
        assert len(df) == len(sample_ohlcv_large) - 10

    def test_clean_and_scale_features_train_test(self, sample_ohlcv_large):
        pipe = MLFeaturePipeline()
        df = pipe.feature_engineering(sample_ohlcv_large.copy())
        df = pipe.generate_target_variable(df, future_bars=5)

        split = int(len(df) * 0.8)
        train = df.iloc[:split].copy()
        test = df.iloc[split:].copy()

        train = pipe.clean_and_scale_features(train, fit_scaler=True)
        test = pipe.clean_and_scale_features(test, fit_scaler=False)

        assert pipe.scaler is not None
        assert pipe.feature_columns is not None

    def test_exclude_cols_not_in_features(self, sample_ohlcv_large):
        pipe = MLFeaturePipeline()
        df = pipe.feature_engineering(sample_ohlcv_large.copy())
        df = pipe.generate_target_variable(df, future_bars=5)
        df = pipe.clean_and_scale_features(df, fit_scaler=True)
        assert not any(c in pipe.feature_columns for c in _EXCLUDE_COLS)

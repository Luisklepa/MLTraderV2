"""Tests for ml/signal_filter.py — market regime detection and signal filtering."""

import pandas as pd
import pytest

from ml.signal_filter import MarketRegime, MLSignalFilter


@pytest.fixture
def filter_config():
    """Minimal config for MLSignalFilter."""
    side_config = {
        "filters": {
            "volatility": {
                "enabled": True,
                "atr_column": "atr_14",
                "atr_percentile_min": 0.1,
                "atr_percentile_max": 0.9,
            },
            "volume": {"enabled": True, "volume_ratio_min": 0.5},
            "trend": {"enabled": True},
        }
    }
    return {"long": side_config, "short": side_config}


@pytest.fixture
def df_for_filter(sample_ohlcv_large):
    """DataFrame with required columns for signal filter."""
    df = sample_ohlcv_large.copy()
    df["atr_14"] = (df["high"] - df["low"]).rolling(14).mean()
    df = df.dropna()
    return df


class TestMarketRegime:
    def test_identify_regime_returns_series(self, df_for_filter):
        detector = MarketRegime()
        regimes = detector.identify_regime(df_for_filter)
        assert isinstance(regimes, pd.Series)
        assert len(regimes) == len(df_for_filter)

    def test_regime_labels_valid(self, df_for_filter):
        detector = MarketRegime()
        regimes = detector.identify_regime(df_for_filter)
        valid_labels = {
            "neutral",
            "volatile_bullish",
            "volatile_bearish",
            "volatile_sideways",
            "low_vol_bullish",
            "low_vol_bearish",
            "low_vol_sideways",
        }
        assert set(regimes.unique()).issubset(valid_labels)

    def test_custom_lookback(self, df_for_filter):
        detector = MarketRegime(lookback_period=30)
        regimes = detector.identify_regime(df_for_filter)
        assert len(regimes) == len(df_for_filter)


class TestMLSignalFilter:
    def test_apply_filters_returns_bool_series(self, filter_config, df_for_filter):
        sf = MLSignalFilter(filter_config)
        mask = sf.apply_filters(df_for_filter, "long")
        assert mask.dtype == bool
        assert len(mask) == len(df_for_filter)

    def test_apply_filters_short(self, filter_config, df_for_filter):
        sf = MLSignalFilter(filter_config)
        mask = sf.apply_filters(df_for_filter, "short")
        assert mask.dtype == bool

    def test_regime_filter_long(self, filter_config):
        sf = MLSignalFilter(filter_config)
        regimes = pd.Series(["low_vol_bullish", "volatile_bearish", "low_vol_sideways"])
        result = sf._apply_regime_filter(regimes, "long")
        assert result.iloc[0] is True or result.iloc[0]
        assert result.iloc[1] is False or not result.iloc[1]

    def test_regime_filter_short(self, filter_config):
        sf = MLSignalFilter(filter_config)
        regimes = pd.Series(["low_vol_bearish", "volatile_bullish", "low_vol_sideways"])
        result = sf._apply_regime_filter(regimes, "short")
        assert result.iloc[0] is True or result.iloc[0]
        assert result.iloc[1] is False or not result.iloc[1]

    def test_trend_filter_long(self, filter_config, df_for_filter):
        sf = MLSignalFilter(filter_config)
        mask = sf._apply_trend_filter(df_for_filter, "long")
        assert mask.dtype == bool

    def test_volume_filter(self, filter_config, df_for_filter):
        sf = MLSignalFilter(filter_config)
        mask = sf._apply_volume_filter(df_for_filter, "long")
        assert mask.dtype == bool

    def test_filters_disabled(self, df_for_filter):
        """When all filters disabled, should pass everything through regime filter."""
        config = {
            "long": {
                "filters": {
                    "volatility": {"enabled": False},
                    "volume": {"enabled": False},
                    "trend": {"enabled": False},
                }
            },
            "short": {
                "filters": {
                    "volatility": {"enabled": False},
                    "volume": {"enabled": False},
                    "trend": {"enabled": False},
                }
            },
        }
        sf = MLSignalFilter(config)
        mask = sf.apply_filters(df_for_filter, "long")
        assert mask.dtype == bool

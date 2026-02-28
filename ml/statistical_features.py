"""
Statistical feature generation for ML trading strategies.

Used by ml/pipeline.py via config-driven approach. For the standalone
pipeline, use ml/feature_pipeline.py instead — it generates equivalent
features directly.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from scipy import stats
import logging

logger = logging.getLogger(__name__)


def _safe_div(a, b, fill: float = 0.0):
    """Element-wise division with zero/inf protection."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(np.abs(b) < 1e-12, fill, a / b)
    return np.where(np.isfinite(result), result, fill)


# ---- Config-driven feature generators (used by pipeline.py) ----

def add_return_features(
    df: pd.DataFrame,
    config: Dict,
    price_col: str = "close",
) -> pd.DataFrame:
    """Add return-based features from config."""
    params = config["returns"]["params"]
    windows = params["window"]
    return_type = params["type"]

    for window in windows:
        if return_type == "simple":
            df[f"return_{window}"] = df[price_col].pct_change(window)
        else:
            df[f"return_{window}"] = np.log(df[price_col] / df[price_col].shift(window))

    return df


def add_volatility_features(
    df: pd.DataFrame,
    config: Dict,
    price_col: str = "close",
) -> pd.DataFrame:
    """Add volatility-based features from config."""
    params = config["volatility"]["params"]
    windows = params["window"]
    vol_type = params["type"]

    for window in windows:
        if vol_type == "realized":
            returns = np.log(df[price_col] / df[price_col].shift(1))
            df[f"volatility_{window}"] = returns.rolling(window).std() * np.sqrt(252)
        elif vol_type == "parkinson":
            high_low = np.log(df["high"] / df["low"])
            df[f"volatility_{window}"] = np.sqrt(
                (1 / (4 * np.log(2)))
                * high_low.pow(2).rolling(window).mean()
                * np.sqrt(252)
            )

    return df


def add_zscore_features(
    df: pd.DataFrame,
    config: Dict,
) -> pd.DataFrame:
    """Add z-score features with zero-std protection."""
    params = config["zscore"]["params"]
    window = params["window"]
    columns = params["columns"]

    for col in columns:
        if col not in df.columns:
            continue
        rolling_mean = df[col].rolling(window=window).mean()
        rolling_std = df[col].rolling(window=window).std().replace(0, np.nan)
        df[f"zscore_{col}_{window}"] = (df[col] - rolling_mean) / rolling_std

    return df


def add_correlation_features(
    df: pd.DataFrame,
    window: int = 20,
    price_col: str = "close",
    volume_col: str = "volume",
) -> pd.DataFrame:
    """Add correlation-based features."""
    df["price_volume_corr"] = df[price_col].rolling(window).corr(df[volume_col])
    high_low_range = df["high"] - df["low"]
    df["range_volume_corr"] = high_low_range.rolling(window).corr(df[volume_col])
    return df


def add_distribution_features(
    df: pd.DataFrame,
    window: int = 20,
    price_col: str = "close",
) -> pd.DataFrame:
    """Add distribution-based features (skew, kurtosis)."""
    returns = df[price_col].pct_change()
    df["returns_skew"] = returns.rolling(window).skew()
    df["returns_kurt"] = returns.rolling(window).kurt()
    return df


def add_entropy_features(
    df: pd.DataFrame,
    window: int = 20,
    bins: int = 10,
) -> pd.DataFrame:
    """Add entropy-based features."""
    def _entropy(x):
        hist, _ = np.histogram(x, bins=bins)
        p = hist / len(x)
        return -np.sum(p * np.log(p + 1e-10))

    df["price_entropy"] = df["close"].rolling(window).apply(_entropy, raw=True)
    df["volume_entropy"] = df["volume"].rolling(window).apply(_entropy, raw=True)
    returns = df["close"].pct_change()
    df["return_entropy"] = returns.rolling(window).apply(_entropy, raw=True)
    return df


def add_all_statistical_features(
    df: pd.DataFrame,
    features_config: Dict,
    data_config: Optional[Dict] = None,
) -> pd.DataFrame:
    """Add all statistical features based on configuration dict."""
    result = df.copy()
    price_col = data_config.get("price_column", "close") if data_config else "close"
    volume_col = data_config.get("volume_column", "volume") if data_config else "volume"

    if "returns" in features_config:
        result = add_return_features(result, features_config, price_col)
    if "volatility" in features_config:
        result = add_volatility_features(result, features_config, price_col)
    if "zscore" in features_config:
        result = add_zscore_features(result, features_config)

    result = add_correlation_features(result, window=20, price_col=price_col, volume_col=volume_col)
    result = add_distribution_features(result, window=20, price_col=price_col)
    result = add_entropy_features(result, window=20)
    return result


# ---- Standalone convenience functions ----

def calculate_zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score with zero-std protection."""
    mean = series.rolling(window).mean()
    std = series.rolling(window).std().replace(0, np.nan)
    return (series - mean) / std


def calculate_efficiency_ratio(prices: pd.Series, window: int) -> pd.Series:
    """Price efficiency ratio (direction / volatility)."""
    direction = abs(prices - prices.shift(window))
    path = abs(prices - prices.shift(1)).rolling(window).sum()
    return pd.Series(_safe_div(direction.values, path.values), index=prices.index)

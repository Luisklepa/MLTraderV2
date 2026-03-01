"""
Config-driven technical indicator generation using TA-Lib.

Used by ml/pipeline.py. For the standalone pipeline, use
ml/feature_pipeline.py instead.
"""

import numpy as np
import pandas as pd
import talib


def _safe_div(a, b, fill: float = 0.0):
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(np.abs(b) < 1e-12, fill, a / b)
    return np.where(np.isfinite(result), result, fill)


def add_momentum_indicators(
    df: pd.DataFrame,
    config: dict,
    price_col: str = "close",
) -> pd.DataFrame:
    """Add momentum indicators from config dict."""
    if "rsi" in config:
        p = config["rsi"]["params"]
        df[f"rsi_{p['window']}"] = talib.RSI(df[price_col].values, timeperiod=p["window"])

    if "macd" in config:
        p = config["macd"]["params"]
        macd, signal, hist = talib.MACD(
            df[price_col].values,
            fastperiod=p["fast"],
            slowperiod=p["slow"],
            signalperiod=p["signal"],
        )
        df["macd"] = macd
        df["macd_signal"] = signal
        df["macd_histogram"] = hist

    if "stoch" in config:
        p = config["stoch"]["params"]
        k, d = talib.STOCH(
            df["high"].values,
            df["low"].values,
            df[price_col].values,
            fastk_period=p["k_window"],
            slowk_period=p["k_window"],
            slowd_period=p["d_window"],
        )
        df["stoch_k"] = k
        df["stoch_d"] = d

    return df


def add_volatility_indicators(
    df: pd.DataFrame,
    config: dict,
    price_col: str = "close",
) -> pd.DataFrame:
    """Add volatility indicators from config dict."""
    if "atr" in config:
        p = config["atr"]["params"]
        df[f"atr_{p['window']}"] = talib.ATR(
            df["high"].values,
            df["low"].values,
            df[price_col].values,
            timeperiod=p["window"],
        )

    if "bbands" in config:
        p = config["bbands"]["params"]
        upper, middle, lower = talib.BBANDS(
            df[price_col].values,
            timeperiod=p["window"],
            nbdevup=p["num_std"],
            nbdevdn=p["num_std"],
        )
        w = p["window"]
        df[f"bb_upper_{w}"] = upper
        df[f"bb_middle_{w}"] = middle
        df[f"bb_lower_{w}"] = lower
        df[f"bb_width_{w}"] = _safe_div(upper - lower, middle)

    return df


def add_volume_indicators(
    df: pd.DataFrame,
    config: dict,
    price_col: str = "close",
    volume_col: str = "volume",
) -> pd.DataFrame:
    """Add volume-based indicators from config dict."""
    if "obv" in config:
        df["obv"] = talib.OBV(df[price_col].values, df[volume_col].values)

    if "vwap" in config:
        p = config["vwap"]["params"]
        w = p["window"]
        tp = (df["high"] + df["low"] + df[price_col]) / 3
        df["vwap"] = (tp * df[volume_col]).rolling(w).sum() / df[volume_col].rolling(w).sum()
        df["vwap_distance"] = _safe_div(
            (df[price_col] - df["vwap"]).values,
            df["vwap"].values,
        )

    return df


def add_trend_indicators(
    df: pd.DataFrame,
    config: dict,
    price_col: str = "close",
) -> pd.DataFrame:
    """Add trend indicators."""
    for period in [20, 50]:
        df[f"ema_{period}"] = talib.EMA(df[price_col].values, timeperiod=period)
        df[f"price_ema_{period}_ratio"] = _safe_div(
            df[price_col].values,
            df[f"ema_{period}"].values,
        )
    return df


def add_pattern_recognition(df: pd.DataFrame) -> pd.DataFrame:
    """Add candlestick pattern recognition."""
    o, h, low_vals, c = (
        df["open"].values,
        df["high"].values,
        df["low"].values,
        df["close"].values,
    )
    df["doji"] = talib.CDLDOJI(o, h, low_vals, c)
    df["hammer"] = talib.CDLHAMMER(o, h, low_vals, c)
    df["engulfing"] = talib.CDLENGULFING(o, h, low_vals, c)
    return df


def add_support_resistance(
    df: pd.DataFrame,
    window: int = 20,
    price_col: str = "close",
) -> pd.DataFrame:
    """Add support/resistance levels and breakout signals."""
    df["resistance"] = df["high"].rolling(window=window).max()
    df["support"] = df["low"].rolling(window=window).min()
    df["resistance_distance"] = _safe_div(
        (df["resistance"] - df[price_col]).values,
        df[price_col].values,
    )
    df["support_distance"] = _safe_div(
        (df[price_col] - df["support"]).values,
        df[price_col].values,
    )
    df["breakout_high"] = (df[price_col] > df["resistance"].shift(1)).astype(int)
    df["breakout_low"] = (df[price_col] < df["support"].shift(1)).astype(int)
    return df


def add_all_indicators(
    df: pd.DataFrame,
    features_config: dict,
    data_config: dict | None = None,
) -> pd.DataFrame:
    """Add all technical indicators based on configuration dict."""
    result = df.copy()
    price_col = data_config.get("price_column", "close") if data_config else "close"
    volume_col = data_config.get("volume_column", "volume") if data_config else "volume"

    result = add_momentum_indicators(result, features_config, price_col)
    result = add_volatility_indicators(result, features_config, price_col)
    result = add_volume_indicators(result, features_config, price_col, volume_col)
    result = add_trend_indicators(result, features_config, price_col)
    result = add_pattern_recognition(result)
    result = add_support_resistance(result, window=20, price_col=price_col)
    return result

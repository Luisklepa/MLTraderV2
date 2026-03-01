"""
Unified feature engineering pipeline for ML trading strategies.

Single source of truth for all feature generation — avoids duplication
across technical_indicators.py, statistical_features.py, and pipeline.py.
"""

import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

try:
    import talib
except ImportError as e:
    raise ImportError("Could not import 'talib'. Make sure TA-Lib is installed in your venv.") from e


# ---------------------------------------------------------------------------
# Helper: safe division (vectorized, avoids inf/nan)
# ---------------------------------------------------------------------------
def _safe_div(a, b, fill: float = 0.0):
    """Element-wise a/b, returning *fill* where b is zero or near-zero."""
    b_safe = np.where(np.abs(b) < 1e-12, np.nan, b)
    result = a / b_safe
    return np.where(np.isfinite(result), result, fill)


# ---------------------------------------------------------------------------
# Vectorized rolling percentile rank (replaces slow lambda-based version)
# ---------------------------------------------------------------------------
def _rolling_pct_rank(series: pd.Series, window: int) -> pd.Series:
    """Vectorized rolling percentile rank using pandas built-in."""
    return series.rolling(window).rank(pct=True)


# ---------------------------------------------------------------------------
# Vectorized rolling linear slope (replaces rolling lambda + polyfit)
# ---------------------------------------------------------------------------
def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling OLS slope using vectorized operations."""
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    def _slope(y):
        return np.sum((x - x_mean) * (y - y.mean())) / x_var

    return series.rolling(window).apply(_slope, raw=True)


# =========================================================================
#  Standalone feature-group generators (no class needed)
# =========================================================================


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Basic price and return features."""
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
    df["price_change"] = _safe_div(df["close"] - df["open"], df["open"])
    df["gap"] = _safe_div(df["open"] - df["close"].shift(1), df["close"].shift(1))
    for w in [1, 5, 10]:
        df[f"return_{w}"] = df["close"].pct_change(w)
    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Volume-derived features. Uses rolling VWAP (not cumulative)."""
    df["volume_sma_10"] = df["volume"].rolling(10).mean()
    df["volume_sma_20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = _safe_div(df["volume"], df["volume_sma_10"])
    df["volume_price_trend"] = df["volume"] * df["returns"]
    # Rolling VWAP (20-bar) instead of cumulative
    tp = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap"] = (tp * df["volume"]).rolling(20).sum() / df["volume"].rolling(20).sum()
    df["vwap_distance"] = _safe_div(df["close"] - df["vwap"], df["vwap"])
    return df


def add_moving_average_features(df: pd.DataFrame) -> pd.DataFrame:
    """Moving averages and crossovers (reduced from 8 to 5 key periods)."""
    for period in [10, 20, 50, 100, 200]:
        df[f"sma_{period}"] = df["close"].rolling(period).mean()
        df[f"ema_{period}"] = df["close"].ewm(span=period).mean()
        df[f"price_sma_{period}_ratio"] = _safe_div(df["close"], df[f"sma_{period}"])
        df[f"price_ema_{period}_ratio"] = _safe_div(df["close"], df[f"ema_{period}"])
    # Key crossovers only
    df["sma_cross_20_50"] = (df["sma_20"] > df["sma_50"]).astype(int)
    df["ema_cross_20_50"] = (df["ema_20"] > df["ema_50"]).astype(int)
    return df


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Momentum indicators via TA-Lib."""
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    for period in [14, 21]:
        df[f"rsi_{period}"] = talib.RSI(close, timeperiod=period)

    df["macd"], df["macd_signal"], df["macd_histogram"] = talib.MACD(close)
    df["macd_cross"] = (df["macd"] > df["macd_signal"]).astype(int)

    df["stoch_k"], df["stoch_d"] = talib.STOCH(high, low, close)
    df["williams_r"] = talib.WILLR(high, low, close)
    df["cci"] = talib.CCI(high, low, close)
    df["adx_14"] = talib.ADX(high, low, close, timeperiod=14)

    for period in [10, 20]:
        df[f"momentum_{period}"] = talib.MOM(close, timeperiod=period)
        df[f"roc_{period}"] = talib.ROC(close, timeperiod=period)

    # Advanced momentum
    rolling_std = df["returns"].rolling(20).std().replace(0, np.nan)
    df["return_zscore_20"] = (df["returns"] - df["returns"].rolling(20).mean()) / rolling_std

    rm20 = df["close"].rolling(20).max()
    rn20 = df["close"].rolling(20).min()
    df["close_to_max_20"] = _safe_div(df["close"] - rm20, rm20)
    df["close_to_min_20"] = _safe_div(df["close"] - rn20, rn20)

    df["rsi_overbought"] = (df["rsi_14"] > 70).astype(int)
    df["rsi_oversold"] = (df["rsi_14"] < 30).astype(int)

    return df


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Volatility indicators via TA-Lib."""
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    for period in [14, 50]:
        df[f"atr_{period}"] = talib.ATR(high, low, close, timeperiod=period)

    for period in [20, 50]:
        upper, middle, lower = talib.BBANDS(close, timeperiod=period)
        df[f"bb_upper_{period}"] = upper
        df[f"bb_lower_{period}"] = lower
        bb_range = upper - lower
        df[f"bb_width_{period}"] = _safe_div(bb_range, middle)
        df[f"bb_position_{period}"] = _safe_div(df["close"].values - lower, bb_range)

    for period in [10, 20, 50]:
        df[f"volatility_{period}"] = df["returns"].rolling(period).std()

    df["true_range"] = talib.TRANGE(high, low, close)
    df["intraday_range"] = _safe_div(df["high"] - df["low"], df["close"])
    df["vol_of_vol_20"] = df["volatility_20"].rolling(20).std()
    df["volatility_custom_10"] = _safe_div(df["close"].rolling(10).std(), df["close"].rolling(10).mean())

    # Percentile ranks (vectorized, no lambda)
    df["atr_14_pctrank_20"] = _rolling_pct_rank(df["atr_14"], 20)
    df["atr_14_high"] = (df["atr_14_pctrank_20"] > 0.9).astype(int)
    df["atr_14_low"] = (df["atr_14_pctrank_20"] < 0.1).astype(int)

    df["bb_width_20_pctrank"] = _rolling_pct_rank(df["bb_width_20"], 20)

    # Donchian Channel
    df["donchian_high_20"] = df["high"].rolling(20).max()
    df["donchian_low_20"] = df["low"].rolling(20).min()
    df["donchian_width"] = df["donchian_high_20"] - df["donchian_low_20"]

    return df


def add_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    """Candlestick patterns and candle anatomy."""
    o, h, low_vals, c = (
        df["open"].values,
        df["high"].values,
        df["low"].values,
        df["close"].values,
    )
    df["doji"] = talib.CDLDOJI(o, h, low_vals, c)
    df["hammer"] = talib.CDLHAMMER(o, h, low_vals, c)
    df["engulfing"] = talib.CDLENGULFING(o, h, low_vals, c)
    df["morning_star"] = talib.CDLMORNINGSTAR(o, h, low_vals, c)
    df["evening_star"] = talib.CDLEVENINGSTAR(o, h, low_vals, c)

    df["body_size"] = _safe_div(np.abs(df["close"] - df["open"]), df["open"])
    df["upper_shadow"] = _safe_div(df["high"] - np.maximum(df["open"], df["close"]), df["open"])
    df["lower_shadow"] = _safe_div(np.minimum(df["open"], df["close"]) - df["low"], df["open"])
    df["candle_type"] = (df["close"] > df["open"]).astype(int)

    return df


def add_market_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Support/resistance, breakouts, trend strength."""
    h20 = df["high"].rolling(20).max()
    l20 = df["low"].rolling(20).min()
    df["high_20"] = h20
    df["low_20"] = l20
    df["resistance_distance"] = _safe_div(h20 - df["close"], df["close"])
    df["support_distance"] = _safe_div(df["close"] - l20, df["close"])
    df["breakout_high"] = (df["close"] > h20.shift(1)).astype(int)
    df["breakout_low"] = (df["close"] < l20.shift(1)).astype(int)
    # Vectorized trend strength (rolling OLS slope)
    df["trend_strength"] = _rolling_slope(df["close"], 20)
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar features from datetime index or column."""
    if "timestamp" in df.columns:
        dt = pd.to_datetime(df["timestamp"])
        hour = dt.dt.hour
        dow = dt.dt.dayofweek
    elif df.index.dtype.kind == "M":
        hour = df.index.hour
        dow = df.index.dayofweek
    else:
        return df

    df["hour"] = hour.values if hasattr(hour, "values") else hour
    df["day_of_week"] = dow.values if hasattr(dow, "values") else dow
    h = df["hour"]
    df["is_asian_session"] = ((h >= 0) & (h <= 8)).astype(int)
    df["is_london_session"] = ((h >= 8) & (h <= 16)).astype(int)
    df["is_ny_session"] = ((h >= 13) & (h <= 21)).astype(int)

    return df


def add_cross_features(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Cross features between indicators. Only creates feature if BOTH columns exist."""
    cross_specs = [
        ("ratio", "rsi_14", "atr_14"),
        ("ratio", "macd", "volatility_20"),
        ("diff", "rsi_14", "rsi_21"),
        ("diff", "ema_20", "sma_50"),
        ("prod", "rsi_14", "macd"),
        ("prod", "returns", "volume"),
    ]

    generated = []
    for op, f1, f2 in cross_specs:
        if f1 not in df.columns or f2 not in df.columns:
            if verbose:
                logger.info("Skipping cross feature %s_%s_%s: missing columns", f1, f2, op)
            continue

        if op == "ratio":
            name = f"{f1}_{f2}_ratio"
            df[name] = _safe_div(df[f1], df[f2])
        elif op == "diff":
            name = f"{f1}_{f2}_diff"
            df[name] = df[f1] - df[f2]
        elif op == "prod":
            name = f"{f1}_{f2}_prod"
            df[name] = df[f1] * df[f2]
        generated.append(name)

    if generated:
        df[generated] = df[generated].replace([np.inf, -np.inf], np.nan).fillna(0)
        df[generated] = df[generated].clip(-1e6, 1e6)

    return df


def add_advanced_cross_features(df: pd.DataFrame) -> pd.DataFrame:
    """Advanced cross-features. Only creates feature if base columns exist."""

    def _cross(name, f1, f2, op):
        if f1 not in df.columns or f2 not in df.columns:
            return
        if op == "div":
            df[name] = _safe_div(df[f1], df[f2])
        elif op == "mul":
            df[name] = df[f1] * df[f2]
        elif op == "sub":
            df[name] = df[f1] - df[f2]

    _cross("return1_over_atr14", "returns", "atr_14", "div")
    _cross("ema20_over_ema50", "ema_20", "ema_50", "div")

    if "macd" in df.columns and "volume" in df.columns:
        df["macd_vol_roc"] = df["macd"] * df["volume"].pct_change().fillna(0)

    if "rsi_14" in df.columns and "volume" in df.columns:
        vol_mean = df["volume"].rolling(20).mean()
        df["rsi_overbought_spike"] = ((df["rsi_14"] > 70) & (df["volume"] > vol_mean)).astype(int)

    if "close" in df.columns and "open" in df.columns:
        body = df["close"] - df["open"]
        df["wick_to_body_ratio"] = _safe_div(df["high"] - df["low"], np.abs(body))

    if "rsi_14" in df.columns and "atr_14" in df.columns:
        df["rsi_norm_by_atr"] = _safe_div(df["rsi_14"] - 50, df["atr_14"])

    if "volatility_50" in df.columns:
        vol_mean_100 = df["volatility_50"].rolling(100).mean()
        df["vol_jump"] = _safe_div(df["volatility_50"], vol_mean_100)
        df["vol_jump_flag"] = (df["vol_jump"] > 1.5).astype(int)

    # Cleanup
    cross_cols = [
        c
        for c in df.columns
        if c
        in [
            "return1_over_atr14",
            "ema20_over_ema50",
            "macd_vol_roc",
            "rsi_overbought_spike",
            "wick_to_body_ratio",
            "rsi_norm_by_atr",
            "vol_jump",
            "vol_jump_flag",
        ]
    ]
    if cross_cols:
        df[cross_cols] = df[cross_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    return df


def add_anti_failure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Volume spike, volatility squeeze, trend confirmation, return streaks."""
    vol_mean_20 = df["volume"].rolling(20).mean()
    df["vol_spike_20"] = _safe_div(df["volume"], vol_mean_20)

    if "atr_14" in df.columns:
        df["atr_squeeze"] = _safe_div(df["atr_14"], df["atr_14"].rolling(50).mean())

    if all(c in df.columns for c in ["ema_20", "ema_50", "macd", "rsi_14"]):
        df["trend_confirm"] = ((df["ema_20"] > df["ema_50"]) & (df["macd"] > 0) & (df["rsi_14"] > 50)).astype(int)

    if "returns" in df.columns:
        pos = (df["returns"] > 0).astype(int)
        neg = (df["returns"] < 0).astype(int)
        df["pos_streak"] = pos.groupby((pos != pos.shift()).cumsum()).cumsum()
        df["neg_streak"] = neg.groupby((neg != neg.shift()).cumsum()).cumsum()

    return df


def add_lag_features(
    df: pd.DataFrame,
    lag_periods: list[int] | None = None,
) -> pd.DataFrame:
    """Create lagged versions of key features."""
    lag_periods = lag_periods or [1, 2, 3, 5]
    lag_cols = ["returns", "volume_ratio", "rsi_14", "macd", "atr_14", "bb_position_20"]

    for col in lag_cols:
        if col not in df.columns:
            continue
        for lag in lag_periods:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)

    return df


def add_conditional_features(df: pd.DataFrame, target_type: str) -> pd.DataFrame:
    """Features specific to long or short target."""
    if target_type == "long":
        if "macd" in df.columns:
            df["bullish_momentum"] = (df["macd"] > 0).astype(int)
        if "rsi_14" in df.columns:
            df["rsi_above_60"] = (df["rsi_14"] > 60).astype(int)
        if "close" in df.columns and "open" in df.columns:
            df["vol_up_on_green"] = (
                (df["close"] > df["open"]) & (df["volume"] > df["volume"].rolling(10).mean())
            ).astype(int)
        if "engulfing" in df.columns:
            df["bullish_engulfing"] = (df["engulfing"] > 0).astype(int)
        if "low_20" in df.columns:
            df["dist_to_support"] = _safe_div(df["close"] - df["low_20"], df["close"])
        if "volatility_20" in df.columns:
            df["low_volatility"] = (df["volatility_20"] < df["volatility_20"].rolling(20).mean()).astype(int)

    elif target_type == "short":
        if "macd" in df.columns:
            df["bearish_momentum"] = (df["macd"] < 0).astype(int)
        if "rsi_14" in df.columns:
            df["rsi_below_40"] = (df["rsi_14"] < 40).astype(int)
        if "close" in df.columns and "open" in df.columns:
            df["vol_up_on_red"] = (
                (df["close"] < df["open"]) & (df["volume"] > df["volume"].rolling(10).mean())
            ).astype(int)
        if "engulfing" in df.columns:
            df["bearish_engulfing"] = (df["engulfing"] < 0).astype(int)
        if "high_20" in df.columns:
            df["dist_to_resistance"] = _safe_div(df["high_20"] - df["close"], df["close"])
        if "volatility_20" in df.columns:
            df["high_volatility"] = (df["volatility_20"] > df["volatility_20"].rolling(20).mean()).astype(int)

    return df


# =========================================================================
#  MLFeaturePipeline — orchestrator class
# =========================================================================

# Columns that are NEVER features (targets, identifiers, raw OHLCV)
_EXCLUDE_COLS = frozenset(
    [
        "target",
        "target_long",
        "target_short",
        "future_return",
        "future_return_long",
        "future_return_short",
        "future_max",
        "ema_vs_fut_return5",
        "position",
        "position_long",
        "position_short",
        "datetime",
        "timestamp",
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
)


class MLFeaturePipeline:
    """Orchestrates feature generation, target creation, and scaling."""

    def __init__(self, target_profit_pct: float = 2.0, target_bars: int = 10):
        self.target_profit_pct = target_profit_pct
        self.target_bars = target_bars
        self.scaler: StandardScaler | None = None
        self.feature_columns: list[str] | None = None

    # ---- Data loading ----

    def load_and_prepare_data(self, file_path: str) -> pd.DataFrame:
        """Load CSV and validate required columns."""
        logger.info("Loading data from %s", file_path)
        df = pd.read_csv(file_path)
        required = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Dataset must contain: {missing}")
        df = df.dropna(subset=required)
        df = df[df["volume"] > 0]
        logger.info("Loaded %d rows", len(df))
        return df

    # ---- Full feature engineering ----

    def feature_engineering(self, df: pd.DataFrame, target_type: str = "long") -> pd.DataFrame:
        """Run the full feature engineering pipeline."""
        df = add_price_features(df)
        df = add_volume_features(df)
        df = add_moving_average_features(df)
        df = add_momentum_features(df)
        df = add_volatility_features(df)
        df = add_pattern_features(df)
        df = add_market_structure_features(df)
        df = add_temporal_features(df)
        df = add_cross_features(df)
        df = add_advanced_cross_features(df)
        df = add_anti_failure_features(df)
        df = add_conditional_features(df, target_type)
        df = add_lag_features(df)
        return df

    # Keep legacy name as alias
    def feature_engineering_conditional(self, df, target_type="long"):
        return self.feature_engineering(df, target_type)

    def create_lag_features(self, df, lag_periods=None):
        return add_lag_features(df, lag_periods or [1, 2, 3, 5, 10])

    # ---- Target generation ----

    def generate_target_variable(
        self,
        df: pd.DataFrame,
        future_bars: int = 10,
        threshold: float = 0.01,
        target_type: str = "long",
    ) -> pd.DataFrame:
        """Create target variable. Drops lookahead rows at the end."""
        df["future_return"] = df["close"].shift(-future_bars) / df["close"] - 1
        if target_type == "long":
            df["target"] = (df["future_return"] > threshold).astype(int)
        elif target_type == "short":
            df["target"] = (df["future_return"] < -threshold).astype(int)
        else:
            raise ValueError("target_type must be 'long' or 'short'")
        df = df.iloc[:-future_bars]
        dist = df["target"].value_counts()
        logger.info("Target distribution:\n%s", dist)
        return df

    # ---- Cleaning and scaling ----

    def clean_and_scale_features(
        self,
        df: pd.DataFrame,
        fit_scaler: bool | None = True,
    ) -> pd.DataFrame:
        """Clean inf/NaN and optionally scale features.

        Args:
            df: DataFrame with features.
            fit_scaler: True  -> fit StandardScaler on this data (training).
                        False -> transform with previously fitted scaler (test/production).
                        None  -> skip scaling entirely.
        """
        feature_cols = [c for c in df.columns if c not in _EXCLUDE_COLS]

        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

        if "target" in df.columns:
            df = df.dropna(subset=["target"])

        df[feature_cols] = df[feature_cols].fillna(0)

        if fit_scaler is not None:
            if fit_scaler:
                self.scaler = StandardScaler()
                df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
            else:
                if self.scaler is None:
                    raise RuntimeError("Scaler not fitted yet. Call with fit_scaler=True first.")
                df[feature_cols] = self.scaler.transform(df[feature_cols])

        self.feature_columns = feature_cols
        logger.info("Total features: %d", len(feature_cols))
        return df

    # ---- High-level dataset generators ----

    def generate_complete_dataset(
        self,
        file_path: str,
        output_path: str = "ml_dataset.csv",
        threshold: float = 0.01,
        window: int = 10,
        target_type: str = "long",
    ) -> None:
        """Full pipeline: load -> features -> target -> scale -> save.

        IMPORTANT: scaler is fit ONLY on training portion (first 80%).
        """
        logger.info("=== STARTING FEATURE PIPELINE for target: %s ===", target_type)
        df = self.load_and_prepare_data(file_path)
        df = self.feature_engineering(df, target_type=target_type)
        df = self.generate_target_variable(df, future_bars=window, threshold=threshold, target_type=target_type)

        # Temporal split for scaler: fit on first 80%, transform rest
        split_idx = int(len(df) * 0.8)
        feature_cols = [c for c in df.columns if c not in _EXCLUDE_COLS]
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

        self.scaler = StandardScaler()
        df.iloc[:split_idx, df.columns.get_indexer(feature_cols)] = self.scaler.fit_transform(
            df.iloc[:split_idx][feature_cols]
        )
        df.iloc[split_idx:, df.columns.get_indexer(feature_cols)] = self.scaler.transform(
            df.iloc[split_idx:][feature_cols]
        )
        self.feature_columns = feature_cols

        if target_type == "long":
            df["position"] = df["target"]
        elif target_type == "short":
            df["position"] = -df["target"]
        else:
            df["position"] = 0

        df.to_csv(output_path, index=False)
        logger.info("Dataset saved to %s — shape %s", output_path, df.shape)

    def generate_combined_dataset(
        self,
        file_path: str,
        output_path: str = "ml_dataset_combined.csv",
        window_long: int = 10,
        threshold_long: float = 0.01,
        window_short: int = 10,
        threshold_short: float = 0.01,
    ) -> None:
        """Combined long+short dataset with proper scaler handling."""
        logger.info("=== STARTING COMBINED PIPELINE (LONG+SHORT) ===")
        df = self.load_and_prepare_data(file_path)
        df = self.feature_engineering(df, target_type="long")

        df["future_return_long"] = df["close"].shift(-window_long) / df["close"] - 1
        df["future_return_short"] = df["close"].shift(-window_short) / df["close"] - 1
        df["target_long"] = (df["future_return_long"] > threshold_long).astype(int)
        df["target_short"] = (df["future_return_short"] < -threshold_short).astype(int)

        min_bars = max(window_long, window_short)
        df = df.iloc[:-min_bars]

        df["position_long"] = df["target_long"]
        df["position_short"] = -df["target_short"]
        df["position"] = df["position_long"] + df["position_short"]

        # Temporal-split scaler
        feature_cols = [c for c in df.columns if c not in _EXCLUDE_COLS]
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

        split_idx = int(len(df) * 0.8)
        self.scaler = StandardScaler()
        df.iloc[:split_idx, df.columns.get_indexer(feature_cols)] = self.scaler.fit_transform(
            df.iloc[:split_idx][feature_cols]
        )
        df.iloc[split_idx:, df.columns.get_indexer(feature_cols)] = self.scaler.transform(
            df.iloc[split_idx:][feature_cols]
        )
        self.feature_columns = feature_cols

        df.to_csv(output_path, index=False)
        logger.info("Combined dataset saved to %s — shape %s", output_path, df.shape)


def select_important_cross_features(
    df: pd.DataFrame,
    target: str,
    train_end_idx: int,
    importance_threshold: float = 0.005,
    top_n: int | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Select important cross features using only TRAINING data.

    Args:
        df: Full DataFrame.
        target: Target column name.
        train_end_idx: Index up to which data is training (exclusive).
        importance_threshold: Minimum importance to keep.
        top_n: If set, keep only top_n cross features.

    Returns:
        DataFrame with unimportant cross features dropped, and list of kept cross features.
    """
    from sklearn.ensemble import RandomForestClassifier

    cross_feats = [c for c in df.columns if any(s in c for s in ["_ratio", "_diff", "_prod", "_logic"])]
    if not cross_feats:
        return df, []

    base_feats = [c for c in df.columns if c not in cross_feats + [target] and c not in _EXCLUDE_COLS]
    all_feats = base_feats + cross_feats

    train_df = df.iloc[:train_end_idx]
    X_train = train_df[all_feats].fillna(0)
    y_train = train_df[target]

    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    importances = pd.Series(rf.feature_importances_, index=all_feats)
    cross_importances = importances[cross_feats].sort_values(ascending=False)

    if top_n:
        selected = cross_importances.head(top_n).index.tolist()
    else:
        selected = cross_importances[cross_importances > importance_threshold].index.tolist()

    to_drop = [f for f in cross_feats if f not in selected]
    df = df.drop(columns=to_drop)
    logger.info("Kept %d / %d cross features", len(selected), len(cross_feats))
    return df, selected


# Ejecutar pipeline
if __name__ == "__main__":
    from core.logging_config import setup_logging

    setup_logging()
    pipeline = MLFeaturePipeline()
    pipeline.generate_complete_dataset(
        "btcusdt_prices.csv",
        output_path="btcusdt_ml_dataset.csv",
        threshold=0.01,
        window=10,
    )
    logger.info("Feature pipeline completed!")

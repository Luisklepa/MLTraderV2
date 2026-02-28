"""Data validation utilities for OHLCV DataFrames."""
import pandas as pd
import numpy as np
import logging
from typing import Optional

from core.exceptions import DataError

logger = logging.getLogger(__name__)

REQUIRED_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def validate_ohlcv(
    df: pd.DataFrame,
    timeframe_seconds: Optional[float] = None,
    max_gap_factor: float = 2.5,
) -> pd.DataFrame:
    """Validate an OHLCV DataFrame and return a cleaned copy.

    Raises DataError for unrecoverable issues.
    Logs warnings for fixable issues.
    """
    if df is None or df.empty:
        raise DataError("DataFrame is empty or None")

    missing = [c for c in REQUIRED_OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f"Missing required columns: {missing}")

    df = df.copy()

    numeric_cols = [c for c in REQUIRED_OHLCV_COLUMNS if c in df.columns]
    inf_mask = df[numeric_cols].isin([np.inf, -np.inf]).any(axis=1)
    if inf_mask.any():
        n_inf = inf_mask.sum()
        logger.warning(f"Dropping {n_inf} rows with infinite values")
        df = df[~inf_mask]

    if not pd.api.types.is_numeric_dtype(df["close"]):
        raise DataError("Column 'close' is not numeric")

    if (df["close"] <= 0).any():
        n_bad = (df["close"] <= 0).sum()
        logger.warning(f"Dropping {n_bad} rows with non-positive close prices")
        df = df[df["close"] > 0]

    if (df["volume"] < 0).any():
        n_bad = (df["volume"] < 0).sum()
        logger.warning(f"Dropping {n_bad} rows with negative volume")
        df = df[df["volume"] >= 0]

    hl_violations = df["high"] < df["low"]
    if hl_violations.any():
        n_bad = hl_violations.sum()
        logger.warning(f"Fixing {n_bad} rows where high < low (swapping)")
        df.loc[hl_violations, ["high", "low"]] = df.loc[hl_violations, ["low", "high"]].values

    if timeframe_seconds and hasattr(df.index, 'freq'):
        idx = pd.to_datetime(df.index)
        gaps = np.diff(idx.values).astype("timedelta64[s]").astype(float)
        max_allowed = timeframe_seconds * max_gap_factor
        large_gaps = gaps > max_allowed
        if large_gaps.any():
            n_gaps = large_gaps.sum()
            logger.warning(
                f"Found {n_gaps} time gaps exceeding {max_gap_factor}x the timeframe"
            )

    if df.empty:
        raise DataError("DataFrame is empty after validation")

    return df

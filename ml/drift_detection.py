"""Feature drift detection for production monitoring."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Calculate Population Stability Index between two distributions.

    PSI < 0.1: no significant shift
    PSI 0.1-0.25: moderate shift, investigate
    PSI > 0.25: significant shift, likely need retraining
    """
    expected = np.array(expected, dtype=float)
    actual = np.array(actual, dtype=float)

    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    breakpoints = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    breakpoints = np.unique(breakpoints)

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    expected_pct = (expected_counts + 1e-6) / (len(expected) + 1e-6 * len(expected_counts))
    actual_pct = (actual_counts + 1e-6) / (len(actual) + 1e-6 * len(actual_counts))

    eps = 1e-10
    psi = np.sum((actual_pct - expected_pct) * np.log((actual_pct + eps) / (expected_pct + eps)))
    return float(psi)


def detect_drift(
    reference_df: pd.DataFrame,
    production_df: pd.DataFrame,
    feature_cols: list[str],
    psi_threshold: float = 0.25,
    alert_ratio: float = 0.1,
    alert_count: int = 5,
) -> dict[str, any]:
    """Detect feature drift between reference (training) and production data.

    Returns a dict with:
        - drifted_features: list of features with PSI above threshold
        - psi_scores: dict of feature -> PSI value
        - alert: bool indicating whether retraining is recommended
    """
    psi_scores = {}
    drifted = []

    for col in feature_cols:
        if col not in reference_df.columns or col not in production_df.columns:
            continue

        psi = calculate_psi(
            reference_df[col].values,
            production_df[col].values,
        )
        psi_scores[col] = psi

        if psi > psi_threshold:
            drifted.append(col)

    drift_ratio = len(drifted) / max(len(feature_cols), 1)
    alert = drift_ratio > alert_ratio or len(drifted) > alert_count

    if drifted:
        logger.warning(
            f"Drift detected in {len(drifted)}/{len(feature_cols)} features "
            f"(PSI > {psi_threshold}): {drifted[:5]}{'...' if len(drifted) > 5 else ''}"
        )
    else:
        logger.info("No significant feature drift detected")

    return {
        "drifted_features": drifted,
        "psi_scores": psi_scores,
        "drift_ratio": drift_ratio,
        "alert": alert,
    }

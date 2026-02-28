"""Automated feature selection for production ML pipelines."""
import pandas as pd
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


def remove_highly_correlated(
    df: pd.DataFrame,
    feature_cols: List[str],
    threshold: float = 0.95,
    importance: Optional[pd.Series] = None,
) -> List[str]:
    """Remove features with pairwise correlation above *threshold*.

    When *importance* is provided (a Series mapping feature name -> score),
    the less important feature in each correlated pair is dropped.
    Otherwise the second column in the pair (by column order) is dropped.
    """
    valid_cols = [c for c in feature_cols if c in df.columns]
    corr_matrix = df[valid_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop: set = set()
    for col in upper.columns:
        correlated = upper.index[upper[col] > threshold].tolist()
        for corr_col in correlated:
            if importance is not None and col in importance.index and corr_col in importance.index:
                drop = col if importance[col] < importance[corr_col] else corr_col
            else:
                drop = col
            to_drop.add(drop)

    kept = [c for c in feature_cols if c not in to_drop]
    logger.info("Removed %d correlated features (r > %.2f), kept %d", len(to_drop), threshold, len(kept))
    return kept


def select_top_features_by_importance(
    model,
    feature_cols: List[str],
    top_n: int = 50,
) -> List[str]:
    """Select top N features by model feature importance."""
    if not hasattr(model, "feature_importances_"):
        logger.warning("Model has no feature_importances_, returning all features")
        return feature_cols

    importances = pd.Series(model.feature_importances_, index=feature_cols)
    top_n = min(top_n, len(feature_cols))
    selected = importances.nlargest(top_n).index.tolist()
    logger.info("Selected top %d features by importance", len(selected))
    return selected


def auto_select_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    model=None,
    correlation_threshold: float = 0.95,
    max_features: int = 50,
) -> List[str]:
    """Full feature selection pipeline: correlation filter + importance ranking.

    If *model* is provided, its feature importances are used both for
    importance-aware correlation removal and for final top-N ranking.
    """
    importance = None
    if model is not None and hasattr(model, "feature_importances_"):
        importance = pd.Series(model.feature_importances_, index=feature_cols)

    cols = remove_highly_correlated(
        df, feature_cols, threshold=correlation_threshold, importance=importance,
    )

    if model is not None and len(cols) > max_features:
        cols = select_top_features_by_importance(model, cols, top_n=max_features)

    logger.info("Final feature set: %d features", len(cols))
    return cols

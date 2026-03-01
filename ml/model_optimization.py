"""
Model optimization utilities for the ML trading system.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


class ModelOptimizer:
    """Handles model optimization and maintenance."""

    def __init__(self, config: dict):
        """Initialize optimizer with configuration."""
        self.config = config
        self.feature_importance_threshold = config.get("min_feature_importance", 0.01)
        self.max_features = config.get("max_features", 50)
        self.n_trials = config.get("optimization_trials", 100)
        self.cv_folds = config.get("cv_folds", 5)
        self.model_dir = Path(config.get("model_dir", "models"))
        self.model_dir.mkdir(exist_ok=True)

    def optimize_feature_selection(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, list[str]]:
        """Select most important features using model-based selection."""
        selector = SelectFromModel(
            estimator=xgb.XGBClassifier(n_estimators=100),
            threshold=self.feature_importance_threshold,
            max_features=self.max_features,
            prefit=False,
        )

        selector.fit(X, y)
        selected_features = X.columns[selector.get_support()].tolist()

        logger.info(f"Selected {len(selected_features)} features")
        return X[selected_features], selected_features

    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Optimize model hyperparameters and threshold using Optuna."""

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
                "gamma": trial.suggest_float("gamma", 0, 1),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
                "reg_lambda": trial.suggest_float("reg_lambda", 0, 1),
            }
            skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            f1_scores = []
            thresholds = []
            for train_idx, valid_idx in skf.split(X, y):
                X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
                y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
                model = xgb.XGBClassifier(**params, eval_metric="logloss")
                model.fit(X_train, y_train)
                y_proba = model.predict_proba(X_valid)[:, 1]
                best_f1 = 0
                best_thr = 0.5
                for thr in np.arange(0.1, 0.91, 0.01):
                    y_pred = (y_proba > thr).astype(int)
                    f1 = f1_score(y_valid, y_pred)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_thr = thr
                f1_scores.append(best_f1)
                thresholds.append(best_thr)
            avg_f1 = np.mean(f1_scores)
            avg_thr = float(np.mean(thresholds))
            trial.set_user_attr("best_threshold", avg_thr)
            return avg_f1

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials)
        logger.info(f"Best trial: score={study.best_value:.4f}")
        for key, value in study.best_params.items():
            logger.info(f"    {key}: {value}")
        best_threshold = study.best_trial.user_attrs.get("best_threshold", 0.5)
        best_params = study.best_params
        return {"best_params": best_params, "best_threshold": best_threshold}

    def evaluate_model_drift(
        self, model: xgb.XGBClassifier, X_recent: pd.DataFrame, y_recent: pd.Series, threshold: float = 0.1
    ) -> bool:
        """Evaluate if model performance has drifted significantly."""
        y_pred = model.predict_proba(X_recent)[:, 1]
        recent_auc = roc_auc_score(y_recent, y_pred)
        recent_f1 = f1_score(y_recent, y_pred > 0.5)

        # Load historical metrics
        metrics_file = self.model_dir / "historical_metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                historical_metrics = json.load(f)
            historical_f1 = historical_metrics["f1"][-1]

            # Check for significant degradation
            if (historical_f1 - recent_f1) > threshold:
                logger.warning(f"Model performance degraded: {historical_f1:.4f} -> {recent_f1:.4f}")
                return True

        # Save current metrics
        self.save_metrics({"timestamp": datetime.now().isoformat(), "auc": recent_auc, "f1": recent_f1})

        return False

    def save_metrics(self, metrics: dict) -> None:
        """Save model metrics to historical record."""
        metrics_file = self.model_dir / "historical_metrics.json"

        if metrics_file.exists():
            with open(metrics_file) as f:
                historical_metrics = json.load(f)
        else:
            historical_metrics = {"timestamp": [], "auc": [], "f1": []}

        # Update metrics
        historical_metrics["timestamp"].append(metrics["timestamp"])
        historical_metrics["auc"].append(metrics["auc"])
        historical_metrics["f1"].append(metrics["f1"])

        # Save updated metrics
        with open(metrics_file, "w") as f:
            json.dump(historical_metrics, f, indent=4)

    def save_model(
        self, model: xgb.XGBClassifier, features: list[str], metrics: dict, threshold: float | None = None
    ) -> None:
        """Save model and its metadata. threshold is the optimal prediction threshold (e.g. from Optuna)."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save model
        model_path = self.model_dir / f"model_{timestamp}.pkl"
        joblib.dump(model, model_path)

        # Save metadata
        metadata = {
            "timestamp": timestamp,
            "features": features,
            "metrics": metrics,
            "model_params": model.get_params(),
        }
        if threshold is not None:
            metadata["best_threshold"] = float(threshold)

        metadata_path = self.model_dir / f"metadata_{timestamp}.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        logger.info(f"Model saved to {model_path}")
        logger.info(f"Metadata saved to {metadata_path}")

    def load_latest_model(self) -> tuple[xgb.XGBClassifier, list[str], dict, float]:
        """Load the latest model and its metadata. Returns (model, features, metrics, threshold)."""
        model_files = list(self.model_dir.glob("model_*.pkl"))
        if not model_files:
            raise FileNotFoundError("No saved models found")

        latest_model_file = max(model_files, key=lambda x: x.stat().st_mtime)
        return self.load_model_by_path(latest_model_file)

    def load_model_by_path(self, model_path: Path) -> tuple[xgb.XGBClassifier, list[str], dict, float]:
        """Load model and metadata by path. Returns (model, features, metrics, threshold)."""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # timestamp from filename: model_20250101_120000.pkl -> 20250101_120000
        timestamp = "_".join(model_path.stem.split("_")[1:])

        model = joblib.load(model_path)
        metadata_file = self.model_dir / f"metadata_{timestamp}.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata not found for {model_path}")

        with open(metadata_file, encoding="utf-8") as f:
            metadata = json.load(f)

        features = metadata.get("features", [])
        metrics = metadata.get("metrics", {})
        threshold = metadata.get("best_threshold", 0.5)
        return model, features, metrics, threshold

    def analyze_feature_importance(self, model: xgb.XGBClassifier, feature_names: list[str]) -> pd.DataFrame:
        """Analyze and return feature importance."""
        importance_df = pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
        importance_df = importance_df.sort_values("importance", ascending=False)

        return importance_df

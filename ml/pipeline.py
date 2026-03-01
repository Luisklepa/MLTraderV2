"""
ML Pipeline — config-driven training, prediction, and signal generation.

Decomposed into focused components:
  - PipelineConfig: YAML configuration
  - ModelConfig: single model hyperparameters
  - FeatureEngine: feature generation (delegates to technical_indicators + statistical_features)
  - ModelTrainer: training + evaluation
  - SignalGenerator: signal generation with position sizing
  - MLPipeline: orchestrator that ties everything together
"""

import logging
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import talib
import xgboost as xgb
import yaml
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .statistical_features import add_all_statistical_features
from .technical_indicators import add_all_indicators

logger = logging.getLogger(__name__)


# =========================================================================
#  Configuration
# =========================================================================


@dataclass
class ModelConfig:
    """Hyperparameters for a single XGBoost model."""

    n_estimators: int = 100
    max_depth: int = 3
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 1
    gamma: float = 0
    scale_pos_weight: float = 1

    def to_dict(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "gamma": self.gamma,
            "scale_pos_weight": self.scale_pos_weight,
        }


class PipelineConfig:
    """Load and expose YAML configuration."""

    def __init__(self, config_path: str):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        self.model_config = config["model_config"]
        self.features_config = config["features_config"]
        self.data_config = config["data_config"]

    @property
    def long_params(self) -> dict[str, Any]:
        return self.model_config["ensemble"]["models"][0]["params"]

    @property
    def short_params(self) -> dict[str, Any]:
        return self.model_config["ensemble"]["models"][0]["params"]

    @property
    def selected_features(self) -> list[str]:
        features = []
        for indicator in self.features_config["technical_indicators"]:
            features.append(indicator["name"])
        features.extend(
            [
                "returns_1",
                "returns_5",
                "returns_10",
                "volatility_10",
                "volatility_20",
                "zscore_close_20",
                "zscore_volume_20",
                "hour",
                "weekday",
                "month",
                "is_weekend",
                "session_progress",
            ]
        )
        return features

    @property
    def use_smote(self) -> bool:
        return False

    @property
    def threshold(self) -> float:
        return self.model_config["prediction"]["threshold"]

    @property
    def min_prediction_confidence(self) -> float:
        return self.model_config["prediction"]["min_prediction_confidence"]


# =========================================================================
#  FeatureEngine — delegates to technical_indicators + statistical_features
# =========================================================================


def _safe_div(a, b, fill: float = 0.0):
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(np.abs(b) < 1e-12, fill, a / b)
    return np.where(np.isfinite(result), result, fill)


class FeatureEngine:
    """Generates features using config-driven indicators."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._cache: dict[int, pd.DataFrame] = {}

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add all configured technical + statistical features."""
        df = data.copy()

        # Config-driven indicators
        df = add_all_indicators(df, self.config.features_config, self.config.data_config)
        df = add_all_statistical_features(df, self.config.features_config, self.config.data_config)

        # Extra statistical features
        df["returns_1"] = df["close"].pct_change()
        df["returns_5"] = df["close"].pct_change(5)
        df["returns_10"] = df["close"].pct_change(10)
        df["volatility_10"] = df["returns_1"].rolling(10).std()
        df["volatility_20"] = df["returns_1"].rolling(20).std()

        std_20 = df["close"].rolling(20).std().replace(0, np.nan)
        df["zscore_close_20"] = (df["close"] - df["close"].rolling(20).mean()) / std_20
        vol_std = df["volume"].rolling(20).std().replace(0, np.nan)
        df["zscore_volume_20"] = (df["volume"] - df["volume"].rolling(20).mean()) / vol_std

        self._add_temporal_features(df)
        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> None:
        """Extract temporal features from datetime index."""
        if not hasattr(df.index, "hour"):
            return
        df["hour"] = df.index.hour
        df["weekday"] = df.index.weekday
        df["month"] = df.index.month
        df["is_weekend"] = df.index.weekday.isin([5, 6]).astype(int)
        df["session_progress"] = (df["hour"] * 60 + df.index.minute) / (24 * 60)

    def calculate_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced features with caching."""
        df = data.copy()
        close = np.asarray(df["close"].values, dtype=np.float64)
        high = np.asarray(df["high"].values, dtype=np.float64)
        low = np.asarray(df["low"].values, dtype=np.float64)
        volume = np.asarray(df["volume"].values, dtype=np.float64)
        o = np.asarray(df["open"].values, dtype=np.float64)

        cache_key = hash(close.tobytes())
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            for col in cached.columns:
                if col not in df.columns:
                    df[col] = cached[col]
            return df

        for period in [20, 50, 200]:
            col = f"ema_{period}"
            if col not in df.columns:
                df[col] = talib.EMA(close, timeperiod=period)
                ratio_col = f"price_ema_{period}_ratio"
                if ratio_col not in df.columns:
                    df[ratio_col] = pd.Series(
                        _safe_div(close, df[col].values),
                        index=df.index,
                    )

        if "trend_strength" not in df.columns and "price_ema_20_ratio" in df.columns:
            df["trend_strength"] = np.abs(df["price_ema_20_ratio"] - 1) * 100
        if "trend_direction" not in df.columns and "price_ema_20_ratio" in df.columns:
            df["trend_direction"] = np.sign(df["price_ema_20_ratio"] - 1)

        if "volatility_regime" not in df.columns:
            returns = np.diff(np.log(close))
            returns = np.insert(returns, 0, 0)
            vol_win = 20
            shape = (len(returns) - vol_win + 1, vol_win)
            strides = (returns.strides[0], returns.strides[0])
            wins = np.lib.stride_tricks.as_strided(returns, shape=shape, strides=strides)
            vol = np.std(wins, axis=1)
            vol = np.pad(vol, (vol_win - 1, 0), mode="edge")
            vol_sma = np.convolve(vol, np.ones(100) / 100, mode="valid")
            vol_sma = np.pad(vol_sma, (99, 0), mode="edge")
            df["volatility_regime"] = np.where(vol > vol_sma, 1, -1)
            df["volatility_ratio"] = pd.Series(_safe_div(vol, vol_sma), index=df.index)

        if "volume_ratio" not in df.columns:
            vol_sma = np.convolve(volume, np.ones(20) / 20, mode="valid")
            vol_sma = np.pad(vol_sma, (19, 0), mode="edge")
            df["volume_ratio"] = pd.Series(_safe_div(volume, vol_sma), index=df.index)

        patterns = [
            talib.CDL3BLACKCROWS,
            talib.CDL3WHITESOLDIERS,
            talib.CDLENGULFING,
            talib.CDLHARAMI,
            talib.CDLMORNINGSTAR,
            talib.CDLEVENINGSTAR,
        ]
        for func in patterns:
            name = f"pattern_{func.__name__.replace('CDL', '').lower()}"
            if name not in df.columns:
                try:
                    df[name] = func(o, high, low, close)
                except Exception as e:
                    logger.debug("Candlestick pattern %s failed: %s", name, e)
                    df[name] = 0

        new_cols = [c for c in df.columns if c not in data.columns]
        if new_cols:
            self._cache[cache_key] = df[new_cols].copy()

        return df


# =========================================================================
#  ModelTrainer — training + evaluation
# =========================================================================


class ModelTrainer:
    """Handles model training, evaluation, and walk-forward."""

    def __init__(self, config: PipelineConfig, random_state: int = 42):
        self.config = config
        self.random_state = random_state
        self.preprocessors: dict[str, Pipeline | None] = {"long": None, "short": None}

    def prepare_data(
        self,
        df: pd.DataFrame,
        side: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Temporal train/test split with scaler fitted ONLY on train."""
        X = df[self.config.selected_features]
        y = df[f"{side}_target"]

        n = len(X)
        split_idx = int(n * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        preprocessor = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]
        )
        X_train_np = preprocessor.fit_transform(X_train)
        X_test_np = preprocessor.transform(X_test)
        self.preprocessors[side] = preprocessor

        if self.config.use_smote:
            minority = int((y_train == 1).sum())
            if minority > 1 and int((y_train == 0).sum()) > 1:
                k = min(5, minority - 1)
                if k >= 1:
                    smote = SMOTE(random_state=self.random_state, k_neighbors=k)
                    X_train_np, y_train = smote.fit_resample(X_train_np, y_train)

        return X_train_np, X_test_np, y_train, y_test

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        side: str,
    ) -> tuple[xgb.XGBClassifier, dict[str, float]]:
        """Train a single XGBoost model."""
        params = self.config.long_params if side == "long" else self.config.short_params

        pos = int((y_train == 1).sum())
        neg = int((y_train == 0).sum())
        if pos > 0 and neg > 0:
            spw = max(1.0, neg / max(pos, 1))
            params = {**params, "scale_pos_weight": float(spw)}

        params = {**params, "random_state": self.random_state}
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)

        pred = model.predict(X_train)
        tp = int((pred & y_train).sum())
        metrics = {
            "accuracy": float((pred == y_train).mean()),
            "precision": tp / max(int(pred.sum()), 1),
            "recall": tp / max(int(y_train.sum()), 1),
            "f1": 2 * tp / max(int(pred.sum()) + int(y_train.sum()), 1),
        }
        return model, metrics

    def evaluate_model(
        self,
        model: xgb.XGBClassifier,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict[str, float]:
        """Evaluate on test set."""
        pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
        }
        try:
            proba = model.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = roc_auc_score(y_test, proba)
        except Exception as e:
            logger.warning("Could not compute ROC-AUC: %s", e)
            metrics["roc_auc"] = float("nan")
        return metrics

    def walk_forward_oos(
        self,
        df: pd.DataFrame,
        side: str,
        n_splits: int = 5,
        min_train_frac: float = 0.5,
    ) -> pd.DataFrame:
        """Walk-forward out-of-sample predictions."""
        X_full = df[self.config.selected_features]
        y_full = df[f"{side}_target"]
        n = len(X_full)
        start = int(n * min_train_frac)
        if start <= 1:
            raise ValueError("Dataset too small for walk-forward")

        step = max(1, (n - start) // max(n_splits, 1))
        slices = []
        pos = start
        while pos < n:
            end = min(n, pos + step)
            slices.append((pos, end))
            pos = end

        preds = []
        for t0, t1 in slices:
            X_train = X_full.iloc[:t0]
            X_test = X_full.iloc[t0:t1]
            if len(X_test) == 0:
                continue
            pp = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler()),
                ]
            )
            X_tr = pp.fit_transform(X_train)
            X_te = pp.transform(X_test)
            y_tr = y_full.iloc[:t0]
            model, _ = self.train_model(X_tr, y_tr, side)
            proba = model.predict_proba(X_te)[:, 1]
            fold_df = pd.DataFrame({f"{side}_prob": proba}, index=X_test.index)
            preds.append(fold_df)

        if not preds:
            raise ValueError("No OOS splits generated")

        out = pd.concat(preds).sort_index()
        th = self.config.threshold
        mc = self.config.min_prediction_confidence
        out[f"{side}_signal"] = ((out[f"{side}_prob"] > th) & (out[f"{side}_prob"] > mc)).astype(int)
        return out


# =========================================================================
#  SignalGenerator — position sizing and signal filtering
# =========================================================================


class SignalGenerator:
    """Generate trading signals with position sizing."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def calculate_position_size(
        self,
        probability: float,
        model_config: dict,
        current_drawdown: float = 0,
        market_data: pd.Series | None = None,
    ) -> float:
        """Calculate position size based on confidence and risk."""
        pos_cfg = model_config["position_sizing"]
        risk_cfg = model_config["risk_management"]

        base = pos_cfg["base_size"]
        conf_mult = pos_cfg["confidence_multiplier"]
        conf_adj = 1 + (probability - model_config["threshold"]) * conf_mult

        dd_factor = max(0, 1 + current_drawdown / risk_cfg["max_drawdown"]) if current_drawdown < 0 else 1.0

        vol_factor = 1.0
        if pos_cfg.get("volatility_adjustment") and market_data is not None:
            if market_data.get("volatility_regime", 0) == 1:
                vol_factor = 0.75
            if abs(market_data.get("trend_direction", 0)) == 1 and market_data.get(
                "volume_trend_confirm", 0
            ) == market_data.get("trend_direction", 0):
                vol_factor *= 1.25
            if market_data.get("trend_exhaustion", 0) == 1:
                vol_factor *= 0.75
            er = market_data.get("efficiency_ratio", 1.0)
            vol_factor *= min(1.25, max(0.75, er))

        size = base * conf_adj * dd_factor * vol_factor
        return min(size, pos_cfg["max_size"])

    def generate_signals(
        self,
        X_test: pd.DataFrame,
        long_probs: np.ndarray,
        short_probs: np.ndarray,
        current_drawdown: float = 0,
    ) -> pd.DataFrame:
        """Generate signals with position sizing and market filters."""
        signals = pd.DataFrame(index=X_test.index)
        signals["long_prob"] = long_probs
        signals["short_prob"] = short_probs

        long_th = self.config.model_config["long_model"]["threshold"]
        short_th = self.config.model_config["short_model"]["threshold"]
        long_min = self.config.model_config["long_model"]["min_probability"]
        short_min = self.config.model_config["short_model"]["min_probability"]

        long_sizes = []
        short_sizes = []
        for idx in X_test.index:
            row = X_test.loc[idx]
            lp = signals.loc[idx, "long_prob"]
            sp = signals.loc[idx, "short_prob"]

            ls = (
                self.calculate_position_size(
                    lp,
                    self.config.model_config["long_model"],
                    current_drawdown,
                    row,
                )
                if lp >= long_th and lp >= long_min
                else 0.0
            )

            ss = (
                self.calculate_position_size(
                    sp,
                    self.config.model_config["short_model"],
                    current_drawdown,
                    row,
                )
                if sp >= short_th and sp >= short_min
                else 0.0
            )

            # Trend filter
            td = row.get("trend_direction", 0)
            if abs(td) == 1:
                if td == 1:
                    ss = 0.0
                else:
                    ls = 0.0

            # Volatility filter
            if row.get("volatility_regime", 0) == 1:
                ls *= 0.75
                ss *= 0.75

            # RSI extremes
            rsi = row.get("rsi_14", 50)
            if rsi > 80:
                ls = 0.0
            elif rsi < 20:
                ss = 0.0

            long_sizes.append(ls)
            short_sizes.append(ss)

        signals["long_size"] = long_sizes
        signals["short_size"] = short_sizes
        signals["position"] = signals["long_size"] - signals["short_size"]
        signals["prediction"] = np.sign(signals["position"])
        return signals


# =========================================================================
#  ResultAnalyzer — plotting and reporting
# =========================================================================


class ResultAnalyzer:
    """Analyze and visualize pipeline results."""

    def __init__(self, output_dir: str = "ml_pipeline_results/plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def analyze_results(self, results: dict, save_plots: bool = True) -> None:
        """Generate analysis plots for pipeline results."""
        import matplotlib.pyplot as plt

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if "feature_importance" in results and not results["feature_importance"].empty:
            plt.figure(figsize=(12, 8))
            importance = results["feature_importance"]
            for i, col in enumerate(importance.columns[:2]):
                plt.subplot(1, 2, i + 1)
                importance.sort_values(col, ascending=True).tail(20)[col].plot(kind="barh")
                plt.title(f"Top 20 Features ({col})")
                plt.tight_layout()
            if save_plots:
                plt.savefig(self.output_dir / f"feature_importance_{timestamp}.png")
            plt.close()

    def get_feature_importance(
        self,
        models: dict[str, xgb.XGBClassifier],
        feature_names: list[str],
    ) -> pd.DataFrame:
        """Extract feature importance from both models."""
        frames = []
        for side, model in models.items():
            if model is not None:
                imp = pd.Series(
                    model.feature_importances_,
                    index=feature_names,
                    name=f"{side}_importance",
                )
                frames.append(imp)
        return pd.concat(frames, axis=1) if frames else pd.DataFrame()


# =========================================================================
#  MLPipeline — orchestrator
# =========================================================================


class MLPipeline:
    """Orchestrates feature engineering, training, and prediction."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models: dict[str, xgb.XGBClassifier | None] = {"long": None, "short": None}
        self.random_state = 42
        np.random.seed(self.random_state)
        random.seed(self.random_state)

        self.feature_engine = FeatureEngine(config)
        self.trainer = ModelTrainer(config, self.random_state)
        self.signal_gen = SignalGenerator(config)
        self.analyzer = ResultAnalyzer()

    # ---- Delegated feature methods ----

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.feature_engine.prepare_features(data)

    def calculate_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.feature_engine.calculate_advanced_features(data)

    # ---- Delegated trainer methods ----

    def prepare_data(self, df, side):
        return self.trainer.prepare_data(df, side)

    def train_model(self, X_train, y_train, side):
        return self.trainer.train_model(X_train, y_train, side)

    def evaluate_model(self, model, X_test, y_test):
        return self.trainer.evaluate_model(model, X_test, y_test)

    def walk_forward_oos(self, df, side, n_splits=5, min_train_frac=0.5):
        return self.trainer.walk_forward_oos(df, side, n_splits, min_train_frac)

    # ---- Delegated signal methods ----

    def calculate_position_size(self, probability, model_config, current_drawdown=0, market_data=None):
        return self.signal_gen.calculate_position_size(probability, model_config, current_drawdown, market_data)

    def generate_signals(self, X_test, long_probs, short_probs, current_drawdown=0):
        return self.signal_gen.generate_signals(X_test, long_probs, short_probs, current_drawdown)

    # ---- High-level methods ----

    def train(self, df: pd.DataFrame) -> dict[str, Any]:
        """Train both long and short models."""
        results = {
            "long_metrics": {},
            "short_metrics": {},
            "feature_importance": pd.DataFrame(),
            "test_predictions": pd.DataFrame(),
        }

        for side in ["long", "short"]:
            self.logger.info("Training %s model...", side)

            X_train, X_test, y_train, y_test = self.trainer.prepare_data(df, side)
            model, train_metrics = self.trainer.train_model(X_train, y_train, side)
            self.models[side] = model

            test_metrics = self.trainer.evaluate_model(model, X_test, y_test)
            results[f"{side}_metrics"] = test_metrics

            importance = self.analyzer.get_feature_importance(
                {side: model},
                self.config.selected_features,
            )
            results["feature_importance"] = pd.concat(
                [results["feature_importance"], importance],
                axis=1,
            )

            pp = self.trainer.preprocessors[side]
            if pp is None:
                raise RuntimeError(f"Preprocessor for {side} not fitted")
            X_all = pp.transform(df[self.config.selected_features])
            proba = model.predict_proba(X_all)[:, 1]
            results["test_predictions"][f"{side}_prob"] = proba

            self.logger.info("%s metrics: %s", side, test_metrics)

        return results

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions for new data."""
        df = self.feature_engine.prepare_features(df)
        predictions = pd.DataFrame(index=df.index)

        for side in ["long", "short"]:
            if self.models[side] is None:
                raise ValueError(f"No trained {side} model available")

            pp = self.trainer.preprocessors[side]
            if pp is None:
                raise RuntimeError(f"Preprocessor for {side} not fitted")

            X = df[self.config.selected_features]
            X_scaled = pp.transform(X)
            proba = self.models[side].predict_proba(X_scaled)[:, 1]

            predictions[f"{side}_probability"] = proba
            th = self.config.threshold
            mc = self.config.min_prediction_confidence
            predictions[f"{side}_signal"] = ((proba > th) & (proba > mc)).astype(int)

        return predictions

    def analyze_results(self, results: dict, save_plots: bool = True) -> None:
        self.analyzer.analyze_results(results, save_plots)

    def get_feature_importance(self, model, side):
        return pd.Series(
            model.feature_importances_,
            index=self.config.selected_features,
            name=f"{side}_importance",
        )

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }

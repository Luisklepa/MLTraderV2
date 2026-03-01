import glob
import json
import logging
import os
from datetime import datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

# === MODO RÁPIDO PARA PRUEBAS BETA ===
FAST_BETA = False  # Cambia a False para entrenamiento completo

if FAST_BETA:
    N_ROWS = 300
    OPTIMIZE_PARAMS = False
    N_FOLDS = 2
    N_ESTIMATORS = 20
    BALANCE_STRATEGY = None
else:
    N_ROWS = None
    OPTIMIZE_PARAMS = True
    N_FOLDS = 5
    N_ESTIMATORS = 200
    BALANCE_STRATEGY = "smote"


class MLTradingTrainer:
    def __init__(
        self,
        dataset_path,
        selected_features=None,
        n_estimators=100,
        n_folds=5,
        balance_strategy="smote",
        metric="f1",
        auto_select: bool = True,
        max_features: int = 60,
        correlation_threshold: float = 0.95,
    ):
        self.dataset_path = dataset_path
        self.selected_features = selected_features
        self.n_estimators = n_estimators
        self.n_folds = n_folds
        self.balance_strategy = balance_strategy
        self.metric = metric
        self.auto_select = auto_select
        self.max_features = max_features
        self.correlation_threshold = correlation_threshold
        self.model_long = None
        self.model_short = None
        self.feature_columns = None
        self.best_threshold_long = 0.5
        self.best_threshold_short = 0.5
        self.best_metric_score_long = None
        self.best_metric_score_short = None
        self.test_predictions_long = None
        self.test_predictions_short = None

    def load_selected_features(self):
        # Busca el archivo de features más reciente generado por ml_feature_selection.py
        files = glob.glob("log_importance_rf_*.csv")
        if files:
            latest = max(files, key=os.path.getctime)
            logger.info("Usando top features de: %s", latest)
            top_features = pd.read_csv(latest, index_col=0).index.tolist()
            return top_features
        else:
            logger.info("No se encontró archivo de top features, usando todos los features.")
            return None

    def load_and_prepare_data(self):
        """
        Carga y prepara los datos para entrenamiento.

        Returns:
            X: Features
            y_long: Target para señales long
            y_short: Target para señales short
        """
        logger.info("Cargando dataset...")
        df = pd.read_csv(self.dataset_path)

        # Seleccionar features
        if self.selected_features is None:
            # Excluir columnas que no son features
            exclude_cols = [
                "datetime",
                "timestamp",
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "target",
                "target_long",
                "target_short",
                "future_return",
                "future_return_long",
                "future_return_short",
                "position",
                "position_long",
                "position_short",
            ]
            self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        else:
            self.feature_columns = self.selected_features

        # Preparar X, y_long, y_short
        X = df[self.feature_columns]
        y_long = df["target_long"]
        y_short = df["target_short"]

        # Eliminar filas con NaN
        mask = ~(X.isna().any(axis=1) | y_long.isna() | y_short.isna())
        X = X[mask]
        y_long = y_long[mask]
        y_short = y_short[mask]

        return X, y_long, y_short

    def create_balanced_dataset(self, X, y, strategy=None):
        """Crea dataset balanceado"""
        if strategy is None:
            logger.info("[FAST_BETA] Sin balanceo de clases para pruebas rápidas.")
            return X, y
        logger.info("Aplicando balanceo de clases: %s", strategy)

        if strategy == "smote":
            # SMOTE para oversampling
            sampler = SMOTE(random_state=42, k_neighbors=5)
            X_balanced, y_balanced = sampler.fit_resample(X, y)

        elif strategy == "undersample":
            # Random undersampling
            sampler = RandomUnderSampler(random_state=42)
            X_balanced, y_balanced = sampler.fit_resample(X, y)

        elif strategy == "combined":
            # Combinación de over y undersampling
            over = SMOTE(sampling_strategy=0.5, random_state=42)
            under = RandomUnderSampler(sampling_strategy=0.8, random_state=42)

            pipeline = ImbPipeline([("over", over), ("under", under)])

            X_balanced, y_balanced = pipeline.fit_resample(X, y)

        else:
            # Sin balanceo
            X_balanced, y_balanced = X, y

        logger.info("Dataset balanceado: %s", pd.Series(y_balanced).value_counts().to_dict())
        return X_balanced, y_balanced

    def split_data_temporal(self, X, y, test_size=0.2):
        """Split temporal para datos de series temporales"""
        split_idx = int(len(X) * (1 - test_size))

        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]

        logger.info("Train set: %d samples", len(X_train))
        logger.info("Test set: %d samples", len(X_test))

        return X_train, X_test, y_train, y_test

    def optimize_hyperparameters(self, X_train, y_train):
        """Optimización de hiperparámetros con GridSearch"""
        logger.info("Optimizando hiperparámetros...")
        param_grid = {
            "n_estimators": [self.n_estimators],
            "max_depth": [10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt"],
            "class_weight": ["balanced"],
        }
        tscv = TimeSeriesSplit(n_splits=self.n_folds)
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(rf, param_grid, cv=tscv, scoring="roc_auc", n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        logger.info("Mejores parámetros: %s", grid_search.best_params_)
        logger.info("Mejor score: %.4f", grid_search.best_score_)
        return grid_search.best_estimator_

    def train_model(self, X=None, y_long=None, y_short=None):
        """
        Entrena los modelos long y short.

        Args:
            X: Features (opcional, se carga del dataset si no se proporciona)
            y_long: Target long (opcional)
            y_short: Target short (opcional)

        Returns:
            model_long: Modelo entrenado para señales long
            model_short: Modelo entrenado para señales short
        """
        logger.info("=== INICIANDO ENTRENAMIENTO DE MODELOS LONG/SHORT ===")

        if X is None or y_long is None or y_short is None:
            X, y_long, y_short = self.load_and_prepare_data()

        # Automated feature selection (correlation + importance-based)
        if self.auto_select and len(X.columns) > self.max_features:
            from ml.feature_selection import auto_select_features

            logger.info("Running auto_select_features on %d features...", len(X.columns))
            selected = auto_select_features(
                X,
                list(X.columns),
                model=None,
                correlation_threshold=self.correlation_threshold,
                max_features=self.max_features,
            )
            X = X[selected]
            self.feature_columns = selected
            logger.info("Reduced to %d features after auto-selection", len(selected))

        logger.info("Entrenando modelo LONG...")
        self.model_long = self._train_single_model(X, y_long, "LONG")

        # Entrenar modelo short
        logger.info("Entrenando modelo SHORT...")
        self.model_short = self._train_single_model(X, y_short, "SHORT")

        return self.model_long, self.model_short

    def _train_single_model(self, X: pd.DataFrame, y: pd.Series, side: str) -> Any:
        """
        Entrena un modelo individual (long o short).

        Args:
            X: Features
            y: Target
            side: 'LONG' o 'SHORT'

        Returns:
            Modelo entrenado
        """
        logger.info(f"\nEntrenando modelo {side}...")

        # Asegurar que las etiquetas sean enteros
        y = y.astype(int)

        # Inicializar modelo
        model = XGBClassifier(n_estimators=self.n_estimators, random_state=42, eval_metric="logloss")

        # Temporal cross-validation (preserves time ordering)
        n_folds = self.n_folds
        kf = TimeSeriesSplit(n_splits=n_folds)

        logger.info(f"\nIniciando cross-validation para modelo {side}...")

        best_f1 = 0
        best_threshold = 0

        # Cross-validation
        for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            if len(y_train.unique()) < 2:
                logger.warning("Fold %d: only one class in training set, skipping", fold)
                continue

            if self.balance_strategy == "smote":
                minority_count = int((y_train == 1).sum())
                k = min(5, minority_count - 1)
                if k >= 1:
                    smote = SMOTE(random_state=42, k_neighbors=k)
                    X_train, y_train = smote.fit_resample(X_train, y_train)
                else:
                    logger.warning("Fold %d: minority class has %d samples, skipping SMOTE", fold, minority_count)

            model.fit(X_train, y_train)

            # Predecir probabilidades
            y_proba = model.predict_proba(X_test)[:, 1]

            # Buscar mejor threshold
            thresholds = np.linspace(0.1, 0.9, 9)
            for threshold in thresholds:
                y_pred = (y_proba >= threshold).astype(int)
                score = f1_score(y_test, y_pred)

                if score > best_f1:
                    best_f1 = score
                    best_threshold = threshold

            logger.info(f"Fold {fold}/{n_folds} - Mejor F1: {best_f1:.4f} (threshold: {best_threshold:.2f})")

        logger.info(f"\nEntrenamiento {side} completado:")
        logger.info(f"Mejor F1: {best_f1:.4f}")
        logger.info(f"Mejor threshold: {best_threshold:.2f}\n")

        # Persist best threshold for this side
        if side == "LONG":
            self.best_threshold_long = best_threshold
            self.best_metric_score_long = best_f1
        else:
            self.best_threshold_short = best_threshold
            self.best_metric_score_short = best_f1

        # Train final model on full dataset
        if self.balance_strategy == "smote":
            minority_count = int((y == 1).sum())
            k = min(5, minority_count - 1)
            if k >= 1:
                smote = SMOTE(random_state=42, k_neighbors=k)
                X_resampled, y_resampled = smote.fit_resample(X, y)
                model.fit(X_resampled, y_resampled)
            else:
                logger.warning(f"{side}: minority class too small ({minority_count}), training without SMOTE")
                model.fit(X, y)
        else:
            model.fit(X, y)

        return model

    def evaluate_model(self):
        """Evalúa los modelos long y short"""
        logger.info("=== EVALUACIÓN MODELO LONG ===")
        logger.info(
            "\n%s", classification_report(self.test_predictions_long["y_true"], self.test_predictions_long["y_pred"])
        )

        logger.info("=== EVALUACIÓN MODELO SHORT ===")
        logger.info(
            "\n%s", classification_report(self.test_predictions_short["y_true"], self.test_predictions_short["y_pred"])
        )

    def analyze_feature_importance(self):
        """Analiza la importancia de features para ambos modelos"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Importancia features modelo LONG
        importance_long = pd.DataFrame(
            {"feature": self.feature_columns, "importance": self.model_long.feature_importances_}
        ).sort_values("importance", ascending=False)

        importance_long.to_csv(f"models/feature_importance_long_{timestamp}.csv", index=False)
        logger.info("Top 10 features más importantes (LONG):\n%s", importance_long.head(10))

        # Importancia features modelo SHORT
        importance_short = pd.DataFrame(
            {"feature": self.feature_columns, "importance": self.model_short.feature_importances_}
        ).sort_values("importance", ascending=False)

        importance_short.to_csv(f"models/feature_importance_short_{timestamp}.csv", index=False)
        logger.info("Top 10 features más importantes (SHORT):\n%s", importance_short.head(10))

    def simulate_trading_performance(self):
        """Simula performance de trading con ambos modelos"""
        # Implementar simulación de trading considerando señales long y short
        pass  # TODO: Implementar simulación combinada

    def save_model(self):
        """Guarda los modelos long y short"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path_long = f"models/trading_model_long_{timestamp}.pkl"
        model_path_short = f"models/trading_model_short_{timestamp}.pkl"

        joblib.dump(self.model_long, model_path_long)
        joblib.dump(self.model_short, model_path_short)

        logger.info("Modelo LONG guardado en: %s", model_path_long)
        logger.info("Modelo SHORT guardado en: %s", model_path_short)

        # Guardar thresholds y métricas
        threshold_path = f"models/thresholds_{timestamp}.json"
        threshold_data = {
            "long": {"threshold": self.best_threshold_long, "metric_score": self.best_metric_score_long},
            "short": {"threshold": self.best_threshold_short, "metric_score": self.best_metric_score_short},
        }
        with open(threshold_path, "w") as f:
            json.dump(threshold_data, f, indent=4)
        logger.info("Thresholds guardados en: %s", threshold_path)


# Ejecutar entrenamiento
if __name__ == "__main__":
    from core.logging_config import setup_logging

    setup_logging()
    trainer = MLTradingTrainer(
        dataset_path="btcusdt_ml_dataset.csv",
        balance_strategy="smote",
    )
    model_long, model_short = trainer.train_model()
    logger.info(
        "Entrenamiento completado! Thresholds: LONG=%s, SHORT=%s",
        trainer.best_threshold_long,
        trainer.best_threshold_short,
    )

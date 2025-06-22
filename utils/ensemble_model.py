import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib
import logging
import shap
from datetime import datetime

logger = logging.getLogger(__name__)

class EnsembleModel:
    """Ensemble model combining XGBoost, LightGBM, and CatBoost with dynamic weighting."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the ensemble model with configuration."""
        self.config = config or {}
        
        # Model parameters
        self.xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.01,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'tree_method': 'hist',
            'random_state': 42
        }
        
        self.lgb_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'max_depth': 6,
            'learning_rate': 0.01,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_samples': 20,
            'random_state': 42
        }
        
        self.cat_params = {
            'objective': 'Logloss',
            'eval_metric': 'Logloss',
            'depth': 6,
            'learning_rate': 0.01,
            'subsample': 0.8,
            'colsample_bylevel': 0.8,
            'min_data_in_leaf': 20,
            'random_state': 42
        }
        
        # Model weights (will be updated dynamically)
        self.model_weights = {
            'xgb': 1/3,
            'lgb': 1/3,
            'cat': 1/3
        }
        
        # Performance history
        self.performance_history = []
        
        # Initialize models
        self.models = {
            'xgb': None,
            'lgb': None,
            'cat': None
        }
        
        # Feature importance
        self.feature_importance = {}
        
        # SHAP values
        self.shap_values = {}
    
    def prepare_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Prepare data for training and testing."""
        # Use time series split
        n_samples = len(X)
        train_size = int(n_samples * (1 - test_size))
        
        X_train = X.iloc[:train_size]
        X_test = X.iloc[train_size:]
        y_train = y.iloc[:train_size]
        y_test = y.iloc[train_size:]
        
        return X_train, X_test, y_train, y_test
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> None:
        """Train all models in the ensemble."""
        logger.info("Training ensemble models...")
        
        # Prepare validation data if not provided
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = self.prepare_data(X_train, y_train)
        
        # Train XGBoost
        logger.info("Training XGBoost...")
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        self.models['xgb'] = xgb.train(
            self.xgb_params,
            dtrain,
            num_boost_round=1000,
            evals=[(dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        # Train LightGBM
        logger.info("Training LightGBM...")
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)
        self.models['lgb'] = lgb.train(
            self.lgb_params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        # Train CatBoost
        logger.info("Training CatBoost...")
        self.models['cat'] = cb.CatBoostClassifier(**self.cat_params)
        self.models['cat'].fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Calculate feature importance
        self._calculate_feature_importance(X_train)
        
        # Calculate SHAP values
        self._calculate_shap_values(X_train)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate weighted ensemble predictions."""
        predictions = {}
        
        # XGBoost predictions
        dmatrix = xgb.DMatrix(X)
        predictions['xgb'] = self.models['xgb'].predict(dmatrix)
        
        # LightGBM predictions
        predictions['lgb'] = self.models['lgb'].predict(X)
        
        # CatBoost predictions
        predictions['cat'] = self.models['cat'].predict_proba(X)[:, 1]
        
        # Weighted ensemble prediction
        weighted_pred = sum(
            pred * self.model_weights[model]
            for model, pred in predictions.items()
        )
        
        return weighted_pred
    
    def update_weights(self, X_recent: pd.DataFrame, y_recent: pd.Series) -> None:
        """Update model weights based on recent performance."""
        performance = {}
        
        # Calculate performance for each model
        for model_name in self.models.keys():
            if model_name == 'xgb':
                dmatrix = xgb.DMatrix(X_recent)
                pred = self.models[model_name].predict(dmatrix)
            elif model_name == 'lgb':
                pred = self.models[model_name].predict(X_recent)
            else:  # catboost
                pred = self.models[model_name].predict_proba(X_recent)[:, 1]
            
            # Calculate F1 score
            pred_binary = (pred > 0.5).astype(int)
            f1 = f1_score(y_recent, pred_binary)
            performance[model_name] = f1
        
        # Update weights based on relative performance
        total_score = sum(performance.values())
        if total_score > 0:
            self.model_weights = {
                model: score / total_score
                for model, score in performance.items()
            }
        
        # Store performance history
        self.performance_history.append({
            'timestamp': datetime.now(),
            'performance': performance,
            'weights': self.model_weights.copy()
        })
    
    def _calculate_feature_importance(self, X: pd.DataFrame) -> None:
        """Calculate and store feature importance for each model."""
        feature_names = X.columns
        
        # XGBoost feature importance
        self.feature_importance['xgb'] = pd.Series(
            self.models['xgb'].get_score(importance_type='gain'),
            index=feature_names
        ).sort_values(ascending=False)
        
        # LightGBM feature importance
        self.feature_importance['lgb'] = pd.Series(
            self.models['lgb'].feature_importance(importance_type='gain'),
            index=feature_names
        ).sort_values(ascending=False)
        
        # CatBoost feature importance
        self.feature_importance['cat'] = pd.Series(
            self.models['cat'].get_feature_importance(),
            index=feature_names
        ).sort_values(ascending=False)
        
        # Ensemble feature importance (weighted average)
        self.feature_importance['ensemble'] = pd.DataFrame(self.feature_importance).apply(
            lambda x: np.average(x, weights=list(self.model_weights.values())),
            axis=1
        ).sort_values(ascending=False)
    
    def _calculate_shap_values(self, X: pd.DataFrame) -> None:
        """Calculate and store SHAP values for each model."""
        # XGBoost SHAP values
        explainer = shap.TreeExplainer(self.models['xgb'])
        self.shap_values['xgb'] = explainer.shap_values(X)
        
        # LightGBM SHAP values
        explainer = shap.TreeExplainer(self.models['lgb'])
        self.shap_values['lgb'] = explainer.shap_values(X)
        
        # CatBoost SHAP values
        explainer = shap.TreeExplainer(self.models['cat'])
        self.shap_values['cat'] = explainer.shap_values(X)
    
    def save_models(self, path_prefix: str) -> None:
        """Save all models and their configurations."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save individual models
        joblib.dump(self.models['xgb'], f'{path_prefix}_xgb_{timestamp}.pkl')
        joblib.dump(self.models['lgb'], f'{path_prefix}_lgb_{timestamp}.pkl')
        joblib.dump(self.models['cat'], f'{path_prefix}_cat_{timestamp}.pkl')
        
        # Save model weights and performance history
        metadata = {
            'weights': self.model_weights,
            'performance_history': self.performance_history,
            'feature_importance': {
                k: v.to_dict() for k, v in self.feature_importance.items()
            }
        }
        joblib.dump(metadata, f'{path_prefix}_metadata_{timestamp}.pkl')
    
    def load_models(self, path_prefix: str, timestamp: str) -> None:
        """Load all models and their configurations."""
        # Load individual models
        self.models['xgb'] = joblib.load(f'{path_prefix}_xgb_{timestamp}.pkl')
        self.models['lgb'] = joblib.load(f'{path_prefix}_lgb_{timestamp}.pkl')
        self.models['cat'] = joblib.load(f'{path_prefix}_cat_{timestamp}.pkl')
        
        # Load metadata
        metadata = joblib.load(f'{path_prefix}_metadata_{timestamp}.pkl')
        self.model_weights = metadata['weights']
        self.performance_history = metadata['performance_history']
        self.feature_importance = {
            k: pd.Series(v) for k, v in metadata['feature_importance'].items()
        } 
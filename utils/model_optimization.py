"""
Model optimization utilities for the ML trading system.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import joblib
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class ModelOptimizer:
    """Handles model optimization and maintenance."""
    
    def __init__(self, config: Dict):
        """Initialize optimizer with configuration."""
        self.config = config
        self.feature_importance_threshold = config.get('min_feature_importance', 0.01)
        self.max_features = config.get('max_features', 50)
        self.n_trials = config.get('optimization_trials', 100)
        self.cv_folds = config.get('cv_folds', 5)
        self.model_dir = Path(config.get('model_dir', 'models'))
        self.model_dir.mkdir(exist_ok=True)
    
    def optimize_feature_selection(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, List[str]]:
        """Select most important features using model-based selection."""
        selector = SelectFromModel(
            estimator=xgb.XGBClassifier(n_estimators=100),
            threshold=self.feature_importance_threshold,
            max_features=self.max_features,
            prefit=False
        )
        
        selector.fit(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        logger.info(f"Selected {len(selected_features)} features")
        return X[selected_features], selected_features
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Optimize model hyperparameters using Optuna."""
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'gamma': trial.suggest_float('gamma', 0, 1),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1)
            }
            
            model = xgb.XGBClassifier(**params)
            scores = cross_val_score(
                model, X, y,
                cv=self.cv_folds,
                scoring='f1',
                n_jobs=-1
            )
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        
        logger.info(f"Best trial: score={study.best_value:.4f}")
        for key, value in study.best_params.items():
            logger.info(f"    {key}: {value}")
        
        return study.best_params
    
    def evaluate_model_drift(
        self,
        model: xgb.XGBClassifier,
        X_recent: pd.DataFrame,
        y_recent: pd.Series,
        threshold: float = 0.1
    ) -> bool:
        """Evaluate if model performance has drifted significantly."""
        y_pred = model.predict_proba(X_recent)[:, 1]
        recent_auc = roc_auc_score(y_recent, y_pred)
        recent_f1 = f1_score(y_recent, y_pred > 0.5)
        
        # Load historical metrics
        metrics_file = self.model_dir / 'historical_metrics.json'
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                historical_metrics = json.load(f)
            historical_f1 = historical_metrics['f1'][-1]
            
            # Check for significant degradation
            if (historical_f1 - recent_f1) > threshold:
                logger.warning(f"Model performance degraded: {historical_f1:.4f} -> {recent_f1:.4f}")
                return True
        
        # Save current metrics
        self.save_metrics({
            'timestamp': datetime.now().isoformat(),
            'auc': recent_auc,
            'f1': recent_f1
        })
        
        return False
    
    def save_metrics(self, metrics: Dict) -> None:
        """Save model metrics to historical record."""
        metrics_file = self.model_dir / 'historical_metrics.json'
        
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                historical_metrics = json.load(f)
        else:
            historical_metrics = {
                'timestamp': [],
                'auc': [],
                'f1': []
            }
        
        # Update metrics
        historical_metrics['timestamp'].append(metrics['timestamp'])
        historical_metrics['auc'].append(metrics['auc'])
        historical_metrics['f1'].append(metrics['f1'])
        
        # Save updated metrics
        with open(metrics_file, 'w') as f:
            json.dump(historical_metrics, f, indent=4)
    
    def save_model(self, model: xgb.XGBClassifier, features: List[str], metrics: Dict) -> None:
        """Save model and its metadata."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save model
        model_path = self.model_dir / f'model_{timestamp}.pkl'
        joblib.dump(model, model_path)
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'features': features,
            'metrics': metrics,
            'model_params': model.get_params()
        }
        
        metadata_path = self.model_dir / f'metadata_{timestamp}.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Metadata saved to {metadata_path}")
    
    def load_latest_model(self) -> Tuple[xgb.XGBClassifier, List[str], Dict]:
        """Load the latest model and its metadata."""
        model_files = list(self.model_dir.glob('model_*.pkl'))
        if not model_files:
            raise FileNotFoundError("No saved models found")
        
        latest_model_file = max(model_files, key=lambda x: x.stat().st_mtime)
        timestamp = latest_model_file.stem.split('_')[1]
        
        # Load model
        model = joblib.load(latest_model_file)
        
        # Load metadata
        metadata_file = self.model_dir / f'metadata_{timestamp}.json'
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        return model, metadata['features'], metadata['metrics']
    
    def analyze_feature_importance(self, model: xgb.XGBClassifier, feature_names: List[str]) -> pd.DataFrame:
        """Analyze and return feature importance."""
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        })
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df 
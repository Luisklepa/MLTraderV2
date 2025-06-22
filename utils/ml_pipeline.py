import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, List, Tuple, Any
from pathlib import Path
import logging
import yaml
from dataclasses import dataclass
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.impute import SimpleImputer
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import talib
import pywt
from scipy import stats
from scipy.signal import find_peaks
from statsmodels.tsa.stattools import adfuller
from functools import lru_cache
import numpy.lib.stride_tricks as stride_tricks

from .technical_indicators import add_all_indicators
from .statistical_features import add_all_statistical_features

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for a single model."""
    n_estimators: int = 100
    max_depth: int = 3
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 1
    gamma: float = 0
    scale_pos_weight: float = 1
    
    def to_dict(self) -> Dict:
        """Convert ModelConfig to dictionary."""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'min_child_weight': self.min_child_weight,
            'gamma': self.gamma,
            'scale_pos_weight': self.scale_pos_weight
        }

class PipelineConfig:
    """Configuration class for ML pipeline."""
    
    def __init__(self, config_path: str):
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("[DEBUG] Contenido del YAML cargado:")
        print(config)
        self.model_config = config['model_config']
        self.features_config = config['features_config']
        self.data_config = config['data_config']
        
    @property
    def long_params(self) -> Dict[str, Any]:
        """Get long model parameters."""
        return self.model_config['ensemble']['models'][0]['params']
        
    @property
    def short_params(self) -> Dict[str, Any]:
        """Get short model parameters."""
        return self.model_config['ensemble']['models'][0]['params']
        
    @property
    def selected_features(self) -> list:
        """Get list of selected features."""
        features = []
        
        # Add technical indicators
        for indicator in self.features_config['technical_indicators']:
            features.append(indicator['name'])
            
        # Add statistical features
        features.extend([
            'returns_1',
            'returns_5',
            'returns_10',
            'volatility_10',
            'volatility_20',
            'zscore_close_20',
            'zscore_volume_20'
        ])
        
        # Add temporal features
        features.extend([
            'hour',
            'weekday',
            'month',
            'is_weekend',
            'session_progress'
        ])
        
        return features
        
    @property
    def use_smote(self) -> bool:
        """Get SMOTE configuration."""
        return True
        
    @property
    def threshold(self) -> float:
        """Get prediction threshold."""
        return self.model_config['prediction']['threshold']
        
    @property
    def min_prediction_confidence(self) -> float:
        """Get minimum prediction confidence."""
        return self.model_config['prediction']['min_prediction_confidence']

class MLPipeline:
    """ML pipeline for training and prediction."""
    
    def __init__(self, config: PipelineConfig):
        """Initialize pipeline with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models = {'long': None, 'short': None}
        self.scalers = {'long': StandardScaler(), 'short': StandardScaler()}
        
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for prediction."""
        df = data.copy()
        
        # Add technical indicators
        for indicator in self.config.features_config['technical_indicators']:
            name = indicator['name']
            params = indicator.get('params', {})
            
            if name == 'rsi':
                df[f'rsi_{params["timeperiod"]}'] = talib.RSI(df['close'], timeperiod=params['timeperiod'])
            elif name == 'macd':
                macd, signal, hist = talib.MACD(
                    df['close'],
                    fastperiod=params['fastperiod'],
                    slowperiod=params['slowperiod'],
                    signalperiod=params['signalperiod']
                )
                df['macd'] = macd
                df['macd_signal'] = signal
                df['macd_hist'] = hist
            elif name == 'bbands':
                upper, middle, lower = talib.BBANDS(
                    df['close'],
                    timeperiod=params['timeperiod'],
                    nbdevup=params['nbdevup'],
                    nbdevdn=params['nbdevdn']
                )
                df['bb_upper'] = upper
                df['bb_middle'] = middle
                df['bb_lower'] = lower
            elif name == 'atr':
                df[f'atr_{params["timeperiod"]}'] = talib.ATR(
                    df['high'],
                    df['low'],
                    df['close'],
                    timeperiod=params['timeperiod']
                )
            elif name == 'stoch':
                slowk, slowd = talib.STOCH(
                    df['high'],
                    df['low'],
                    df['close'],
                    fastk_period=params['fastk_period'],
                    slowk_period=params['slowk_period'],
                    slowd_period=params['slowd_period']
                )
                df['stoch_k'] = slowk
                df['stoch_d'] = slowd
        
        # Add statistical features
        df['returns_1'] = df['close'].pct_change()
        df['returns_5'] = df['close'].pct_change(5)
        df['returns_10'] = df['close'].pct_change(10)
        
        df['volatility_10'] = df['returns_1'].rolling(10).std()
        df['volatility_20'] = df['returns_1'].rolling(20).std()
        
        df['zscore_close_20'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
        df['zscore_volume_20'] = (df['volume'] - df['volume'].rolling(20).mean()) / df['volume'].rolling(20).std()
        
        # Add temporal features
        self.add_temporal_features(df)
        
        return df
        
    def add_temporal_features(self, df: pd.DataFrame) -> None:
        """Add temporal features to the dataframe."""
        # Print index type and sample
        print("Index type:", type(df.index))
        print("Sample index:")
        print(df.index[:5])
        
        print("\nExtracting temporal features...")
        
        # Extract temporal features
        df['hour'] = df.index.hour
        print("Added hour")
        
        df['weekday'] = df.index.weekday
        print("Added weekday")
        
        df['month'] = df.index.month
        print("Added month")
        
        df['is_weekend'] = df.index.weekday.isin([5, 6]).astype(int)
        print("Added is_weekend")
        
        # Calculate session progress (0 to 1)
        df['session_progress'] = (df['hour'] * 60 + df.index.minute) / (24 * 60)
        print("Added session_progress")
        
        # Print sample of temporal features
        print("\nTemporal features added:")
        temporal_features = ['hour', 'weekday', 'month', 'is_weekend', 'session_progress']
        print(df[temporal_features].head())
        
        # Print final columns
        print("\nAfter temporal features:")
        print(df.columns)
        
    def prepare_data(
        self,
        df: pd.DataFrame,
        side: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training."""
        # Select features and target
        X = df[self.config.selected_features]
        y = df[f'{side}_target']
        
        # Split data (estratificado)
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42
            )
        except ValueError as e:
            print(f"[ADVERTENCIA] No se pudo estratificar el split: {e}. Usando split sin stratify.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=True, random_state=42
            )

        # Advertencia si alguna clase no está presente
        for clase in [0, 1]:
            if clase not in y_train.values:
                print(f"[ADVERTENCIA] La clase {clase} no está presente en el set de entrenamiento para {side}.")
            if clase not in y_test.values:
                print(f"[ADVERTENCIA] La clase {clase} no está presente en el set de test para {side}.")
        
        # Impute missing values
        imputer = SimpleImputer(strategy='mean')
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)
        
        # Scale features
        X_train = self.scalers[side].fit_transform(X_train)
        X_test = self.scalers[side].transform(X_test)
        
        # Handle class imbalance
        if self.config.use_smote:
            # Solo aplicar SMOTE si hay al menos 2 muestras de la clase minoritaria
            if sum(y_train == 1) > 1 and sum(y_train == 0) > 1:
                smote = SMOTE(random_state=42)
                X_train, y_train = smote.fit_resample(X_train, y_train)
            else:
                print(f"[ADVERTENCIA] No se puede aplicar SMOTE para {side}: muy pocos ejemplos de la clase minoritaria.")
        
        return X_train, X_test, y_train, y_test
        
    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        side: str
    ) -> Tuple[xgb.XGBClassifier, Dict[str, float]]:
        """Train model and return metrics."""
        # Get model parameters
        params = (
            self.config.long_params if side == 'long'
            else self.config.short_params
        )
        
        # Create and train model
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train)],
            verbose=False
        )
        
        # Calculate metrics
        train_pred = model.predict(X_train)
        train_prob = model.predict_proba(X_train)[:, 1]
        
        metrics = {
            'accuracy': (train_pred == y_train).mean(),
            'precision': (train_pred & y_train).sum() / train_pred.sum(),
            'recall': (train_pred & y_train).sum() / y_train.sum(),
            'f1': 2 * (train_pred & y_train).sum() / (train_pred.sum() + y_train.sum()),
            'roc_auc': np.nan  # Placeholder for ROC-AUC
        }
        
        return model, metrics
        
    def evaluate_model(
        self,
        model: xgb.XGBClassifier,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate model on test set."""
        test_pred = model.predict(X_test)
        test_prob = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': (test_pred == y_test).mean(),
            'precision': (test_pred & y_test).sum() / max(test_pred.sum(), 1),
            'recall': (test_pred & y_test).sum() / max(y_test.sum(), 1),
            'f1': 2 * (test_pred & y_test).sum() / (test_pred.sum() + y_test.sum()),
            'roc_auc': np.nan  # Placeholder for ROC-AUC
        }
        
        return metrics
        
    def get_feature_importance(
        self,
        model: xgb.XGBClassifier,
        side: str
    ) -> pd.Series:
        """Get feature importance scores."""
        importance = model.feature_importances_
        return pd.Series(
            importance,
            index=self.config.selected_features,
            name=f'{side}_importance'
        )
        
    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train both long and short models."""
        results = {
            'long_metrics': {},
            'short_metrics': {},
            'feature_importance': pd.DataFrame(),
            'test_predictions': pd.DataFrame()
        }
        
        for side in ['long', 'short']:
            self.logger.info(f"\nTraining {side} model...")
            
            # Prepare data
            X_train, X_test, y_train, y_test = self.prepare_data(df, side)
            
            # Train model
            model, train_metrics = self.train_model(X_train, y_train, side)
            self.models[side] = model
            
            # Evaluate model
            test_metrics = self.evaluate_model(model, X_test, y_test)
            
            # Store results
            results[f'{side}_metrics'] = test_metrics
            
            # Get feature importance
            importance = self.get_feature_importance(model, side)
            results['feature_importance'] = pd.concat(
                [results['feature_importance'], importance],
                axis=1
            )
            
            # Generate predictions
            X_scaled = self.scalers[side].transform(df[self.config.selected_features])
            predictions = model.predict_proba(X_scaled)[:, 1]
            results['test_predictions'][f'{side}_prob'] = predictions
            
            # Log metrics
            self.logger.info(f"\n{side.capitalize()} Model Metrics:")
            for metric, value in test_metrics.items():
                self.logger.info(f"{metric}: {value:.4f}")
        
        return results
        
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions for new data."""
        # Prepare features
        df = self.prepare_features(df)
        
        predictions = pd.DataFrame(index=df.index)
        
        for side in ['long', 'short']:
            if self.models[side] is None:
                raise ValueError(f"No trained {side} model available")
            
            # Scale features
            X = df[self.config.selected_features]
            X_scaled = self.scalers[side].transform(X)
            
            # Get probabilities
            probs = self.models[side].predict_proba(X_scaled)[:, 1]
            predictions[f'{side}_probability'] = probs
            
            # Apply threshold
            threshold = self.config.threshold
            min_confidence = self.config.min_prediction_confidence
            
            predictions[f'{side}_signal'] = (
                (probs > threshold) &
                (probs > min_confidence)
            ).astype(int)
        
        return predictions

    def _calculate_ema(self, prices: tuple, period: int) -> np.ndarray:
        """Calculate EMA using TA-Lib."""
        return talib.EMA(np.array(prices, dtype=np.float64), timeperiod=period)
    
    def _rolling_window(self, a: np.ndarray, window: int) -> np.ndarray:
        """Create rolling window views of array."""
        shape = (a.shape[0] - window + 1, window)
        strides = (a.strides[0], a.strides[0])
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    
    def calculate_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced technical features with optimizations."""
        df = data.copy()
        
        # Ensure data types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert to numpy arrays with correct type
        close_array = np.asarray(df['close'].values, dtype=np.float64)
        high_array = np.asarray(df['high'].values, dtype=np.float64)
        low_array = np.asarray(df['low'].values, dtype=np.float64)
        volume_array = np.asarray(df['volume'].values, dtype=np.float64)
        open_array = np.asarray(df['open'].values, dtype=np.float64)
        
        # Cache key for this data
        cache_key = hash(close_array.tobytes())
        if cache_key in self._feature_cache:
            # Merge cached features with existing features
            cached_features = self._feature_cache[cache_key]
            for col in cached_features.columns:
                if col not in df.columns:
                    df[col] = cached_features[col]
            return df
        
        # Calculate EMAs if not present
        ema_periods = [20, 50, 200]
        for period in ema_periods:
            col_name = f'ema_{period}'
            if col_name not in df.columns:
                df[col_name] = self._calculate_ema(tuple(close_array), period)
                ratio_col = f'price_ema_{period}_ratio'
                if ratio_col not in df.columns:
                    df[ratio_col] = close_array / df[col_name].to_numpy()
        
        # Calculate trend strength and direction if not present
        if 'trend_strength' not in df.columns:
            if 'price_ema_20_ratio' in df.columns:
                df['trend_strength'] = np.abs(df['price_ema_20_ratio'] - 1) * 100
        if 'trend_direction' not in df.columns:
            if 'price_ema_20_ratio' in df.columns:
                df['trend_direction'] = np.sign(df['price_ema_20_ratio'] - 1)
        
        # Calculate volatility features if not present
        if 'volatility_regime' not in df.columns or 'volatility_ratio' not in df.columns:
            returns = np.diff(np.log(close_array))
            returns = np.insert(returns, 0, 0)  # Add 0 at the beginning to maintain length
            vol_window = 20
            volatility = np.std(self._rolling_window(returns, vol_window), axis=1)
            volatility = np.pad(volatility, (vol_window-1, 0), mode='edge')
            
            vol_sma = np.convolve(volatility, np.ones(100)/100, mode='valid')
            vol_sma = np.pad(vol_sma, (99, 0), mode='edge')
            
            if 'volatility_regime' not in df.columns:
                df['volatility_regime'] = np.where(volatility > vol_sma, 1, -1)
            if 'volatility_ratio' not in df.columns:
                df['volatility_ratio'] = volatility / vol_sma
        
        # Calculate volume ratio if not present
        if 'volume_ratio' not in df.columns:
            volume_sma = np.convolve(volume_array, np.ones(20)/20, mode='valid')
            volume_sma = np.pad(volume_sma, (19, 0), mode='edge')
            df['volume_ratio'] = volume_array / volume_sma
        
        # Calculate pattern recognition if not present
        pattern_funcs = [
            talib.CDL3BLACKCROWS, talib.CDL3WHITESOLDIERS,
                           talib.CDLENGULFING, talib.CDLHARAMI, 
            talib.CDLMORNINGSTAR, talib.CDLEVENINGSTAR
        ]
        
        for pattern_func in pattern_funcs:
            pattern_name = pattern_func.__name__.replace('CDL', '').lower()
            pattern_col = f'pattern_{pattern_name}'
            if pattern_col not in df.columns:
                try:
                    df[pattern_col] = pattern_func(
                        open_array, high_array, low_array, close_array
                    )
                except Exception as e:
                    print(f"Error calculating pattern {pattern_name}: {str(e)}")
                    df[pattern_col] = np.zeros_like(close_array)
        
        # Cache the new features
        new_features = df[[col for col in df.columns if col not in data.columns]]
        if not new_features.empty:
            self._feature_cache[cache_key] = new_features
        
        return df
    
    def calculate_position_size(
        self,
        probability: float,
        model_config: Dict,
        current_drawdown: float = 0,
        market_data: Optional[pd.Series] = None
    ) -> float:
        """Calculate position size based on prediction confidence and risk management."""
        position_config = model_config['position_sizing']
        risk_config = model_config['risk_management']
        
        # Base position size
        base_size = position_config['base_size']
        
        # Adjust for prediction confidence
        confidence_multiplier = position_config['confidence_multiplier']
        confidence_adjustment = 1 + (probability - model_config['threshold']) * confidence_multiplier
        
        # Adjust for drawdown
        if current_drawdown < 0:
            drawdown_factor = max(0, 1 + current_drawdown / risk_config['max_drawdown'])
        else:
            drawdown_factor = 1
        
        # Volatility adjustment if enabled and market data available
        volatility_factor = 1.0
        if position_config.get('volatility_adjustment', False) and market_data is not None:
            # Reduce position size in high volatility environments
            if market_data['volatility_regime'] == 1:  # High volatility
                volatility_factor = 0.75
            # Increase position size in trending markets with confirmation
            if (abs(market_data['trend_direction']) == 1 and 
                market_data['volume_trend_confirm'] == market_data['trend_direction']):
                volatility_factor *= 1.25
            # Reduce size in trend exhaustion
            if market_data['trend_exhaustion'] == 1:
                volatility_factor *= 0.75
            # Adjust for market efficiency
            volatility_factor *= min(1.25, max(0.75, market_data['efficiency_ratio']))
        
        # Calculate final position size
        position_size = base_size * confidence_adjustment * drawdown_factor * volatility_factor
        
        # Cap at maximum size
        return min(position_size, position_config['max_size'])
    
    def generate_signals(
        self,
        X_test: pd.DataFrame,
        long_probs: np.ndarray,
        short_probs: np.ndarray,
        current_drawdown: float = 0
    ) -> pd.DataFrame:
        """Generate trading signals with position sizing."""
        signals = pd.DataFrame(index=X_test.index)
        
        # Get probabilities and thresholds
        long_threshold = self.config.model_config['long_model']['threshold']
        short_threshold = self.config.model_config['short_model']['threshold']
        
        # Generate signals
        signals['long_prob'] = long_probs
        signals['short_prob'] = short_probs
        
        # Apply minimum probability filter
        long_min_prob = self.config.model_config['long_model']['min_probability']
        short_min_prob = self.config.model_config['short_model']['min_probability']
        
        # Calculate position sizes with market context
        signals['long_size'] = signals.apply(
            lambda row: self.calculate_position_size(
                row['long_prob'],
                self.config.model_config['long_model'],
                current_drawdown,
                X_test.loc[row.name]
            ) if row['long_prob'] >= long_threshold and row['long_prob'] >= long_min_prob else 0,
            axis=1
        )
        
        signals['short_size'] = signals.apply(
            lambda row: self.calculate_position_size(
                row['short_prob'],
                self.config.model_config['short_model'],
                current_drawdown,
                X_test.loc[row.name]
            ) if row['short_prob'] >= short_threshold and row['short_prob'] >= short_min_prob else 0,
            axis=1
        )
        
        # Apply additional filters based on market conditions
        for idx in signals.index:
            market_data = X_test.loc[idx]
            
            # Don't trade against strong trends
            if abs(market_data['trend_direction']) == 1:
                if market_data['trend_direction'] == 1:  # Uptrend
                    signals.loc[idx, 'short_size'] = 0
                else:  # Downtrend
                    signals.loc[idx, 'long_size'] = 0
            
            # Reduce position sizes in high volatility
            if market_data['volatility_regime'] == 1:
                signals.loc[idx, 'long_size'] *= 0.75
                signals.loc[idx, 'short_size'] *= 0.75
            
            # Don't trade in extreme RSI conditions
            if market_data['rsi_14'] > 80:
                signals.loc[idx, 'long_size'] = 0
            elif market_data['rsi_14'] < 20:
                signals.loc[idx, 'short_size'] = 0
            
            # Consider candlestick patterns
            if market_data['pattern_3whitesoldiers'] > 0:
                signals.loc[idx, 'short_size'] = 0
            elif market_data['pattern_3blackcrows'] > 0:
                signals.loc[idx, 'long_size'] = 0
            
            # Consider support/resistance levels
            if market_data['price_to_resistance'] < 0.01:  # Near resistance
                signals.loc[idx, 'long_size'] *= 0.5
            if market_data['price_to_support'] < 0.01:  # Near support
                signals.loc[idx, 'short_size'] *= 0.5
        
        # Combine into final signal
        signals['position'] = signals['long_size'] - signals['short_size']
        signals['prediction'] = np.sign(signals['position'])
        
        return signals
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate classification metrics."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred)
        }
        
        return metrics
    
    def run(self, data: pd.DataFrame) -> Dict:
        """Run the complete ML pipeline."""
        print("Adding advanced features...")
        df = self.calculate_advanced_features(data)
        
        print("Preparing features...")
        X = self.prepare_features(df)
        
        print("Preparing target...")
        y = self.prepare_target(df)
        
        print("Preparing train/test split...")
        splits = self.prepare_train_test_split(X, y)
        
        results = {}
        for i, (train_idx, test_idx) in enumerate(splits):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Impute missing values
            X_train_imputed = pd.DataFrame(
                self.imputer.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            X_test_imputed = pd.DataFrame(
                self.imputer.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
            
            # Train models
            print("Training long model...")
            self.long_model = self.train_model(
                X_train_imputed, 
                y_train, 
                'long'
            )
            
            print("Training short model...")
            self.short_model = self.train_model(
                X_train_imputed, 
                y_train, 
                'short'
            )
            
            # Make predictions
            long_probs = self.long_model[0].predict_proba(X_test_imputed)[:, 1]
            short_probs = self.short_model[0].predict_proba(X_test_imputed)[:, 1]
            
            # Calculate positions
            positions = self.calculate_positions(long_probs, short_probs)
            
            # Store predictions
            predictions = pd.DataFrame({
                'long_prob': long_probs,
                'short_prob': short_probs,
                'position': positions
            }, index=X_test.index)
            
            # Calculate metrics
            long_metrics = self.calculate_metrics(y_test == 1, long_probs > 0.5)
            short_metrics = self.calculate_metrics(y_test == -1, short_probs > 0.5)
            
            # Store results
            results[f'fold_{i}'] = {
                'predictions': predictions,
                'long_metrics': long_metrics,
                'short_metrics': short_metrics
            }
        
        # Combine results
        all_predictions = pd.concat([r['predictions'] for r in results.values()])
        
        # Calculate feature importance
        feature_importance = pd.DataFrame({
            'long_importance': self.long_model[0].feature_importances_,
            'short_importance': self.short_model[0].feature_importances_
        }, index=X.columns)
        
        return {
            'test_predictions': all_predictions,
            'feature_importance': feature_importance,
            'long_metrics': results['fold_0']['long_metrics'],
            'short_metrics': results['fold_0']['short_metrics']
        }
    
    def analyze_results(self, results: Dict, save_plots: bool = True):
        """Generate analysis plots."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path('ml_pipeline_results/plots')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Feature importance plot
        plt.figure(figsize=(12, 8))
        importance = results['feature_importance'].sort_values('long_importance', ascending=True)
        
        ax1 = plt.subplot(121)
        importance.tail(20)['long_importance'].plot(kind='barh')
        plt.title('Top 20 Important Features (Long)')
        plt.tight_layout()
        
        ax2 = plt.subplot(122)
        importance.tail(20)['short_importance'].plot(kind='barh')
        plt.title('Top 20 Important Features (Short)')
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(output_dir / f'feature_importance_{timestamp}.png')
        plt.close()
        
        # ROC curves
        plt.figure(figsize=(12, 6))
        
        # Long model ROC
        plt.subplot(121)
        fpr, tpr, _ = roc_curve(
            (results['test_predictions']['target'] == 1).astype(int),
            results['test_predictions']['long_probability']
        )
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Long Model ROC Curve')
        
        # Short model ROC
        plt.subplot(122)
        fpr, tpr, _ = roc_curve(
            (results['test_predictions']['target'] == -1).astype(int),
            results['test_predictions']['short_probability']
        )
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Short Model ROC Curve')
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(output_dir / f'roc_curves_{timestamp}.png')
        plt.close()
        
        # Prediction distribution
        plt.figure(figsize=(12, 6))
        
        plt.subplot(121)
        sns.histplot(
            data=results['test_predictions'],
            x='long_probability',
            hue='target',
            bins=50
        )
        plt.title('Long Model Probability Distribution')
        
        plt.subplot(122)
        sns.histplot(
            data=results['test_predictions'],
            x='short_probability',
            hue='target',
            bins=50
        )
        plt.title('Short Model Probability Distribution')
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(output_dir / f'prediction_dist_{timestamp}.png')
        plt.close()
    
    def _get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from both models."""
        long_importance = pd.Series(
            self.long_model[0].feature_importances_,
            index=self.feature_names,
            name='long_importance'
        )
        short_importance = pd.Series(
            self.short_model[0].feature_importances_,
            index=self.feature_names,
            name='short_importance'
        )
        return pd.concat([long_importance, short_importance], axis=1)
    
    def _get_test_predictions(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """Get predictions for test data."""
        # Impute missing values
        X_imputed = self.imputer.transform(X_test)
        
        # Scale features
        X_scaled = self.scaler.transform(X_imputed)
        
        # Get probabilities
        long_prob = self.long_model[0].predict_proba(X_scaled)[:, 1]
        short_prob = self.short_model[0].predict_proba(X_scaled)[:, 1]
        
        # Apply probability thresholds
        long_threshold = self.config.model_config['long_model']['threshold']
        short_threshold = self.config.model_config['short_model']['threshold']
        
        # Create predictions DataFrame
        predictions = pd.DataFrame({
            'target': self.y_test,  # Add target
            'long_probability': long_prob,
            'short_probability': short_prob,
            'prediction': 0
        }, index=X_test.index)
        
        # Generate final predictions
        predictions.loc[long_prob >= long_threshold, 'prediction'] = 1
        predictions.loc[short_prob >= short_threshold, 'prediction'] = -1
        
        return predictions 
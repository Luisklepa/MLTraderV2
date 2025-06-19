import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, List
from pathlib import Path
import logging
import yaml
from dataclasses import dataclass
from sklearn.model_selection import TimeSeriesSplit
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

from .technical_indicators import add_all_indicators
from .statistical_features import add_all_statistical_features

logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Configuration for ML pipeline."""
    data_config: Dict
    features_config: Dict
    target_config: Dict
    model_config: Dict
    
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> 'PipelineConfig':
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(**config)

class MLPipeline:
    """ML pipeline for trading signal generation."""
    
    def __init__(self, config: PipelineConfig):
        """Initialize pipeline with configuration."""
        self.config = config
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.long_model = None
        self.short_model = None
        self.feature_names = None
    
    def calculate_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced technical features."""
        df = data.copy()
        
        # Market Regime Features
        # 1. Trend Analysis
        df['ema_20'] = talib.EMA(df['close'], timeperiod=20)
        df['ema_50'] = talib.EMA(df['close'], timeperiod=50)
        df['ema_200'] = talib.EMA(df['close'], timeperiod=200)
        df['price_ema_20_ratio'] = df['close'] / df['ema_20']
        df['price_ema_50_ratio'] = df['close'] / df['ema_50']
        df['price_ema_200_ratio'] = df['close'] / df['ema_200']
        
        # Trend strength and direction
        df['trend_strength'] = abs(talib.ROC(df['close'], timeperiod=20))
        df['trend_direction'] = np.where(df['ema_20'] > df['ema_50'], 1,
                                       np.where(df['ema_20'] < df['ema_50'], -1, 0))
        
        # 2. Volatility Analysis
        df['atr_14'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        volatility = df['close'].pct_change().rolling(window=20).std()
        volatility_sma = volatility.rolling(window=100).mean()
        df['volatility_regime'] = np.where(volatility > volatility_sma, 1, -1)
        df['volatility_ratio'] = volatility / volatility_sma
        
        # 3. Volume Analysis
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['volume_trend_confirm'] = np.where(
            (df['close'] > df['close'].shift(1)) & (df['volume'] > df['volume_sma']), 1,
            np.where((df['close'] < df['close'].shift(1)) & (df['volume'] > df['volume_sma']), -1, 0)
        )
        
        # 4. Market Momentum
        df['rsi_14'] = talib.RSI(df['close'], timeperiod=14)
        df['rsi_divergence'] = df['rsi_14'] - df['rsi_14'].rolling(window=5).mean()
        
        # 5. Pattern Recognition
        for pattern_func in [talib.CDL3BLACKCROWS, talib.CDL3WHITESOLDIERS, 
                           talib.CDLENGULFING, talib.CDLHARAMI, 
                           talib.CDLMORNINGSTAR, talib.CDLEVENINGSTAR]:
            pattern_name = pattern_func.__name__.replace('CDL', '').lower()
            df[f'pattern_{pattern_name}'] = pattern_func(df['open'], df['high'], 
                                                       df['low'], df['close'])
        
        # 6. Support/Resistance Levels
        df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
        df['support_1'] = df['pivot'] - (df['high'] - df['low'])
        df['resistance_1'] = df['pivot'] + (df['high'] - df['low'])
        df['price_to_support'] = (df['close'] - df['support_1']) / df['support_1']
        df['price_to_resistance'] = (df['resistance_1'] - df['close']) / df['close']
        
        # 7. Market Efficiency
        df['efficiency_ratio'] = abs(df['close'] - df['close'].shift(20)) / \
                                df['close'].diff().abs().rolling(window=20).sum()
        
        # 8. Mean Reversion
        df['zscore_price'] = (df['close'] - df['close'].rolling(window=20).mean()) / \
                            df['close'].rolling(window=20).std()
        
        # 9. Trend Exhaustion
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        df['trend_exhaustion'] = np.where(df['adx'] > 40, 1, 0)
        
        # 10. Volatility Breakout
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
            df['close'], timeperiod=20, nbdevup=2, nbdevdn=2
        )
        df['bb_breakout'] = np.where(df['close'] > df['bb_upper'], 1,
                                    np.where(df['close'] < df['bb_lower'], -1, 0))
        
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
        long_threshold = self.config['model_config']['long_model']['threshold']
        short_threshold = self.config['model_config']['short_model']['threshold']
        
        # Generate signals
        signals['long_prob'] = long_probs
        signals['short_prob'] = short_probs
        
        # Apply minimum probability filter
        long_min_prob = self.config['model_config']['long_model']['min_probability']
        short_min_prob = self.config['model_config']['short_model']['min_probability']
        
        # Calculate position sizes with market context
        signals['long_size'] = signals.apply(
            lambda row: self.calculate_position_size(
                row['long_prob'],
                self.config['model_config']['long_model'],
                current_drawdown,
                X_test.loc[row.name]
            ) if row['long_prob'] >= long_threshold and row['long_prob'] >= long_min_prob else 0,
            axis=1
        )
        
        signals['short_size'] = signals.apply(
            lambda row: self.calculate_position_size(
                row['short_prob'],
                self.config['model_config']['short_model'],
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
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare feature set from raw data."""
        df = data.copy()
        
        # Add technical indicators
        for feature_name, feature_config in self.config['features_config'].items():
            if feature_config['type'] == 'technical':
                params = feature_config.get('params', {})
                if feature_name == 'rsi':
                    df[f'rsi_{params["window"]}'] = talib.RSI(df['close'], timeperiod=params['window'])
                elif feature_name == 'macd':
                    macd, signal, hist = talib.MACD(df['close'], 
                                                  fastperiod=params['fast'],
                                                  slowperiod=params['slow'],
                                                  signalperiod=params['signal'])
                    df['macd'] = macd
                    df['macd_signal'] = signal
                    df['macd_histogram'] = hist
                elif feature_name == 'stoch':
                    k, d = talib.STOCH(df['high'], df['low'], df['close'],
                                     fastk_period=params['k_window'],
                                     slowk_period=params['k_window'],
                                     slowd_period=params['d_window'])
                    df['stoch_k'] = k
                    df['stoch_d'] = d
                elif feature_name == 'atr':
                    df[f'atr_{params["window"]}'] = talib.ATR(df['high'], df['low'], df['close'],
                                                             timeperiod=params['window'])
                elif feature_name == 'bbands':
                    upper, middle, lower = talib.BBANDS(df['close'],
                                                      timeperiod=params['window'],
                                                      nbdevup=params['num_std'],
                                                      nbdevdn=params['num_std'])
                    df[f'bb_upper'] = upper
                    df[f'bb_middle'] = middle
                    df[f'bb_lower'] = lower
                elif feature_name == 'obv':
                    df['obv'] = talib.OBV(df['close'], df['volume'])
                elif feature_name == 'vwap':
                    df['vwap'] = (df['volume'] * df['close']).rolling(window=params['window']).sum() / df['volume'].rolling(window=params['window']).sum()
                    df['vwap_distance'] = (df['close'] - df['vwap']) / df['vwap']
        
        # Add temporal features
        if 'temporal' in self.config['features_config']:
            temporal_config = self.config['features_config']['temporal']
            if 'hour' in temporal_config['params']['features']:
                df['hour'] = pd.to_datetime(df.index).hour
            if 'weekday' in temporal_config['params']['features']:
                df['weekday'] = pd.to_datetime(df.index).weekday
            if 'month' in temporal_config['params']['features']:
                df['month'] = pd.to_datetime(df.index).month
            if 'is_weekend' in temporal_config['params']['features']:
                df['is_weekend'] = pd.to_datetime(df.index).weekday >= 5
            if 'session_progress' in temporal_config['params']['features']:
                df['session_progress'] = (pd.to_datetime(df.index).hour * 60 + pd.to_datetime(df.index).minute) / (24 * 60)
        
        # Add statistical features
        for feature_name, feature_config in self.config['features_config'].items():
            if feature_config['type'] == 'statistical':
                params = feature_config.get('params', {})
                if feature_name == 'returns':
                    for window in params['window']:
                        if params['type'] == 'log':
                            df[f'return_{window}'] = np.log(df['close'] / df['close'].shift(window))
                        else:
                            df[f'return_{window}'] = df['close'].pct_change(window)
                elif feature_name == 'volatility':
                    for window in params['window']:
                        if params['type'] == 'realized':
                            df[f'volatility_{window}'] = df['close'].pct_change().rolling(window=window).std()
                elif feature_name == 'zscore':
                    for col in params['columns']:
                        df[f'zscore_{col}_{params["window"]}'] = (df[col] - df[col].rolling(window=params['window']).mean()) / df[col].rolling(window=params['window']).std()
        
        # Select features
        feature_cols = self.config['model_config']['selected_features']
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing features: {missing_cols}")
            feature_cols = [col for col in feature_cols if col in df.columns]
        
        # Ensure all features are numeric
        for col in feature_cols:
            if not np.issubdtype(df[col].dtype, np.number):
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle NaN values
        df_features = df[feature_cols].copy()
        
        # Forward fill NaN values first
        df_features = df_features.fillna(method='ffill')
        
        # For any remaining NaN values (at the start of the series), backward fill
        df_features = df_features.fillna(method='bfill')
        
        # For any still remaining NaN values, fill with 0
        df_features = df_features.fillna(0)
        
        # Remove any infinite values
        df_features = df_features.replace([np.inf, -np.inf], np.nan)
        df_features = df_features.fillna(0)
        
        return df_features
    
    def prepare_target(self, data: pd.DataFrame) -> pd.Series:
        """Prepare target variable with adaptive thresholds and market regime consideration."""
        df = data.copy()
        
        # Calculate multiple horizons for adaptive targets
        horizons = [5, 10, 20]
        future_returns = pd.DataFrame(index=df.index)
        
        for horizon in horizons:
            future_returns[f'return_{horizon}'] = df['close'].pct_change(horizon).shift(-horizon)
        
        # Calculate dynamic threshold based on volatility
        volatility = df['close'].pct_change().rolling(window=50).std()
        atr_volatility = df['atr_14'] / df['close']
        
        # Combine different volatility measures
        combined_volatility = (volatility + atr_volatility) / 2
        
        # Calculate adaptive thresholds - more aggressive
        base_threshold = 0.5  # Reduced from 1.0
        threshold = base_threshold * combined_volatility
        
        # Adjust threshold based on market regime
        if 'trend_strength' in df.columns:
            threshold = threshold * (1 + 0.25 * df['trend_strength'])  # Reduced from 0.5
        
        if 'volatility_regime' in df.columns:
            threshold = threshold * (1 + 0.15 * df['volatility_regime'])  # Reduced from 0.25
        
        # Create target variable
        target = pd.Series(0, index=df.index)
        
        # Weight different horizons
        weights = [0.5, 0.3, 0.2]  # Weights for 5, 10, 20 periods
        weighted_returns = pd.Series(0.0, index=df.index)
        
        for horizon, weight in zip(horizons, weights):
            weighted_returns += future_returns[f'return_{horizon}'] * weight
        
        # Generate signals based on weighted returns and threshold
        target[weighted_returns > threshold] = 1  # Long signals
        target[weighted_returns < -threshold] = -1  # Short signals
        
        # Additional filters for target generation - less restrictive
        if 'trend_direction' in df.columns and 'volume_trend_confirm' in df.columns:
            # Only generate long signals in uptrend with volume confirmation
            uptrend_mask = (df['trend_direction'] == 1) & (df['volume_trend_confirm'] >= 0)  # Changed from == 1
            target[~uptrend_mask & (target == 1)] = 0
            
            # Only generate short signals in downtrend with volume confirmation
            downtrend_mask = (df['trend_direction'] == -1) & (df['volume_trend_confirm'] <= 0)  # Changed from == -1
            target[~downtrend_mask & (target == -1)] = 0
        
        # Don't generate signals near support/resistance levels - more lenient
        if 'price_to_support' in df.columns and 'price_to_resistance' in df.columns:
            near_support = df['price_to_support'] < 0.005  # Reduced from 0.01
            near_resistance = df['price_to_resistance'] < 0.005  # Reduced from 0.01
            target[near_support & (target == -1)] = 0
            target[near_resistance & (target == 1)] = 0
        
        # Consider RSI extremes - more lenient
        if 'rsi_14' in df.columns:
            target[(df['rsi_14'] > 85) & (target == 1)] = 0  # Increased from 80
            target[(df['rsi_14'] < 15) & (target == -1)] = 0  # Decreased from 20
        
        # Consider pattern recognition - more lenient
        pattern_columns = [col for col in df.columns if col.startswith('pattern_')]
        if pattern_columns:
            for col in pattern_columns:
                if '3whitesoldiers' in col or 'morningstar' in col:
                    target[(df[col] > 0) & (target == -1)] = 0
                elif '3blackcrows' in col or 'eveningstar' in col:
                    target[(df[col] > 0) & (target == 1)] = 0
        
        # Forward fill NaN values that might have been created
        target = target.fillna(0)
        
        # Remove signals too close to the end of the dataset
        target.iloc[-max(horizons):] = 0
        
        return target
    
    def prepare_train_test_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        test_size: float = 0.2
    ) -> List[tuple]:
        """Prepare time series cross-validation splits."""
        # Calculate the size of each fold
        n_samples = len(X)
        test_samples = int(n_samples * test_size)
        train_samples = n_samples - test_samples
        
        # Create splits
        splits = []
        for i in range(n_splits):
            # Calculate indices for this fold
            fold_size = test_samples // n_splits
            test_start = train_samples + i * fold_size
            test_end = min(test_start + fold_size, n_samples)
            
            # Skip if we've reached the end of the data
            if test_start >= n_samples:
                break
                
            # Create train/test indices
            train_idx = list(range(test_start))
            test_idx = list(range(test_start, test_end))
            splits.append((train_idx, test_idx))
        
        return splits
    
    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_config: Dict,
        positive_class: int = 1
    ) -> xgb.XGBClassifier:
        """Train model with given configuration."""
        # Convert target to binary for specific class
        y_tr = (y_train == positive_class).astype(int)
        
        # Scale features while preserving feature names
        X_tr_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        # Handle class imbalance
        if model_config.get('use_smote', False):
            try:
                # Count samples in minority class
                minority_samples = y_tr.sum() if y_tr.mean() < 0.5 else len(y_tr) - y_tr.sum()
                
                if minority_samples > 5:  # Only use SMOTE if we have enough minority samples
                    smote = SMOTE(random_state=42)
                    X_resampled_array, y_resampled = smote.fit_resample(X_tr_scaled, y_tr)
                    X_resampled = pd.DataFrame(X_resampled_array, columns=X_tr_scaled.columns)
                else:
                    # If too few samples, use random oversampling
                    from imblearn.over_sampling import RandomOverSampler
                    ros = RandomOverSampler(random_state=42)
                    X_resampled_array, y_resampled = ros.fit_resample(X_tr_scaled, y_tr)
                    X_resampled = pd.DataFrame(X_resampled_array, columns=X_tr_scaled.columns)
            except Exception as e:
                print(f"Warning: Resampling failed ({str(e)}), proceeding with original data")
                X_resampled, y_resampled = X_tr_scaled, y_tr
        else:
            X_resampled, y_resampled = X_tr_scaled, y_tr
        
        # Initialize and train model
        model = xgb.XGBClassifier(
            **model_config['model_params'],
            random_state=42
        )
        
        # Add class weights if specified
        if model_config.get('class_weight') == 'balanced':
            scale_pos_weight = (len(y_resampled) - sum(y_resampled)) / sum(y_resampled)
            model.set_params(scale_pos_weight=scale_pos_weight)
        
        # Train model
        model.fit(
            X_resampled,
            y_resampled,
            eval_set=[(X_resampled, y_resampled)],
            verbose=False
        )
        
        return model
    
    def evaluate_model(
        self,
        model: xgb.XGBClassifier,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        positive_class: int = 1
    ) -> Dict:
        """Evaluate model performance."""
        # Prepare binary target
        y_binary = (y_test == positive_class).astype(int)
        
        # Get predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Initialize metrics dictionary
        metrics = {}
        
        # Basic metrics with zero_division parameter
        metrics['accuracy'] = accuracy_score(y_binary, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_binary, y_pred)
        metrics['precision'] = precision_score(y_binary, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_binary, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_binary, y_pred, zero_division=0)
        
        # ROC-AUC score (only if both classes are present)
        if len(np.unique(y_binary)) > 1:
            metrics['roc_auc'] = roc_auc_score(y_binary, y_prob)
        else:
            metrics['roc_auc'] = 0.5  # Default for single-class case
        
        # Confusion matrix (ensure both classes are represented)
        cm = confusion_matrix(y_binary, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        metrics['true_negative'] = int(tn)
        metrics['false_positive'] = int(fp)
        metrics['false_negative'] = int(fn)
        metrics['true_positive'] = int(tp)
        
        # Additional metrics
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['total_samples'] = len(y_test)
        metrics['positive_samples'] = int(y_binary.sum())
        metrics['negative_samples'] = int(len(y_binary) - y_binary.sum())
        
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
            self.long_model = self.train_model(X_train_imputed, y_train, self.config['model_config']['long_model'], 1)
            
            print("Training short model...")
            self.short_model = self.train_model(X_train_imputed, y_train, self.config['model_config']['short_model'], -1)
            
            # Generate predictions
            print("Generating predictions...")
            long_probs = self.long_model.predict_proba(X_test_imputed)[:, 1]
            short_probs = self.short_model.predict_proba(X_test_imputed)[:, 1]
            
            # Generate signals
            signals = self.generate_signals(X_test_imputed, long_probs, short_probs)
            
            # Store results
            if i == len(splits) - 1:  # Only store results for the last split
                results['long_metrics'] = self.evaluate_model(self.long_model, X_test_imputed, y_test, 1)
                results['short_metrics'] = self.evaluate_model(self.short_model, X_test_imputed, y_test, -1)
                results['feature_importance'] = self._get_feature_importance()
                results['test_predictions'] = signals
        
        # Store feature names
        self.feature_names = self.config['model_config']['selected_features']
        
        # Train models
        print("Training long model...")
        self.long_model = self.train_model(X_train, y_train, self.config['model_config']['long_model'], 1)
        
        print("Training short model...")
        self.short_model = self.train_model(X_train, y_train, self.config['model_config']['short_model'], -1)
        
        # Generate predictions
        print("Generating predictions...")
        X_test_imputed = self.imputer.transform(X_test)
        X_test_scaled = self.scaler.transform(X_test_imputed)
        
        long_probs = self.long_model.predict_proba(X_test_scaled)[:, 1]
        short_probs = self.short_model.predict_proba(X_test_scaled)[:, 1]
        
        # Generate signals with position sizing
        signals = self.generate_signals(X_test, long_probs, short_probs)
        
        # Calculate metrics
        print("Calculating metrics...")
        long_metrics = self.evaluate_model(self.long_model, X_test, y_test, 1)
        short_metrics = self.evaluate_model(self.short_model, X_test, y_test, -1)
        
        # Calculate feature importance
        feature_importance = pd.DataFrame({
            'long_importance': self.long_model.feature_importances_,
            'short_importance': self.short_model.feature_importances_
        }, index=self.feature_names)
        
        return {
            'long_metrics': long_metrics,
            'short_metrics': short_metrics,
            'feature_importance': feature_importance,
            'test_predictions': signals,
            'test_index': X_test.index
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
            self.long_model.feature_importances_,
            index=self.feature_names,
            name='long_importance'
        )
        short_importance = pd.Series(
            self.short_model.feature_importances_,
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
        long_prob = self.long_model.predict_proba(X_scaled)[:, 1]
        short_prob = self.short_model.predict_proba(X_scaled)[:, 1]
        
        # Apply probability thresholds
        long_threshold = self.config['model_config']['long_model']['threshold']
        short_threshold = self.config['model_config']['short_model']['threshold']
        
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
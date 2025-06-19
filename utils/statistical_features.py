import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from scipy import stats

def add_return_features(
    df: pd.DataFrame,
    config: Dict,
    price_col: str = 'close'
) -> pd.DataFrame:
    """Add return-based features."""
    params = config['returns']['params']
    windows = params['window']
    return_type = params['type']
    
    for window in windows:
        if return_type == 'simple':
            df[f'return_{window}'] = df[price_col].pct_change(window)
        else:  # log returns
            df[f'return_{window}'] = np.log(df[price_col] / df[price_col].shift(window))
    
    return df

def add_volatility_features(
    df: pd.DataFrame,
    config: Dict,
    price_col: str = 'close'
) -> pd.DataFrame:
    """Add volatility-based features."""
    params = config['volatility']['params']
    windows = params['window']
    vol_type = params['type']
    
    for window in windows:
        if vol_type == 'realized':
            # Realized volatility
            returns = np.log(df[price_col] / df[price_col].shift(1))
            df[f'volatility_{window}'] = returns.rolling(window).std() * np.sqrt(252)
        elif vol_type == 'parkinson':
            # Parkinson volatility
            high_low = np.log(df['high'] / df['low'])
            df[f'volatility_{window}'] = np.sqrt(
                (1 / (4 * np.log(2))) * 
                high_low.pow(2).rolling(window).mean() * 
                np.sqrt(252)
            )
    
    return df

def add_zscore_features(
    df: pd.DataFrame,
    config: Dict
) -> pd.DataFrame:
    """Add z-score based features."""
    params = config['zscore']['params']
    window = params['window']
    columns = params['columns']
    
    for col in columns:
        if col in df.columns:
            rolling_mean = df[col].rolling(window=window).mean()
            rolling_std = df[col].rolling(window=window).std()
            df[f'zscore_{col}_{window}'] = (df[col] - rolling_mean) / rolling_std
    
    return df

def add_correlation_features(
    df: pd.DataFrame,
    window: int = 20,
    price_col: str = 'close',
    volume_col: str = 'volume'
) -> pd.DataFrame:
    """Add correlation-based features."""
    # Price-volume correlation
    df['price_volume_corr'] = df[price_col].rolling(window).corr(df[volume_col])
    
    # High-low range correlation
    high_low_range = df['high'] - df['low']
    df['range_volume_corr'] = high_low_range.rolling(window).corr(df[volume_col])
    
    return df

def add_distribution_features(
    df: pd.DataFrame,
    window: int = 20,
    price_col: str = 'close'
) -> pd.DataFrame:
    """Add distribution-based features."""
    returns = df[price_col].pct_change()
    
    # Skewness
    df['returns_skew'] = returns.rolling(window).skew()
    
    # Kurtosis
    df['returns_kurt'] = returns.rolling(window).kurt()
    
    # Normality test (Jarque-Bera)
    def rolling_jb_test(x):
        if len(x) < 3:
            return np.nan
        return stats.jarque_bera(x)[0]
    
    df['returns_normality'] = returns.rolling(window).apply(rolling_jb_test)
    
    return df

def add_entropy_features(
    df: pd.DataFrame,
    window: int = 20,
    bins: int = 10
) -> pd.DataFrame:
    """Add entropy-based features."""
    def calculate_entropy(x):
        hist, _ = np.histogram(x, bins=bins)
        hist = hist / len(x)
        return -np.sum(hist * np.log(hist + 1e-10))
    
    # Price entropy
    df['price_entropy'] = df['close'].rolling(window).apply(calculate_entropy)
    
    # Volume entropy
    df['volume_entropy'] = df['volume'].rolling(window).apply(calculate_entropy)
    
    # Return entropy
    returns = df['close'].pct_change()
    df['return_entropy'] = returns.rolling(window).apply(calculate_entropy)
    
    return df

def add_all_statistical_features(
    df: pd.DataFrame,
    features_config: Dict,
    data_config: Optional[Dict] = None
) -> pd.DataFrame:
    """Add all statistical features based on configuration."""
    result = df.copy()
    
    # Get column names from config if provided
    price_col = data_config.get('price_column', 'close') if data_config else 'close'
    volume_col = data_config.get('volume_column', 'volume') if data_config else 'volume'
    
    # Add return features
    if 'returns' in features_config:
        result = add_return_features(result, features_config, price_col)
    
    # Add volatility features
    if 'volatility' in features_config:
        result = add_volatility_features(result, features_config, price_col)
    
    # Add z-score features
    if 'zscore' in features_config:
        result = add_zscore_features(result, features_config)
    
    # Add correlation features
    result = add_correlation_features(result, window=20, price_col=price_col, volume_col=volume_col)
    
    # Add distribution features
    result = add_distribution_features(result, window=20, price_col=price_col)
    
    # Add entropy features
    result = add_entropy_features(result, window=20)
    
    return result 
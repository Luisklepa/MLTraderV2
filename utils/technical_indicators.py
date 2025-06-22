import pandas as pd
import numpy as np
import talib
from typing import Dict, Optional, Any

def add_momentum_indicators(
    df: pd.DataFrame,
    config: Dict,
    price_col: str = 'close'
) -> pd.DataFrame:
    """Add momentum indicators to DataFrame."""
    # RSI
    if 'rsi' in config:
        params = config['rsi']['params']
        df[f'rsi_{params["window"]}'] = talib.RSI(
            df[price_col],
            timeperiod=params['window']
        )
    
    # MACD
    if 'macd' in config:
        params = config['macd']['params']
        macd, signal, hist = talib.MACD(
            df[price_col],
            fastperiod=params['fast'],
            slowperiod=params['slow'],
            signalperiod=params['signal']
        )
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_histogram'] = hist
    
    # Stochastic
    if 'stoch' in config:
        params = config['stoch']['params']
        k, d = talib.STOCH(
            df['high'],
            df['low'],
            df[price_col],
            fastk_period=params['k_window'],
            slowk_period=params['k_window'],
            slowd_period=params['d_window']
        )
        df['stoch_k'] = k
        df['stoch_d'] = d
    
    return df

def add_volatility_indicators(
    df: pd.DataFrame,
    config: Dict,
    price_col: str = 'close'
) -> pd.DataFrame:
    """Add volatility indicators to DataFrame."""
    # ATR
    if 'atr' in config:
        params = config['atr']['params']
        df[f'atr_{params["window"]}'] = talib.ATR(
            df['high'],
            df['low'],
            df[price_col],
            timeperiod=params['window']
        )
    
    # Bollinger Bands
    if 'bbands' in config:
        params = config['bbands']['params']
        upper, middle, lower = talib.BBANDS(
            df[price_col],
            timeperiod=params['window'],
            nbdevup=params['num_std'],
            nbdevdn=params['num_std']
        )
        df[f'bb_upper_{params["window"]}'] = upper
        df[f'bb_middle_{params["window"]}'] = middle
        df[f'bb_lower_{params["window"]}'] = lower
        df[f'bb_width_{params["window"]}'] = (upper - lower) / middle
    
    return df

def add_volume_indicators(
    df: pd.DataFrame,
    config: Dict,
    price_col: str = 'close',
    volume_col: str = 'volume'
) -> pd.DataFrame:
    """Add volume-based indicators to DataFrame."""
    # OBV
    if 'obv' in config:
        df['obv'] = talib.OBV(df[price_col], df[volume_col])
    
    # VWAP
    if 'vwap' in config:
        params = config['vwap']['params']
        df['vwap'] = talib.SMA(df[price_col] * df[volume_col], params['window']) / \
                     talib.SMA(df[volume_col], params['window'])
        df['vwap_distance'] = (df[price_col] - df['vwap']) / df['vwap']
    
    return df

def add_trend_indicators(
    df: pd.DataFrame,
    config: Dict,
    price_col: str = 'close'
) -> pd.DataFrame:
    """Add trend indicators to DataFrame."""
    # EMAs
    for period in [20, 50]:
        df[f'ema_{period}'] = talib.EMA(df[price_col], timeperiod=period)
        df[f'price_ema_{period}_ratio'] = df[price_col] / df[f'ema_{period}']
    
    return df

def add_pattern_recognition(
    df: pd.DataFrame
) -> pd.DataFrame:
    """Add candlestick pattern recognition."""
    # Single candlestick patterns
    df['doji'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
    df['hammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
    
    # Multiple candlestick patterns
    df['engulfing'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
    
    return df

def add_support_resistance(
    df: pd.DataFrame,
    window: int = 20,
    price_col: str = 'close'
) -> pd.DataFrame:
    """Add support and resistance levels."""
    # Rolling max/min
    df['resistance'] = df['high'].rolling(window=window).max()
    df['support'] = df['low'].rolling(window=window).min()
    
    # Distance to levels
    df['resistance_distance'] = (df['resistance'] - df[price_col]) / df[price_col]
    df['support_distance'] = (df[price_col] - df['support']) / df[price_col]
    
    # Breakout signals
    df['breakout_high'] = (df[price_col] > df['resistance'].shift(1)).astype(int)
    df['breakout_low'] = (df[price_col] < df['support'].shift(1)).astype(int)
    
    return df

def add_all_indicators(
    df: pd.DataFrame,
    features_config: Dict,
    data_config: Optional[Dict] = None
) -> pd.DataFrame:
    """Add all technical indicators based on configuration."""
    result = df.copy()
    
    # Get column names from config if provided
    price_col = data_config.get('price_column', 'close') if data_config else 'close'
    volume_col = data_config.get('volume_column', 'volume') if data_config else 'volume'
    
    # Add momentum indicators
    result = add_momentum_indicators(result, features_config, price_col)
    
    # Add volatility indicators
    result = add_volatility_indicators(result, features_config, price_col)
    
    # Add volume indicators
    result = add_volume_indicators(result, features_config, price_col, volume_col)
    
    # Add trend indicators
    result = add_trend_indicators(result, features_config, price_col)
    
    # Add pattern recognition
    result = add_pattern_recognition(result)
    
    # Add support/resistance
    result = add_support_resistance(result, window=20, price_col=price_col)
    
    return result

def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate RSI."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    """Calculate MACD."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return {
        'macd': macd,
        'signal': signal_line,
        'histogram': histogram
    }

def calculate_stochastic(df: pd.DataFrame, k_window: int = 14, d_window: int = 3) -> Dict[str, pd.Series]:
    """Calculate Stochastic Oscillator."""
    low_min = df['low'].rolling(k_window).min()
    high_max = df['high'].rolling(k_window).max()
    
    k = 100 * (df['close'] - low_min) / (high_max - low_min)
    d = k.rolling(d_window).mean()
    
    return {'k': k, 'd': d}

def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window).mean()

def calculate_bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2.0) -> Dict[str, pd.Series]:
    """Calculate Bollinger Bands."""
    middle = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    
    return {
        'upper': upper,
        'middle': middle,
        'lower': lower,
        'breakout': (prices - middle) / (std * num_std)
    }

def calculate_obv(df: pd.DataFrame) -> pd.Series:
    """Calculate On-Balance Volume."""
    return (np.sign(df['close'].diff()) * df['volume']).cumsum()

def calculate_vwap(df: pd.DataFrame, window: int = 20) -> Dict[str, pd.Series]:
    """Calculate VWAP and related metrics."""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical_price * df['volume']).rolling(window).sum() / df['volume'].rolling(window).sum()
    distance = (df['close'] - vwap) / vwap
    
    return {
        'vwap': vwap,
        'distance': distance
    }

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators to the dataframe."""
    # RSI
    df['rsi_14'] = calculate_rsi(df['close'])
    
    # MACD
    macd_data = calculate_macd(df['close'])
    df['macd'] = macd_data['macd']
    df['macd_signal'] = macd_data['signal']
    df['macd_histogram'] = macd_data['histogram']
    
    # Stochastic
    stoch_data = calculate_stochastic(df)
    df['stoch_k'] = stoch_data['k']
    df['stoch_d'] = stoch_data['d']
    
    # ATR
    df['atr_14'] = calculate_atr(df)
    
    # Bollinger Bands
    bb_data = calculate_bollinger_bands(df['close'])
    df['bb_upper'] = bb_data['upper']
    df['bb_lower'] = bb_data['lower']
    df['bb_breakout'] = bb_data['breakout']
    
    # OBV
    df['obv'] = calculate_obv(df)
    
    # VWAP
    vwap_data = calculate_vwap(df)
    df['vwap'] = vwap_data['vwap']
    df['vwap_distance'] = vwap_data['distance']
    
    return df 
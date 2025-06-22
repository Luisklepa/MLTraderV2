import sys
from pathlib import Path
# --- Agregado: añadir la raíz del proyecto al sys.path ---
sys.path.append(str(Path(__file__).parent.parent))
import pandas as pd
import numpy as np
import argparse
import logging
from typing import Tuple
import optuna

# --- AGREGADO: Importar el pipeline avanzado ---
from utils.ml_feature_pipeline import MLFeaturePipeline

def setup_logging() -> logging.Logger:
    """Setup logging configuration."""
    # Remove existing handlers
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    logger.setLevel(logging.INFO)
    
    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    return logger

def load_data(file_path: str) -> pd.DataFrame:
    """Load and preprocess price data."""
    logger = logging.getLogger()
    logger.info("Loading data...")
    
    # Load data
    df = pd.read_csv(file_path)
    
    # Convert timestamp
    if 'open_time' in df.columns:
        df['timestamp'] = pd.to_datetime(df['open_time'])
        df.drop('open_time', axis=1, inplace=True)
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    elif 'date' in df.columns:
        df['timestamp'] = pd.to_datetime(df['date'])
        df.drop('date', axis=1, inplace=True)
    else:
        raise ValueError("DataFrame must have a timestamp, date, or open_time column")
    
    df.set_index('timestamp', inplace=True)
    
    # Sort by datetime
    df.sort_index(inplace=True)
    
    # Remove duplicates
    df = df[~df.index.duplicated(keep='first')]
    
    logger.info(f"Loaded {len(df)} rows of data")
    return df

def calculate_targets(
    df: pd.DataFrame,
    window: int,
    threshold: float
) -> Tuple[pd.Series, pd.Series]:
    """Calculate binary targets for long and short positions."""
    logger = logging.getLogger()
    logger.info(f"Calculating targets (window: {window}, threshold: {threshold})")
    
    # Calculate future returns
    future_returns = df['close'].pct_change(window).shift(-window)
    
    # Generate binary targets
    long_target = (future_returns > threshold).astype(int)
    short_target = (future_returns < -threshold).astype(int)
    
    logger.info(f"Long signals: {long_target.sum()}")
    logger.info(f"Short signals: {short_target.sum()}")
    
    return long_target, short_target

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators."""
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # Stochastic
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    
    # ATR
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift())
    tr3 = abs(df['low'] - df['close'].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(window=14).mean()
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_breakout'] = (df['close'] - df['bb_middle']) / (bb_std * 2)
    
    # OBV
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
    
    # VWAP
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (typical_price * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
    df['vwap_distance'] = (df['close'] - df['vwap']) / df['vwap']
    
    return df

def add_market_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add market regime related features."""
    # EMAs
    df['ema_20'] = df['close'].ewm(span=20).mean()
    df['ema_50'] = df['close'].ewm(span=50).mean()
    
    # Price to EMA ratios
    df['price_ema_20_ratio'] = df['close'] / df['ema_20'] - 1
    df['price_ema_50_ratio'] = df['close'] / df['ema_50'] - 1
    
    # Trend strength and direction
    df['trend_strength'] = abs(df['ema_20'] / df['ema_50'] - 1) * 100
    df['trend_direction'] = np.where(df['ema_20'] > df['ema_50'], 1, -1)
    
    # Volatility regime
    returns = df['close'].pct_change()
    df['volatility_20'] = returns.rolling(20).std() * np.sqrt(252)
    df['volatility_50'] = returns.rolling(50).std() * np.sqrt(252)
    df['volatility_ratio'] = df['volatility_20'] / df['volatility_50']
    
    # Volume analysis
    df['volume_ma20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma20']
    df['volume_trend'] = df['volume_ma20'].pct_change(20)
    df['volume_trend_confirm'] = np.where(
        (df['trend_direction'] == 1) & (df['volume_trend'] > 0) |
        (df['trend_direction'] == -1) & (df['volume_trend'] < 0),
        1, 0
    )
    
    return df

def add_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add statistical features."""
    # Returns
    for window in [1, 5, 10]:
        df[f'return_{window}'] = df['close'].pct_change(window)
    
    # Volatility
    returns = df['close'].pct_change()
    for window in [10, 20]:
        df[f'volatility_{window}'] = returns.rolling(window).std() * np.sqrt(252)
    
    # Z-scores
    for col in ['close', 'volume', 'rsi_14']:
        mean = df[col].rolling(20).mean()
        std = df[col].rolling(20).std()
        df[f'zscore_{col}_20'] = (df[col] - mean) / std
    
    # Efficiency ratio
    direction = abs(df['close'] - df['close'].shift(20))
    volatility = abs(df['close'] - df['close'].shift(1)).rolling(20).sum()
    df['efficiency_ratio'] = direction / volatility
    
    return df

def prepare_ml_dataset(
    input_file: str,
    output_file: str,
    window: int = 10,
    threshold: float = 0.01
) -> None:
    """Prepare dataset for ML training using the advanced MLFeaturePipeline."""
    logger = setup_logging()
    try:
        logger.info(f"[ADVANCED PIPELINE] Loading data from {input_file}...")
        # Usar el pipeline avanzado
        pipeline = MLFeaturePipeline()
        # El pipeline espera listas de thresholds/windows
        pipeline.generate_complete_dataset(
            file_path=input_file,
            output_path=output_file,
            thresholds=[threshold],
            windows=[window]
        )
        logger.info(f"[ADVANCED PIPELINE] Dataset created and saved to {output_file}")
    except Exception as e:
        logger.error(f"Error preparing dataset: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def objective_long(trial):
    window = trial.suggest_int('window', 5, 30)
    threshold = trial.suggest_float('threshold', 0.001, 0.03)
    df = objective_long.df.copy()
    long_target, _ = calculate_targets(df, window, threshold)
    long_ratio = long_target.mean()
    # Penalizar si hay muy pocos positivos
    if long_ratio < 0.01:
        return 0.0
    return long_ratio

def objective_short(trial):
    window = trial.suggest_int('window', 5, 30)
    threshold = trial.suggest_float('threshold', 0.001, 0.03)
    df = objective_short.df.copy()
    _, short_target = calculate_targets(df, window, threshold)
    short_ratio = short_target.mean()
    if short_ratio < 0.01:
        return 0.0
    return short_ratio

def bayesian_optimize_dataset_separated(input_file, n_trials=50):
    print(f"Cargando datos de {input_file}...")
    df = load_data(input_file)
    # --- Optimización LONG ---
    print("Optimizando parámetros para LONG...")
    objective_long.df = df
    study_long = optuna.create_study(direction='maximize')
    study_long.optimize(objective_long, n_trials=n_trials)
    best_win_long = study_long.best_params['window']
    best_thr_long = study_long.best_params['threshold']
    print(f"Mejor combinación LONG: window={best_win_long}, threshold={best_thr_long:.4f}, ratio positivo={study_long.best_value:.4f}")
    # --- Optimización SHORT ---
    print("Optimizando parámetros para SHORT...")
    objective_short.df = df
    study_short = optuna.create_study(direction='maximize')
    study_short.optimize(objective_short, n_trials=n_trials)
    best_win_short = study_short.best_params['window']
    best_thr_short = study_short.best_params['threshold']
    print(f"Mejor combinación SHORT: window={best_win_short}, threshold={best_thr_short:.4f}, ratio positivo={study_short.best_value:.4f}")
    return (best_win_long, best_thr_long, study_long.best_value), (best_win_short, best_thr_short, study_short.best_value)

def main():
    parser = argparse.ArgumentParser(description='Prepare ML dataset (long/short separados, optimización separada)')
    parser.add_argument('--input-file', required=True, help='Input price data file')
    parser.add_argument('--output-long', required=False, help='Output ML dataset file for LONG signals')
    parser.add_argument('--output-short', required=False, help='Output ML dataset file for SHORT signals')
    parser.add_argument('--output-combined', required=False, help='Output ML dataset file for combined LONG+SHORT signals')
    parser.add_argument('--window-long', type=int, default=None, help='Target window size for LONG (si None, optimiza)')
    parser.add_argument('--threshold-long', type=float, default=None, help='Target threshold for LONG (si None, optimiza)')
    parser.add_argument('--window-short', type=int, default=None, help='Target window size for SHORT (si None, optimiza)')
    parser.add_argument('--threshold-short', type=float, default=None, help='Target threshold for SHORT (si None, optimiza)')
    parser.add_argument('--n-trials', type=int, default=30, help='Nº de trials Optuna para cada target')
    args = parser.parse_args()

    logger = setup_logging()
    try:
        # Si se solicita dataset combinado
        if args.output_combined:
            # Si algún parámetro es None, optimizarlo
            if args.window_long is None or args.threshold_long is None or args.window_short is None or args.threshold_short is None:
                (best_win_long, best_thr_long, _), (best_win_short, best_thr_short, _) = bayesian_optimize_dataset_separated(
                    args.input_file, n_trials=args.n_trials)
                window_long = best_win_long if args.window_long is None else args.window_long
                threshold_long = best_thr_long if args.threshold_long is None else args.threshold_long
                window_short = best_win_short if args.window_short is None else args.window_short
                threshold_short = best_thr_short if args.threshold_short is None else args.threshold_short
            else:
                window_long = args.window_long
                threshold_long = args.threshold_long
                window_short = args.window_short
                threshold_short = args.threshold_short
            logger.info(f"[ADVANCED PIPELINE] Generando dataset COMBINADO...")
            pipeline = MLFeaturePipeline()
            pipeline.generate_combined_dataset(
                file_path=args.input_file,
                output_path=args.output_combined,
                window_long=window_long,
                threshold_long=threshold_long,
                window_short=window_short,
                threshold_short=threshold_short
            )
            logger.info(f"[ADVANCED PIPELINE] Dataset combinado generado correctamente.")
        else:
            # Dataset separados (lógica anterior)
            if args.window_long is None or args.threshold_long is None or args.window_short is None or args.threshold_short is None:
                (best_win_long, best_thr_long, best_ratio_long), (best_win_short, best_thr_short, best_ratio_short) = bayesian_optimize_dataset_separated(
                    args.input_file, n_trials=args.n_trials)
                window_long = best_win_long if args.window_long is None else args.window_long
                threshold_long = best_thr_long if args.threshold_long is None else args.threshold_long
                window_short = best_win_short if args.window_short is None else args.window_short
                threshold_short = best_thr_short if args.threshold_short is None else args.threshold_short
            else:
                window_long = args.window_long
                threshold_long = args.threshold_long
                window_short = args.window_short
                threshold_short = args.threshold_short
            if args.output_long:
                logger.info(f"[ADVANCED PIPELINE] Generando dataset LONG...")
                pipeline = MLFeaturePipeline()
                pipeline.generate_complete_dataset(
                    file_path=args.input_file,
                    output_path=args.output_long,
                    threshold=threshold_long,
                    window=window_long,
                    target_type='long'
                )
            if args.output_short:
                logger.info(f"[ADVANCED PIPELINE] Generando dataset SHORT...")
                pipeline = MLFeaturePipeline()
                pipeline.generate_complete_dataset(
                    file_path=args.input_file,
                    output_path=args.output_short,
                    threshold=threshold_short,
                    window=window_short,
                    target_type='short'
                )
            logger.info(f"[ADVANCED PIPELINE] Datasets generados correctamente.")
    except Exception as e:
        logger.error(f"Error preparing datasets: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == '__main__':
    main() 
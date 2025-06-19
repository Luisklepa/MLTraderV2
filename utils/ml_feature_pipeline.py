import pandas as pd
import numpy as np
import talib
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings('ignore')

# === FEATURES AVANZADOS DE MOMENTUM ===
def add_advanced_momentum_features(df):
    """
    Añade features avanzados de momentum y reversión.
    """
    # Z-score de retornos (normaliza el retorno respecto a su rolling window)
    df['return_1'] = df['close'].pct_change()
    df['return_1_zscore'] = (df['return_1'] - df['return_1'].rolling(20).mean()) / df['return_1'].rolling(20).std()
    # Diferencia respecto a máximos/mínimos recientes
    df['close_rolling_max_20'] = df['close'].rolling(20).max()
    df['close_rolling_min_20'] = df['close'].rolling(20).min()
    df['close_to_max_20'] = (df['close'] - df['close_rolling_max_20']) / df['close_rolling_max_20']
    df['close_to_min_20'] = (df['close'] - df['close_rolling_min_20']) / df['close_rolling_min_20']
    # Señales booleanas de RSI extremos
    if 'rsi_14' not in df.columns:
        import talib
        df['rsi_14'] = talib.RSI(df['close'], timeperiod=14)
    df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
    df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
    # Momentum respecto a rolling max/min
    df['momentum_to_max_20'] = df['close'] - df['close_rolling_max_20']
    df['momentum_to_min_20'] = df['close'] - df['close_rolling_min_20']
    # Señal de reversión: ¿precio en percentil bajo/alto de la ventana?
    df['close_percentile_20'] = df['close'].rolling(20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    df['close_in_top_10pct_20'] = (df['close_percentile_20'] > 0.9).astype(int)
    df['close_in_bottom_10pct_20'] = (df['close_percentile_20'] < 0.1).astype(int)
    # Asserts para evitar NaN excesivos
    assert df['return_1_zscore'].isnull().mean() < 0.2, "Demasiados nulos en return_1_zscore"
    assert df['close_to_max_20'].isnull().mean() < 0.2, "Demasiados nulos en close_to_max_20"
    return df

# === FEATURES AVANZADOS DE VOLATILIDAD ===
def add_advanced_volatility_features(df):
    """
    Añade features avanzados de volatilidad.
    """
    # Volatilidad de rango intradiario
    df['intraday_range'] = (df['high'] - df['low']) / df['close']
    # Volatilidad de volumen
    df['volume_volatility_20'] = df['volume'].rolling(20).std()
    # Volatilidad de la volatilidad (rolling std de la volatilidad)
    if 'volatility_20' not in df.columns:
        df['volatility_20'] = df['close'].rolling(20).std()
    df['vol_of_vol_20'] = df['volatility_20'].rolling(20).std()
    # Señales booleanas: ATR y BB width en percentil alto/bajo
    if 'atr_14' not in df.columns:
        import talib
        df['atr_14'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['atr_14_percentile_20'] = df['atr_14'].rolling(20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    df['atr_14_high'] = (df['atr_14_percentile_20'] > 0.9).astype(int)
    df['atr_14_low'] = (df['atr_14_percentile_20'] < 0.1).astype(int)
    if 'bb_width_20' not in df.columns:
        import talib
        upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20)
        df['bb_width_20'] = (upper - lower) / middle
    df['bb_width_20_percentile'] = df['bb_width_20'].rolling(20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    df['bb_width_20_high'] = (df['bb_width_20_percentile'] > 0.9).astype(int)
    df['bb_width_20_low'] = (df['bb_width_20_percentile'] < 0.1).astype(int)
    # Asserts para evitar NaN excesivos
    assert df['intraday_range'].isnull().mean() < 0.2, "Demasiados nulos en intraday_range"
    return df

def add_cross_features(df, config=None, clip_value=1e6, dropna=False, verbose=True):
    """
    Añade features cruzados (ratios, diferencias, productos, lógicos) entre indicadores clave.
    - clip_value: recorta valores extremos para robustez.
    - dropna: elimina filas con NaN en features cruzados.
    - verbose: imprime advertencias y resumen.
    """
    default_config = [
        ('ratio', 'rsi_14', 'atr_14'),
        ('ratio', 'macd', 'volatility_20'),
        ('ratio', 'return_5', 'volume_rolling_20'),
        ('diff', 'rsi_14', 'rsi_50'),
        ('diff', 'ema_20', 'sma_50'),
        ('diff', 'bb_width_20', 'atr_14'),
        ('prod', 'rsi_14', 'macd'),
        ('prod', 'return_1', 'volume'),
        ('logic', 'rsi_14', 'atr_14'),
        ('logic', 'macd', 'volume'),
    ]
    config = config or default_config
    generated = []

    for tipo, f1, f2 in config:
        if tipo == 'ratio':
            name = f"{f1}_{f2}_ratio"
            df[name] = df.get(f1, 0) / (df.get(f2, 1) + 1e-6)
        elif tipo == 'diff':
            name = f"{f1}_{f2}_diff"
            df[name] = df.get(f1, 0) - df.get(f2, 0)
        elif tipo == 'prod':
            name = f"{f1}_{f2}_prod"
            df[name] = df.get(f1, 0) * df.get(f2, 0)
        elif tipo == 'logic':
            name = f"{f1}_{f2}_logic_high"
            df[name] = ((df.get(f1, 0) > df.get(f1, 0).rolling(20).mean()) &
                        (df.get(f2, 0) > df.get(f2, 0).rolling(20).mean())).astype(int)
        generated.append(name)
        if verbose and (f1 not in df.columns or f2 not in df.columns):
            print(f"Advertencia: {f1} o {f2} no existen, {name} se rellena con 0.")

    cruzadas = generated
    df[cruzadas] = df[cruzadas].replace([np.inf, -np.inf], np.nan)
    df[cruzadas] = df[cruzadas].clip(-clip_value, clip_value)
    df[cruzadas] = df[cruzadas].fillna(0)
    if verbose:
        max_abs = df[cruzadas].abs().max()
        if (max_abs > (clip_value * 0.9)).any():
            print("Advertencia: Feature cruzado cerca del límite de clipping.")
        print(f"Features cruzados generados: {cruzadas}")
    if dropna:
        df = df.dropna(subset=cruzadas)
    return df

def select_important_cross_features(df, target, importance_threshold=0.005, top_n=None, verbose=True):
    """
    Selecciona y conserva solo los features cruzados más importantes según Random Forest.
    """
    cross_feats = [col for col in df.columns if any(s in col for s in ['_ratio', '_diff', '_prod', '_logic'])]
    base_feats = [col for col in df.columns if col not in cross_feats + [target]]
    X = df[base_feats + cross_feats]
    y = df[target]
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    importances = rf.feature_importances_
    feat_importance = pd.Series(importances, index=X.columns).sort_values(ascending=False)
    if top_n:
        selected_cross = feat_importance[cross_feats].head(top_n).index.tolist()
    else:
        selected_cross = feat_importance[cross_feats][feat_importance[cross_feats] > importance_threshold].index.tolist()
    if verbose:
        print(f"Features cruzados seleccionados: {selected_cross}")
        print(f"Features cruzados eliminados: {[f for f in cross_feats if f not in selected_cross]}")
    to_drop = [f for f in cross_feats if f not in selected_cross]
    df = df.drop(columns=to_drop)
    return df, selected_cross

def add_advanced_cross_features(df):
    """
    Añade features cruzados avanzados inspirados en prácticas cuant y papers top.
    Robustez: chequea que las columnas base existan antes de crear el feature cruzado.
    """
    def safe_cross(name, f1, f2, op, fallback=0):
        if f1 in df.columns and f2 in df.columns:
            if op == 'div':
                df[name] = df[f1] / (df[f2] + 1e-6)
            elif op == 'mul':
                df[name] = df[f1] * df[f2]
            elif op == 'sub':
                df[name] = df[f1] - df[f2]
            elif op == 'add':
                df[name] = df[f1] + df[f2]
        else:
            df[name] = fallback
            print(f"Advertencia: {f1} o {f2} no existen, {name} se rellena con {fallback}.")

    # A. Cruzados con contexto de volatilidad
    safe_cross('return10_over_vol50', 'return_10', 'volatility_50', 'div')
    safe_cross('return1_over_atr14', 'return_1', 'atr_14', 'div')
    safe_cross('atr14_times_vol', 'atr_14', 'volume', 'mul')
    # B. Momentos e interacciones de tendencia
    if 'macd' in df.columns and 'volume' in df.columns:
        df['macd_times_vol_roc'] = df['macd'] * df['volume'].pct_change().fillna(0)
    else:
        df['macd_times_vol_roc'] = 0
        print("Advertencia: macd o volume no existen, macd_times_vol_roc se rellena con 0.")
    safe_cross('ema20_over_ema50', 'ema_20', 'ema_50', 'div')
    if 'ema_20' in df.columns and 'close' in df.columns and 'return_5' in df.columns:
        df['ema_vs_fut_return5'] = (df['ema_20'] - df['close']) * df['return_5'].shift(-5)
    else:
        df['ema_vs_fut_return5'] = 0
        print("Advertencia: ema_20, close o return_5 no existen, ema_vs_fut_return5 se rellena con 0.")
    # C. Señales condicionales/lógicas
    if 'rsi_14' in df.columns and 'volume' in df.columns:
        df['rsi_overbought_spike'] = ((df['rsi_14'] > 70) & (df['volume'] > df['volume'].rolling(20).mean())).astype(int)
    else:
        df['rsi_overbought_spike'] = 0
        print("Advertencia: rsi_14 o volume no existen, rsi_overbought_spike se rellena con 0.")
    if 'volatility_50' in df.columns and 'macd' in df.columns:
        df['volatility_and_bullish'] = ((df['volatility_50'] > df['volatility_50'].rolling(100).mean()) & (df['macd'] > 0)).astype(int)
    else:
        df['volatility_and_bullish'] = 0
        print("Advertencia: volatility_50 o macd no existen, volatility_and_bullish se rellena con 0.")
    # D. Microestructura/Velas/Patrones
    if 'close' in df.columns and 'open' in df.columns and 'return_1' in df.columns:
        df['body_return1'] = (df['close'] - df['open']) * df['return_1']
    else:
        df['body_return1'] = 0
        print("Advertencia: close, open o return_1 no existen, body_return1 se rellena con 0.")
    if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns and 'open' in df.columns:
        df['wick_to_body_ratio'] = ((df['high'] - df['low']) / ((df['close'] - df['open']).abs() + 1e-6))
    else:
        df['wick_to_body_ratio'] = 0
        print("Advertencia: high, low, close o open no existen, wick_to_body_ratio se rellena con 0.")
    if 'rsi_14' in df.columns and 'atr_14' in df.columns:
        df['rsi_norm_by_atr'] = (df['rsi_14'] - 50) / (df['atr_14'] + 1e-6)
    else:
        df['rsi_norm_by_atr'] = 0
        print("Advertencia: rsi_14 o atr_14 no existen, rsi_norm_by_atr se rellena con 0.")
    # E. Señales de Regime Change
    if 'volatility_50' in df.columns:
        df['vol_jump'] = df['volatility_50'] / (df['volatility_50'].rolling(100).mean() + 1e-6)
        df['vol_jump_flag'] = (df['vol_jump'] > 1.5).astype(int)
    else:
        df['vol_jump'] = 0
        df['vol_jump_flag'] = 0
        print("Advertencia: volatility_50 no existe, vol_jump y vol_jump_flag se rellenan con 0.")
    if 'return_1' in df.columns and 'vol_jump' in df.columns:
        df['return1_times_vol_jump'] = df['return_1'] * df['vol_jump']
    else:
        df['return1_times_vol_jump'] = 0
        print("Advertencia: return_1 o vol_jump no existen, return1_times_vol_jump se rellena con 0.")
    # F. Secuenciales y patrones avanzados
    if 'body_return1' in df.columns:
        df['body_return5_mean'] = df['body_return1'].rolling(5).mean()
    else:
        df['body_return5_mean'] = 0
        print("Advertencia: body_return1 no existe, body_return5_mean se rellena con 0.")
    # Lags básicos para persistencia
    if 'vol_jump' in df.columns:
        df['vol_jump_lag1'] = df['vol_jump'].shift(1)
        df['vol_jump_lag2'] = df['vol_jump'].shift(2)
    else:
        df['vol_jump_lag1'] = 0
        df['vol_jump_lag2'] = 0
        print("Advertencia: vol_jump no existe, vol_jump_lag1 y vol_jump_lag2 se rellenan con 0.")
    # Manejo de NaN/inf
    cruzados = [col for col in df.columns if col in [
        'return10_over_vol50','return1_over_atr14','atr14_times_vol','macd_times_vol_roc','ema20_over_ema50','ema_vs_fut_return5',
        'rsi_overbought_spike','volatility_and_bullish','body_return1','wick_to_body_ratio','rsi_norm_by_atr','vol_jump','vol_jump_flag',
        'return1_times_vol_jump','body_return5_mean','vol_jump_lag1','vol_jump_lag2']]
    df[cruzados] = df[cruzados].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df

def add_anti_fallo_features(df):
    """
    Añade features diseñados para reducir falsos negativos y falsos positivos:
    - Spike de volumen
    - Squeeze de volatilidad
    - Confirmación de tendencia
    - Secuencia de retornos
    - Contexto temporal
    """
    # Spike de volumen
    df['vol_spike_20'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-6)
    df['vol_spike_50'] = df['volume'] / (df['volume'].rolling(50).mean() + 1e-6)
    # Squeeze de volatilidad
    df['atr_14_squeeze'] = df['atr_14'] / (df['atr_14'].rolling(50).mean() + 1e-6)
    df['bb_width_20_squeeze'] = df['bb_width_20'] / (df['bb_width_20'].rolling(50).mean() + 1e-6) if 'bb_width_20' in df.columns else 0
    # Confirmación de tendencia
    df['trend_confirm'] = (
        (df['ema_10'] > df['ema_20']) &
        (df['ema_20'] > df['ema_50']) &
        (df['macd'] > 0) &
        (df['rsi_14'] > 50)
    ).astype(int) if all(col in df.columns for col in ['ema_10','ema_20','ema_50','macd','rsi_14']) else 0
    # Secuencia de retornos positivos/negativos
    df['pos_streak'] = (df['return_1'] > 0).astype(int).groupby((df['return_1'] <= 0).astype(int).cumsum()).cumsum()
    df['neg_streak'] = (df['return_1'] < 0).astype(int).groupby((df['return_1'] >= 0).astype(int).cumsum()).cumsum()
    # Contexto temporal (si existe 'hour' y 'day_of_week')
    if 'hour' in df.columns:
        df['is_high_liquidity_hour'] = df['hour'].between(8, 18).astype(int)
    else:
        df['is_high_liquidity_hour'] = 0
    if 'day_of_week' in df.columns:
        df['is_week_start'] = (df['day_of_week'] == 0).astype(int)
        df['is_week_end'] = (df['day_of_week'] == 4).astype(int)
    else:
        df['is_week_start'] = 0
        df['is_week_end'] = 0
    # Robustez: reemplazar inf/nan
    anti_fallo_cols = ['vol_spike_20','vol_spike_50','atr_14_squeeze','bb_width_20_squeeze','trend_confirm','pos_streak','neg_streak','is_high_liquidity_hour','is_week_start','is_week_end']
    df[anti_fallo_cols] = df[anti_fallo_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df

class MLFeaturePipeline:
    def __init__(self, target_profit_pct=2.0, target_bars=10):
        """Inicializa el pipeline de features ML"""
        self.target_profit_pct = target_profit_pct
        self.target_bars = target_bars
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def load_and_prepare_data(self, file_path):
        """Carga y prepara datos iniciales"""
        print("Cargando datos...")
        df = pd.read_csv(file_path)
        
        # Asegurar formato correcto
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Dataset debe contener: {required_cols}")
        
        # Limpiar datos
        df = df.dropna()
        df = df[df['volume'] > 0]  # Filtrar volumen cero
        
        print(f"Datos cargados: {len(df)} registros")
        return df
    
    def generate_technical_features(self, df):
        """Genera features técnicos avanzados"""
        print("Generando features técnicos...")
        
        data = df.copy()
        
        # FEATURES BÁSICOS DE PRECIO
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        data['price_change'] = (data['close'] - data['open']) / data['open']
        data['gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
        # Retornos de diferentes ventanas
        data['return_1'] = data['close'].pct_change(1)
        data['return_5'] = data['close'].pct_change(5)
        data['return_10'] = data['close'].pct_change(10)
        
        # FEATURES DE VOLUMEN
        data['volume_sma_10'] = data['volume'].rolling(10).mean()
        data['volume_ratio'] = data['volume'] / data['volume_sma_10']
        data['volume_price_trend'] = data['volume'] * data['returns']
        data['vwap'] = (data['volume'] * (data['high'] + data['low'] + data['close']) / 3).cumsum() / data['volume'].cumsum()
        data['vwap_distance'] = (data['close'] - data['vwap']) / data['vwap']
        
        # MEDIAS MÓVILES Y CROSSOVERS
        for period in [5, 10, 12, 20, 26, 50, 100, 200]:
            data[f'sma_{period}'] = data['close'].rolling(period).mean()
            data[f'ema_{period}'] = data['close'].ewm(span=period).mean()
            data[f'price_sma_{period}_ratio'] = data['close'] / data[f'sma_{period}']
            data[f'price_ema_{period}_ratio'] = data['close'] / data[f'ema_{period}']
        
        # Crossovers importantes
        data['sma_cross_5_20'] = np.where(data['sma_5'] > data['sma_20'], 1, 0)
        data['sma_cross_20_50'] = np.where(data['sma_20'] > data['sma_50'], 1, 0)
        data['ema_cross_12_26'] = np.where(data['ema_12'] > data['ema_26'], 1, 0)
        
        return data
    
    def generate_momentum_features(self, df):
        """Genera features de momentum"""
        print("Generando features de momentum...")
        
        # RSI múltiples períodos
        for period in [7, 14, 21, 50]:
            df[f'rsi_{period}'] = talib.RSI(df['close'].values, timeperiod=period)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_histogram'] = talib.MACD(df['close'].values)
        df['macd_cross'] = np.where(df['macd'] > df['macd_signal'], 1, 0)
        
        # Stochastic
        df['stoch_k'], df['stoch_d'] = talib.STOCH(df['high'].values, df['low'].values, df['close'].values)
        
        # Williams %R
        df['williams_r'] = talib.WILLR(df['high'].values, df['low'].values, df['close'].values)
        
        # CCI
        df['cci'] = talib.CCI(df['high'].values, df['low'].values, df['close'].values)
        
        # Momentum
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = talib.MOM(df['close'].values, timeperiod=period)
            df[f'roc_{period}'] = talib.ROC(df['close'].values, timeperiod=period)
        
        return df
    
    def generate_volatility_features(self, df):
        """Genera features de volatilidad"""
        print("Generando features de volatilidad...")
        
        # ATR múltiples períodos
        for period in [7, 14, 21, 50]:
            df[f'atr_{period}'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=period)
        
        # Bollinger Bands
        for period in [14, 20, 50]:
            upper, middle, lower = talib.BBANDS(df['close'].values, timeperiod=period)
            df[f'bb_upper_{period}'] = upper
            df[f'bb_middle_{period}'] = middle
            df[f'bb_lower_{period}'] = lower
            df[f'bb_width_{period}'] = (upper - lower) / middle
            df[f'bb_position_{period}'] = (df['close'] - lower) / (upper - lower)
        
        # Volatilidad realizada
        for period in [5, 10, 20, 50]:
            df[f'volatility_{period}'] = df['returns'].rolling(period).std()
        
        # True Range
        df['true_range'] = talib.TRANGE(df['high'].values, df['low'].values, df['close'].values)

        # === FEATURES NO CONVENCIONALES Y DE VOLATILIDAD ===
        try:
            import ta
            # Keltner Channel (usando ta)
            df['keltner_hband'] = ta.volatility.keltner_channel_hband(df['high'], df['low'], df['close'])
            df['keltner_lband'] = ta.volatility.keltner_channel_lband(df['high'], df['low'], df['close'])
            df['keltner_avg'] = (df['keltner_hband'] + df['keltner_lband']) / 2
        except ImportError:
            # Si no hay ta, usar rolling mean y ATR para aproximar
            df['keltner_hband'] = df['close'].rolling(20).mean() + 2 * df['atr_14']
            df['keltner_lband'] = df['close'].rolling(20).mean() - 2 * df['atr_14']
            df['keltner_avg'] = (df['keltner_hband'] + df['keltner_lband']) / 2
        # Donchian Channel
        df['donchian_high_20'] = df['high'].rolling(20).max()
        df['donchian_low_20'] = df['low'].rolling(20).min()
        df['donchian_channel_width'] = df['donchian_high_20'] - df['donchian_low_20']
        # ADX (fuerza de tendencia)
        try:
            import ta
            df['adx_14'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
        except ImportError:
            df['adx_14'] = talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
        # Volatilidad custom
        df['volatility_custom_10'] = df['close'].rolling(10).std() / df['close'].rolling(10).mean()
        return df
    
    def generate_pattern_features(self, df):
        """Genera features de patrones de candlesticks"""
        print("Generando features de patrones...")
        
        # Patrones básicos
        df['doji'] = talib.CDLDOJI(df['open'].values, df['high'].values, df['low'].values, df['close'].values)
        df['hammer'] = talib.CDLHAMMER(df['open'].values, df['high'].values, df['low'].values, df['close'].values)
        df['engulfing'] = talib.CDLENGULFING(df['open'].values, df['high'].values, df['low'].values, df['close'].values)
        df['morning_star'] = talib.CDLMORNINGSTAR(df['open'].values, df['high'].values, df['low'].values, df['close'].values)
        df['evening_star'] = talib.CDLEVENINGSTAR(df['open'].values, df['high'].values, df['low'].values, df['close'].values)
        
        # Características de la vela
        df['body_size'] = abs(df['close'] - df['open']) / df['open']
        df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['open']
        df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / df['open']
        df['candle_type'] = np.where(df['close'] > df['open'], 1, 0)  # 1 = bullish, 0 = bearish
        
        return df
    
    def generate_market_structure_features(self, df):
        """Genera features de estructura de mercado"""
        print("Generando features de estructura de mercado...")
        
        # Soporte y resistencia aproximados
        df['high_20'] = df['high'].rolling(20).max()
        df['low_20'] = df['low'].rolling(20).min()
        df['resistance_distance'] = (df['high_20'] - df['close']) / df['close']
        df['support_distance'] = (df['close'] - df['low_20']) / df['close']
        
        # Breakouts
        df['breakout_high'] = np.where(df['close'] > df['high_20'].shift(1), 1, 0)
        df['breakout_low'] = np.where(df['close'] < df['low_20'].shift(1), 1, 0)
        
        # Trend strength
        df['trend_strength'] = df['close'].rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        
        return df
    
    def generate_temporal_features(self, df):
        """Genera features temporales si hay timestamp"""
        if 'timestamp' in df.columns or df.index.dtype.kind == 'M':
            print("Generando features temporales...")
            
            if 'timestamp' in df.columns:
                df['datetime'] = pd.to_datetime(df['timestamp'])
            else:
                df['datetime'] = df.index
            
            df['hour'] = df['datetime'].dt.hour
            df['day_of_week'] = df['datetime'].dt.dayofweek
            df['month'] = df['datetime'].dt.month
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            
            # Sessions de trading (ajustar según mercado)
            df['is_asian_session'] = df['hour'].between(0, 8).astype(int)
            df['is_london_session'] = df['hour'].between(8, 16).astype(int)
            df['is_ny_session'] = df['hour'].between(13, 21).astype(int)
            
        return df
    
    def generate_contextual_features(self, df):
        """Features contextuales y de volatilidad intrabar"""
        print("Generando features contextuales...")
        # Volatilidad intrabar
        df['intrabar_range'] = (df['high'] - df['low']) / df['close']
        df['intrabar_body'] = abs(df['close'] - df['open']) / df['close']
        df['intrabar_upper_wick'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['close']
        df['intrabar_lower_wick'] = (np.minimum(df['open'], df['close']) - df['low']) / df['close']
        # Rolling min/max
        for win in [5, 10, 20, 50]:
            df[f'rolling_max_{win}'] = df['close'].rolling(win).max()
            df[f'rolling_min_{win}'] = df['close'].rolling(win).min()
            df[f'close_to_max_{win}'] = (df['close'] - df[f'rolling_max_{win}']) / df[f'rolling_max_{win}']
            df[f'close_to_min_{win}'] = (df['close'] - df[f'rolling_min_{win}']) / df[f'rolling_min_{win}']
        # Contexto de mercado
        df['above_ema_200'] = (df['close'] > df['ema_200']).astype(int) if 'ema_200' in df.columns else 0
        df['above_sma_200'] = (df['close'] > df['sma_200']).astype(int) if 'sma_200' in df.columns else 0
        return df
    
    def generate_target_variable(self, df, future_bars=10, threshold=0.01):
        """Genera la variable objetivo basada en un movimiento real configurable"""
        print(f"Generando target: movimiento > {threshold*100:.2f}% en {future_bars} barras...")
        df['future_return'] = df['close'].shift(-future_bars) / df['close'] - 1
        df['target'] = (df['future_return'] > threshold).astype(int)
        df = df[:-future_bars]
        target_distribution = df['target'].value_counts()
        print(f"Distribución del target:\n{target_distribution}")
        print(f"Ratio positivo: {target_distribution.get(1,0) / len(df):.2%}")
        return df
    
    def create_lag_features(self, df, lag_periods=[1, 2, 3, 5, 10]):
        """Crea features con lag temporal"""
        print("Creando features con lag temporal...")
        
        # Features clave para lag
        lag_features = ['returns', 'volume_ratio', 'rsi_14', 'macd', 'atr_14', 'bb_position_20']
        
        for feature in lag_features:
            if feature in df.columns:
                for lag in lag_periods:
                    df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
        
        return df
    
    def clean_and_scale_features(self, df):
        """Limpia y escala las features"""
        print("Limpiando y escalando features...")
        
        # Identificar columnas de features (excluir target y metadatos)
        exclude_cols = ['target', 'future_max', 'datetime', 'timestamp', 'open_time', 'open', 'high', 'low', 'close', 'volume']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Reemplazar infinitos y NaN
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        
        # Eliminar filas con demasiados NaN
        df = df.dropna(subset=['target'])  # Mantener target válido
        
        # Llenar NaN restantes con forward fill y después con 0
        df[feature_cols] = df[feature_cols].fillna(method='ffill').fillna(0)
        
        # Escalar features
        self.scaler = StandardScaler()
        df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        
        self.feature_columns = feature_cols
        print(f"Total features creadas: {len(feature_cols)}")
        
        return df
    
    def generate_complete_dataset(self, file_path, output_path='btcusdt_ml_dataset.csv', thresholds=[0.01], windows=[10]):
        """Pipeline completo de generación de dataset para grid de targets"""
        print("=== INICIANDO PIPELINE DE FEATURES ML ===")
        df = self.load_and_prepare_data(file_path)
        df = self.generate_technical_features(df)
        df = self.generate_momentum_features(df)
        df = add_advanced_momentum_features(df)
        df = self.generate_volatility_features(df)
        df = add_advanced_volatility_features(df)
        df = add_cross_features(df, clip_value=1e6, dropna=False, verbose=True)
        df = add_advanced_cross_features(df)
        df = add_anti_fallo_features(df)
        # Selección automática de features cruzados relevantes
        if 'target' in df.columns:
            df, selected_cross = select_important_cross_features(df, target='target', importance_threshold=0.005, verbose=True)
        # === INTEGRACIÓN DE FEATURES AVANZADOS ===
        df = self.generate_pattern_features(df)
        df = self.generate_market_structure_features(df)
        df = self.generate_temporal_features(df)
        df = self.generate_contextual_features(df)
        df_all = df.copy()
        for thr in thresholds:
            for win in windows:
                print(f"\n--- Generando dataset para threshold={thr}, window={win} ---")
                df = df_all.copy()
                df = self.create_lag_features(df)
                df = self.generate_target_variable(df, future_bars=win, threshold=thr)
                df = self.clean_and_scale_features(df)
                out_name = output_path.replace('.csv', f'_win{win}_thr{int(thr*10000)}.csv')
                df.to_csv(out_name, index=False)
                print(f"Dataset guardado en: {out_name}")
                print(f"Shape final: {df.shape}")
                print(f"Distribución target: {df['target'].value_counts().to_dict()}")
                print(f"Ratio positivo: {df['target'].mean():.2%}")
        return None

# Ejecutar pipeline
if __name__ == "__main__":
    pipeline = MLFeaturePipeline()
    # Grid de thresholds y ventanas
    thresholds = [0.003, 0.005, 0.007, 0.01, 0.015]
    windows = [5, 10, 20]
    pipeline.generate_complete_dataset('btcusdt_prices.csv', output_path='btcusdt_ml_dataset.csv', thresholds=thresholds, windows=windows)
    print("Pipeline de features completado!") 
import pandas as pd
import numpy as np

def analyze_signals_with_filters(df, signal_type='all'):
    """Analiza el impacto de cada filtro en las señales."""
    if signal_type == 'long':
        signals = df[df['target'] == 1].copy()
    elif signal_type == 'short':
        signals = df[df['target'] == -1].copy()
    else:
        signals = df[df['target'] != 0].copy()
    
    total_signals = len(signals)
    if total_signals == 0:
        return
    
    print(f"\n=== Análisis de Señales {signal_type.upper()} (Total: {total_signals}) ===")
    
    # Análisis progresivo de filtros
    remaining = signals.copy()
    
    # 1. Filtro EMA
    if signal_type == 'long':
        ema_filter = (remaining['ema_5'] > remaining['ema_20']) & (remaining['ema_20'] > remaining['ema_50'])
    else:  # short o all
        ema_filter = remaining['ema_5'] < remaining['ema_50']
    
    remaining = remaining[ema_filter]
    print(f"\n1. Filtro EMA:")
    print(f"Señales que pasan: {len(remaining)} ({len(remaining)/total_signals*100:.1f}%)")
    if len(remaining) > 0:
        print(f"Future return promedio: {remaining['future_return'].mean():.2%}")
        print(f"Win rate esperado: {(remaining['future_return'] * np.where(signal_type=='short', -1, 1) > 0).mean()*100:.1f}%")
    
    # 2. Filtro de Volumen
    vol_threshold = 0.25 if signal_type == 'short' else 1.0
    volume_filter = (1 + remaining['volume_ratio']) > vol_threshold
    remaining = remaining[volume_filter]
    print(f"\n2. Filtro de Volumen (threshold: {vol_threshold}):")
    print(f"Señales que pasan: {len(remaining)} ({len(remaining)/total_signals*100:.1f}%)")
    if len(remaining) > 0:
        print(f"Future return promedio: {remaining['future_return'].mean():.2%}")
        print(f"Win rate esperado: {(remaining['future_return'] * np.where(signal_type=='short', -1, 1) > 0).mean()*100:.1f}%")
    
    # 3. Filtro ATR
    atr_percentile = remaining['atr_14'].rank(pct=True)
    atr_filter = (atr_percentile >= 0.2) & (atr_percentile <= 0.8)
    remaining = remaining[atr_filter]
    print(f"\n3. Filtro ATR (20-80 percentil):")
    print(f"Señales que pasan: {len(remaining)} ({len(remaining)/total_signals*100:.1f}%)")
    if len(remaining) > 0:
        print(f"Future return promedio: {remaining['future_return'].mean():.2%}")
        print(f"Win rate esperado: {(remaining['future_return'] * np.where(signal_type=='short', -1, 1) > 0).mean()*100:.1f}%")
    
    # Análisis de rentabilidad por umbral de señal
    if len(remaining) > 0:
        print("\nAnálisis por umbral de señal:")
        # Usar percentiles fijos en lugar de qcut
        thresholds = [0.25, 0.5, 0.75, 1.0]
        labels = ['Muy Débil', 'Débil', 'Fuerte', 'Muy Fuerte']
        
        signal_abs = abs(remaining['target'])
        max_signal = signal_abs.max()
        
        for i, (lower, upper) in enumerate(zip([0] + thresholds[:-1], thresholds)):
            mask = (signal_abs >= lower * max_signal) & (signal_abs <= upper * max_signal)
            if mask.any():
                signals_in_group = remaining[mask]
                win_rate = (signals_in_group['future_return'] * np.where(signal_type=='short', -1, 1) > 0).mean()*100
                avg_return = signals_in_group['future_return'].mean()
                print(f"\n{labels[i]}:")
                print(f"Número de señales: {len(signals_in_group)}")
                print(f"Win rate: {win_rate:.1f}%")
                print(f"Retorno promedio: {avg_return:.2%}")

# Cargar datos
df = pd.read_csv('data/processed/ml_signals_shifted.csv')

# Analizar cada tipo de señal
analyze_signals_with_filters(df, 'long')
analyze_signals_with_filters(df, 'short') 
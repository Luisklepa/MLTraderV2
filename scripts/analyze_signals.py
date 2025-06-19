import pandas as pd
import numpy as np

# Cargar datos
df = pd.read_csv('data/processed/ml_signals_shifted.csv')

# Filtrar señales short
short_signals = df[df['target'] == -1].copy()

print("\n=== Análisis de Señales SHORT ===")
print(f"Total señales short: {len(short_signals)}")

# Aplicar filtros uno por uno y mostrar cuántas señales quedan después de cada uno
signals_remaining = short_signals.copy()

# 1. Filtro de EMAs simplificado
ema_filter = signals_remaining['ema_5'] < signals_remaining['ema_50']
signals_remaining = signals_remaining[ema_filter]
print(f"\n1. Después del filtro EMA (fast < slow):")
print(f"Señales restantes: {len(signals_remaining)} ({len(signals_remaining)/len(short_signals)*100:.1f}%)")

# 2. Filtro de volumen más permisivo
volume_threshold = 0.5  # Mitad del umbral original
volume_filter = (1 + signals_remaining['volume_ratio']) > volume_threshold
signals_remaining = signals_remaining[volume_filter]
print(f"\n2. Después del filtro de volumen (ratio > {volume_threshold}):")
print(f"Señales restantes: {len(signals_remaining)} ({len(signals_remaining)/len(short_signals)*100:.1f}%)")

# 3. Filtro de ATR
atr_percentile = signals_remaining['atr_14'].rank(pct=True)
atr_filter = (atr_percentile >= 0.2) & (atr_percentile <= 0.8)
signals_remaining = signals_remaining[atr_filter]
print(f"\n3. Después del filtro de ATR (20-80 percentil):")
print(f"Señales restantes: {len(signals_remaining)} ({len(signals_remaining)/len(short_signals)*100:.1f}%)")

# Análisis temporal de las señales finales
print(f"\nDistribución temporal de las señales válidas:")
signals_remaining['hour'] = pd.to_datetime(signals_remaining['datetime']).dt.hour
print("\nSeñales por hora:")
print(signals_remaining['hour'].value_counts().sort_index())

# Análisis de rentabilidad potencial
print("\nAnálisis de rentabilidad potencial:")
future_returns = signals_remaining['future_return']
print(f"Retorno promedio: {future_returns.mean():.2%}")
print(f"Retorno mediano: {future_returns.median():.2%}")
print(f"% Señales rentables: {(future_returns < 0).mean()*100:.1f}%") 
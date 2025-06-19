"""
utils/trading_utils.py
Funciones utilitarias para estrategias de trading en Backtrader.
Centraliza lógica de indicadores, stops, gestión de riesgo y prints de depuración.
"""
import backtrader as bt

# ================= INDICADORES =================
def get_atr(data, period=14):
    """Devuelve el indicador ATR de Backtrader."""
    return bt.ind.ATR(data, period=period)

def get_ema(data, period=20):
    """Devuelve el indicador EMA de Backtrader."""
    return bt.ind.EMA(data.close, period=period)

def get_rsi(data, period=14):
    """Devuelve el indicador RSI de Backtrader."""
    return bt.ind.RSI(data.close, period=period)

# ================= STOPS Y RIESGO ===============
def calculate_stop_loss(entry_price, atr, atr_mult=1.0, direction='long'):
    """Calcula el nivel de stop loss usando ATR."""
    if direction == 'long':
        return entry_price - atr * atr_mult
    else:
        return entry_price + atr * atr_mult

def calculate_take_profit(entry_price, stop_loss, rr_ratio=2.0, direction='long'):
    """Calcula el nivel de take profit usando RR ratio."""
    risk = abs(entry_price - stop_loss)
    if direction == 'long':
        return entry_price + risk * rr_ratio
    else:
        return entry_price - risk * rr_ratio

def calculate_position_size(account_value, risk_per_trade, stop_distance):
    """Calcula el tamaño de la posición según el riesgo y la distancia al stop."""
    if stop_distance == 0:
        return 0
    # Calcular el tamaño en BTC (permitiendo fracciones)
    size_btc = (account_value * 0.2 * risk_per_trade) / stop_distance
    # Redondear a 4 decimales para evitar problemas de precisión
    return round(size_btc, 4)

# ================= PRINTS DE DEPURACIÓN ===============
def print_debug(msg):
    print(f"[DEBUG] {msg}")

# ================= EJEMPLO DE USO ===============
# from utils.trading_utils import get_atr, calculate_stop_loss, calculate_position_size 
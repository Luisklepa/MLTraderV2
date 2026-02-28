import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Parámetros óptimos y gestión de riesgo
WINDOW = 15
THRESHOLD = 0.003
PROBA_THR = 0.5
CAPITAL_INICIAL = 10000
TAM_POS = 1.0
STOP_LOSS = -0.02  # -2%
TAKE_PROFIT = 0.03  # +3%

# 1. Cargar datos y modelo
DATA_PATH = 'btcusdt_ml_dataset.csv'
df = pd.read_csv(DATA_PATH)
features = [
    'open', 'high', 'low', 'close', 'volume',
    'ema_10', 'ema_20', 'ema_50', 'rsi_14', 'atr_14',
    'macd', 'macd_signal', 'bb_high', 'bb_low',
    'return_1', 'return_5', 'return_10'
]
model = joblib.load('rf_trading_model.pkl')

# 2. Generar target y features para todo el histórico
future_return = df['close'].shift(-WINDOW) / df['close'] - 1
df['target'] = (future_return > THRESHOLD).astype(int)
df = df.dropna(subset=features + ['target']).reset_index(drop=True)
X = df[features].values
y_prob = model.predict_proba(X)[:,1]
df['ml_signal'] = (y_prob > PROBA_THR).astype(int)

# 3. Simular equity curve con SL/TP
capital = CAPITAL_INICIAL
equity_curve = [capital]
trade_returns = []
trade_outcomes = []
for i, row in df.iterrows():
    if row['ml_signal'] == 1 and i + WINDOW < len(df):
        entry = row['close']
        # Simular barra a barra para aplicar SL/TP
        ret = None
        for j in range(1, WINDOW+1):
            idx = i + j
            if idx >= len(df):
                break
            price = df.loc[idx, 'close']
            r = (price - entry) / entry
            if r <= STOP_LOSS:
                ret = STOP_LOSS
                break
            if r >= TAKE_PROFIT:
                ret = TAKE_PROFIT
                break
        if ret is None:
            # Si no se tocó SL/TP, usar retorno final
            exit_ = df.loc[i+WINDOW, 'close']
            ret = (exit_ - entry) / entry
        capital *= (1 + ret * TAM_POS)
        equity_curve.append(capital)
        trade_returns.append(ret)
        trade_outcomes.append(ret > 0)

# 4. Métricas
if trade_returns:
    total_return = (capital - CAPITAL_INICIAL) / CAPITAL_INICIAL
    avg_return = np.mean(trade_returns)
    winrate = np.mean(trade_outcomes)
    n_trades = len(trade_returns)
    max_dd = np.max(1 - np.array(equity_curve)/np.maximum.accumulate(equity_curve))
    print(f"Nº trades: {n_trades}")
    print(f"Retorno total (compuesto): {total_return*100:.2f}%")
    print(f"Retorno promedio por trade: {avg_return*100:.2f}%")
    print(f"Winrate: {winrate*100:.2f}%")
    print(f"Max Drawdown: {max_dd*100:.2f}%")
else:
    print("No se generaron trades.")

# 5. Graficar equity curve
plt.figure(figsize=(12,5))
plt.plot(equity_curve)
plt.title('Equity Curve - ML Signal + Risk Management')
plt.xlabel('Trade #')
plt.ylabel('Capital')
plt.tight_layout()
plt.show() 
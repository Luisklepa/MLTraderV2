import pandas as pd
import numpy as np
import joblib
import torch
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from ml_lstm_pipeline_torch import LSTMClassifier, TIMESTEPS, features as lstm_features

# Configuración
MODEL_TYPE = 'rf'  # 'rf' para Random Forest, 'lstm' para LSTM
WINDOW = 15
THRESHOLD = 0.003
PROBA_THR = 0.5
CAPITAL_INICIAL = 10000
TAM_POS = 1.0
COMISION = 0.001  # 0.1% por trade

# 1. Cargar datos
DATA_PATH = 'btcusdt_ml_dataset.csv'
df = pd.read_csv(DATA_PATH)
features = lstm_features
future_return = df['close'].shift(-WINDOW) / df['close'] - 1
df['target'] = (future_return > THRESHOLD).astype(int)
df = df.dropna(subset=features + ['target']).reset_index(drop=True)

# 2. Preparar modelo y señales
if MODEL_TYPE == 'rf':
    model = joblib.load('rf_trading_model.pkl')
    X = df[features].values
    y_prob = model.predict_proba(X)[:,1]
    df['ml_signal'] = (y_prob > PROBA_THR).astype(int)
elif MODEL_TYPE == 'lstm':
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features].values)
    X_seq = []
    for i in range(len(X_scaled) - TIMESTEPS):
        X_seq.append(X_scaled[i:i+TIMESTEPS])
    X_seq = np.array(X_seq, dtype=np.float32)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMClassifier(input_dim=len(features))
    model.load_state_dict(torch.load('lstm_trading_model.pt', map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        y_prob = model(torch.tensor(X_seq).to(DEVICE)).cpu().numpy().flatten()
    # Ajustar el dataframe para el offset de secuencias
    df = df.iloc[TIMESTEPS:].reset_index(drop=True)
    df['ml_signal'] = (y_prob > PROBA_THR).astype(int)
else:
    raise ValueError('MODEL_TYPE debe ser "rf" o "lstm"')

# 3. Simular equity curve con gestión de capital y comisiones
capital = CAPITAL_INICIAL
equity_curve = [capital]
trade_returns = []
trade_outcomes = []
for i, row in df.iterrows():
    if row['ml_signal'] == 1 and i + WINDOW < len(df):
        entry = row['close']
        exit_ = df.loc[i+WINDOW, 'close']
        ret = (exit_ - entry) / entry - 2*COMISION  # comisión de entrada y salida
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
plt.title(f'Equity Curve - ML Signal ({MODEL_TYPE.upper()})')
plt.xlabel('Trade #')
plt.ylabel('Capital')
plt.tight_layout()
plt.show() 
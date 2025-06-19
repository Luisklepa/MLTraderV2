import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

# Parámetros
WINDOW = 15
THRESHOLD = 0.003
TIMESTEPS = 20  # Ventana para LSTM

# 1. Cargar datos y generar target
DATA_PATH = 'btcusdt_ml_dataset.csv'
df = pd.read_csv(DATA_PATH)
features = [
    'open', 'high', 'low', 'close', 'volume',
    'ema_10', 'ema_20', 'ema_50', 'rsi_14', 'atr_14',
    'macd', 'macd_signal', 'bb_high', 'bb_low',
    'return_1', 'return_5', 'return_10'
]
future_return = df['close'].shift(-WINDOW) / df['close'] - 1
df['target'] = (future_return > THRESHOLD).astype(int)
df = df.dropna(subset=features + ['target']).reset_index(drop=True)

# 2. Normalizar features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features].values)
y = df['target'].values

# 3. Crear secuencias para LSTM
X_seq = []
y_seq = []
for i in range(len(X_scaled) - TIMESTEPS):
    X_seq.append(X_scaled[i:i+TIMESTEPS])
    y_seq.append(y[i+TIMESTEPS])
X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

# 4. Split train/test
split_idx = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

# 5. Definir y entrenar modelo LSTM
model = keras.Sequential([
    layers.LSTM(32, input_shape=(TIMESTEPS, len(features))),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test))

# 6. Evaluar
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.3f}")
y_pred = (model.predict(X_test) > 0.5).astype(int)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred)) 
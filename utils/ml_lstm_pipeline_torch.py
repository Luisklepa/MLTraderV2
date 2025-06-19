import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader

# Parámetros
WINDOW = 15
THRESHOLD = 0.003
TIMESTEPS = 20  # Ventana para LSTM
BATCH_SIZE = 32
EPOCHS = 15
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
X_seq = np.array(X_seq, dtype=np.float32)
y_seq = np.array(y_seq, dtype=np.float32)

# 4. Split train/test
split_idx = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

# 5. Convertir a tensores
X_train_t = torch.tensor(X_train).to(DEVICE)
y_train_t = torch.tensor(y_train).unsqueeze(1).to(DEVICE)
X_test_t = torch.tensor(X_test).to(DEVICE)
y_test_t = torch.tensor(y_test).unsqueeze(1).to(DEVICE)

# 5. Dataset y DataLoader balanceado
class BalancedSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        # Indices de cada clase
        self.idx_0 = np.where(y == 0)[0]
        self.idx_1 = np.where(y == 1)[0]
        self.n = max(len(self.idx_0), len(self.idx_1)) * 2
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        if idx % 2 == 0:
            i = np.random.choice(self.idx_0)
        else:
            i = np.random.choice(self.idx_1)
        return torch.tensor(self.X[i]), torch.tensor(self.y[i]).unsqueeze(0)

train_dataset = BalancedSequenceDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 6. Definir modelo LSTM
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return self.sigmoid(out)

model = LSTMClassifier(input_dim=len(features)).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# 7. Entrenamiento balanceado
for epoch in range(EPOCHS):
    model.train()
    sum_loss = 0
    for xb, yb in train_loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        sum_loss += loss.item() * xb.size(0)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {sum_loss/len(train_loader.dataset):.4f}")

# 8. Evaluación
model.eval()
with torch.no_grad():
    y_pred = (model(X_test_t) > 0.5).cpu().numpy().astype(int)
    y_true = y_test_t.cpu().numpy().astype(int)
    print(classification_report(y_true, y_pred))

# Guardar solo los pesos del modelo entrenado para comparación
torch.save(model.state_dict(), 'lstm_trading_model.pt')
print('Pesos del modelo LSTM guardados en lstm_trading_model.pt') 
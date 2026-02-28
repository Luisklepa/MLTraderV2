import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
import datetime
import os

# Configuración
DATASET = 'btcusdt_ml_dataset_win10_thr100.csv'
MODEL_PATH = sorted([f for f in os.listdir('experiment_results') if f.startswith('xgb_best_model_')], reverse=True)[0]
MODEL_PATH = f'experiment_results/{MODEL_PATH}'
TARGET = 'target'
TEST_SIZE = 0.2
REPORT_PATH = f'experiment_results/xgb_feature_analysis_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
PLOT_PATH = f'experiment_results/xgb_feature_importance_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
TRADING_PLOT_PATH = f'experiment_results/xgb_trading_sim_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png'

# 1. Cargar datos y modelo
df = pd.read_csv(DATASET)
exclude_cols = ['target', 'future_return', 'datetime', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
features = [col for col in df.columns if col not in exclude_cols]
X = df[features]
y = df[TARGET]

# Split temporal
split = int(len(df) * (1 - TEST_SIZE))
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# Cargar modelo
model = joblib.load(MODEL_PATH)

# 2. Importancia de features
importances = model.feature_importances_
feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)
plt.figure(figsize=(10,6))
feat_imp.head(20).plot(kind='bar')
plt.title('Importancia de features XGBoost (top 20)')
plt.tight_layout()
plt.savefig(PLOT_PATH)
plt.close()

# 3. Evaluación out-of-sample
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, digits=4)
cm = confusion_matrix(y_test, y_pred)

with open(REPORT_PATH, 'w') as f:
    f.write('--- Importancia de features (top 20) ---\n')
    f.write(str(feat_imp.head(20)))
    f.write('\n\n--- Classification report (out-of-sample) ---\n')
    f.write(report)
    f.write('\n\n--- Confusion matrix ---\n')
    f.write(str(cm))

print(f'Reporte guardado en: {REPORT_PATH}')
print(f'Gráfico de importancia de features guardado en: {PLOT_PATH}')

# 4. Simulación básica de trading
df_test = df.iloc[split:].copy()
df_test['signal'] = y_pred
# Supón que solo operas cuando signal==1, y tomas el future_return como resultado
# (puedes ajustar la lógica según tu pipeline real)
df_test['strategy_return'] = df_test['signal'] * df_test['future_return']
df_test['cumulative_return'] = (1 + df_test['strategy_return']).cumprod()
plt.figure(figsize=(10,6))
plt.plot(df_test['cumulative_return'].values, label='Modelo XGBoost')
plt.title('Simulación de equity curve (out-of-sample)')
plt.xlabel('Trades')
plt.ylabel('Cumulative Return')
plt.legend()
plt.tight_layout()
plt.savefig(TRADING_PLOT_PATH)
plt.close()
print(f'Gráfico de simulación de trading guardado en: {TRADING_PLOT_PATH}') 
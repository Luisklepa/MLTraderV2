import pandas as pd
import numpy as np
from utils.ml_train_model import MLTradingTrainer
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Parámetros
DATASET_PATH = 'data/btcusdt_ml_dataset.csv'
OUT_SAMPLE_RATIO = 0.2  # 20% final reservado para test out-of-sample

# 1. Cargar dataset completo
print(f"Cargando dataset: {DATASET_PATH}")
df = pd.read_csv(DATASET_PATH)

# 1b. Extraer features temporales de 'timestamp'
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['weekday'] = df['timestamp'].dt.weekday
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)

# 2. Split temporal: 80% train, 20% out-of-sample
n_total = len(df)
n_out = int(n_total * OUT_SAMPLE_RATIO)
n_train = n_total - n_out

df_train = df.iloc[:n_train].reset_index(drop=True)
df_out = df.iloc[n_train:].reset_index(drop=True)

print(f"Total muestras: {n_total}")
print(f"Train: {len(df_train)} | Out-of-sample: {len(df_out)}")

# 3. Entrenar pipeline SOLO con train
trainer = MLTradingTrainer(dataset_path=DATASET_PATH)

def load_and_prepare_data_train_only(self):
    df = df_train.copy()
    # Excluir columnas no numéricas y 'timestamp'
    exclude_cols = ['timestamp']
    if 'target' in df.columns:
        exclude_cols.append('target')
    X = df.drop(columns=exclude_cols, errors='ignore')
    # Solo columnas numéricas
    X = X.select_dtypes(include=[np.number])
    y = df['target']
    # Asignar lista de features usados
    self.feature_columns = list(X.columns)
    return X, y
trainer.load_and_prepare_data = load_and_prepare_data_train_only.__get__(trainer)

trainer.train_model(balance_strategy='smote', optimize_params=True, auto_threshold=True)

# === FILTRADO POR IMPORTANCIA MÍNIMA ===
importances = trainer.feature_importance
min_importance = 0.01
selected_features = importances[importances['importance'] > min_importance]['feature'].tolist()

# Asegurar que los features existan en ambos conjuntos
selected_features = [f for f in selected_features if f in df_train.columns and f in df_out.columns]
print(f"\nFeatures seleccionados por importancia > {min_importance} y presentes en train/test: {selected_features}")

if len(selected_features) < len(importances):
    def load_and_prepare_data_selected(self):
        df = df_train.copy()
        X = df[selected_features]
        y = df['target']
        self.feature_columns = selected_features
        return X, y
    trainer.load_and_prepare_data = load_and_prepare_data_selected.__get__(trainer)
    print("\nRe-entrenando modelo solo con features seleccionados...")
    trainer.train_model(balance_strategy='smote', optimize_params=True, auto_threshold=True)

    # Re-evaluar in-sample
    X_train = df_train[selected_features]
    y_train = df_train['target']
    proba_train = trainer.model.predict_proba(X_train)[:, 1]
    threshold = trainer.best_threshold if hasattr(trainer, 'best_threshold') else 0.5
    y_pred_train = (proba_train >= threshold).astype(int)
    print("\n=== MÉTRICAS IN-SAMPLE (TRAIN, features filtrados) ===")
    print(classification_report(y_train, y_pred_train))
    print("Confusion Matrix:")
    print(confusion_matrix(y_train, y_pred_train))
    print(f"ROC-AUC: {roc_auc_score(y_train, proba_train):.4f}")
    returns_train = []
    for i, signal in enumerate(y_pred_train):
        if signal == 1:
            if y_train.iloc[i] == 1:
                returns_train.append(0.015)
            else:
                returns_train.append(-0.01)
        else:
            returns_train.append(0)
    equity_curve_train = np.cumprod(1 + np.array(returns_train))

    # Re-evaluar out-of-sample
    X_out = df_out[selected_features]
    y_out = df_out['target']
    proba_out = trainer.model.predict_proba(X_out)[:, 1]
    y_pred_out = (proba_out >= threshold).astype(int)
    print("\n=== MÉTRICAS OUT-OF-SAMPLE (features filtrados) ===")
    print(classification_report(y_out, y_pred_out))
    print("Confusion Matrix:")
    print(confusion_matrix(y_out, y_pred_out))
    print(f"ROC-AUC: {roc_auc_score(y_out, proba_out):.4f}")
    returns_out = []
    for i, signal in enumerate(y_pred_out):
        if signal == 1:
            if y_out.iloc[i] == 1:
                returns_out.append(0.015)
            else:
                returns_out.append(-0.01)
        else:
            returns_out.append(0)
    equity_curve_out = np.cumprod(1 + np.array(returns_out))

    # Guardar nuevas curvas
    plt.figure(figsize=(14, 7))
    plt.plot(equity_curve_train, label='In-sample (Train, filtrados)')
    plt.plot(range(len(equity_curve_train), len(equity_curve_train) + len(equity_curve_out)), equity_curve_out, label='Out-of-sample (Test, filtrados)')
    plt.title(f'Equity Curve: In-sample vs Out-of-sample (features filtrados, thr={threshold})')
    plt.xlabel('Trade Number')
    plt.ylabel('Equity')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig('equity_curve_in_vs_out_of_sample_filtered.png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve_train)
    plt.title('Equity Curve IN-SAMPLE (Train, filtrados)')
    plt.xlabel('Trade Number')
    plt.ylabel('Equity')
    plt.grid(alpha=0.3)
    plt.savefig('equity_curve_in_sample_filtered.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve_out)
    plt.title('Equity Curve OUT-OF-SAMPLE (Test, filtrados)')
    plt.xlabel('Trade Number')
    plt.ylabel('Equity')
    plt.grid(alpha=0.3)
    plt.savefig('equity_curve_out_of_sample_filtered.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Resumen comparativo
    print("\n=== RESUMEN COMPARATIVO (features filtrados) ===")
    print(f"IN-SAMPLE: Total trades: {sum(y_pred_train)}, Win rate: {np.mean((y_pred_train == 1) & (y_train == 1)):.2%}, Total return: {sum(returns_train):.2%}, Return per trade: {np.mean([r for r in returns_train if r != 0]):.3%}, Final equity: {equity_curve_train[-1]:.3f}")
    print(f"OUT-OF-SAMPLE: Total trades: {sum(y_pred_out)}, Win rate: {np.mean((y_pred_out == 1) & (y_out == 1)):.2%}, Total return: {sum(returns_out):.2%}, Return per trade: {np.mean([r for r in returns_out if r != 0]):.3%}, Final equity: {equity_curve_out[-1]:.3f}")

# 4. Evaluar in-sample (train)
def get_X_y(df, trainer):
    exclude_cols = ['timestamp']
    if 'target' in df.columns:
        exclude_cols.append('target')
    X = df.drop(columns=exclude_cols, errors='ignore')
    X = X.select_dtypes(include=[np.number])
    y = df['target']
    return X, y

X_train, y_train = get_X_y(df_train, trainer)
proba_train = trainer.model.predict_proba(X_train)[:, 1]
threshold = trainer.best_threshold if hasattr(trainer, 'best_threshold') else 0.5
y_pred_train = (proba_train >= threshold).astype(int)

print("\n=== MÉTRICAS IN-SAMPLE (TRAIN) ===")
print(classification_report(y_train, y_pred_train))
print("Confusion Matrix:")
print(confusion_matrix(y_train, y_pred_train))
print(f"ROC-AUC: {roc_auc_score(y_train, proba_train):.4f}")

returns_train = []
for i, signal in enumerate(y_pred_train):
    if signal == 1:
        if y_train.iloc[i] == 1:
            returns_train.append(0.015)
        else:
            returns_train.append(-0.01)
    else:
        returns_train.append(0)
equity_curve_train = np.cumprod(1 + np.array(returns_train))

# 5. Evaluar out-of-sample (test)
X_out, y_out = get_X_y(df_out, trainer)
proba_out = trainer.model.predict_proba(X_out)[:, 1]
y_pred_out = (proba_out >= threshold).astype(int)

print("\n=== MÉTRICAS OUT-OF-SAMPLE ===")
print(classification_report(y_out, y_pred_out))
print("Confusion Matrix:")
print(confusion_matrix(y_out, y_pred_out))
print(f"ROC-AUC: {roc_auc_score(y_out, proba_out):.4f}")

returns_out = []
for i, signal in enumerate(y_pred_out):
    if signal == 1:
        if y_out.iloc[i] == 1:
            returns_out.append(0.015)
        else:
            returns_out.append(-0.01)
    else:
        returns_out.append(0)
equity_curve_out = np.cumprod(1 + np.array(returns_out))

# 6. Comparar y graficar ambas curvas de equity
plt.figure(figsize=(14, 7))
plt.plot(equity_curve_train, label='In-sample (Train)')
plt.plot(range(len(equity_curve_train), len(equity_curve_train) + len(equity_curve_out)), equity_curve_out, label='Out-of-sample (Test)')
plt.title(f'Equity Curve: In-sample vs Out-of-sample (Threshold óptimo: {threshold})')
plt.xlabel('Trade Number')
plt.ylabel('Equity')
plt.grid(alpha=0.3)
plt.legend()
plt.savefig('equity_curve_in_vs_out_of_sample.png', dpi=300, bbox_inches='tight')
plt.show()

# Guardar curvas individuales
plt.figure(figsize=(12, 6))
plt.plot(equity_curve_train)
plt.title('Equity Curve IN-SAMPLE (Train)')
plt.xlabel('Trade Number')
plt.ylabel('Equity')
plt.grid(alpha=0.3)
plt.savefig('equity_curve_in_sample.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(12, 6))
plt.plot(equity_curve_out)
plt.title('Equity Curve OUT-OF-SAMPLE (Test)')
plt.xlabel('Trade Number')
plt.ylabel('Equity')
plt.grid(alpha=0.3)
plt.savefig('equity_curve_out_of_sample.png', dpi=300, bbox_inches='tight')
plt.close()

# 7. Resumen comparativo
print("\n=== RESUMEN COMPARATIVO ===")
print(f"IN-SAMPLE: Total trades: {sum(y_pred_train)}, Win rate: {np.mean((y_pred_train == 1) & (y_train == 1)):.2%}, Total return: {sum(returns_train):.2%}, Return per trade: {np.mean([r for r in returns_train if r != 0]):.3%}, Final equity: {equity_curve_train[-1]:.3f}")
print(f"OUT-OF-SAMPLE: Total trades: {sum(y_pred_out)}, Win rate: {np.mean((y_pred_out == 1) & (y_out == 1)):.2%}, Total return: {sum(returns_out):.2%}, Return per trade: {np.mean([r for r in returns_out if r != 0]):.3%}, Final equity: {equity_curve_out[-1]:.3f}") 
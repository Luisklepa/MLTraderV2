import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
import datetime
import os

# Configuración de experimentos
DATA_PATH = 'btcusdt_ml_dataset.csv'
THRESHOLDS = [0.005, 0.01, 0.015]  # 0.5%, 1%, 1.5%
WINDOWS = [5, 10, 20]
TOP_N = 10
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Cargar datos base
base = pd.read_csv(DATA_PATH)
exclude_cols = ['target', 'future_return', 'datetime', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
features_all = [col for col in base.columns if col not in exclude_cols]

# Crear carpeta de resultados
RESULTS_DIR = 'experiment_results'
os.makedirs(RESULTS_DIR, exist_ok=True)

for window in WINDOWS:
    for thr in THRESHOLDS:
        # 1. Generar target dinámico
        df = base.copy()
        df['future_return'] = df['close'].shift(-window) / df['close'] - 1
        df['target'] = (df['future_return'] > thr).astype(int)
        df = df.dropna(subset=features_all + ['target']).reset_index(drop=True)
        X = df[features_all].values
        y = df['target'].values
        split_idx = int(len(df) * (1 - TEST_SIZE))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        # 2. Selección de features
        rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(X_train, y_train)
        importances = pd.Series(rf.feature_importances_, index=features_all).sort_values(ascending=False)
        top_features = importances.head(TOP_N).index.tolist()
        # 3. Entrenamiento con top features
        X_train_top = df[top_features].values[:split_idx]
        X_test_top = df[top_features].values[split_idx:]
        rf_top = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
        rf_top.fit(X_train_top, y_train)
        y_pred = rf_top.predict(X_test_top)
        y_prob = rf_top.predict_proba(X_test_top)[:,1]
        # 4. Métricas y logs
        report = classification_report(y_test, y_pred, output_dict=True)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        # 5. Análisis de errores
        df_test = df.iloc[split_idx:].copy()
        df_test['y_true'] = y_test
        df_test['y_pred'] = y_pred
        df_test['y_prob'] = y_prob
        df_fp = df_test[(df_test['y_true'] == 0) & (df_test['y_pred'] == 1)]
        df_fn = df_test[(df_test['y_true'] == 1) & (df_test['y_pred'] == 0)]
        # 6. Guardar resultados
        date_str = datetime.datetime.now().strftime('%Y%m%d')
        base_name = f'win{window}_thr{int(thr*10000)}_{date_str}'
        pd.DataFrame(importances).to_csv(f'{RESULTS_DIR}/importances_{base_name}.csv')
        pd.DataFrame(report).to_csv(f'{RESULTS_DIR}/report_{base_name}.csv')
        pd.DataFrame(cm).to_csv(f'{RESULTS_DIR}/cm_{base_name}.csv')
        df_fp.to_csv(f'{RESULTS_DIR}/false_positives_{base_name}.csv', index=False)
        df_fn.to_csv(f'{RESULTS_DIR}/false_negatives_{base_name}.csv', index=False)
        with open(f'{RESULTS_DIR}/summary_{base_name}.txt', 'w') as f:
            f.write(f'Window: {window}\nThreshold: {thr}\nAUC: {auc:.4f}\n')
            f.write(f'Accuracy: {report["accuracy"]:.4f}\n')
            f.write(f'Precision (1): {report["1"]["precision"]:.4f}\nRecall (1): {report["1"]["recall"]:.4f}\n')
            f.write(f'False Positives: {len(df_fp)}\nFalse Negatives: {len(df_fn)}\n')
        print(f'Experimento window={window}, thr={thr} completado. AUC={auc:.4f}, FP={len(df_fp)}, FN={len(df_fn)}') 
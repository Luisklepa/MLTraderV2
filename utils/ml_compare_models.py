import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import datetime

# Configuración
DATASET = 'btcusdt_ml_dataset_win10_thr100.csv'  # Puedes cambiar el dataset aquí
TARGET = 'target'
TEST_SIZE = 0.2
RANDOM_STATE = 42
REPORT_PATH = f'experiment_results/compare_rf_xgb_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'

# 1. Cargar datos
df = pd.read_csv(DATASET)
exclude_cols = ['target', 'future_return', 'datetime', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
features = [col for col in df.columns if col not in exclude_cols]
X = df[features]
y = df[TARGET]

# 2. Train/test split
df_train, df_test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
X_train, y_train = df_train[features], df_train[TARGET]
X_test, y_test = df_test[features], df_test[TARGET]

# 3. Balanceo con SMOTE
sm = SMOTE(random_state=RANDOM_STATE)
X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)

# 4. Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced')
rf.fit(X_train_bal, y_train_bal)
y_pred_rf = rf.predict(X_test)
report_rf = classification_report(y_test, y_pred_rf, output_dict=True)
importances_rf = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)

# 5. XGBoost
xgb_clf = xgb.XGBClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1, scale_pos_weight=(y_train == 0).sum()/(y_train == 1).sum())
xgb_clf.fit(X_train_bal, y_train_bal)
y_pred_xgb = xgb_clf.predict(X_test)
report_xgb = classification_report(y_test, y_pred_xgb, output_dict=True)
importances_xgb = pd.Series(xgb_clf.feature_importances_, index=features).sort_values(ascending=False)

# 6. Guardar reporte comparativo
with open(REPORT_PATH, 'w', encoding='utf-8') as f:
    f.write('=== COMPARATIVA RANDOM FOREST vs XGBOOST ===\n')
    f.write(f'Dataset: {DATASET}\n\n')
    f.write('--- Random Forest ---\n')
    f.write(classification_report(y_test, y_pred_rf))
    f.write('\nTop 10 features RF:\n')
    f.write(importances_rf.head(10).to_string())
    f.write('\n\n--- XGBoost ---\n')
    f.write(classification_report(y_test, y_pred_xgb))
    f.write('\nTop 10 features XGB:\n')
    f.write(importances_xgb.head(10).to_string())
    f.write('\n')
print(f'Reporte comparativo guardado en: {REPORT_PATH}')

# 7. Mostrar resumen en consola
print('\n=== RANDOM FOREST ===')
print(classification_report(y_test, y_pred_rf))
print('Top 10 features RF:')
print(importances_rf.head(10))
print('\n=== XGBOOST ===')
print(classification_report(y_test, y_pred_xgb))
print('Top 10 features XGB:')
print(importances_xgb.head(10)) 
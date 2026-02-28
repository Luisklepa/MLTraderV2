import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import classification_report, make_scorer, f1_score, recall_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import datetime

# Configuración
DATASET = 'btcusdt_ml_dataset_win10_thr100.csv'  # Cambia aquí si lo deseas
TARGET = 'target'
N_SPLITS = 5
RANDOM_STATE = 42
RESULTS_PATH = f'experiment_results/xgb_gridsearch_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
BEST_MODEL_PATH = f'experiment_results/xgb_best_model_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'

# 1. Cargar datos
df = pd.read_csv(DATASET)
exclude_cols = ['target', 'future_return', 'datetime', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
features = [col for col in df.columns if col not in exclude_cols]
X = df[features]
y = df[TARGET]

# 2. Balanceo con SMOTE
sm = SMOTE(random_state=RANDOM_STATE)
X_bal, y_bal = sm.fit_resample(X, y)

# 3. Definir TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

# 4. Definir el modelo y el grid de hiperparámetros
xgb_clf = xgb.XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1, scale_pos_weight=(y == 0).sum()/(y == 1).sum())
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# 5. Métrica principal: f1-score de la clase minoritaria
scorer = make_scorer(f1_score, pos_label=1)

# 6. GridSearchCV
print('Iniciando GridSearchCV con validación cruzada temporal...')
gs = GridSearchCV(xgb_clf, param_grid, scoring=scorer, cv=tscv, verbose=2, n_jobs=-1)
gs.fit(X_bal, y_bal)

# 7. Guardar resultados
gs_results = pd.DataFrame(gs.cv_results_)
gs_results.to_csv(RESULTS_PATH, index=False)
print(f'Resultados del grid search guardados en: {RESULTS_PATH}')

# 8. Guardar el mejor modelo
joblib.dump(gs.best_estimator_, BEST_MODEL_PATH)
print(f'Mejor modelo guardado en: {BEST_MODEL_PATH}')

# 9. Mostrar el mejor resultado
print('--- Mejor combinación de hiperparámetros ---')
print(gs.best_params_)
print('Mejor f1-score (minoritaria):', gs.best_score_) 
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from imblearn.over_sampling import SMOTE

# Configuración
DATA_PATH = 'btcusdt_ml_dataset.csv'
TOP_N = 10  # Número de features a seleccionar
TEST_SIZE = 0.2
RANDOM_STATE = 42

# 1. Cargar datos
print('Cargando dataset...')
df = pd.read_csv(DATA_PATH)
# Eliminar features redundantes por alta correlación
features_to_drop = ['open', 'high', 'low', 'ema_10', 'ema_20', 'ema_50', 'bb_high', 'bb_low', 'macd_signal']
features_to_drop = [col for col in features_to_drop if col in df.columns]
df = df.drop(columns=features_to_drop)
print(f"Features eliminados por alta correlación: {features_to_drop}")
exclude_cols = ['target', 'future_return', 'datetime', 'timestamp', 'close', 'volume']
features = [col for col in df.columns if col not in exclude_cols]
X = df[features].values
y = df['target'].values

# 2. Visualización con PCA
def plot_pca(X, y):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(8,6))
    plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='coolwarm', alpha=0.5)
    plt.title('PCA - Primeros 2 Componentes')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.colorbar(label='Target')
    plt.tight_layout()
    plt.show()
    importances = pd.Series(pca.components_[0], index=features).abs().sort_values(ascending=False)
    print('\nTop 10 features más importantes según PCA:')
    print(importances.head(10))
    return importances

# 3. Visualización con t-SNE
def plot_tsne(X, y):
    tsne = TSNE(n_components=2, perplexity=30, random_state=RANDOM_STATE)
    X_tsne = tsne.fit_transform(X)
    plt.figure(figsize=(8,6))
    plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y, cmap='coolwarm', alpha=0.5)
    plt.title('t-SNE - Proyección 2D')
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.colorbar(label='Target')
    plt.tight_layout()
    plt.show()

# 4. Selección de features por importancia de Random Forest
def select_top_features(X, y, features, top_n=10):
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    print('\nTop {} features más importantes según Random Forest:'.format(top_n))
    print(importances.head(top_n))
    return importances.head(top_n).index.tolist()

# 5. Entrenamiento y comparación de modelos
def train_and_compare(df, features, target, top_features):
    X = df[features].values
    y = df[target].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=False)
    # --- Balanceo con SMOTE ---
    unique, counts = np.unique(y_train, return_counts=True)
    min_count = counts.min()
    if min_count > 1:
        k_neighbors = min(5, min_count - 1)
        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)
        X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
        print(f"[SMOTE] Dataset balanceado: {dict(zip(*np.unique(y_train_bal, return_counts=True)))}")
    else:
        print("[SMOTE] Muy pocos ejemplos para aplicar SMOTE. Usando datos originales.")
        X_train_bal, y_train_bal = X_train, y_train
    # Modelo con todos los features
    rf_all = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced')
    rf_all.fit(X_train_bal, y_train_bal)
    y_pred_all = rf_all.predict(X_test)
    acc_all = accuracy_score(y_test, y_pred_all)
    print('\n--- Modelo con TODOS los features ---')
    print(classification_report(y_test, y_pred_all))
    # Modelo con top N features (solo los que existen en el DataFrame)
    valid_top_features = [f for f in top_features if f in features]
    if len(valid_top_features) < len(top_features):
        print(f"[WARN] Solo {len(valid_top_features)} de {len(top_features)} top features existen en el dataset. Usando solo los válidos.")
    X_top = df[valid_top_features].values
    X_train_top, X_test_top = X_top[:len(X_train)], X_top[len(X_train):]
    # --- Balanceo con SMOTE para top features ---
    if min_count > 1:
        smote_top = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)
        X_train_top_bal, y_train_top_bal = smote_top.fit_resample(X_train_top, y_train)
    else:
        X_train_top_bal, y_train_top_bal = X_train_top, y_train
    rf_top = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced')
    rf_top.fit(X_train_top_bal, y_train_top_bal)
    y_pred_top = rf_top.predict(X_test_top)
    acc_top = accuracy_score(y_test, y_pred_top)
    print('\n--- Modelo con TOP {} features ---'.format(len(valid_top_features)))
    print(classification_report(y_test, y_pred_top))
    print(f'Accuracy TODOS: {acc_all:.4f} | Accuracy TOP {len(valid_top_features)}: {acc_top:.4f}')
    return acc_all, acc_top

def save_metadata_and_logs(importances_rf, importances_pca, top_features, auc_all, auc_top, params):
    date_str = datetime.datetime.now().strftime('%Y%m%d')
    meta = {
        'date': date_str,
        'params': params,
        'top_features': top_features,
        'auc_all': auc_all,
        'auc_top': auc_top
    }
    # Guardar logs de importancia
    importances_rf.to_csv(f'log_importance_rf_{date_str}.csv')
    if importances_pca is not None:
        importances_pca.to_csv(f'log_importance_pca_{date_str}.csv')
    # Guardar metadatos
    pd.Series(meta).to_json(f'metadata_feature_selection_{date_str}.json')
    print(f'Logs y metadatos guardados con fecha {date_str}')

def auto_feature_selection_on_top_datasets(top_csv='ml_experiments_top10.csv', top_n=20):
    df_top = pd.read_csv(top_csv)
    datasets = df_top['dataset'].unique()
    for dataset in datasets:
        print(f"\n=== Selección de features para: {dataset} ===")
        df = pd.read_csv(dataset)
        # Eliminar features redundantes por alta correlación
        features_to_drop = ['open', 'high', 'low', 'ema_10', 'ema_20', 'ema_50', 'bb_high', 'bb_low', 'macd_signal']
        features_to_drop = [col for col in features_to_drop if col in df.columns]
        df = df.drop(columns=features_to_drop)
        print(f"Features eliminados por alta correlación: {features_to_drop}")
        exclude_cols = ['target', 'future_return', 'datetime', 'timestamp', 'close', 'volume']
        features = [col for col in df.columns if col not in exclude_cols]
        target = 'target'
        # Selección top-N
        top_features = select_top_features(df[features].values, df[target].values, features, top_n=top_n)
        # Comparar performance
        acc_all, acc_top = train_and_compare(df, features, target, top_features)
        # Guardar logs
        date_str = datetime.datetime.now().strftime('%Y%m%d')
        pd.Series(top_features).to_csv(f'feature_selection_{dataset.replace(".csv","")}_top{top_n}_{date_str}.csv')
        print(f"Top-{top_n} features guardados para {dataset}")

if __name__ == '__main__':
    print('Visualizando con PCA...')
    importances_pca = plot_pca(X, y)
    print('Visualizando con t-SNE...')
    plot_tsne(X, y)
    print('Seleccionando top features...')
    top_features = select_top_features(X, y, features, top_n=TOP_N)
    print('Entrenando y comparando modelos...')
    auc_all, auc_top = train_and_compare(df, features, 'target', top_features)
    print('Guardando metadatos y logs...')
    save_metadata_and_logs(importances_pca, None, top_features, auc_all, auc_top, {'top_n': TOP_N, 'test_size': TEST_SIZE, 'random_state': RANDOM_STATE})
    auto_feature_selection_on_top_datasets(top_n=20) 
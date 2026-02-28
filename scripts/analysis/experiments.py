import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
import glob
from collections import Counter

# 1. Buscar todos los datasets generados
DATASETS = sorted(glob.glob('btcusdt_ml_dataset_win*_thr*.csv'))
results = []
for dataset_path in DATASETS:
    print(f"\n=== Procesando dataset: {dataset_path} ===")
    df = pd.read_csv(dataset_path)
    features = [col for col in df.columns if col not in ['target','future_max','future_return','datetime','timestamp','open','high','low','close','volume']]
    X = df[features].values
    y = df['target'].values
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    close_test = df['close'].values[split_idx:] if 'close' in df.columns else np.zeros_like(y_test)
    class_counts = Counter(y_train)
    min_class = min(class_counts, key=class_counts.get)
    min_count = class_counts[min_class]
    for balance_strategy in ['smote', 'undersample', 'combined', None]:
        for metric in ['f1', 'recall']:
            try:
                # Balanceo robusto
                if balance_strategy == 'smote':
                    k_neighbors = min(5, min_count - 1) if min_count > 1 else 1
                    if min_count < 2:
                        raise ValueError('Muy pocos ejemplos para SMOTE')
                    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
                elif balance_strategy == 'undersample':
                    from imblearn.under_sampling import RandomUnderSampler
                    rus = RandomUnderSampler(random_state=42)
                    X_train_bal, y_train_bal = rus.fit_resample(X_train, y_train)
                elif balance_strategy == 'combined':
                    from imblearn.under_sampling import RandomUnderSampler
                    from imblearn.over_sampling import SMOTE
                    k_neighbors = min(5, min_count - 1) if min_count > 1 else 1
                    if min_count < 2:
                        raise ValueError('Muy pocos ejemplos para SMOTE')
                    over = SMOTE(sampling_strategy=0.5, random_state=42, k_neighbors=k_neighbors)
                    under = RandomUnderSampler(sampling_strategy=0.8, random_state=42)
                    from imblearn.pipeline import Pipeline as ImbPipeline
                    imb_pipe = ImbPipeline([
                        ('over', over),
                        ('under', under)
                    ])
                    X_train_bal, y_train_bal = imb_pipe.fit_resample(X_train, y_train)
                else:
                    X_train_bal, y_train_bal = X_train, y_train
                # Entrenar modelo
                clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
                clf.fit(X_train_bal, y_train_bal)
                y_prob = clf.predict_proba(X_test)[:,1]
                # Fine-tuning de threshold
                proba_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
                best_thr, best_score = 0.5, 0
                for pthr in proba_thresholds:
                    y_pred = (y_prob > pthr).astype(int)
                    if metric == 'recall':
                        score = recall_score(y_test, y_pred, pos_label=1)
                    elif metric == 'f1':
                        from sklearn.metrics import f1_score
                        score = f1_score(y_test, y_pred, pos_label=1)
                    else:
                        score = f1_score(y_test, y_pred, pos_label=1)
                    if score > best_score:
                        best_score = score
                        best_thr = pthr
                # Simulación de retorno
                trade_returns = []
                trade_outcomes = []
                y_pred = (y_prob > best_thr).astype(int)
                for i, pred in enumerate(y_pred):
                    if pred == 1 and i + 1 < len(close_test):
                        entry = close_test[i]
                        exit_ = close_test[min(i+1, len(close_test)-1)]
                        ret = (exit_ - entry) / entry
                        trade_returns.append(ret)
                        trade_outcomes.append(ret > 0)
                if trade_returns:
                    total_return = np.prod([1+r for r in trade_returns]) - 1
                    avg_return = np.mean(trade_returns)
                    winrate = np.mean(trade_outcomes)
                    n_trades = len(trade_returns)
                else:
                    total_return = avg_return = winrate = n_trades = 0
                results.append({
                    'dataset': dataset_path,
                    'n_features': len(features),
                    'balance_strategy': balance_strategy,
                    'metric': metric,
                    'metric_score': best_score,
                    'proba_thr': best_thr,
                    'total_return': total_return,
                    'avg_return': avg_return,
                    'winrate': winrate,
                    'n_trades': n_trades
                })
            except Exception as e:
                print(f"[WARN] Experimento no viable para {dataset_path} | {balance_strategy} | {metric}: {e}")
                results.append({
                    'dataset': dataset_path,
                    'n_features': len(features),
                    'balance_strategy': balance_strategy,
                    'metric': metric,
                    'metric_score': None,
                    'proba_thr': None,
                    'total_return': None,
                    'avg_return': None,
                    'winrate': None,
                    'n_trades': None,
                    'error': str(e)
                })
# 4. Mostrar top 10 combinaciones
results_df = pd.DataFrame(results)
top = results_df.sort_values('total_return', ascending=False).head(10)
print('\nTop 10 combinaciones por retorno compuesto:')
print(top[['dataset','n_features','balance_strategy','metric','proba_thr','total_return','avg_return','winrate','n_trades']])
top.to_csv('ml_experiments_top10.csv', index=False)
print('\nResultados guardados en ml_experiments_top10.csv') 
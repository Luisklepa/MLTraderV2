import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

RESULTS_DIR = 'experiment_results'

# Buscar el mejor experimento con dos clases
from ml_experiment_analysis import find_best_experiment_with_two_classes, load_experiment_data

if __name__ == '__main__':
    best_file, best_auc = find_best_experiment_with_two_classes()
    if best_file is None:
        print('No se encontraron experimentos con al menos dos clases en el test set.')
        exit(0)
    print(f'Mejor experimento (con dos clases): {best_file} (AUC={best_auc:.4f})')
    base_name = os.path.basename(best_file).replace('summary_', '').replace('.txt', '')
    report, importances, df_fp, df_fn, df_test = load_experiment_data(base_name)
    # Cargar probabilidades y verdaderos
    y_true = df_test['y_true'].values
    y_prob = df_test['y_prob'].values
    thresholds = np.arange(0.1, 0.91, 0.05)
    precisions, recalls, f1s, n_signals = [], [], [], []
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        n_signals.append(y_pred.sum())
    # Graficar
    plt.figure(figsize=(10,6))
    plt.plot(thresholds, precisions, label='Precision', marker='o')
    plt.plot(thresholds, recalls, label='Recall', marker='o')
    plt.plot(thresholds, f1s, label='F1-score', marker='o')
    plt.plot(thresholds, n_signals, label='Nº Señales', marker='o', linestyle='--', color='gray')
    plt.xlabel('Threshold de probabilidad')
    plt.ylabel('Métrica')
    plt.title(f'Tuning de Threshold - {base_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/threshold_tuning_{base_name}.png', dpi=200)
    plt.show()
    # Guardar resultados
    df_thr = pd.DataFrame({'threshold': thresholds, 'precision': precisions, 'recall': recalls, 'f1': f1s, 'n_signals': n_signals})
    df_thr.to_csv(f'{RESULTS_DIR}/threshold_tuning_{base_name}.csv', index=False)
    print('¡Listo! Resultados y gráfico guardados en experiment_results/.') 
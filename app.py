import glob
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix

from backtest.robustness_metrics import calculate_robustness_metrics
from backtest.visualization import plot_backtest_results
from backtest.walk_forward import WalkForwardAnalyzer
from core.logging_config import setup_logging
from ml.model_optimization import ModelOptimizer

setup_logging()

# -----------------------------------------------------------------------------
# PIPELINE DE USO (evaluación de coherencia)
# -----------------------------------------------------------------------------
# Flujo esperado: 0 → 1 → 2 → 3 → [4] → 5 → 6
# - 0: Crear dataset (opcional si ya hay datos): descargar raw Binance → preparar ML (L/S).
# - 1: Seleccionar datasets Long y Short (o combinados). Requiere data/processed/*.csv.
# - 2: Entrenar y validar (Debug = holdout 80/20, WF = CV temporal). Salida: modelos guardados.
# - 3: Comparar modelos guardados (métricas, features, gráficos). Requiere al menos 2 modelos.
# - 4: Simulación de trading (en desarrollo): usar modelos + thresholds para simular PnL.
# - 5: Recordatorio de que los modelos se guardan automáticamente en 2.
# - 6: Logs de entrenamiento (experiment_results/, logs/).
# Coherencia: el flujo lleva a "tener modelos Long/Short validados" y poder compararlos.
# -----------------------------------------------------------------------------

st.set_page_config(page_title="ML Trading App", layout="wide")
st.title("ML Trading App: Modelos Long y Short Separados")

# Asegurar que existan directorios necesarios
for d in ("data/raw", "data/processed", "models", "ml_pipeline_results/plots", "experiment_results", "logs"):
    Path(d).mkdir(parents=True, exist_ok=True)

# --- Selector de modo ---
st.sidebar.header("Modo de entrenamiento y validación")
modo = st.sidebar.radio("Selecciona el modo:", ["Debug/Ajuste", "Walk-Forward (Robustez)"])

# Parámetros Walk-Forward (visibles cuando el modo es WF)
st.sidebar.header("Walk-Forward (solo si usas modo WF)")
train_size = st.sidebar.number_input("Tamaño ventana entrenamiento", 100, 2000, 500, 50, key="wf_train")
test_size = st.sidebar.number_input("Tamaño ventana test", 20, 500, 100, 10, key="wf_test")
gap = st.sidebar.number_input("Gap entre ventanas", 0, 50, 0, 1, key="wf_gap")
expanding = st.sidebar.checkbox("Ventana expandible", value=False, key="wf_expanding")


# --- Cargar configuración y pipeline ---
def cargar_config():
    archivos = []
    for f in os.listdir("config"):
        if f.endswith(".yaml"):
            with open(os.path.join("config", f)) as file:
                contenido = file.read()
                if "model_config" in contenido:
                    archivos.append(f)
    if not archivos:
        st.error("No se encontró ningún archivo de configuración válido (con 'model_config').")
        st.stop()
    archivo = st.sidebar.selectbox("Selecciona archivo de configuración:", archivos)
    config_path = os.path.join("config", archivo)
    # Leer YAML como dict para ModelOptimizer
    import yaml

    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    return config_dict


config_dict = cargar_config()
optimizer = ModelOptimizer(config_dict)


# --- Dataset loader ---
def cargar_dataset():
    archivos = [f for f in glob.glob("data/processed/*.csv")]
    if not archivos:
        st.warning("No hay datasets ML generados en data/processed/. Por favor, crea uno primero.")
        return None
    archivo = st.selectbox("Selecciona el dataset:", archivos)
    df = pd.read_csv(archivo)
    st.write("Vista previa del dataset:", df.head())
    return df


# --- Sidebar (parámetros manuales, opcional) ---
st.sidebar.header("Configuración avanzada (opcional)")
# Determinar el máximo de features dinámicamente si hay dataset cargado
try:
    archivos_procesados = [f for f in glob.glob("data/processed/*.csv")]
    if archivos_procesados:
        df_temp = pd.read_csv(archivos_procesados[0])
        n_features_max = max(5, min(50, df_temp.select_dtypes(include=["number"]).shape[1] - 2))  # -2 por los targets
    else:
        n_features_max = 50
except Exception as e:
    logging.getLogger(__name__).warning("Could not determine max features dynamically: %s", e)
    n_features_max = 50
min_feature_importance = st.sidebar.number_input("Min feature importance", 0.0, 0.1, 0.01, 0.001)
max_features = st.sidebar.number_input("Max features", 1, n_features_max, min(50, n_features_max), 1)
n_trials = st.sidebar.number_input("Nº de trials Optuna", 10, 500, 50, 1)
cv_folds = st.sidebar.number_input("Folds CV", 2, 10, 5, 1)


# Actualizar config del optimizador si el usuario cambia algo
def update_optimizer_config():
    optimizer.feature_importance_threshold = min_feature_importance
    optimizer.max_features = max_features
    optimizer.n_trials = n_trials
    optimizer.cv_folds = cv_folds


update_optimizer_config()


def get_X_y(df: pd.DataFrame, side: str):
    """Obtiene X e y de un dataset Long/Short o combinado. side in ('long','short')."""
    drop_cols = ["target_long", "target_short", "position_long", "position_short", "position", "timestamp", "datetime"]
    if side == "long" and "target_long" in df.columns:
        y = df["target_long"]
        X = df.drop(columns=[c for c in drop_cols + ["target"] if c in df.columns])
    elif side == "short" and "target_short" in df.columns:
        y = df["target_short"]
        X = df.drop(columns=[c for c in drop_cols + ["target"] if c in df.columns])
    else:
        if "target" not in df.columns:
            raise ValueError("Dataset sin columna 'target' ni target_long/target_short")
        y = df["target"]
        X = df.drop(columns=[c for c in ["target", "timestamp", "datetime"] if c in df.columns])
    return X, y


# --- Main ---
st.header("0. Creación de Dataset desde Binance")
with st.expander("Crear nuevo dataset ML a partir de datos históricos", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        symbol = st.selectbox("Símbolo", ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT"])
        interval = st.selectbox("Timeframe", ["15m", "1h", "4h", "1d"])
        days = st.number_input("Días de histórico a descargar", 1, 365, 90)
        raw_file = st.text_input("Archivo de salida (datos crudos)", f"data/raw/{symbol.lower()}_{interval}.csv")
        if st.button("Descargar datos históricos"):
            cmd = [
                sys.executable,
                "scripts/download_data.py",
                "--symbol",
                symbol,
                "--interval",
                interval,
                "--days",
                str(days),
                "--output-file",
                raw_file,
            ]
            with st.spinner("Descargando datos..."):
                result = subprocess.run(cmd, capture_output=True, text=True)
            st.subheader("Log de la descarga:")
            st.code(result.stdout + "\n" + result.stderr)
            if result.returncode == 0:
                try:
                    df_raw = pd.read_csv(raw_file)
                    st.success(f"Datos descargados correctamente en {raw_file}")
                    st.write(f"Total de velas descargadas: {len(df_raw)}")
                    st.write(f"Rango de fechas: {df_raw['timestamp'].min()} a {df_raw['timestamp'].max()}")
                    st.write(df_raw.head())
                except Exception as e:
                    st.error(f"No se pudo leer el archivo generado: {e}")
            else:
                st.error("Error al descargar datos. Revisa el log arriba.")
    with col2:
        # Botón para refrescar archivos
        if "refresh_files" not in st.session_state:
            st.session_state["refresh_files"] = 0
        if st.button("Refrescar archivos"):
            st.session_state["refresh_files"] += 1
        archivos_crudos = glob.glob("data/raw/*.csv")
        # Forzar refresco usando key dependiente del contador
        input_file = st.selectbox(
            "Archivo de entrada (datos crudos)", archivos_crudos, key=f"input_file_{st.session_state['refresh_files']}"
        )
        # Usar solo el nombre base para el archivo de salida
        if input_file:
            base = os.path.basename(input_file).replace(".csv", "")
            output_sugerido = f"data/processed/{base}_ml_opt.csv"
        else:
            output_sugerido = "data/processed/ml_dataset_opt.csv"
        output_file = st.text_input("Archivo de salida (dataset ML)", output_sugerido)
        if st.button("Preparar dataset ML (optimización automática)"):
            if not input_file or not output_file:
                st.warning("Debes especificar los archivos de entrada y salida.")
            else:
                # Generar nombres para long y short automáticamente
                base = os.path.splitext(output_file)[0]
                output_long = base + "_L.csv"
                output_short = base + "_S.csv"
                cmd = [
                    sys.executable,
                    "scripts/prepare_ml_dataset.py",
                    "--input-file",
                    input_file,
                    "--output-long",
                    output_long,
                    "--output-short",
                    output_short,
                ]
                with st.spinner("Optimizando window y threshold y preparando datasets ML..."):
                    result = subprocess.run(cmd, capture_output=True, text=True)
                st.subheader("Log de la optimización y preparación:")
                st.code(result.stdout + "\n" + result.stderr)
                if result.returncode == 0:
                    try:
                        df_long = pd.read_csv(output_long)
                        df_short = pd.read_csv(output_short)
                        st.success(f"Datasets ML generados:\nLONG: {output_long}\nSHORT: {output_short}")
                        st.write("LONG (primeras filas):", df_long.head())
                        st.write("SHORT (primeras filas):", df_short.head())
                    except Exception as e:
                        st.error(f"No se pudo leer alguno de los archivos generados: {e}")
                else:
                    st.error("Error al preparar datasets ML. Revisa el log arriba.")

st.header("1. Selección de Datasets")
all_files = [f.name for f in Path("data/processed").glob("*.csv")]
if not all_files:
    st.warning(
        "No hay datasets ML en `data/processed/`. Crea datos en la **sección 0** (expandir «Crear nuevo dataset ML»)."
    )
    st.info(
        "Flujo: Descargar datos históricos → Refrescar archivos → Archivo de entrada (crudos) → Preparar dataset ML."
    )
    st.stop()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Dataset para Modelo Long")
    dataset_long = st.selectbox("Selecciona el dataset LONG o COMBINADO", all_files, key="long_dataset")
    df_long = pd.read_csv(f"data/processed/{dataset_long}")
    st.write(df_long.head())

with col2:
    st.subheader("Dataset para Modelo Short")
    dataset_short = st.selectbox("Selecciona el dataset SHORT o COMBINADO", all_files, key="short_dataset")
    df_short = pd.read_csv(f"data/processed/{dataset_short}")
    st.write(df_short.head())

# --- Entrenamiento de modelos ---
st.header("2. Entrenamiento y Validación de Modelos")

if modo == "Debug/Ajuste":
    col_btn, col_prog = st.columns([2, 1])
    with col_btn:
        run_debug = st.button("Entrenar y validar (debug)")
    with col_prog:
        progress = st.empty()
    if run_debug:
        with col_prog:
            bar = st.progress(0, text="Iniciando entrenamiento... 🧠")
        st.subheader("Entrenamiento y validación rápida (holdout temporal)")
        # Split temporal 80/20
        split_idx_long = int(len(df_long) * 0.8)
        X_long, y_long = get_X_y(df_long, "long")
        X_train_long, X_test_long = X_long.iloc[:split_idx_long], X_long.iloc[split_idx_long:]
        y_train_long, y_test_long = y_long.iloc[:split_idx_long], y_long.iloc[split_idx_long:]
        optimizer_long = ModelOptimizer(config_dict)
        bar.progress(10, text="Seleccionando features (LONG)... ✨")
        X_train_long_sel, selected_features_long = optimizer_long.optimize_feature_selection(X_train_long, y_train_long)
        X_test_long_sel = X_test_long[selected_features_long]
        bar.progress(30, text="Optimizando hiperparámetros (LONG)... 🔎")
        opt_result_long = optimizer_long.optimize_hyperparameters(X_train_long_sel, y_train_long)
        best_params_long = opt_result_long["best_params"]
        best_threshold_long = opt_result_long["best_threshold"]
        bar.progress(50, text="Entrenando modelo (LONG)... 🚀")
        model_long = xgb.XGBClassifier(**best_params_long, eval_metric="logloss")
        model_long.fit(X_train_long_sel, y_train_long)
        y_pred_long = (model_long.predict_proba(X_test_long_sel)[:, 1] > best_threshold_long).astype(int)
        metrics_long = classification_report(y_test_long, y_pred_long, output_dict=True)
        bar.progress(60, text="Analizando importancia de features (LONG)... 🧬")
        fi_long = optimizer_long.analyze_feature_importance(model_long, selected_features_long)
        st.success("Modelo Long entrenado y validado (debug)!")
        st.write("Mejores features:", selected_features_long)
        st.write("Mejores hiperparámetros:", best_params_long)
        st.write("Threshold óptimo:", best_threshold_long)
        st.write("Métricas (test):")
        st.json(metrics_long)
        st.write("Matriz de confusión (test):")
        st.write(confusion_matrix(y_test_long, y_pred_long))
        st.write("Importancia de features (test):")
        st.dataframe(fi_long.head(15))
        # --- Visualización avanzada LONG ---
        st.markdown("### Gráficos de performance modelo LONG")
        equity_long = pd.Series(
            (model_long.predict_proba(X_test_long_sel)[:, 1] > best_threshold_long).cumsum(), name="Equity"
        )
        returns_long = pd.Series(model_long.predict_proba(X_test_long_sel)[:, 1] - 0.5, name="Returns")
        results_long = pd.DataFrame({"equity": equity_long, "returns": returns_long})
        results_long["drawdown"] = results_long["equity"] - results_long["equity"].cummax()
        if "position_long" in df_long.columns:
            results_long["position"] = df_long["position_long"].iloc[split_idx_long:].reset_index(drop=True)
        elif "position" in df_long.columns:
            results_long["position"] = df_long["position"].iloc[split_idx_long:].reset_index(drop=True)
        else:
            results_long["position"] = 0
        plot_backtest_results(results_long, "ml_pipeline_results/plots/debug_long.html")
        try:
            with open("ml_pipeline_results/plots/debug_long.html", encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=600)
        except Exception as e:
            st.warning(f"No se pudo mostrar el gráfico: {e}")
        # Métricas de robustez LONG
        st.markdown("#### Métricas de robustez (LONG)")
        metrics_long_robust = calculate_robustness_metrics(
            [
                {
                    "pnl_total": returns_long.sum(),
                    "win_rate": metrics_long["1"]["precision"],
                    "total_trades": len(returns_long),
                    "max_drawdown": results_long["drawdown"].min(),
                }
            ]
        )
        st.json(metrics_long_robust)
        optimizer_long.save_model(model_long, selected_features_long, metrics_long, threshold=best_threshold_long)
        # --- Short ---
        split_idx_short = int(len(df_short) * 0.8)
        X_short, y_short = get_X_y(df_short, "short")
        X_train_short, X_test_short = X_short.iloc[:split_idx_short], X_short.iloc[split_idx_short:]
        y_train_short, y_test_short = y_short.iloc[:split_idx_short], y_short.iloc[split_idx_short:]
        optimizer_short = ModelOptimizer(config_dict)
        bar.progress(80, text="Seleccionando features (SHORT)... ✨")
        X_train_short_sel, selected_features_short = optimizer_short.optimize_feature_selection(
            X_train_short, y_train_short
        )
        X_test_short_sel = X_test_short[selected_features_short]
        bar.progress(85, text="Optimizando hiperparámetros (SHORT)... 🔎")
        opt_result_short = optimizer_short.optimize_hyperparameters(X_train_short_sel, y_train_short)
        best_params_short = opt_result_short["best_params"]
        best_threshold_short = opt_result_short["best_threshold"]
        bar.progress(90, text="Entrenando modelo (SHORT)... 🚀")
        model_short = xgb.XGBClassifier(**best_params_short, eval_metric="logloss")
        model_short.fit(X_train_short_sel, y_train_short)
        y_pred_short = (model_short.predict_proba(X_test_short_sel)[:, 1] > best_threshold_short).astype(int)
        metrics_short = classification_report(y_test_short, y_pred_short, output_dict=True)
        bar.progress(95, text="Analizando importancia de features (SHORT)... 🧬")
        fi_short = optimizer_short.analyze_feature_importance(model_short, selected_features_short)
        st.success("Modelo Short entrenado y validado (debug)!")
        st.write("Mejores features:", selected_features_short)
        st.write("Mejores hiperparámetros:", best_params_short)
        st.write("Threshold óptimo:", best_threshold_short)
        st.write("Métricas (test):")
        st.json(metrics_short)
        st.write("Matriz de confusión (test):")
        st.write(confusion_matrix(y_test_short, y_pred_short))
        st.write("Importancia de features (test):")
        st.dataframe(fi_short.head(15))
        st.markdown("### Gráficos de performance modelo SHORT")
        equity_short = pd.Series(
            (model_short.predict_proba(X_test_short_sel)[:, 1] > best_threshold_short).cumsum(), name="Equity"
        )
        returns_short = pd.Series(model_short.predict_proba(X_test_short_sel)[:, 1] - 0.5, name="Returns")
        results_short = pd.DataFrame({"equity": equity_short, "returns": returns_short})
        results_short["drawdown"] = results_short["equity"] - results_short["equity"].cummax()
        if "position_short" in df_short.columns:
            results_short["position"] = df_short["position_short"].iloc[split_idx_short:].reset_index(drop=True)
        elif "position" in df_short.columns:
            results_short["position"] = df_short["position"].iloc[split_idx_short:].reset_index(drop=True)
        else:
            results_short["position"] = 0
        plot_backtest_results(results_short, "ml_pipeline_results/plots/debug_short.html")
        try:
            with open("ml_pipeline_results/plots/debug_short.html", encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=600)
        except Exception as e:
            st.warning(f"No se pudo mostrar el gráfico: {e}")
        st.markdown("#### Métricas de robustez (SHORT)")
        metrics_short_robust = calculate_robustness_metrics(
            [
                {
                    "pnl_total": returns_short.sum(),
                    "win_rate": metrics_short["1"]["precision"],
                    "total_trades": len(returns_short),
                    "max_drawdown": results_short["drawdown"].min(),
                }
            ]
        )
        st.json(metrics_short_robust)
        optimizer_short.save_model(model_short, selected_features_short, metrics_short, threshold=best_threshold_short)
        bar.progress(100, text="¡Entrenamiento y validación completados! 🎉")
        bar.empty()

if modo == "Walk-Forward (Robustez)":
    col_btn, col_prog = st.columns([2, 1])
    with col_btn:
        run_wf = st.button("Entrenar y validar (walk-forward)")
    with col_prog:
        progress = st.empty()
    if run_wf:
        with col_prog:
            bar = st.progress(0, text="Iniciando walk-forward... 🧠")
        st.subheader("Validación cruzada temporal (walk-forward)")
        # --- Long ---
        st.write("\n--- Walk-forward Modelo Long ---")
        wf_long = WalkForwardAnalyzer(df_long, train_size=train_size, test_size=test_size, gap=gap, expanding=expanding)
        st.info("Walk-forward para ML: Entrenando y validando modelo en cada ventana...")
        wf_results_long = []
        total_long = len(wf_long.windows)
        for i, window in enumerate(wf_long.windows, 1):
            train_data = df_long[window["train_start"] : window["train_end"]].copy()
            test_data = df_long[window["test_start"] : window["test_end"]].copy()
            if train_data.empty or test_data.empty:
                continue
            X_train, y_train = get_X_y(train_data, "long")
            X_test, y_test = get_X_y(test_data, "long")
            optimizer_wf = ModelOptimizer(config_dict)
            X_train_sel, selected_features = optimizer_wf.optimize_feature_selection(X_train, y_train)
            X_test_sel = X_test[selected_features]
            opt_result = optimizer_wf.optimize_hyperparameters(X_train_sel, y_train)
            best_params = opt_result["best_params"]
            best_threshold = opt_result["best_threshold"]
            model = xgb.XGBClassifier(**best_params, eval_metric="logloss")
            model.fit(X_train_sel, y_train)
            y_pred = (model.predict_proba(X_test_sel)[:, 1] > best_threshold).astype(int)
            metrics = classification_report(y_test, y_pred, output_dict=True)
            wf_results_long.append(metrics["1"]["f1-score"])
            bar.progress(int(i / total_long * 50), text=f"LONG: Ventana {i}/{total_long} 🚀")
        # Guardar último modelo LONG del walk-forward
        optimizer_wf.save_model(model, selected_features, metrics, threshold=best_threshold)
        st.write(f"F1-score promedio (LONG, test): {pd.Series(wf_results_long).mean():.4f}")
        st.line_chart(wf_results_long, use_container_width=True)
        # --- Visualización avanzada y robustez LONG ---
        st.markdown("### Gráficos de performance modelo LONG (Walk-Forward)")
        equity_long = pd.Series(wf_results_long).cumsum()
        returns_long = pd.Series(wf_results_long) - 0.5
        results_long = pd.DataFrame({"equity": equity_long, "returns": returns_long})
        results_long["drawdown"] = results_long["equity"] - results_long["equity"].cummax()
        if "position" not in results_long.columns:
            results_long["position"] = 0
        plot_backtest_results(results_long, "ml_pipeline_results/plots/wf_long.html")
        try:
            with open("ml_pipeline_results/plots/wf_long.html", encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=600)
        except Exception as e:
            st.warning(f"No se pudo mostrar el gráfico LONG: {e}")
        st.markdown("#### Métricas de robustez (LONG)")
        metrics_long = calculate_robustness_metrics(
            [
                {
                    "pnl_total": returns_long.sum(),
                    "win_rate": float(pd.Series(wf_results_long).mean()),
                    "total_trades": len(returns_long),
                    "max_drawdown": results_long["drawdown"].min(),
                }
            ]
        )
        st.json(metrics_long)
        # --- Short ---
        st.write("\n--- Walk-forward Modelo Short ---")
        wf_short = WalkForwardAnalyzer(
            df_short, train_size=train_size, test_size=test_size, gap=gap, expanding=expanding
        )
        wf_results_short = []
        total_short = len(wf_short.windows)
        for i, window in enumerate(wf_short.windows, 1):
            train_data = df_short[window["train_start"] : window["train_end"]].copy()
            test_data = df_short[window["test_start"] : window["test_end"]].copy()
            if train_data.empty or test_data.empty:
                continue
            X_train, y_train = get_X_y(train_data, "short")
            X_test, y_test = get_X_y(test_data, "short")
            optimizer_wf = ModelOptimizer(config_dict)
            X_train_sel, selected_features = optimizer_wf.optimize_feature_selection(X_train, y_train)
            X_test_sel = X_test[selected_features]
            opt_result = optimizer_wf.optimize_hyperparameters(X_train_sel, y_train)
            best_params = opt_result["best_params"]
            best_threshold = opt_result["best_threshold"]
            model = xgb.XGBClassifier(**best_params, eval_metric="logloss")
            model.fit(X_train_sel, y_train)
            y_pred = (model.predict_proba(X_test_sel)[:, 1] > best_threshold).astype(int)
            metrics = classification_report(y_test, y_pred, output_dict=True)
            wf_results_short.append(metrics["1"]["f1-score"])
            bar.progress(50 + int(i / total_short * 50), text=f"SHORT: Ventana {i}/{total_short} 🚀")
        # Guardar último modelo SHORT del walk-forward
        optimizer_wf.save_model(model, selected_features, metrics, threshold=best_threshold)
        st.write(f"F1-score promedio (SHORT, test): {pd.Series(wf_results_short).mean():.4f}")
        st.line_chart(wf_results_short, use_container_width=True)
        # --- Visualización avanzada y robustez SHORT ---
        st.markdown("### Gráficos de performance modelo SHORT (Walk-Forward)")
        equity_short = pd.Series(wf_results_short).cumsum()
        returns_short = pd.Series(wf_results_short) - 0.5
        results_short = pd.DataFrame({"equity": equity_short, "returns": returns_short})
        results_short["drawdown"] = results_short["equity"] - results_short["equity"].cummax()
        if "position" not in results_short.columns:
            results_short["position"] = 0
        plot_backtest_results(results_short, "ml_pipeline_results/plots/wf_short.html")
        try:
            with open("ml_pipeline_results/plots/wf_short.html", encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=600)
        except Exception as e:
            st.warning(f"No se pudo mostrar el gráfico SHORT: {e}")
        st.markdown("#### Métricas de robustez (SHORT)")
        metrics_short = calculate_robustness_metrics(
            [
                {
                    "pnl_total": returns_short.sum(),
                    "win_rate": float(pd.Series(wf_results_short).mean()),
                    "total_trades": len(returns_short),
                    "max_drawdown": results_short["drawdown"].min(),
                }
            ]
        )
        st.json(metrics_short)
        bar.progress(100, text="¡Walk-forward completado! 🎉")
        bar.empty()

# --- Visualización y comparación ---
st.header("3. Comparación y Análisis de Modelos Guardados")
model_files = sorted(glob.glob("models/model_*.pkl"))
metadata_files = sorted(glob.glob("models/metadata_*.json"))

if len(model_files) < 2:
    st.info("Debes entrenar y guardar al menos dos modelos para comparar.")
else:
    colA, colB = st.columns(2)
    with colA:
        st.subheader("Modelo A")
        modelA_file = st.selectbox("Selecciona modelo A", model_files, key="modelA")
        metaA_file = modelA_file.replace(".pkl", ".json").replace("model_", "metadata_")
        if Path(metaA_file).exists():
            with open(metaA_file, encoding="utf-8") as f:
                metaA = json.load(f)
            st.write(f"**Timestamp:** {metaA['timestamp']}")
            st.write(f"**Features:** {metaA['features']}")
            st.write("**Métricas:**")
            st.json(metaA["metrics"])
            # Gráfico de performance si existe
            plotA = f"ml_pipeline_results/plots/{metaA['timestamp']}_equity.html"
            if Path(plotA).exists():
                with open(plotA, encoding="utf-8") as f:
                    st.components.v1.html(f.read(), height=400)
            else:
                st.info("No hay gráfico de performance guardado para este modelo.")
        else:
            st.warning("No se encontró metadata para este modelo.")
    with colB:
        st.subheader("Modelo B")
        modelB_file = st.selectbox("Selecciona modelo B", model_files, key="modelB")
        metaB_file = modelB_file.replace(".pkl", ".json").replace("model_", "metadata_")
        if Path(metaB_file).exists():
            with open(metaB_file, encoding="utf-8") as f:
                metaB = json.load(f)
            st.write(f"**Timestamp:** {metaB['timestamp']}")
            st.write(f"**Features:** {metaB['features']}")
            st.write("**Métricas:**")
            st.json(metaB["metrics"])
            plotB = f"ml_pipeline_results/plots/{metaB['timestamp']}_equity.html"
            if Path(plotB).exists():
                with open(plotB, encoding="utf-8") as f:
                    st.components.v1.html(f.read(), height=400)
            else:
                st.info("No hay gráfico de performance guardado para este modelo.")
        else:
            st.warning("No se encontró metadata para este modelo.")

st.header("4. Simulación de Trading")
st.caption("Simula PnL usando dos modelos guardados (Long y Short), con capital inicial y comisión por operación.")

sim_model_files = sorted(glob.glob("models/model_*.pkl"))
if not sim_model_files:
    st.warning("No hay modelos guardados. Entrena modelos en la **sección 2**.")
else:
    col_sim1, col_sim2 = st.columns(2)
    with col_sim1:
        model_long_sim = st.selectbox("Modelo Long", sim_model_files, key="sim_long")
    with col_sim2:
        model_short_sim = st.selectbox("Modelo Short", sim_model_files, key="sim_short")

    sim_dataset = st.selectbox("Dataset para simular (mismo formato que entrenamiento)", all_files, key="sim_dataset")
    capital = st.number_input("Capital inicial", 100, 100000, 10000, key="sim_capital")
    tam_pos = st.number_input("Tamaño de posición (fracción del capital por trade)", 0.1, 10.0, 1.0, key="sim_tam")
    comision_pct = st.number_input("Comisión por trade (%)", 0.0, 1.0, 0.1, step=0.01, key="sim_comision")

    if st.button("Simular trading"):
        try:
            # Cargar modelos y metadatos (features + threshold)
            model_long, feats_long, _, th_long = optimizer.load_model_by_path(Path(model_long_sim))
            model_short, feats_short, _, th_short = optimizer.load_model_by_path(Path(model_short_sim))

            df_sim = pd.read_csv(f"data/processed/{sim_dataset}")
            # Features presentes en el dataset
            feats_ok_long = [f for f in feats_long if f in df_sim.columns]
            feats_ok_short = [f for f in feats_short if f in df_sim.columns]
            if len(feats_ok_long) != len(feats_long) or len(feats_ok_short) != len(feats_short):
                missing_long = set(feats_long) - set(df_sim.columns)
                missing_short = set(feats_short) - set(df_sim.columns)
                st.error(
                    f"El dataset no tiene todas las features. Faltan para Long: {missing_long}. Faltan para Short: {missing_short}."
                )
                st.stop()

            X_long = df_sim[feats_long]
            X_short = df_sim[feats_short]
            prob_long = model_long.predict_proba(X_long)[:, 1]
            prob_short = model_short.predict_proba(X_short)[:, 1]
            signal_long = (prob_long > th_long).astype(int)
            signal_short = (prob_short > th_short).astype(int)
            # Posición: +1 long, -1 short, 0 si ambos o ninguno (evitar conflicto)
            position = np.where(signal_long & ~signal_short, 1, np.where(signal_short & ~signal_long, -1, 0))

            # Retorno por barra: preferir columna existente
            if "future_return" in df_sim.columns:
                bar_return = df_sim["future_return"].values
            elif "future_return_long" in df_sim.columns and "future_return_short" in df_sim.columns:
                bar_return = np.where(
                    position == 1,
                    df_sim["future_return_long"].values,
                    np.where(position == -1, -df_sim["future_return_short"].values, 0.0),
                )
            else:
                bar_return = np.zeros(len(df_sim))
                st.info("Dataset sin columna de retorno (future_return). Mostrando solo curva de posiciones.")

            notional = capital * tam_pos
            commission_per_trade = notional * (comision_pct / 100)
            n_trades = np.abs(np.diff(position, prepend=position[0])).sum()
            trade_pnl = (
                position * bar_return * notional
                - (np.abs(np.diff(position, prepend=position[0])) > 0) * commission_per_trade
            )
            equity = capital + np.cumsum(trade_pnl)

            st.success("Simulación completada.")
            st.metric("Capital final", f"{equity[-1]:,.2f}")
            st.metric("Retorno total (%)", f"{(equity[-1] / capital - 1) * 100:.2f}%")
            st.metric("Nº de trades", int(n_trades))
            st.metric("Drawdown máximo", f"{(equity - np.maximum.accumulate(equity)).min():,.2f}")

            st.subheader("Curva de equity")
            st.line_chart(pd.DataFrame({"equity": equity}))
            st.subheader("Posiciones (muestra)")
            st.dataframe(
                pd.DataFrame({"position": position, "prob_long": prob_long, "prob_short": prob_short}).head(100)
            )
        except FileNotFoundError as e:
            st.error(f"No se encontró modelo o metadata: {e}")
        except Exception as e:
            logging.getLogger(__name__).exception("Error en simulación")
            st.error(f"Error en simulación: {e}")

st.header("5. Guardar modelos")
st.info("Los modelos se guardan automáticamente tras la optimización y entrenamiento.")

st.caption("Desarrollado con ❤️ usando Streamlit, XGBoost y Optuna.")

# --- Sección de logs de entrenamiento ---
st.header("6. Logs de Entrenamiento y Validación")
log_files = sorted(glob.glob("experiment_results/*.txt") + glob.glob("logs/*.log"), reverse=True)
if log_files:
    log_to_show = st.selectbox("Selecciona un log para ver detalles:", log_files)
    with open(log_to_show, encoding="utf-8", errors="replace") as f:
        st.text_area(f"Contenido de {log_to_show}", f.read(), height=400)
else:
    st.info("No se encontraron logs de entrenamiento recientes en experiment_results/ o logs/.")

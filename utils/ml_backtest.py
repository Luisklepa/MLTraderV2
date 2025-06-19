"""
Script para realizar backtesting del modelo ML.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import glob
import os
from xgboost import XGBClassifier
from ml_feature_pipeline import (
    MLFeaturePipeline,
    add_advanced_momentum_features,
    add_advanced_volatility_features,
    add_cross_features,
    add_advanced_cross_features,
    add_anti_fallo_features
)
warnings.filterwarnings('ignore')

class MLBacktester:
    def __init__(self, price_data='btcusdt_prices.csv', model_path=None, threshold=0.45):
        self.price_data = price_data
        self.model_path = model_path or self.get_latest_model()
        self.threshold = threshold
        self.model = None
        self.scaler = StandardScaler()
        self.trades = []
        self.equity_curve = []
        
    def get_latest_model(self):
        """Obtiene el modelo más reciente"""
        files = glob.glob('xgboost_model_*.pkl')
        if not files:
            raise ValueError("No se encontró ningún modelo entrenado.")
        latest = max(files, key=os.path.getctime)
        print(f"Usando modelo: {latest}")
        return latest
    
    def load_model(self):
        """Carga el modelo y metadata"""
        print("Cargando modelo...")
        self.model = joblib.load(self.model_path)
        metadata_path = self.model_path.replace('xgboost_model', 'model_metadata')
        if os.path.exists(metadata_path):
            self.metadata = joblib.load(metadata_path)
            print("Metadata cargada.")
        else:
            print("No se encontró metadata del modelo.")
    
    def prepare_features(self, df):
        """Prepara features para el modelo"""
        from ml_feature_pipeline import MLFeaturePipeline
        
        print("Preparando features...")
        pipeline = MLFeaturePipeline()
        df = pipeline.generate_technical_features(df)
        df = pipeline.generate_momentum_features(df)
        df = add_advanced_momentum_features(df)
        df = pipeline.generate_volatility_features(df)
        df = add_advanced_volatility_features(df)
        df = add_cross_features(df, clip_value=1e6, dropna=False, verbose=True)
        df = add_advanced_cross_features(df)
        df = add_anti_fallo_features(df)
        df = pipeline.generate_pattern_features(df)
        df = pipeline.generate_market_structure_features(df)
        df = pipeline.generate_temporal_features(df)
        df = pipeline.generate_contextual_features(df)
        df = pipeline.create_lag_features(df)
        
        # Excluir columnas no numéricas y de precio/volumen
        exclude_cols = ['datetime', 'timestamp', 'open_time', 'open', 'high', 'low', 'close', 'volume']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Limpiar y escalar features
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        df[feature_cols] = df[feature_cols].fillna(method='ffill').fillna(0)
        df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        
        return df, feature_cols
    
    def run_backtest(self, initial_capital=10000, risk_per_trade=0.02, stop_loss=0.02):
        """Ejecuta el backtesting"""
        print("\n=== INICIANDO BACKTEST ===")
        
        # Cargar datos y modelo
        print("Cargando datos...")
        df = pd.read_csv(self.price_data)
        print(f"Datos cargados: {len(df)} registros")
        
        print("Cargando modelo...")
        self.load_model()
        
        # Preparar features
        print("Preparando features...")
        df, feature_cols = self.prepare_features(df)
        print(f"Features preparados: {len(feature_cols)}")
        
        # Generar predicciones
        print("Generando predicciones...")
        X = df[feature_cols]
        print(f"Shape de X: {X.shape}")
        df['prediction_proba'] = self.model.predict_proba(X)[:, 1]
        df['signal'] = (df['prediction_proba'] > self.threshold).astype(int)
        print(f"Señales generadas: {df['signal'].sum()}")
        
        # Simular trading
        print("\nSimulando operaciones...")
        capital = initial_capital
        position = None
        entry_price = None
        
        for i in range(1, len(df)):
            close = df['close'].iloc[i]
            signal = df['signal'].iloc[i]
            
            # Si no hay posición abierta y hay señal de compra
            if position is None and signal == 1:
                # Calcular tamaño de posición
                position_size = capital * risk_per_trade
                entry_price = close
                position = position_size / close
                
                self.trades.append({
                    'entry_date': df.index[i],
                    'entry_price': entry_price,
                    'position_size': position_size,
                    'type': 'long'
                })
                
            # Si hay posición abierta, revisar si cerrar
            elif position is not None:
                # Calcular P&L
                pnl = position * (close - entry_price)
                pnl_pct = (close - entry_price) / entry_price
                
                # Cerrar si:
                # 1. Stop loss hit
                # 2. Take profit hit (2x risk)
                # 3. Señal de venta
                if (pnl_pct <= -stop_loss or 
                    pnl_pct >= risk_per_trade * 2 or 
                    signal == 0):
                    
                    capital += pnl
                    self.trades[-1].update({
                        'exit_date': df.index[i],
                        'exit_price': close,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct
                    })
                    position = None
                    entry_price = None
            
            self.equity_curve.append(capital)
        
        print(f"Trades realizados: {len(self.trades)}")
        print(f"Capital final: ${capital:.2f}")
        
        # Análisis de resultados
        self.analyze_results(df)
    
    def analyze_results(self, df):
        """Analiza resultados del backtest"""
        print("\n=== RESULTADOS DEL BACKTEST ===")
        
        # Estadísticas de trades
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            win_rate = winning_trades / total_trades
            
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean()
            avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean()
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            
            print(f"\nTotal trades: {total_trades}")
            print(f"Win rate: {win_rate:.2%}")
            print(f"Profit factor: {profit_factor:.2f}")
            print(f"Average win: ${avg_win:.2f}")
            print(f"Average loss: ${avg_loss:.2f}")
            
            # Equity curve
            equity_curve = pd.Series(self.equity_curve)
            total_return = (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]
            max_drawdown = (equity_curve / equity_curve.cummax() - 1).min()
            
            print(f"\nTotal return: {total_return:.2%}")
            print(f"Max drawdown: {max_drawdown:.2%}")
            print(f"Return/Drawdown ratio: {abs(total_return/max_drawdown):.2f}")
            
            # Gráficos
            plt.figure(figsize=(15, 10))
            
            # Equity curve
            plt.subplot(2, 1, 1)
            plt.plot(equity_curve)
            plt.title('Equity Curve')
            plt.grid(True)
            
            # Distribución de retornos
            plt.subplot(2, 1, 2)
            trades_df['pnl_pct'].hist(bins=50)
            plt.title('Distribución de Retornos por Trade')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('backtest_results.png')
            print("\nGráficos guardados en backtest_results.png")
        else:
            print("No se realizaron trades durante el periodo.")

if __name__ == "__main__":
    try:
        # Ejecutar backtest
        print("Iniciando backtester...")
        backtester = MLBacktester(threshold=0.3)  # Usar threshold más bajo para más trades
        print("Ejecutando backtest...")
        backtester.run_backtest()
    except Exception as e:
        print(f"Error durante el backtest: {str(e)}")
        import traceback
        traceback.print_exc()
 
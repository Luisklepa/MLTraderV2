"""
Script para realizar backtesting del modelo ML.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import glob
import os
from xgboost import XGBClassifier
from utils.ml_feature_pipeline import (
    MLFeaturePipeline,
    add_advanced_momentum_features,
    add_advanced_volatility_features,
    add_cross_features,
    add_advanced_cross_features,
    add_anti_fallo_features
)
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class Position:
    """Class to track position information."""
    symbol: str
    type: str  # 'long' or 'short'
    size: float
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: Optional[float]
    pnl: float = 0.0
    unrealized_pnl: float = 0.0

class BacktestEngine:
    """Event-driven backtesting engine for ML strategies."""
    
    def __init__(
        self,
        strategy: Any,
        data: pd.DataFrame,
        reference_data: Dict[str, pd.DataFrame],
        config: Dict[str, Any]
    ):
        """Initialize the backtest engine."""
        self.strategy = strategy
        self.data = data
        self.reference_data = reference_data
        self.config = config
        
        # State variables
        self.positions: Dict[str, Position] = {}
        self.cash = config['risk_config']['initial_capital']
        self.equity = [self.cash]
        self.trades = []
        self.current_time = None
        
        # Performance tracking
        self.returns = []
        self.drawdowns = []
        self.exposure = []
        self.trade_stats = defaultdict(list)
        
        # Commission and slippage
        self.commission_rate = config['risk_config'].get('commission_rate', 0.001)
        self.slippage_rate = config['risk_config'].get('slippage_rate', 0.0005)
        
    def calculate_slippage(self, price: float, size: float, is_buy: bool) -> float:
        """Calculate slippage cost."""
        direction = 1 if is_buy else -1
        return price * (1 + direction * self.slippage_rate)
    
    def calculate_commission(self, price: float, size: float) -> float:
        """Calculate commission cost."""
        return price * size * self.commission_rate
    
    def execute_order(self, order: Dict[str, Any], current_price: float) -> Optional[Position]:
        """Execute a trade order."""
        symbol = order['symbol']
        order_type = order['type']
        size = order['size']
        
        # Apply slippage
        execution_price = self.calculate_slippage(
            current_price,
            size,
            is_buy=(order_type == 'buy')
        )
        
        # Calculate commission
        commission = self.calculate_commission(execution_price, size)
        
        # Check if we have enough cash
        cost = execution_price * size + commission
        if cost > self.cash and order_type == 'buy':
            logger.warning(f"Insufficient cash for order: {order}")
            return None
        
        # Create position
        position = Position(
            symbol=symbol,
            type='long' if order_type == 'buy' else 'short',
            size=size,
            entry_price=execution_price,
            entry_time=self.current_time,
            stop_loss=order.get('stop_loss'),
            take_profit=order.get('take_profit')
        )
        
        # Update cash
        self.cash -= cost if order_type == 'buy' else -cost
        
        # Record trade
        self.trades.append({
            'time': self.current_time,
            'symbol': symbol,
            'type': order_type,
            'size': size,
            'price': execution_price,
            'commission': commission,
            'reason': order.get('reason', 'signal')
        })
        
        return position
    
    def close_position(self, position: Position, current_price: float, reason: str) -> None:
        """Close a position and record the trade."""
        # Calculate PnL
        if position.type == 'long':
            pnl = (current_price - position.entry_price) * position.size
        else:  # short
            pnl = (position.entry_price - current_price) * position.size
        
        # Apply commission
        commission = self.calculate_commission(current_price, position.size)
        pnl -= commission
        
        # Update cash
        self.cash += pnl
        
        # Record trade
        self.trades.append({
            'time': self.current_time,
            'symbol': position.symbol,
            'type': 'sell' if position.type == 'long' else 'buy',
            'size': position.size,
            'price': current_price,
            'commission': commission,
            'pnl': pnl,
            'reason': reason,
            'duration': (self.current_time - position.entry_time).total_seconds() / 3600  # hours
        })
        
        # Update trade statistics
        self.trade_stats['pnl'].append(pnl)
        self.trade_stats['duration'].append(
            (self.current_time - position.entry_time).total_seconds() / 3600
        )
        self.trade_stats['win'].append(pnl > 0)
    
    def update_positions(self, current_bar: pd.Series) -> None:
        """Update all positions with current market data."""
        closed_positions = []
        
        for symbol, position in self.positions.items():
            # Check stop loss
            if position.type == 'long':
                if current_bar['low'] <= position.stop_loss:
                    self.close_position(position, position.stop_loss, 'stop_loss')
                    closed_positions.append(symbol)
                    continue
            else:  # short position
                if current_bar['high'] >= position.stop_loss:
                    self.close_position(position, position.stop_loss, 'stop_loss')
                    closed_positions.append(symbol)
                    continue
            
            # Check take profit
            if position.take_profit is not None:
                if position.type == 'long' and current_bar['high'] >= position.take_profit:
                    self.close_position(position, position.take_profit, 'take_profit')
                    closed_positions.append(symbol)
                    continue
                elif position.type == 'short' and current_bar['low'] <= position.take_profit:
                    self.close_position(position, position.take_profit, 'take_profit')
                    closed_positions.append(symbol)
                    continue
            
            # Update unrealized PnL
            if position.type == 'long':
                position.unrealized_pnl = (current_bar['close'] - position.entry_price) * position.size
            else:
                position.unrealized_pnl = (position.entry_price - current_bar['close']) * position.size
        
        # Remove closed positions
        for symbol in closed_positions:
            del self.positions[symbol]
    
    def update_metrics(self) -> None:
        """Update performance metrics."""
        # Calculate total equity
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        current_equity = self.cash + unrealized_pnl
        
        # Calculate return
        prev_equity = self.equity[-1]
        returns = (current_equity - prev_equity) / prev_equity
        
        # Calculate drawdown
        peak = max(self.equity)
        drawdown = (current_equity - peak) / peak
        
        # Calculate exposure
        total_exposure = sum(abs(pos.size * self.data.loc[self.current_time, 'close'])
                           for pos in self.positions.values())
        exposure = total_exposure / current_equity
        
        # Update metrics
        self.equity.append(current_equity)
        self.returns.append(returns)
        self.drawdowns.append(drawdown)
        self.exposure.append(exposure)
    
    def run(self) -> pd.DataFrame:
        """Run the backtest."""
        logger.info("Starting backtest...")
        
        results = []
        
        for idx, row in self.data.iterrows():
            self.current_time = idx
            
            # Update existing positions
            self.update_positions(row)
            
            # Get new orders from strategy
            try:
                orders = self.strategy.on_data(
                    self.data.loc[:idx].copy()
                )
            except Exception as e:
                logger.error(f"Strategy error at {idx}: {str(e)}")
                continue
            
            # Execute new orders
            for order in orders:
                if order['type'] in ['buy', 'sell']:
                    position = self.execute_order(order, row['close'])
                    if position is not None:
                        self.positions[order['symbol']] = position
            
            # Update metrics
            self.update_metrics()
            
            # Record state
            results.append({
                'timestamp': idx,
                'equity': self.equity[-1],
                'cash': self.cash,
                'returns': self.returns[-1],
                'drawdown': self.drawdowns[-1],
                'exposure': self.exposure[-1],
                'positions': len(self.positions),
                'unrealized_pnl': sum(pos.unrealized_pnl for pos in self.positions.values())
            })
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        results_df.set_index('timestamp', inplace=True)
        
        # Calculate summary statistics
        self.calculate_summary_statistics()
        
        logger.info("Backtest completed successfully")
        return results_df
    
    def calculate_summary_statistics(self) -> None:
        """Calculate and log summary statistics."""
        if not self.trades:
            logger.warning("No trades executed during backtest")
            return
        
        # Trading statistics
        total_trades = len(self.trades)
        profitable_trades = sum(1 for t in self.trades if t.get('pnl', 0) > 0)
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        # PnL statistics
        total_pnl = sum(t.get('pnl', 0) for t in self.trades)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
        
        # Risk statistics
        max_drawdown = min(self.drawdowns)
        avg_exposure = np.mean(self.exposure)
        
        # Returns statistics
        total_return = (self.equity[-1] - self.equity[0]) / self.equity[0]
        annual_return = (1 + total_return) ** (252/len(self.returns)) - 1
        volatility = np.std(self.returns) * np.sqrt(252)
        sharpe_ratio = np.mean(self.returns) / np.std(self.returns) * np.sqrt(252) if np.std(self.returns) > 0 else 0
        
        # Log statistics
        logger.info("\nBacktest Summary Statistics:")
        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"Win Rate: {win_rate:.2%}")
        logger.info(f"Total PnL: ${total_pnl:,.2f}")
        logger.info(f"Average PnL per Trade: ${avg_pnl:,.2f}")
        logger.info(f"Maximum Drawdown: {max_drawdown:.2%}")
        logger.info(f"Average Exposure: {avg_exposure:.2%}")
        logger.info(f"Total Return: {total_return:.2%}")
        logger.info(f"Annualized Return: {annual_return:.2%}")
        logger.info(f"Annualized Volatility: {volatility:.2%}")
        logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")

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
        from utils.ml_feature_pipeline import MLFeaturePipeline
        
        print("Preparando features...")
        pipeline = MLFeaturePipeline()
        
        # Crear una copia del DataFrame original
        df_original = df.copy()
        
        # Lista para seguimiento de características
        feature_sets = []
        
        # Generar características técnicas básicas
        df = pipeline.generate_technical_features(df)
        feature_sets.append(set(df.columns) - set(df_original.columns))
        print(f"Features técnicos añadidos: {len(feature_sets[-1])}")
        
        # Generar características de momentum
        df = pipeline.generate_momentum_features(df)
        feature_sets.append(set(df.columns) - set(df_original.columns) - set.union(*feature_sets))
        print(f"Features de momentum añadidos: {len(feature_sets[-1])}")
        
        # Añadir características avanzadas de momentum
        df = add_advanced_momentum_features(df)
        feature_sets.append(set(df.columns) - set(df_original.columns) - set.union(*feature_sets))
        print(f"Features avanzados de momentum añadidos: {len(feature_sets[-1])}")
        
        # Generar características de volatilidad
        df = pipeline.generate_volatility_features(df)
        feature_sets.append(set(df.columns) - set(df_original.columns) - set.union(*feature_sets))
        print(f"Features de volatilidad añadidos: {len(feature_sets[-1])}")
        
        # Añadir características avanzadas de volatilidad
        df = add_advanced_volatility_features(df)
        feature_sets.append(set(df.columns) - set(df_original.columns) - set.union(*feature_sets))
        print(f"Features avanzados de volatilidad añadidos: {len(feature_sets[-1])}")
        
        # Añadir características cruzadas
        df = add_cross_features(df, clip_value=1e6, dropna=False, verbose=True)
        feature_sets.append(set(df.columns) - set(df_original.columns) - set.union(*feature_sets))
        print(f"Features cruzados añadidos: {len(feature_sets[-1])}")
        
        # Añadir características cruzadas avanzadas
        df = add_advanced_cross_features(df)
        feature_sets.append(set(df.columns) - set(df_original.columns) - set.union(*feature_sets))
        print(f"Features cruzados avanzados añadidos: {len(feature_sets[-1])}")
        
        # Añadir características anti-fallo
        df = add_anti_fallo_features(df)
        feature_sets.append(set(df.columns) - set(df_original.columns) - set.union(*feature_sets))
        print(f"Features anti-fallo añadidos: {len(feature_sets[-1])}")
        
        # Generar características de patrones
        df = pipeline.generate_pattern_features(df)
        feature_sets.append(set(df.columns) - set(df_original.columns) - set.union(*feature_sets))
        print(f"Features de patrones añadidos: {len(feature_sets[-1])}")
        
        # Generar características de estructura de mercado
        df = pipeline.generate_market_structure_features(df)
        feature_sets.append(set(df.columns) - set(df_original.columns) - set.union(*feature_sets))
        print(f"Features de estructura de mercado añadidos: {len(feature_sets[-1])}")
        
        # Generar características temporales
        df = pipeline.generate_temporal_features(df)
        feature_sets.append(set(df.columns) - set(df_original.columns) - set.union(*feature_sets))
        print(f"Features temporales añadidos: {len(feature_sets[-1])}")
        
        # Generar características contextuales
        df = pipeline.generate_contextual_features(df)
        feature_sets.append(set(df.columns) - set(df_original.columns) - set.union(*feature_sets))
        print(f"Features contextuales añadidos: {len(feature_sets[-1])}")
        
        # Crear características de lag
        df = pipeline.create_lag_features(df)
        feature_sets.append(set(df.columns) - set(df_original.columns) - set.union(*feature_sets))
        print(f"Features de lag añadidos: {len(feature_sets[-1])}")
        
        # Excluir columnas no numéricas y de precio/volumen
        exclude_cols = ['datetime', 'timestamp', 'open_time', 'open', 'high', 'low', 'close', 'volume']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Limpiar y escalar features
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        df[feature_cols] = df[feature_cols].fillna(method='ffill').fillna(0)
        
        # Guardar las características antes del escalado
        features_before_scaling = df[feature_cols].copy()
        
        # Escalar features
        df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        
        # Verificar que no se perdieron características durante el escalado
        assert set(df.columns) == set(features_before_scaling.columns).union(set(exclude_cols)), \
            "Se perdieron características durante el escalado"
        
        # Imprimir resumen final
        total_features = len(feature_cols)
        print(f"\nResumen de características:")
        print(f"Total de características generadas: {total_features}")
        print(f"Características por categoría:")
        for i, features in enumerate(feature_sets):
            print(f"  - Set {i+1}: {len(features)} características")
        
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
 
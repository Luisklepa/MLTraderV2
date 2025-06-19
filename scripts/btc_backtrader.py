# ──────────────────────────────────────────────────────────────────────────────
# Imports optimizados y agrupados
# ──────────────────────────────────────────────────────────────────────────────
import os
import sys
import random
import warnings
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import itertools
import time
import hashlib
import json

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import backtrader as bt
from tqdm import tqdm
from colorama import Fore, Style, init
from rich.console import Console
from rich.panel import Panel
import questionary

# Imports locales
# from strategies.mean_reversion import MeanReversionStrategy
# from strategies.breakout_momentum import BreakoutMomentumStrategy
# from utils.data_fetcher import BinanceDataFetcher
# from backtest.engine import BacktestEngine
# from utils.smart_optimizer import SmartOptimizer

warnings.filterwarnings('ignore')

random.seed(42)
np.random.seed(42)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration and Data Classes
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class BacktestResult:
    """Structured result container for backtests"""
    strategy_name: str
    final_value: float
    return_pct: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    win_rate: float
    parameters: Dict[str, Any]

# ──────────────────────────────────────────────────────────────────────────────
# Optimized Data Fetching
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# Base Strategy Class
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# Optimized Trend Following Strategy
# ──────────────────────────────────────────────────────────────────────────────
class CommInfoFractional(bt.CommissionInfo):
    """Comisión personalizada para operar con fracciones de Bitcoin"""
    params = (
        ('commission', 0.001),  # 0.1% por operación
        ('mult', 1.0),          # Multiplicador de tamaño
        ('margin', False),      # No usar margen
        ('commtype', bt.CommInfoBase.COMM_PERC),  # Comisión porcentual
    )
    
    def _getcommission(self, size, price, pseudoexec):
        """Calcula la comisión para una operación"""
        return abs(size * price * self.p.commission)

class BTCStrategy(bt.Strategy):
    # Usar __slots__ para mejor rendimiento de memoria
    __slots__ = ['entry_price', 'stop_loss', 'bars_in_position', 'bars_since_last_trade', 'atr', 'ema_fast', 'ema_slow']
    
    def __init__(self):
        super().__init__()
        # Pre-calcular indicadores una sola vez
        self.atr = bt.ind.ATR(self.data, period=self.p.atr_period)
        self.ema_fast = bt.ind.EMA(self.data.close, period=self.p.ema_fast)
        self.ema_slow = bt.ind.EMA(self.data.close, period=self.p.ema_slow)
        # Inicializar variables
        self.entry_price = None
        self.stop_loss = None
        self.bars_in_position = 0
        self.bars_since_last_trade = 0
    
    def _update_stops(self):
        """Versión optimizada con menos cálculos repetitivos"""
        if not self.position or not self.entry_price:
            return
        current_price = self.data.close[0]
        position_size = self.position.size
        # Calcular PnL una sola vez
        if position_size > 0:
            unrealized_pnl_pct = (current_price - self.entry_price) / self.entry_price
            should_trail = (unrealized_pnl_pct > self.p.min_profit_to_trail and 
                          self.bars_in_position > self.p.min_bars_to_trail)
            if should_trail:
                new_stop = current_price * (1 - self.p.trailing_stop)
                if new_stop > self.stop_loss:
                    self.stop_loss = new_stop
            # Check stop loss
            if current_price < self.stop_loss:
                self.close()
        else:
            unrealized_pnl_pct = (self.entry_price - current_price) / self.entry_price
            should_trail = (unrealized_pnl_pct > self.p.min_profit_to_trail and 
                          self.bars_in_position > self.p.min_bars_to_trail)
            if should_trail:
                new_stop = current_price * (1 + self.p.trailing_stop)
                if new_stop < self.stop_loss:
                    self.stop_loss = new_stop
            if current_price > self.stop_loss:
                self.close()
                
    def next(self):
        """Called for each new candle/period"""
        # Incrementar contadores
        if self.position:
            self.bars_in_position += 1
        self.bars_since_last_trade += 1
        
        # Obtener valores actuales
        price = self.data.close[0]
        atr_val = self.atr[0]
        
        # Actualizar stops si tenemos posición abierta
        self._update_stops()
        
        # Verificar señales solo si no tenemos órdenes pendientes
        if not self.order:
            self._check_signals(price, atr_val)

# ──────────────────────────────────────────────────────────────────────────────
# Mean Reversion Strategy
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# Pandas Data Feed
# ──────────────────────────────────────────────────────────────────────────────
class PandasData(bt.feeds.PandasData):
    """Optimized pandas data feed"""
    params = (
        ('datetime', 'open_time'),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('openinterest', None),
    )

# ──────────────────────────────────────────────────────────────────────────────
# Backtesting Engine
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# Main Application
# ──────────────────────────────────────────────────────────────────────────────
class TradingApp:
    """Aplicación principal de trading"""
    
    def __init__(self):
        self.engine = BacktestEngine()
        self.data_manager = DataManager(cache_timeout=600)
        self.config = self._get_default_config()
        self._load_trend_params()
        self._preload_common_data()
        
    def _preload_common_data(self):
        common_configs = [
            {'symbol': 'BTCUSDT', 'timeframe': '15m', 'limit': 3000},
            {'symbol': 'BTCUSDT', 'timeframe': '15m', 'limit': 2000},
        ]
        self.data_manager.preload_data(common_configs)
    
    def _get_default_config(self):
        """Centralizar configuración por defecto"""
        return {
            'capital_inicial': 10000.0,
            'comision': 0.00075,
            'risk_perc': 0.02,
            'symbol': 'BTCUSDT',
            'timeframe': '15m',
            'max_drawdown': 0.25,  # Permitir mayor drawdown
            'min_trade_size': 1,
            'max_position_size': 100,
            'ema_fast': 12,  # Actualizado para Trend Following
            'ema_slow': 26,  # Actualizado para Trend Following
            'atr_period': 14,  # Actualizado para Trend Following
            'atr_mult': 1.0,  # Stop más ajustado
            'atr_threshold': 0.5,  # Actualizado para Trend Following
            'trend_threshold': 0.3,  # Acepta tendencias débiles
            'distance_threshold': 0.05,  # Acepta precio cerca de EMAs
            'trailing_stop': 0.01,  # Trailing más amplio
            'min_profit_to_trail': 0.001,  # Trailing más fácil de activar
            'min_bars_to_trail': 1,  # Menos barras para trailing
            'min_bars_between_trades': 1,  # Permite operar seguido
            'max_consecutive_losses': 5,  # Permite más pérdidas seguidas
            'risk_reduction_factor': 0.9,  # Menor reducción de riesgo
            'bb_period': 10,  # Bandas de Bollinger más sensibles
            'bb_dev': 1.5,    # Bandas más estrechas
            'rsi_period': 7,  # RSI más rápido
            'rsi_lower': 45,  # Cambiado de rsi_oversold
            'rsi_upper': 55,  # Cambiado de rsi_overbought
        }
        
    @property
    def data_info(self):
        """Cache para información de datos (solo consulta básica)"""
        if not hasattr(self, '_data_info_cache'):
            df = self.data_manager.get_data(
                self.config['symbol'],
                self.config['timeframe'],
                100
            )
            self._data_info_cache = {
                'symbol': self.config['symbol'],
                'timeframe': self.config['timeframe'],
                'available': df is not None and not df.empty
            }
        return self._data_info_cache
    
    def _fetch_and_validate_data(self, limit=3000):
        """Método centralizado para fetch y validación de datos usando cache"""
        df = self.data_manager.get_data(
            self.config['symbol'],
            self.config['timeframe'],
            limit
        )
        if df is None or df.empty:
            print("[ERROR] No se descargaron datos de Binance.")
            return None
        self._print_data_info(df)
        return df
    
    def _print_data_info(self, df):
        """Método centralizado para mostrar info de datos"""
        n_bars = len(df)
        date_start = df['open_time'].iloc[0] if n_bars > 0 else 'N/A'
        date_end = df['open_time'].iloc[-1] if n_bars > 0 else 'N/A'
        print(f"\n{'='*20} DATOS DESCARGADOS {'='*20}")
        print(f"Símbolo: {self.config['symbol']}")
        print(f"Timeframe: {self.config['timeframe']}")
        print(f"Velas: {n_bars}")
        print(f"Rango: {date_start} -> {date_end}")
        print("="*54)
    
    def _setup_backtest_engine(self):
        """Configurar engine una sola vez"""
        self.engine.initial_cash = self.config['capital_inicial']
        self.engine.commission = self.config['comision']
    
    def _print_results(self, result):
        """Imprime los resultados del backtest de forma centralizada"""
        print("\nResultados del backtest:")
        print(f"Final Value: ${result.final_value:.2f}")
        print(f"Return: {result.return_pct:.2f}%")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.3f}")
        print(f"Max Drawdown: {result.max_drawdown:.2f}%")
        print(f"Total Trades: {result.total_trades}")
        print(f"Win Rate: {result.win_rate:.2f}%")
    
    def compare_strategies(self):
        print("\n[INFO] Comparando estrategias (SIN optimización automática)...")
        print("Fetching data...")
        df = self.data_manager.get_data('BTCUSDT', '15m', 3000)
        n_bars = len(df)
        date_start = df['open_time'].iloc[0] if n_bars > 0 else 'N/A'
        date_end = df['open_time'].iloc[-1] if n_bars > 0 else 'N/A'
        print("\n================ DATOS DESCARGADOS ================")
        print(f"Símbolo: BTCUSDT")
        print(f"Timeframe: 15m")
        print(f"Velas descargadas: {n_bars}")
        print(f"Rango de fechas: {date_start} -> {date_end}")
        print("===================================================\n")
        if df is None or df.empty or n_bars == 0:
            print("[ERROR] No se descargaron datos de Binance. La comparación no se ejecutará.")
            return
        if not hasattr(df, 'columns'):
            print("[ERROR] El objeto de datos no es un DataFrame válido.")
            return
        strategies = {
            # 'Mean Reversion': MeanReversionStrategy
        }
        results = {}
        for name, strat in tqdm(strategies.items(), desc="Estrategias", ncols=80):
            print(f"\n--- Ejecutando backtest para: {name} ---")
            try:
                valid_params = list(getattr(strat.params, '_getkeys', lambda: [])())
                params = {k: v for k, v in self.config.items() if k in valid_params}
                params['printlog'] = False
                result = self.engine.run_backtest(strat, df, params)
                results[name] = result
                print(f"✓ {name} completado")
            except Exception as e:
                print(f"Error al ejecutar {name}: {e}")
        print("\nStrategy Comparison:")
        print("--------------------------------------------------------------------------------")
        print(f"{'Strategy':<15}{'Return%':>10}{'Sharpe':>10}{'DrawDown%':>12}{'Trades':>9}{'WinRate%':>10}")
        print("--------------------------------------------------------------------------------")
        for name, res in results.items():
            print(f"{name:<15}{res.return_pct:>10.2f}{res.sharpe_ratio:>10.3f}{res.max_drawdown:>12.2f}{res.total_trades:>9}{res.win_rate:>10.2f}")
        print("--------------------------------------------------------------------------------")
    
    def show_ascii_banner(self):
        banner = '''\033[96m
╔════════════════════════════════════════════════════════════════════╗
║      🚀 OPTIMIZED TRADING ALGORITHM - BACKTESTER PRO v2 🚀       ║
║                    Hecho por ICARUS 🦅                        ║
╚════════════════════════════════════════════════════════════════════╝\033[0m'''
        print(banner)

    def show_tips(self):
        tips = [
            "La gestión del riesgo es más importante que la entrada perfecta.",
            "No pongas todos los huevos en la misma canasta. Diversifica.",
            "El backtesting no garantiza resultados futuros, ¡pero ayuda mucho!",
            "Deja correr las ganancias y corta rápido las pérdidas.",
            "No operes por emociones, sigue tu plan.",
            "Revisa siempre el drawdown, no solo el retorno.",
            "La paciencia es una virtud en el trading.",
            "No existe el santo grial, pero sí la disciplina.",
            "El mejor trade es el que sigue tu estrategia, no el que más gana.",
            "¡Recuerda tomar descansos y cuidar tu salud mental!"
        ]
        print(f"\033[93m💡 TIP: {random.choice(tips)}\033[0m\n")

    # Constantes para evitar strings repetidos
    MENU_OPTIONS = {
        # '1': ('Breakout Momentum 📈', 'run_breakout_momentum'),
        '2': ('Comparar Estrategias ⚔️', 'compare_strategies'),
        '6': ('Ayuda / Tips 🧠', 'show_help'),
        '7': ('Salir ❌', 'exit')
    }
    
    COLORS = {
        'header': '\033[96m',
        'success': '\033[92m',
        'warning': '\033[93m',
        'error': '\033[91m',
        'info': '\033[94m',
        'reset': '\033[0m'
    }

    def show_menu(self):
        self.show_ascii_banner()
        self.show_tips()
        print(f"{self.COLORS['success']}Selecciona una opción:{self.COLORS['reset']}")
        for key, (desc, _) in self.MENU_OPTIONS.items():
            print(f"{self.COLORS['info']}[{key}]{self.COLORS['reset']} {desc}")

    def run(self):
        while True:
            self.show_menu()
            choice = input(f"\n{self.COLORS['header']}Opción: {self.COLORS['reset']}").strip()
            if choice in self.MENU_OPTIONS:
                if choice == '7':
                    print(f"\n{self.COLORS['success']}¡Hasta luego!{self.COLORS['reset']}")
                    break
                _, method_name = self.MENU_OPTIONS[choice]
                method = getattr(self, method_name)
                method()
            else:
                print(f"{self.COLORS['error']}Opción inválida.{self.COLORS['reset']}")

    def _fetch_data(self, limit=2000):
        """Obtiene datos históricos de Binance y los convierte a un feed de Backtrader"""
        print("Descargando datos de Binance...")
        df = self.data_manager.get_data('BTCUSDT', '15m', limit)
        if df is None or df.empty:
            print("No se pudieron obtener datos de Binance.")
            return None
        return PandasData(dataname=df)

    def _trend_params_path(self):
        return os.path.join('config', 'last_trend_params.json')
    
    def _load_trend_params(self):
        path = self._trend_params_path()
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    params = json.load(f)
                self.config.update(params)
                print(f"[INFO] Parámetros óptimos de Trend Following cargados desde {path}")
            except Exception as e:
                print(f"[WARN] No se pudieron cargar los parámetros óptimos: {e}")

    def _save_trend_params(self, params):
        path = self._trend_params_path()
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                json.dump(params, f, indent=2)
            print(f"[INFO] Parámetros óptimos de Trend Following guardados en {path}")
        except Exception as e:
            print(f"[WARN] No se pudieron guardar los parámetros óptimos: {e}")

    def plot_breakout_signals(self, df, strategy_instance):
        plt.figure(figsize=(14, 6))
        plt.plot(df['close'], label='Precio', color='black')
        # Graficar señales de compra
        if hasattr(strategy_instance, 'buy_signals'):
            buy_dates, buy_prices = zip(*strategy_instance.buy_signals) if strategy_instance.buy_signals else ([], [])
            plt.scatter(buy_dates, buy_prices, marker='^', color='green', label='Buy', zorder=5)
        # Graficar señales de venta
        if hasattr(strategy_instance, 'sell_signals'):
            sell_dates, sell_prices = zip(*strategy_instance.sell_signals) if strategy_instance.sell_signals else ([], [])
            plt.scatter(sell_dates, sell_prices, marker='v', color='red', label='Sell', zorder=5)
        plt.title('Breakout Momentum - Señales')
        plt.xlabel('Fecha')
        plt.ylabel('Precio')
        plt.legend()
        plt.tight_layout()
        plt.show()

# Usar un data manager para cachear datos
class DataManager:
    def __init__(self, cache_timeout):
        self._cache = {}
        self._cache_timeout = cache_timeout
    
    def get_data(self, symbol, timeframe, limit):
        cache_key = f"{symbol}_{timeframe}_{limit}"
        if cache_key in self._cache:
            data, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_timeout:
                return data
        # Fetch new data
        fetcher = BinanceDataFetcher()
        data = fetcher.get_klines(symbol, timeframe, limit)
        if data is not None:
            self._cache[cache_key] = (data, time.time())
        return data

    def preload_data(self, configs):
        for config in configs:
            self.get_data(config['symbol'], config['timeframe'], config['limit'])

    def get_cache_info(self):
        return {
            'active': len(self._cache),
            'expired': 0,
            'total': len(self._cache)
        }

    def clear_cache(self):
        self._cache.clear()

console = Console()

def print_header():
    console.clear()
    console.print(Panel.fit(
        "[bold cyan]🤖  MACHINE LEARNING TRADING SUITE - ICARUS v3  🤖[/bold cyan]\n[white]by ICARUS[/white]",
        subtitle="[magenta]100% ML Powered Trading[/magenta]",
        style="bold magenta"))

def main_menu():
    print_header()
    opcion = questionary.select(
        "Selecciona una opción:",
        choices=[
            "1. Generar Features & Dataset ML 📊",
            "2. Entrenar Modelo ML (Random Forest / LSTM) 🏋️‍♂️",
            "3. Comparar Modelos ML ⚔️",
            "4. Backtest con Señales ML 📈",
            "5. Ayuda / Tips 🧠",
            "6. Salir ❌"
        ]).ask()
    return opcion

if __name__ == "__main__":
    while True:
        opcion = main_menu()
        if opcion.startswith("1"):
            os.system('python utils/ml_feature_pipeline.py')
        elif opcion.startswith("2"):
            # Preguntar por balance_strategy y metric
            balance_strategy = questionary.select(
                "Estrategia de balanceo de clases:",
                choices=[
                    "smote",
                    "undersample",
                    "combined",
                    "none (sin balanceo)"
                ]).ask()
            if balance_strategy == "none (sin balanceo)":
                balance_strategy = None
            metric = questionary.select(
                "Métrica objetivo para fine-tuning de threshold:",
                choices=["f1", "recall"]).ask()
            os.system(f'python utils/ml_train_model.py --balance_strategy {balance_strategy} --metric {metric}')
        elif opcion.startswith("3"):
            # Preguntar por balance_strategy y metric
            balance_strategy = questionary.select(
                "Estrategia de balanceo de clases:",
                choices=[
                    "smote",
                    "undersample",
                    "combined",
                    "none (sin balanceo)"
                ]).ask()
            if balance_strategy == "none (sin balanceo)":
                balance_strategy = None
            metric = questionary.select(
                "Métrica objetivo para fine-tuning de threshold:",
                choices=["f1", "recall"]).ask()
            os.system(f'python utils/ml_compare_models.py --balance_strategy {balance_strategy} --metric {metric}')
        elif opcion.startswith("4"):
            os.system('python utils/ml_backtest_signals.py')
        elif opcion.startswith("5"):
            console.print("\n[bold green]💡 TIP:[/bold green] Puedes ajustar los scripts de ML para pruebas rápidas usando menos datos, menos epochs o desactivando la optimización de hiperparámetros.\n")
            input("Presiona Enter para volver al menú...")
        elif opcion.startswith("6"):
            console.print("\n[bold magenta]¡Hasta luego y buenos trades con IA![/bold magenta]\n")
            sys.exit(0)
        else:
            console.print("\n[bold red]Opción inválida. Intenta de nuevo.[/bold red]\n")
            input("Presiona Enter para volver al menú...")
import optuna
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import matplotlib.pyplot as plt

class SmartOptimizer:
    def __init__(self, strategy_class, data, engine=None, n_splits=5, random_state=42, debug=False):
        self.strategy_class = strategy_class
        self.data = data
        self.n_splits = n_splits
        self.random_state = random_state
        self.engine = engine  # Puede ser None, se debe pasar en optimize si no se setea aquí
        self.debug = debug

    def suggest_params_with_relaxation(self, trial, max_retries=3):
        """
        Sugiere parámetros y, si no hay trades, relaja automáticamente los filtros de volumen, RSI, breakout y ATR.
        """
        # Rango base
        params = {
            'lookback_periods': trial.suggest_int('lookback_periods', 5, 100),
            'ema_fast': trial.suggest_int('ema_fast', 10, 60),
            'ema_slow': trial.suggest_int('ema_slow', 30, 120),
            'atr_period': trial.suggest_int('atr_period', 5, 40),
            'atr_multiplier': trial.suggest_float('atr_multiplier', 0.8, 3.5),
            'volume_multiplier': trial.suggest_float('volume_multiplier', 1.0, 2.5),
            'rr_ratio': trial.suggest_float('rr_ratio', 1.2, 5.0),
            'rsi_period': trial.suggest_int('rsi_period', 5, 20),
            'rsi_lower': trial.suggest_int('rsi_lower', 10, 50),
            'rsi_upper': trial.suggest_int('rsi_upper', 50, 90),
            'min_breakout_size': trial.suggest_float('min_breakout_size', 0.0001, 0.01),
            'printlog': False
        }
        for retry in range(max_retries):
            result, profit_factor = self.engine.backtest_on_period(self.strategy_class, params, self.data)
            if result.total_trades > 0:
                return params, result, profit_factor, retry
            # Auto-relajar filtros
            params['volume_multiplier'] = max(0.8, params['volume_multiplier'] * 0.7)
            params['rsi_lower'] = max(5, params['rsi_lower'] - 5)
            params['rsi_upper'] = min(95, params['rsi_upper'] + 5)
            params['min_breakout_size'] = max(0.00005, params['min_breakout_size'] * 0.5)
            params['atr_multiplier'] = max(0.5, params['atr_multiplier'] * 0.7)
            print(f"[AUTO-RELAX] Reintentando con filtros más laxos (intento {retry+1}) para trial {trial.number}")
        # Si sigue sin trades, penalizar
        return params, result, profit_factor, max_retries

    def optimize(self, n_trials=100):
        best_params = None
        best_score = -float('inf')
        for trial_num in range(n_trials):
            params, result, profit_factor, relax_attempts = self.suggest_params_with_relaxation(trial=DummyTrial(trial_num))
            score = self.engine.calculate_robust_score(result, profit_factor)
            # Penalizar estrategias sin trades
            if result.total_trades == 0:
                score -= 100
            if score > best_score:
                best_score = score
                best_params = params
            print(f"[TRIAL {trial_num}] Trades={result.total_trades}, Score={score:.2f}, Relaxed={relax_attempts} veces, Params={params}")
        return best_params, best_score

    @staticmethod
    def _calculate_profit_factor(result):
        if hasattr(result, 'profit_factor'):
            return result.profit_factor
        return 1.0

    @staticmethod
    def _robust_score(sharpe, profit_factor, win_rate, max_dd, total_trades):
        # Score más flexible
        if total_trades < 10 or max_dd > 25 or sharpe < 0.5 or profit_factor < 1.1:
            return -100
        score = (
            sharpe * 2.0 +
            profit_factor * 1.5 +
            (win_rate / 100.0) * 1.0 -
            (max_dd / 25.0) * 1.0
        )
        return score 

    def plot_optimization_results(self, study, top_n=10):
        """Visualiza la distribución de scores y los mejores parámetros de la optimización."""
        scores = [t.value for t in study.trials if t.value is not None and t.value > -100]
        if not scores:
            print("No hay scores válidos para visualizar.")
            return
        plt.figure(figsize=(8, 4))
        plt.hist(scores, bins=30, color='skyblue', edgecolor='black')
        plt.title('Distribución de scores de optimización')
        plt.xlabel('Score')
        plt.ylabel('Frecuencia')
        plt.show()
        # Mostrar los mejores parámetros
        print(f"\nTop {top_n} combinaciones por score:")
        top_trials = sorted([t for t in study.trials if t.value is not None], key=lambda t: t.value, reverse=True)[:top_n]
        for i, t in enumerate(top_trials, 1):
            print(f"#{i}: Score: {t.value:.2f}, Params: {t.params}") 

# DummyTrial para compatibilidad con el loop manual
class DummyTrial:
    def __init__(self, number):
        self.number = number
    def suggest_int(self, name, low, high):
        import random
        return random.randint(low, high)
    def suggest_float(self, name, low, high):
        import random
        return random.uniform(low, high) 
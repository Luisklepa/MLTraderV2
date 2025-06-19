"""
Advanced robustness metrics for strategy evaluation.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from scipy import stats

def calculate_robustness_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate comprehensive robustness metrics from backtest results.
    
    Args:
        results: List of dictionaries containing backtest results
        
    Returns:
        Dictionary with robustness metrics
    """
    # Extract key metrics
    returns = np.array([r['pnl_total'] for r in results])
    win_rates = np.array([r['win_rate'] for r in results])
    trade_counts = np.array([r['total_trades'] for r in results])
    
    # Basic statistical metrics
    metrics = {
        'return_mean': float(np.mean(returns)),
        'return_std': float(np.std(returns)),
        'return_sharpe': float(np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0),
        'win_rate_mean': float(np.mean(win_rates)),
        'win_rate_std': float(np.std(win_rates)),
        'trades_mean': float(np.mean(trade_counts)),
        'trades_std': float(np.std(trade_counts))
    }
    
    # Consistency metrics
    metrics.update({
        'return_cv': float(metrics['return_std'] / abs(metrics['return_mean']) if metrics['return_mean'] != 0 else np.inf),
        'win_rate_cv': float(metrics['win_rate_std'] / metrics['win_rate_mean'] if metrics['win_rate_mean'] > 0 else np.inf),
        'trades_cv': float(metrics['trades_std'] / metrics['trades_mean'] if metrics['trades_mean'] > 0 else np.inf)
    })
    
    # Statistical significance
    if len(returns) > 1:
        t_stat, p_value = stats.ttest_1samp(returns, 0)
        metrics.update({
            'returns_t_stat': float(t_stat),
            'returns_p_value': float(p_value)
        })
    
    # Drawdown consistency
    drawdowns = np.array([r.get('max_drawdown', 0) for r in results])
    metrics.update({
        'drawdown_mean': float(np.mean(drawdowns)),
        'drawdown_std': float(np.std(drawdowns)),
        'drawdown_cv': float(np.std(drawdowns) / abs(np.mean(drawdowns)) if np.mean(drawdowns) != 0 else np.inf)
    })
    
    # Market condition analysis
    long_rates = np.array([r.get('long_win_rate', 0) for r in results])
    short_rates = np.array([r.get('short_win_rate', 0) for r in results])
    metrics.update({
        'long_short_balance': float(np.mean(long_rates) / np.mean(short_rates) if np.mean(short_rates) > 0 else np.inf),
        'direction_consistency': float(np.minimum(np.std(long_rates), np.std(short_rates)))
    })
    
    return metrics

def calculate_monte_carlo_metrics(trades: List[Dict[str, Any]], n_simulations: int = 1000) -> Dict[str, float]:
    """
    Perform Monte Carlo simulation on trade results for robustness analysis.
    
    Args:
        trades: List of trade dictionaries with PnL information
        n_simulations: Number of Monte Carlo simulations
        
    Returns:
        Dictionary with Monte Carlo metrics
    """
    if not trades:
        return {}
    
    # Extract PnL values
    pnls = np.array([t['pnl'] for t in trades])
    
    # Run simulations
    simulation_results = []
    for _ in range(n_simulations):
        # Randomly resample trades with replacement
        sampled_pnls = np.random.choice(pnls, size=len(pnls), replace=True)
        cumulative_return = np.sum(sampled_pnls)
        simulation_results.append(cumulative_return)
    
    simulation_results = np.array(simulation_results)
    
    # Calculate metrics
    metrics = {
        'mc_mean': float(np.mean(simulation_results)),
        'mc_std': float(np.std(simulation_results)),
        'mc_sharpe': float(np.mean(simulation_results) / np.std(simulation_results) if np.std(simulation_results) > 0 else 0),
        'mc_var_95': float(np.percentile(simulation_results, 5)),
        'mc_var_99': float(np.percentile(simulation_results, 1)),
        'mc_positive_prob': float(np.mean(simulation_results > 0))
    }
    
    return metrics

def calculate_regime_metrics(results: List[Dict[str, Any]], market_regimes: pd.Series) -> Dict[str, float]:
    """
    Calculate strategy performance metrics across different market regimes.
    
    Args:
        results: List of backtest results
        market_regimes: Series of market regime labels indexed by datetime
        
    Returns:
        Dictionary with regime-specific metrics
    """
    regime_metrics = {}
    
    # Group results by regime
    for result in results:
        start_date = pd.Timestamp(result['period'].split(' to ')[0])
        end_date = pd.Timestamp(result['period'].split(' to ')[1])
        regime = market_regimes[start_date:end_date].mode().iloc[0]
        
        if regime not in regime_metrics:
            regime_metrics[regime] = []
        regime_metrics[regime].append(result)
    
    # Calculate metrics for each regime
    metrics = {}
    for regime, regime_results in regime_metrics.items():
        regime_returns = np.array([r['pnl_total'] for r in regime_results])
        metrics.update({
            f'{regime}_return_mean': float(np.mean(regime_returns)),
            f'{regime}_return_std': float(np.std(regime_returns)),
            f'{regime}_sharpe': float(np.mean(regime_returns) / np.std(regime_returns) if np.std(regime_returns) > 0 else 0),
            f'{regime}_win_rate': float(np.mean([r['win_rate'] for r in regime_results]))
        })
    
    # Calculate regime consistency
    regime_means = np.array([metrics[f'{r}_return_mean'] for r in regime_metrics.keys()])
    metrics['regime_consistency'] = float(np.std(regime_means) / abs(np.mean(regime_means)) if np.mean(regime_means) != 0 else np.inf)
    
    return metrics

def calculate_robustness_score(metrics: Dict[str, float], config: Any) -> float:
    """
    Calculate an overall robustness score based on multiple metrics.
    
    Args:
        metrics: Dictionary of robustness metrics
        config: Configuration object with thresholds
        
    Returns:
        Float robustness score between 0 and 1
    """
    score_components = []
    
    # Return consistency (weight: 0.3)
    return_score = max(0, 1 - metrics['return_cv'] / 2)
    score_components.append(return_score * 0.3)
    
    # Statistical significance (weight: 0.2)
    if 'returns_p_value' in metrics:
        significance_score = max(0, 1 - metrics['returns_p_value'])
        score_components.append(significance_score * 0.2)
    
    # Win rate consistency (weight: 0.15)
    win_rate_score = max(0, 1 - metrics['win_rate_cv'])
    score_components.append(win_rate_score * 0.15)
    
    # Trade frequency consistency (weight: 0.1)
    trade_score = max(0, 1 - metrics['trades_cv'] / 2)
    score_components.append(trade_score * 0.1)
    
    # Drawdown consistency (weight: 0.15)
    drawdown_score = max(0, 1 - metrics['drawdown_cv'])
    score_components.append(drawdown_score * 0.15)
    
    # Direction balance (weight: 0.1)
    direction_score = max(0, 1 - abs(1 - metrics['long_short_balance']))
    score_components.append(direction_score * 0.1)
    
    # Calculate final score
    final_score = sum(score_components)
    
    return float(final_score) 
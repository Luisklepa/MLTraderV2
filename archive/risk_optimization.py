"""
Risk management optimization utilities for the ML trading system.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from scipy import stats

logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    """Container for risk metrics."""
    volatility: float
    var_95: float
    cvar_95: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    kelly_fraction: float

class RiskOptimizer:
    """Handles risk management optimization."""
    
    def __init__(self, config: Dict):
        """Initialize optimizer with configuration."""
        self.config = config
        self.risk_free_rate = config.get('risk_free_rate', 0.02)
        self.target_sharpe = config.get('target_sharpe', 2.0)
        self.max_position_size = config.get('max_position_size', 1.0)
        self.max_drawdown_limit = config.get('max_drawdown_limit', 0.20)
    
    def calculate_risk_metrics(
        self,
        returns: pd.Series,
        positions: pd.Series
    ) -> RiskMetrics:
        """Calculate comprehensive risk metrics."""
        # Basic return metrics
        volatility = returns.std() * np.sqrt(252)  # Annualized
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        
        # Drawdown analysis
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # Performance ratios
        excess_returns = returns - self.risk_free_rate/252
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / returns.std()
        
        downside_returns = returns[returns < 0]
        sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_returns.std()
        
        # Trading metrics
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        win_rate = len(wins) / len(returns)
        
        if len(losses) > 0:
            profit_factor = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else np.inf
            avg_win = wins.mean() if len(wins) > 0 else 0
            avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
            
            # Kelly Criterion
            kelly_fraction = (win_rate - ((1 - win_rate) / (avg_win/avg_loss))) if avg_loss != 0 else 1
            kelly_fraction = min(max(kelly_fraction, 0), 1)  # Bound between 0 and 1
        else:
            profit_factor = np.inf
            avg_win = wins.mean() if len(wins) > 0 else 0
            avg_loss = 0
            kelly_fraction = 1
        
        return RiskMetrics(
            volatility=volatility,
            var_95=var_95,
            cvar_95=cvar_95,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            kelly_fraction=kelly_fraction
        )
    
    def optimize_position_size(
        self,
        probability: float,
        volatility: float,
        current_drawdown: float,
        risk_metrics: Optional[RiskMetrics] = None
    ) -> float:
        """Calculate optimal position size considering multiple factors."""
        # Base size from probability
        base_size = self.max_position_size * (probability - 0.5) * 2
        
        # Volatility adjustment
        vol_factor = 1.0
        if volatility > 0:
            target_vol = 0.20  # Target annual volatility
            vol_factor = target_vol / volatility
        
        # Drawdown adjustment
        dd_factor = 1.0
        if current_drawdown < 0:
            # Reduce size as drawdown approaches limit
            dd_factor = 1.0 - (abs(current_drawdown) / self.max_drawdown_limit)
            dd_factor = max(0.1, dd_factor)  # Never go below 10% of base size
        
        # Kelly criterion adjustment if risk metrics available
        kelly_factor = 1.0
        if risk_metrics is not None:
            kelly_factor = risk_metrics.kelly_fraction
        
        # Combine all factors
        position_size = base_size * vol_factor * dd_factor * kelly_factor
        
        # Apply limits
        position_size = max(min(position_size, self.max_position_size), -self.max_position_size)
        
        return position_size
    
    def calculate_dynamic_stops(
        self,
        price: float,
        position_size: float,
        volatility: float,
        risk_metrics: Optional[RiskMetrics] = None
    ) -> Tuple[float, float]:
        """Calculate dynamic stop-loss and take-profit levels."""
        # Base stops using ATR-based volatility
        base_stop = 2.0 * volatility
        
        # Adjust based on position size
        size_factor = abs(position_size) / self.max_position_size
        base_stop *= (1 + size_factor)  # Wider stops for larger positions
        
        # Adjust based on win rate and average win/loss if available
        if risk_metrics is not None:
            win_rate = risk_metrics.win_rate
            avg_win = risk_metrics.avg_win
            avg_loss = risk_metrics.avg_loss
            
            if avg_loss > 0:
                # Adjust ratio based on historical performance
                profit_ratio = avg_win / avg_loss
                take_profit = base_stop * profit_ratio
            else:
                take_profit = base_stop * 2
        else:
            take_profit = base_stop * 2
        
        # Calculate actual levels
        if position_size > 0:  # Long position
            stop_loss = price * (1 - base_stop)
            take_profit_level = price * (1 + take_profit)
        else:  # Short position
            stop_loss = price * (1 + base_stop)
            take_profit_level = price * (1 - take_profit)
        
        return stop_loss, take_profit_level
    
    def adjust_for_correlation(
        self,
        position_size: float,
        correlation_matrix: pd.DataFrame,
        current_positions: Dict[str, float]
    ) -> float:
        """Adjust position size based on correlation with existing positions."""
        # If no other positions, return original size
        if not current_positions:
            return position_size
        
        # Calculate total correlation-weighted exposure
        total_exposure = 0
        for asset, pos in current_positions.items():
            if asset in correlation_matrix.columns:
                total_exposure += abs(pos) * correlation_matrix.loc[asset].abs().mean()
        
        # Adjust new position size
        if total_exposure > 1:
            position_size /= total_exposure
        
        return position_size
    
    def calculate_portfolio_var(
        self,
        positions: Dict[str, float],
        covariance_matrix: pd.DataFrame,
        confidence_level: float = 0.95
    ) -> float:
        """Calculate portfolio Value at Risk."""
        weights = pd.Series(positions)
        portfolio_variance = weights.dot(covariance_matrix.dot(weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        z_score = stats.norm.ppf(confidence_level)
        var = portfolio_volatility * z_score
        
        return var
    
    def optimize_portfolio_weights(
        self,
        returns: pd.DataFrame,
        constraints: Optional[Dict] = None
    ) -> Dict[str, float]:
        """Optimize portfolio weights using mean-variance optimization."""
        from scipy.optimize import minimize
        
        def objective(weights):
            portfolio_return = np.sum(returns.mean() * weights) * 252
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
            return -portfolio_return/portfolio_vol  # Negative Sharpe ratio
        
        n_assets = returns.shape[1]
        constraints_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
        ]
        
        if constraints is not None:
            if 'min_weight' in constraints:
                constraints_list.append(
                    {'type': 'ineq', 'fun': lambda x: x - constraints['min_weight']}
                )
            if 'max_weight' in constraints:
                constraints_list.append(
                    {'type': 'ineq', 'fun': lambda x: constraints['max_weight'] - x}
                )
        
        result = minimize(
            objective,
            x0=np.array([1/n_assets] * n_assets),
            method='SLSQP',
            constraints=constraints_list,
            bounds=[(0, 1) for _ in range(n_assets)]
        )
        
        return dict(zip(returns.columns, result.x)) 
"""
Configuration settings for robustness testing.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any

@dataclass
class RobustnessConfig:
    # Testing Parameters
    MIN_TRADES_PER_PERIOD: int = 10
    MIN_PERIODS_WITH_TRADES: float = 0.7  # At least 70% of periods must have trades
    MIN_TOTAL_TRADES: int = 100  # Minimum total trades across all periods
    
    # Performance Thresholds
    MIN_SHARPE_RATIO: float = 0.5
    MIN_WIN_RATE: float = 0.45
    MAX_DRAWDOWN: float = -0.25  # -25% maximum drawdown
    MIN_PROFIT_FACTOR: float = 1.2
    MIN_TRADES_PER_MONTH: float = 5.0
    
    # Consistency Thresholds
    MAX_RETURN_CV: float = 2.0  # Maximum coefficient of variation for returns
    MAX_WIN_RATE_STD: float = 0.15  # Maximum standard deviation for win rate
    MAX_DRAWDOWN_CV: float = 1.5  # Maximum coefficient of variation for drawdowns
    MIN_REGIME_CONSISTENCY: float = 0.6  # Minimum consistency across market regimes
    
    # Monte Carlo Parameters
    N_SIMULATIONS: int = 1000
    MIN_MC_POSITIVE_PROB: float = 0.6  # Minimum probability of positive returns in MC
    MIN_MC_SHARPE: float = 0.3  # Minimum Sharpe ratio in MC simulations
    MAX_MC_VAR_95: float = -0.2  # Maximum 95% Value at Risk
    
    # Market Regime Parameters
    VOLATILITY_WINDOW: int = 20
    TREND_WINDOW: int = 50
    VOLATILITY_THRESHOLD: float = 0.7  # Percentile for high volatility
    TREND_THRESHOLD: float = 0.7  # Percentile for strong trend
    
    # Statistical Significance
    SIGNIFICANCE_LEVEL: float = 0.05  # p-value threshold for statistical significance
    MIN_PERIODS: int = 8  # Minimum number of test periods for statistical validity
    
    # Robustness Score Weights
    SCORE_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        'return_consistency': 0.3,
        'statistical_significance': 0.2,
        'win_rate_consistency': 0.15,
        'trade_consistency': 0.1,
        'drawdown_consistency': 0.15,
        'direction_balance': 0.1
    })
    
    # Paths and Output Settings
    DATA_DIR: str = 'data'
    RESULTS_DIR: str = 'results/robustness'
    PLOT_SETTINGS: Dict[str, Any] = field(default_factory=lambda: {
        'plot_width': 1200,
        'plot_height': 1000,
        'template': 'plotly_white'
    })
    
    def validate_results(self, metrics: Dict[str, float]) -> Dict[str, bool]:
        """
        Validate robustness test results against thresholds.
        
        Args:
            metrics: Dictionary of calculated metrics
            
        Returns:
            Dictionary of validation results
        """
        validations = {
            'sufficient_trades': metrics.get('trades_mean', 0) >= self.MIN_TRADES_PER_PERIOD,
            'sharpe_ratio': metrics.get('return_sharpe', 0) >= self.MIN_SHARPE_RATIO,
            'win_rate': metrics.get('win_rate_mean', 0) >= self.MIN_WIN_RATE,
            'drawdown': metrics.get('drawdown_mean', 0) >= self.MAX_DRAWDOWN,
            'return_consistency': metrics.get('return_cv', float('inf')) <= self.MAX_RETURN_CV,
            'win_rate_consistency': metrics.get('win_rate_std', float('inf')) <= self.MAX_WIN_RATE_STD,
            'regime_consistency': metrics.get('regime_consistency', 0) >= self.MIN_REGIME_CONSISTENCY,
            'monte_carlo': all([
                metrics.get('mc_positive_prob', 0) >= self.MIN_MC_POSITIVE_PROB,
                metrics.get('mc_sharpe', 0) >= self.MIN_MC_SHARPE,
                metrics.get('mc_var_95', float('-inf')) >= self.MAX_MC_VAR_95
            ]),
            'statistical_significance': metrics.get('returns_p_value', 1.0) <= self.SIGNIFICANCE_LEVEL
        }
        
        return validations
    
    def calculate_final_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate final robustness score based on weighted metrics.
        
        Args:
            metrics: Dictionary of calculated metrics
            
        Returns:
            Float score between 0 and 1
        """
        validations = self.validate_results(metrics)
        
        # If any critical validation fails, return 0
        critical_validations = ['sufficient_trades', 'monte_carlo', 'statistical_significance']
        if not all(validations[v] for v in critical_validations):
            return 0.0
        
        # Calculate weighted score components
        score_components = []
        
        # Return consistency (0.3)
        return_score = max(0, 1 - metrics.get('return_cv', float('inf')) / self.MAX_RETURN_CV)
        score_components.append(return_score * self.SCORE_WEIGHTS['return_consistency'])
        
        # Statistical significance (0.2)
        if metrics.get('returns_p_value', 1.0) <= self.SIGNIFICANCE_LEVEL:
            significance_score = 1.0
        else:
            significance_score = 0.0
        score_components.append(significance_score * self.SCORE_WEIGHTS['statistical_significance'])
        
        # Win rate consistency (0.15)
        win_rate_score = max(0, 1 - metrics.get('win_rate_std', float('inf')) / self.MAX_WIN_RATE_STD)
        score_components.append(win_rate_score * self.SCORE_WEIGHTS['win_rate_consistency'])
        
        # Trade consistency (0.1)
        trade_score = max(0, 1 - metrics.get('trades_cv', float('inf')) / 2)
        score_components.append(trade_score * self.SCORE_WEIGHTS['trade_consistency'])
        
        # Drawdown consistency (0.15)
        drawdown_score = max(0, 1 - metrics.get('drawdown_cv', float('inf')) / self.MAX_DRAWDOWN_CV)
        score_components.append(drawdown_score * self.SCORE_WEIGHTS['drawdown_consistency'])
        
        # Direction balance (0.1)
        direction_score = max(0, 1 - abs(1 - metrics.get('long_short_balance', float('inf'))))
        score_components.append(direction_score * self.SCORE_WEIGHTS['direction_balance'])
        
        return sum(score_components) 
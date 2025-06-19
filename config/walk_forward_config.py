"""
Configuration settings for walk-forward analysis.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any

@dataclass
class WalkForwardConfig:
    # Window Parameters
    TRAIN_SIZE: int = 252  # 1 year of trading days
    TEST_SIZE: int = 63    # ~3 months
    GAP_SIZE: int = 5      # 1 week gap
    USE_EXPANDING: bool = False
    MIN_TOTAL_BARS: int = 756  # Minimum total bars needed (3 years)
    
    # Parameter Grid for Optimization
    PARAM_GRID: Dict[str, List[Any]] = field(default_factory=lambda: {
        'atr_multiplier': [1.0, 1.5, 2.0, 2.5],
        'atr_periods': [14, 21, 28],
        'volume_ratio_threshold': [1.0, 1.1, 1.2, 1.3],
        'atr_percentile_lower': [0.1, 0.15, 0.2],
        'atr_percentile_upper': [0.8, 0.85, 0.9]
    })
    
    # Optimization Settings
    OPTIMIZATION_METRIC: str = 'sharpe_ratio'  # Primary metric for optimization
    SECONDARY_METRICS: List[str] = field(default_factory=lambda: [
        'total_return_pct',
        'max_drawdown',
        'profit_factor',
        'win_rate'
    ])
    
    # Performance Criteria
    MIN_TRADES_PER_WINDOW: int = 10
    MIN_SHARPE_RATIO: float = 0.5
    MIN_WIN_RATE: float = 0.4
    MAX_DRAWDOWN_THRESHOLD: float = -0.2  # -20% maximum drawdown
    MIN_PROFIT_FACTOR: float = 1.2
    
    # Robustness Thresholds
    MIN_WINDOWS_ABOVE_CRITERIA: float = 0.7  # 70% of windows must meet criteria
    MAX_TRAIN_TEST_DEVIATION: float = 0.3    # Max 30% deviation between train/test
    MIN_CONSISTENCY_RATIO: float = 0.6       # Min 60% consistency across windows
    
    # Paths
    DATA_PATH: str = 'data/processed/ml_signals.csv'
    RESULTS_DIR: str = 'results/walk_forward'
    
    # Visualization Settings
    PLOT_SETTINGS: Dict[str, Any] = field(default_factory=lambda: {
        'plot_width': 1200,
        'plot_height': 800,
        'template': 'plotly_white',
        'show_individual_trades': True,
        'show_drawdown_overlay': True
    })
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {k: v for k, v in cls.__dict__.items() 
                if not k.startswith('_') and k.isupper()} 
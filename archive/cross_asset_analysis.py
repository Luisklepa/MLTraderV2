import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from scipy.stats import pearsonr
import logging

logger = logging.getLogger(__name__)

class CrossAssetAnalyzer:
    """Analyzes correlations and relationships between different crypto assets."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the cross-asset analyzer with configuration."""
        self.config = config or {}
        self.correlation_lookback = self.config.get('correlation_lookback', 30)
        self.correlation_threshold = self.config.get('correlation_threshold', 0.7)
        self.reference_assets = self.config.get('reference_assets', ['ETH/USDT', 'BNB/USDT', 'XRP/USDT'])
        
    def calculate_correlation_features(self, main_df: pd.DataFrame, reference_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate correlation-based features between the main asset and reference assets."""
        result = main_df.copy()
        
        # Calculate returns
        main_returns = result['close'].pct_change()
        
        for asset, df in reference_dfs.items():
            ref_returns = df['close'].pct_change()
            
            # Rolling correlation
            correlation = main_returns.rolling(self.correlation_lookback).corr(ref_returns)
            result[f'correlation_{asset}'] = correlation
            
            # Correlation regime
            result[f'high_correlation_{asset}'] = (
                np.abs(correlation) > self.correlation_threshold
            ).astype(int)
            
            # Correlation direction
            result[f'correlation_direction_{asset}'] = np.sign(correlation)
            
            # Lead-lag relationship
            lead_lag = []
            for i in range(len(main_returns)):
                if i < self.correlation_lookback:
                    lead_lag.append(0)
                    continue
                
                # Calculate correlation with different lags
                max_corr = 0
                max_lag = 0
                for lag in range(-5, 6):  # Check lags from -5 to +5
                    if i + lag < 0 or i + lag >= len(ref_returns):
                        continue
                    corr = pearsonr(
                        main_returns.iloc[i-self.correlation_lookback:i],
                        ref_returns.iloc[i-self.correlation_lookback+lag:i+lag]
                    )[0]
                    if abs(corr) > abs(max_corr):
                        max_corr = corr
                        max_lag = lag
                
                lead_lag.append(max_lag)
            
            result[f'lead_lag_{asset}'] = lead_lag
            
            # Relative strength
            result[f'relative_strength_{asset}'] = (
                main_returns.rolling(self.correlation_lookback).mean() -
                ref_returns.rolling(self.correlation_lookback).mean()
            )
            
            # Divergence signals
            result[f'divergence_{asset}'] = np.where(
                (main_returns > 0) & (ref_returns < 0),
                1,  # Positive divergence
                np.where(
                    (main_returns < 0) & (ref_returns > 0),
                    -1,  # Negative divergence
                    0
                )
            )
        
        # Aggregate correlation features
        result['avg_correlation'] = np.mean([
            result[f'correlation_{asset}']
            for asset in reference_dfs.keys()
        ], axis=0)
        
        result['correlation_regime'] = pd.qcut(
            result['avg_correlation'],
            q=5,
            labels=['very_low', 'low', 'neutral', 'high', 'very_high']
        )
        
        # Market leadership score
        result['market_leadership'] = np.mean([
            result[f'lead_lag_{asset}']
            for asset in reference_dfs.keys()
        ], axis=0)
        
        # Correlation-based risk score
        result['correlation_risk'] = np.mean([
            result[f'high_correlation_{asset}']
            for asset in reference_dfs.keys()
        ], axis=0)
        
        return result
    
    def calculate_market_impact(self, correlations: pd.Series, position_sizes: pd.Series) -> float:
        """Calculate the market impact score based on correlations and position sizes."""
        # Higher score indicates higher market impact risk
        return np.sum(np.abs(correlations * position_sizes))
    
    def adjust_position_size(self, base_size: float, correlation_risk: float) -> float:
        """Adjust position size based on correlation risk."""
        # Reduce position size when correlation risk is high
        risk_factor = 1 - (correlation_risk * 0.5)  # Max 50% reduction
        return base_size * max(0.5, risk_factor)  # Minimum 50% of base size
    
    def get_correlation_based_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading filters based on correlation analysis."""
        filters = pd.DataFrame(index=df.index)
        
        # Avoid trading when correlation risk is very high
        filters['correlation_filter'] = (df['correlation_risk'] < 0.8).astype(int)
        
        # Filter based on market leadership
        filters['leadership_filter'] = (df['market_leadership'] >= -1).astype(int)
        
        # Combined filter
        filters['combined_filter'] = (
            filters['correlation_filter'] &
            filters['leadership_filter']
        ).astype(int)
        
        return filters 
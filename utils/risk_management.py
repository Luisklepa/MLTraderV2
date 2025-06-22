import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class RiskManager:
    """Advanced risk management system with dynamic position sizing and adaptive stops."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the risk manager with configuration."""
        self.config = config or {}
        
        # Risk limits
        self.max_position_size = self.config.get('max_position_size', 0.1)  # Max 10% of portfolio
        self.max_correlation_exposure = self.config.get('max_correlation_exposure', 0.3)  # Max 30% correlation risk
        self.max_drawdown = self.config.get('max_drawdown', 0.2)  # Max 20% drawdown
        
        # Stop-loss parameters
        self.base_atr_multiplier = self.config.get('base_atr_multiplier', 2.0)
        self.min_stop_distance = self.config.get('min_stop_distance', 0.01)  # 1% minimum stop distance
        
        # Position sizing parameters
        self.volatility_factor = self.config.get('volatility_factor', 1.0)
        self.confidence_factor = self.config.get('confidence_factor', 1.0)
        
        # Portfolio constraints
        self.max_open_positions = self.config.get('max_open_positions', 5)
        self.position_correlation_limit = self.config.get('position_correlation_limit', 0.7)
    
    def calculate_position_size(
        self,
        prediction_probability: float,
        volatility: float,
        correlation_risk: float,
        current_exposure: float,
        account_size: float
    ) -> float:
        """Calculate dynamic position size based on multiple factors."""
        # Base size from prediction confidence
        confidence_multiplier = prediction_probability ** 2  # Square to make it more conservative
        base_size = self.max_position_size * confidence_multiplier
        
        # Adjust for volatility
        volatility_multiplier = 1 / (1 + volatility * self.volatility_factor)
        vol_adjusted_size = base_size * volatility_multiplier
        
        # Adjust for correlation risk
        correlation_multiplier = 1 - (correlation_risk * self.max_correlation_exposure)
        risk_adjusted_size = vol_adjusted_size * max(0.2, correlation_multiplier)  # Minimum 20% of adjusted size
        
        # Account for current exposure
        available_exposure = 1 - current_exposure
        exposure_adjusted_size = min(risk_adjusted_size, available_exposure)
        
        # Convert to absolute position size
        position_size = exposure_adjusted_size * account_size
        
        return position_size
    
    def calculate_adaptive_stops(
        self,
        entry_price: float,
        atr: float,
        volatility: float,
        trend_strength: float,
        position_type: str  # 'long' or 'short'
    ) -> Tuple[float, float]:
        """Calculate adaptive stop-loss and take-profit levels."""
        # Adjust ATR multiplier based on volatility and trend strength
        volatility_factor = 1 + (volatility - volatility.mean()) / volatility.std()
        trend_factor = abs(trend_strength)
        
        adaptive_multiplier = self.base_atr_multiplier * volatility_factor * (1 + trend_factor)
        
        # Calculate stop distances
        stop_distance = max(
            atr * adaptive_multiplier,
            entry_price * self.min_stop_distance
        )
        
        # Set stop-loss and take-profit levels
        if position_type == 'long':
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + (stop_distance * 1.5)  # 1.5:1 reward-risk ratio
        else:  # short
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - (stop_distance * 1.5)
        
        return stop_loss, take_profit
    
    def check_drawdown_breach(self, equity_curve: pd.Series) -> bool:
        """Check if maximum drawdown limit is breached."""
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min())
        
        return max_drawdown > self.max_drawdown
    
    def calculate_portfolio_risk(
        self,
        positions: Dict[str, Dict],
        correlations: pd.DataFrame
    ) -> float:
        """Calculate portfolio-level risk score."""
        if not positions:
            return 0.0
        
        # Calculate position-weighted correlation risk
        total_exposure = sum(pos['size'] for pos in positions.values())
        weighted_correlations = []
        
        for asset1, pos1 in positions.items():
            for asset2, pos2 in positions.items():
                if asset1 >= asset2:  # Avoid double counting
                    continue
                
                correlation = correlations.loc[asset1, asset2]
                weight = (pos1['size'] * pos2['size']) / (total_exposure ** 2)
                weighted_correlations.append(abs(correlation * weight))
        
        return sum(weighted_correlations)
    
    def adjust_for_portfolio_constraints(
        self,
        new_position: Dict,
        current_positions: Dict[str, Dict],
        correlations: pd.DataFrame
    ) -> Dict:
        """Adjust new position based on portfolio constraints."""
        adjusted_position = new_position.copy()
        
        # Check number of open positions
        if len(current_positions) >= self.max_open_positions:
            logger.warning("Maximum number of open positions reached")
            return None
        
        # Calculate portfolio risk with new position
        test_positions = current_positions.copy()
        test_positions[new_position['asset']] = new_position
        portfolio_risk = self.calculate_portfolio_risk(test_positions, correlations)
        
        if portfolio_risk > self.max_correlation_exposure:
            # Reduce position size to meet risk constraints
            risk_ratio = self.max_correlation_exposure / portfolio_risk
            adjusted_position['size'] *= risk_ratio
            logger.info(f"Reduced position size by {(1-risk_ratio)*100:.1f}% to meet portfolio risk constraints")
        
        return adjusted_position
    
    def get_risk_metrics(self, positions: Dict[str, Dict], equity_curve: pd.Series) -> Dict:
        """Calculate current risk metrics."""
        metrics = {}
        
        # Portfolio exposure
        metrics['total_exposure'] = sum(pos['size'] for pos in positions.values())
        
        # Drawdown
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        metrics['current_drawdown'] = abs(drawdown.iloc[-1])
        metrics['max_drawdown'] = abs(drawdown.min())
        
        # Position concentration
        if positions:
            max_position = max(pos['size'] for pos in positions.values())
            metrics['position_concentration'] = max_position / metrics['total_exposure']
        else:
            metrics['position_concentration'] = 0.0
        
        # Risk utilization
        metrics['drawdown_utilization'] = metrics['max_drawdown'] / self.max_drawdown
        metrics['exposure_utilization'] = metrics['total_exposure'] / self.max_position_size
        
        return metrics 
import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class MarketRegime:
    """Market regime detection using volatility and trend metrics."""
    
    def __init__(self, lookback_period: int = 50):
        self.lookback_period = lookback_period
        
    def identify_regime(self, df: pd.DataFrame) -> pd.Series:
        """Identify market regime using volatility and trend metrics."""
        # Calculate volatility metrics
        returns = df['close'].pct_change()
        volatility = returns.rolling(20).std() * np.sqrt(252)
        vol_percentile = volatility.rolling(self.lookback_period).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1]
        )
        
        # Calculate trend metrics
        ema_fast = df['close'].ewm(span=20).mean()
        ema_slow = df['close'].ewm(span=50).mean()
        trend_strength = ((ema_fast / ema_slow - 1) * 100).abs()
        
        # Define regimes
        regimes = pd.Series('neutral', index=df.index)
        
        # High volatility regimes
        high_vol = vol_percentile > 0.7
        
        # Trend regimes
        uptrend = (ema_fast > ema_slow) & (trend_strength > 1.0)
        downtrend = (ema_fast < ema_slow) & (trend_strength > 1.0)
        
        # Assign regime labels
        regimes[high_vol & uptrend] = 'volatile_bullish'
        regimes[high_vol & downtrend] = 'volatile_bearish'
        regimes[high_vol & ~(uptrend | downtrend)] = 'volatile_sideways'
        regimes[~high_vol & uptrend] = 'low_vol_bullish'
        regimes[~high_vol & downtrend] = 'low_vol_bearish'
        regimes[~high_vol & ~(uptrend | downtrend)] = 'low_vol_sideways'
        
        return regimes

class MLSignalFilter:
    """
    Enhanced signal filter with market regime detection and multi-timeframe confirmation.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the signal filter.
        
        Args:
            config: Dictionary with filter configuration
        """
        self.config = config
        self.regime_detector = MarketRegime()
        
    def apply_filters(
        self,
        df: pd.DataFrame,
        side: str,
        higher_tf_data: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """
        Apply enhanced filters to the signals.
        
        Args:
            df: DataFrame with data and signals
            side: 'long' or 'short'
            higher_tf_data: Optional higher timeframe data for confirmation
            
        Returns:
            Boolean Series with filtered signals
        """
        # Initialize mask
        mask = pd.Series(True, index=df.index)
        
        # Get filter configuration
        filters = self.config[side]['filters']
        
        # Detect market regime
        regime = self.regime_detector.identify_regime(df)
        
        # Apply regime-based filters
        mask &= self._apply_regime_filter(regime, side)
        
        # Apply volatility filters
        if filters['volatility']['enabled']:
            mask &= self._apply_volatility_filter(df, side)
            
        # Apply volume filters
        if filters['volume']['enabled']:
            mask &= self._apply_volume_filter(df, side)
            
        # Apply trend filters
        if filters['trend']['enabled']:
            mask &= self._apply_trend_filter(df, side)
            
        # Apply higher timeframe confirmation if available
        if higher_tf_data is not None and filters.get('higher_tf_confirm', {}).get('enabled', False):
            mask &= self._apply_higher_tf_filter(df, higher_tf_data, side)
            
        return mask
        
    def _apply_regime_filter(self, regime: pd.Series, side: str) -> pd.Series:
        """Apply filters based on market regime."""
        if side == 'long':
            # Allow longs in bullish regimes and selective sideways
            return regime.isin(['low_vol_bullish', 'volatile_bullish', 'low_vol_sideways'])
        else:
            # Allow shorts in bearish regimes and selective sideways
            return regime.isin(['low_vol_bearish', 'volatile_bearish', 'low_vol_sideways'])
            
    def _apply_volatility_filter(self, df: pd.DataFrame, side: str) -> pd.Series:
        """
        Apply enhanced volatility filters.
        """
        config = self.config[side]['filters']['volatility']
        
        # ATR-based volatility filter
        atr = df[config['atr_column']]
        atr_percentile = atr.rolling(20).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1]
        )
        
        # Volatility regime filter
        returns = df['close'].pct_change()
        realized_vol = returns.rolling(20).std() * np.sqrt(252)
        vol_regime = realized_vol.rolling(50).mean()
        
        # Combine filters
        return (
            (atr_percentile >= config['atr_percentile_min']) &
            (atr_percentile <= config['atr_percentile_max']) &
            (realized_vol <= vol_regime * 1.5)  # Avoid extreme volatility
        )
        
    def _apply_volume_filter(self, df: pd.DataFrame, side: str) -> pd.Series:
        """
        Apply enhanced volume filters.
        """
        config = self.config[side]['filters']['volume']
        
        # Volume ratio filter
        volume = df['volume']
        volume_ma = volume.rolling(20).mean()
        volume_ratio = volume / volume_ma
        
        # Volume trend filter
        volume_trend = volume_ma.pct_change(20)
        
        # Combine filters
        return (
            (volume_ratio >= config['volume_ratio_min']) &
            (volume_trend > -0.2)  # Avoid declining volume
        )
        
    def _apply_trend_filter(self, df: pd.DataFrame, side: str) -> pd.Series:
        """
        Apply enhanced trend filters with multiple indicators.
        """
        # Calculate EMAs
        ema_20 = df['close'].ewm(span=20).mean()
        ema_50 = df['close'].ewm(span=50).mean()
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9).mean()
        
        if side == 'long':
            return (
                (df['close'] > ema_20) &  # Price above EMA20
                (ema_20 > ema_50) &       # EMA20 above EMA50
                (rsi > 40) &              # RSI not oversold
                (rsi < 70) &              # RSI not overbought
                (macd > macd_signal)      # MACD bullish
            )
        else:
            return (
                (df['close'] < ema_20) &  # Price below EMA20
                (ema_20 < ema_50) &       # EMA20 below EMA50
                (rsi > 30) &              # RSI not oversold
                (rsi < 60) &              # RSI not overbought
                (macd < macd_signal)      # MACD bearish
            )
            
    def _apply_higher_tf_filter(
        self,
        df: pd.DataFrame,
        higher_tf_data: pd.DataFrame,
        side: str
    ) -> pd.Series:
        """
        Apply confirmation filters using higher timeframe data.
        """
        config = self.config[side]['filters'].get('higher_tf_confirm', {})
        
        # Resample current timeframe to match higher timeframe
        higher_tf_close = higher_tf_data['close']
        
        # Calculate higher timeframe trend
        higher_tf_ema_fast = higher_tf_close.ewm(span=10).mean()
        higher_tf_ema_slow = higher_tf_close.ewm(span=30).mean()
        
        # Get trend direction
        if side == 'long':
            higher_tf_trend = higher_tf_ema_fast > higher_tf_ema_slow
        else:
            higher_tf_trend = higher_tf_ema_fast < higher_tf_ema_slow
            
        # Forward fill the higher timeframe signal to match current timeframe
        return higher_tf_trend.reindex(df.index).fillna(method='ffill') 
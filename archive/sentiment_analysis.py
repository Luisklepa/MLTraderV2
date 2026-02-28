import pandas as pd
import numpy as np
from typing import Dict, Optional
import requests
from datetime import datetime, timedelta
import json
import logging

logger = logging.getLogger(__name__)

class MarketSentimentAnalyzer:
    """Analyzes market sentiment from various sources."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the sentiment analyzer with configuration."""
        self.config = config or {}
        self.cache = {}
        
    def calculate_fear_greed_index(self, df: pd.DataFrame) -> pd.Series:
        """Calculate a custom fear/greed index based on market data."""
        # Volatility component (20% weight)
        volatility = df['atr_14'] / df['close']
        volatility_score = 100 - (volatility / volatility.rolling(30).max() * 100)
        
        # Momentum component (20% weight)
        momentum = df['close'].pct_change(5)
        momentum_score = ((momentum + 1) / 2) * 100
        
        # Market strength component (20% weight)
        strength = (df['close'] > df['ema_20']).astype(int)
        strength_score = strength.rolling(10).mean() * 100
        
        # RSI component (20% weight)
        rsi_score = df['rsi_14']
        
        # Volume component (20% weight)
        volume_ratio = df['volume'] / df['volume'].rolling(20).mean()
        volume_score = (volume_ratio / volume_ratio.rolling(30).max()) * 100
        
        # Combine components
        fear_greed = (
            0.2 * volatility_score +
            0.2 * momentum_score +
            0.2 * strength_score +
            0.2 * rsi_score +
            0.2 * volume_score
        )
        
        return fear_greed
    
    def calculate_market_regime(self, df: pd.DataFrame) -> pd.Series:
        """Determine market regime (accumulation, markup, distribution, markdown)."""
        # Price relative to moving averages
        price_trend = np.where(
            df['close'] > df['ema_50'],
            np.where(df['close'] > df['ema_20'], 2, 1),
            np.where(df['close'] > df['ema_200'], 0, -1)
        )
        
        # Volume trend
        volume_trend = np.where(
            df['volume'] > df['volume'].rolling(20).mean(),
            1, -1
        )
        
        # Volatility regime
        volatility = df['atr_14'] / df['close']
        volatility_regime = np.where(
            volatility > volatility.rolling(20).mean(),
            1, -1
        )
        
        # Combine indicators to determine regime
        regime = pd.Series(index=df.index, dtype='str')
        
        # Accumulation: Low volatility, increasing volume, price stabilizing
        regime.loc[(price_trend == -1) & (volume_trend == 1) & (volatility_regime == -1)] = 'accumulation'
        
        # Markup: Strong trend, high volume, moderate volatility
        regime.loc[(price_trend >= 1) & (volume_trend == 1)] = 'markup'
        
        # Distribution: High volatility, decreasing volume, price resistance
        regime.loc[(price_trend == 2) & (volume_trend == -1) & (volatility_regime == 1)] = 'distribution'
        
        # Markdown: Strong downtrend, high volume, high volatility
        regime.loc[(price_trend == -1) & (volatility_regime == 1)] = 'markdown'
        
        # Default to neutral
        regime.fillna('neutral', inplace=True)
        
        return regime
    
    def add_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all sentiment-related features to the dataframe."""
        result = df.copy()
        
        # Calculate fear/greed index
        result['fear_greed_index'] = self.calculate_fear_greed_index(result)
        result['fear_greed_regime'] = pd.qcut(
            result['fear_greed_index'],
            q=5,
            labels=['extreme_fear', 'fear', 'neutral', 'greed', 'extreme_greed']
        )
        
        # Calculate market regime
        result['market_regime'] = self.calculate_market_regime(result)
        
        # Regime transitions
        result['regime_change'] = (result['market_regime'] != result['market_regime'].shift(1)).astype(int)
        
        # Sentiment momentum
        result['fear_greed_momentum'] = result['fear_greed_index'].diff()
        result['fear_greed_zscore'] = (
            (result['fear_greed_index'] - result['fear_greed_index'].rolling(20).mean()) /
            result['fear_greed_index'].rolling(20).std()
        )
        
        # Extreme sentiment signals
        result['extreme_fear'] = (result['fear_greed_index'] < 20).astype(int)
        result['extreme_greed'] = (result['fear_greed_index'] > 80).astype(int)
        
        # Sentiment divergence
        result['sentiment_price_divergence'] = np.where(
            (result['close'] > result['close'].shift(1)) & (result['fear_greed_index'] < result['fear_greed_index'].shift(1)),
            -1,  # Bearish divergence
            np.where(
                (result['close'] < result['close'].shift(1)) & (result['fear_greed_index'] > result['fear_greed_index'].shift(1)),
                1,  # Bullish divergence
                0
            )
        )
        
        return result 
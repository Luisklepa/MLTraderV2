import backtrader as bt
import numpy as np
from utils.trading_utils import calculate_position_size
from collections import defaultdict
import pandas as pd
import logging
from typing import Dict, Optional, List, Tuple, Any
import yaml
from pathlib import Path
from datetime import datetime, timedelta

from utils.sentiment_analysis import MarketSentimentAnalyzer
from utils.cross_asset_analysis import CrossAssetAnalyzer
from utils.risk_management import RiskManager
from utils.ensemble_model import EnsembleModel
from utils.ml_pipeline import MLPipeline, PipelineConfig
from utils.data_feed import DataFeed
import talib

logger = logging.getLogger(__name__)

class MLStrategy(bt.Strategy):
    """
    Strategy that trades based on ML signals with proper risk management.
    Implements ATR-based stops and position sizing for both long and short positions.
    """
    params = (
        ('stop_loss', 0.02),           # Base stop loss percentage
        ('risk_per_trade', 0.02),      # Base risk per trade as fraction of portfolio
        ('signal_threshold', 0.7),      # Minimum signal strength to enter trade
        ('profit_target_r', 2.0),      # Profit target as multiple of risk
        ('max_position_size', 0.25),    # Maximum position size as fraction of portfolio
        ('atr_period', 14),            # Period for ATR calculation
        ('atr_stop_mult', 2.0),        # ATR multiplier for stop loss
        ('vol_adjust_factor', 0.5),     # Factor to adjust position size based on volatility
        ('printlog', False)            # Whether to print trade logs
    )

    def __init__(self):
        """Initialize strategy with optimized components."""
        super().__init__()
        
        # Store lines for easy access
        self.signal = self.data.target
        self.signal_strength = self.data.signal_strength
        self.future_return = self.data.future_return
        
        # Add ATR for adaptive stops
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        
        # Add volatility indicators
        self.vol_fast = bt.indicators.StdDev(self.data.close, period=20)
        self.vol_slow = bt.indicators.StdDev(self.data.close, period=50)
        self.vol_ratio = self.vol_fast / self.vol_slow
        
        # Initialize trade management
        self.orders = {}  # Track open orders
        self.stops = {}   # Track stop orders
        self.active_positions = defaultdict(dict)  # Track position details
        
        # Performance tracking
        self.trade_stats = defaultdict(int)
        self.equity_curve = []
        
    def next(self):
        """Main strategy logic executed on each bar."""
        # Skip if we have pending orders
        if self.orders:
            return
            
        # Update stops for existing positions
        self.update_stops()
        
        # Check for new signals if we can take more positions
        if len(self.active_positions) < 3:  # Maximum 3 concurrent positions
            self.check_signals()
            
        # Track equity curve
        self.equity_curve.append(self.broker.getvalue())
        
    def get_volatility_adjustment(self) -> float:
        """Calculate position size adjustment based on volatility regime."""
        vol_ratio = self.vol_ratio[0]
        
        # Reduce position size in high volatility environments
        if vol_ratio > 1.5:  # High volatility
            return 0.5
        elif vol_ratio > 1.2:  # Moderate-high volatility
            return 0.75
        elif vol_ratio < 0.8:  # Low volatility
            return 1.25
        else:  # Normal volatility
            return 1.0
            
    def calculate_adaptive_stop(self, price: float, side: str) -> float:
        """Calculate adaptive stop loss based on ATR."""
        atr_value = self.atr[0]
        base_stop = price * self.p.stop_loss
        atr_stop = atr_value * self.p.atr_stop_mult
        
        # Use the larger of percentage-based or ATR-based stop
        stop_distance = max(base_stop, atr_stop)
        
        if side == 'long':
            return price - stop_distance
        else:
            return price + stop_distance
            
    def calculate_position_size(self) -> float:
        """Calculate position size based on risk parameters and volatility."""
        # Get current price and volatility adjustment
        price = self.data.close[0]
        vol_adj = self.get_volatility_adjustment()
        
        # Calculate risk amount
        portfolio_value = self.broker.getvalue()
        risk_amount = portfolio_value * self.p.risk_per_trade * vol_adj
        
        # Calculate stop distance using ATR
        stop_distance = self.atr[0] * self.p.atr_stop_mult
        
        # Calculate position size
        size = risk_amount / (stop_distance * price)
        
        # Apply maximum position size limit
        max_size = portfolio_value * self.p.max_position_size / price
        size = min(size, max_size)
        
        return size
        
    def enter_long(self, size):
        """Enter a long position."""
        price = self.data.close[0]
        stop_price = self.calculate_adaptive_stop(price, 'long')
        target_price = price + (price - stop_price) * self.p.profit_target_r
        
        # Place entry order
        self.orders[self.data._name] = self.buy(size=size)
        
        # Store position details
        self.active_positions[self.data._name] = {
            'size': size,
            'entry_price': price,
            'stop_price': stop_price,
            'target_price': target_price,
            'bars_held': 0,
            'initial_stop': stop_price  # Store initial stop for trailing
        }
        
        if self.p.printlog:
            logger.info(f'LONG ENTRY: Price: {price:.2f}, Size: {size:.2f}, Stop: {stop_price:.2f}')
            
    def enter_short(self, size):
        """Enter a short position."""
        price = self.data.close[0]
        stop_price = self.calculate_adaptive_stop(price, 'short')
        target_price = price - (stop_price - price) * self.p.profit_target_r
        
        # Place entry order
        self.orders[self.data._name] = self.sell(size=size)
        
        # Store position details
        self.active_positions[self.data._name] = {
            'size': -size,  # Negative for short positions
            'entry_price': price,
            'stop_price': stop_price,
            'target_price': target_price,
            'bars_held': 0,
            'initial_stop': stop_price  # Store initial stop for trailing
        }
        
        if self.p.printlog:
            logger.info(f'SHORT ENTRY: Price: {price:.2f}, Size: {size:.2f}, Stop: {stop_price:.2f}')
            
    def update_stops(self):
        """Update stop and target orders for all positions."""
        for data_name, pos in list(self.active_positions.items()):
            if not pos:  # Skip if no position
                continue
                
            current_price = self.data.close[0]
            pos['bars_held'] += 1
            
            # Calculate trailing stop
            if pos['size'] > 0:  # Long position
                # Update trailing stop if price has moved in our favor
                if current_price > pos['entry_price']:
                    new_stop = self.calculate_adaptive_stop(current_price, 'long')
                    # Only move stop up, never down
                    pos['stop_price'] = max(new_stop, pos['stop_price'])
                
                # Check stop loss
                if current_price <= pos['stop_price']:
                    self.close(data=self.data)
                    if self.p.printlog:
                        logger.info(f'LONG STOP: Price: {current_price:.2f}, Stop: {pos["stop_price"]:.2f}')
                elif current_price >= pos['target_price']:
                    self.close(data=self.data)
                    if self.p.printlog:
                        logger.info(f'LONG TARGET: Price: {current_price:.2f}, Target: {pos["target_price"]:.2f}')
            else:  # Short position
                # Update trailing stop if price has moved in our favor
                if current_price < pos['entry_price']:
                    new_stop = self.calculate_adaptive_stop(current_price, 'short')
                    # Only move stop down, never up
                    pos['stop_price'] = min(new_stop, pos['stop_price'])
                
                # Check stop loss
                if current_price >= pos['stop_price']:
                    self.close(data=self.data)
                    if self.p.printlog:
                        logger.info(f'SHORT STOP: Price: {current_price:.2f}, Stop: {pos["stop_price"]:.2f}')
                elif current_price <= pos['target_price']:
                    self.close(data=self.data)
                    if self.p.printlog:
                        logger.info(f'SHORT TARGET: Price: {current_price:.2f}, Target: {pos["target_price"]:.2f}')
                        
    def notify_order(self, order):
        """Handle order notifications."""
        if order.status in [order.Submitted, order.Accepted]:
            return  # Order pending
            
        if order.status in [order.Completed]:
            if order.isbuy():
                if self.p.printlog:
                    logger.info(f'BUY EXECUTED: Price: {order.executed.price:.2f}, Size: {order.executed.size:.2f}')
            else:
                if self.p.printlog:
                    logger.info(f'SELL EXECUTED: Price: {order.executed.price:.2f}, Size: {order.executed.size:.2f}')
                    
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            if self.p.printlog:
                logger.warning(f'Order Failed: {order.status}')
                
        # Clear the order
        self.orders[order.data._name] = None
        
    def notify_trade(self, trade):
        """Handle trade notifications."""
        if not trade.isclosed:
            return
            
        # Update trade statistics
        self.trade_stats['total_trades'] += 1
        self.trade_stats['total_pnl'] += trade.pnl
        if trade.pnl > 0:
            self.trade_stats['winning_trades'] += 1
            
        if self.p.printlog:
            logger.info(f'TRADE CLOSED: PnL: {trade.pnl:.2f}, Commission: {trade.commission:.2f}')
            
    def stop(self):
        """Called when backtesting is finished."""
        # Calculate final statistics
        total_trades = self.trade_stats['total_trades']
        if total_trades > 0:
            self.trade_stats['win_rate'] = (self.trade_stats['winning_trades'] / total_trades) * 100
            self.trade_stats['avg_trade'] = self.trade_stats['total_pnl'] / total_trades
            
        if self.p.printlog:
            logger.info('Backtest finished.')
            logger.info(f'Win Rate: {self.trade_stats["win_rate"]:.2f}%')
            logger.info(f'Average Trade: {self.trade_stats["avg_trade"]:.2f}') 

class EnhancedMLStrategy:
    """Simple ML strategy based on technical indicators."""
    
    def __init__(self, config_path: str):
        """Initialize strategy with configuration."""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Initialize state
        self.position = 0
        self.last_signal = 0
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators."""
        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        df = df.copy()
        
        # RSI
        df.loc[:, 'rsi'] = talib.RSI(df['close'], timeperiod=14)
        
        # MACD
        macd, signal, hist = talib.MACD(
            df['close'],
            fastperiod=12,
            slowperiod=26,
            signalperiod=9
        )
        df.loc[:, 'macd'] = macd
        df.loc[:, 'macd_signal'] = signal
        df.loc[:, 'macd_hist'] = hist
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(
            df['close'],
            timeperiod=20,
            nbdevup=2,
            nbdevdn=2
        )
        df.loc[:, 'bb_upper'] = upper
        df.loc[:, 'bb_middle'] = middle
        df.loc[:, 'bb_lower'] = lower
        
        # ATR
        df.loc[:, 'atr'] = talib.ATR(
            df['high'],
            df['low'],
            df['close'],
            timeperiod=14
        )
        
        # Moving averages
        df.loc[:, 'sma_20'] = talib.SMA(df['close'], timeperiod=20)
        df.loc[:, 'sma_50'] = talib.SMA(df['close'], timeperiod=50)
        
        return df
        
    def generate_signals(self, df: pd.DataFrame) -> List[Dict]:
        """Generate trading signals."""
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        # Get latest data
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Initialize signals list
        signals = []
        
        # Check for long entry
        if (
            current['close'] > current['sma_20'] and  # Price above SMA20
            current['rsi'] > 50 and  # RSI above 50
            current['macd'] > current['macd_signal'] and  # MACD crossover
            prev['macd'] <= prev['macd_signal'] and
            self.position <= 0  # Not already long
        ):
            # Calculate position size
            size = self.calculate_position_size(current)
            
            # Calculate stop loss
            stop_loss = current['close'] - current['atr'] * 2
            
            signals.append({
                'type': 'buy',
                'size': size,
                'stop_loss': stop_loss,
                'reason': 'long_entry'
            })
            
        # Check for short entry
        elif (
            current['close'] < current['sma_20'] and  # Price below SMA20
            current['rsi'] < 50 and  # RSI below 50
            current['macd'] < current['macd_signal'] and  # MACD crossover
            prev['macd'] >= prev['macd_signal'] and
            self.position >= 0  # Not already short
        ):
            # Calculate position size
            size = self.calculate_position_size(current)
            
            # Calculate stop loss
            stop_loss = current['close'] + current['atr'] * 2
            
            signals.append({
                'type': 'sell',
                'size': size,
                'stop_loss': stop_loss,
                'reason': 'short_entry'
            })
            
        # Check for exit
        elif self.position != 0:
            if (
                (self.position > 0 and current['close'] < current['sma_20']) or  # Long exit
                (self.position < 0 and current['close'] > current['sma_20'])  # Short exit
            ):
                signals.append({
                    'type': 'sell' if self.position > 0 else 'buy',
                    'size': abs(self.position),
                    'reason': 'exit'
                })
        
        return signals
        
    def calculate_position_size(self, current: pd.Series) -> float:
        """Calculate position size based on risk parameters."""
        # Get risk parameters
        risk_per_trade = self.config['risk_config']['position_sizing']['base_risk_per_trade']
        max_position = self.config['risk_config']['position_sizing']['max_position_size']
        
        # Calculate volatility-adjusted position size
        vol_scale = 0.02 / (current['atr'] / current['close'])  # Scale to 2% volatility
        position_size = risk_per_trade * vol_scale
        
        # Apply maximum position limit
        position_size = min(position_size, max_position)
        
        return position_size
        
    def on_data(self, df: pd.DataFrame) -> List[Dict]:
        """Process new data and generate orders."""
        try:
            # Generate signals
            signals = self.generate_signals(df)
            
            # Update position
            for signal in signals:
                if signal['type'] == 'buy':
                    self.position += signal['size']
                else:  # sell
                    self.position -= signal['size']
            
            return signals
            
        except Exception as e:
            logger.error(f"Error in strategy: {str(e)}")
            return [] 
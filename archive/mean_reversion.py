"""
Mean Reversion Strategy Implementation
"""
import backtrader as bt

class MeanReversionStrategy(bt.Strategy):
    """Optimized mean reversion strategy"""
    
    params = dict(
        # Bollinger Bands Parameters
        bb_period=20,
        bb_dev=2.0,
        
        # RSI Parameters
        rsi_period=14,
        rsi_lower=30,
        rsi_upper=70,
        
        # Risk Management
        atr_mult=2.0,
        risk_perc=0.02,
        printlog=False,
    )
    
    def __init__(self):
        """Initialize strategy specific indicators"""
        super().__init__()
        
        # Mean Reversion Indicators
        self.bb = bt.ind.BollingerBands(
            self.data.close, 
            period=self.p.bb_period, 
            devfactor=self.p.bb_dev
        )
        self.rsi = bt.ind.RSI(self.data.close, period=self.p.rsi_period)
        
        # State Variables
        self.is_mean_reverting = True
    
    def next(self):
        """Main strategy logic executed on each bar"""
        if self.order:
            return
        
        price = self.data.close[0]
        
        if self.position:
            self._manage_position(price)
        else:
            self._check_signals(price)
    
    def _check_signals(self, price: float):
        """Check for mean reversion signals"""
        stop_distance = self.atr[0] * self.p.atr_mult
        size = self.calculate_position_size(stop_distance)
        
        if size <= 0:
            return
        
        # Long signal: Price near lower band and RSI oversold
        if (price <= self.bb.bot[0] * 1.01 and 
            self.rsi[0] <= self.p.rsi_lower):
            self.log(f'BUY @ {price:.2f} (Oversold)')
            self.order = self.buy(size=size)
            self.entry_price = price
            self.stop_loss = price - stop_distance
        
        # Short signal: Price near upper band and RSI overbought
        elif (price >= self.bb.top[0] * 0.99 and 
              self.rsi[0] >= self.p.rsi_upper):
            self.log(f'SELL @ {price:.2f} (Overbought)')
            self.order = self.sell(size=size)
            self.entry_price = price
            self.stop_loss = price + stop_distance
    
    def _manage_position(self, price: float):
        """Manage existing positions"""
        if self.position.size > 0:  # Long position
            if (price >= self.bb.top[0] or 
                self.rsi[0] >= self.p.rsi_upper or
                (self.stop_loss and price <= self.stop_loss)):
                self.log(f'CLOSE LONG @ {price:.2f}')
                self.order = self.close()
        else:  # Short position
            if (price <= self.bb.bot[0] or 
                self.rsi[0] <= self.p.rsi_lower or
                (self.stop_loss and price >= self.stop_loss)):
                self.log(f'CLOSE SHORT @ {price:.2f}')
                self.order = self.close() 
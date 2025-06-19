import backtrader as bt
import numpy as np
from utils.trading_utils import calculate_position_size
from collections import defaultdict

class MLStrategy(bt.Strategy):
    """
    Strategy that trades based on ML signals with proper risk management.
    Implements ATR-based stops and position sizing for both long and short positions.
    """
    params = (
        ('stop_loss', 0.02),
        ('trail_percent', 0.01),
        ('risk_per_trade', 0.02),
        ('profit_target_r', 2.0),
        ('scale_out_r', 2.0),
        ('scale_in_r', 1.0),
        ('ema_fast', 10),
        ('ema_medium', 20),
        ('ema_slow', 50),
        ('volume_ma', 20),
        ('volume_factor_long', 1.5),    # Factor de volumen para longs
        ('volume_factor_short', 1.0),   # Factor de volumen para shorts
        ('atr_period', 14),
        ('atr_threshold', 1.5),
        ('signal_threshold_long', 0.7),   # Umbral para señales long (70%)
        ('signal_threshold_short', 1.5),  # Umbral para señales short (150%)
    )

    def __init__(self):
        """Initialize strategy components"""
        # Store ML signals
        self.target = self.data.target
        
        # Core indicators
        self.atr = bt.indicators.ATR(period=self.p.atr_period)
        self.volume_ratio = bt.indicators.PctChange(self.data.volume, period=1)
        
        # Volatility indicators
        self.atr_percentile = bt.indicators.PercentRank(self.atr, period=100)
        self.atr_ma = bt.indicators.SMA(self.atr, period=50)
        self.vol_ma = bt.indicators.SMA(self.data.volume, period=20)
        
        # Trend indicators
        self.ema_fast = bt.indicators.EMA(self.data.close, period=self.p.ema_fast)
        self.ema_medium = bt.indicators.EMA(self.data.close, period=self.p.ema_medium)
        self.ema_slow = bt.indicators.EMA(self.data.close, period=self.p.ema_slow)
        
        # Trading state
        self.order = None
        self.entry_price = None
        self.stop_loss = None
        self.trailing_stop = None
        self.profit_target = None
        self.bars_in_trade = 0
        self.current_side = None  # 'long' or 'short'
        self.scale_ins = 0       # Número de escalados realizados
        self.initial_size = 0    # Tamaño inicial de la posición
        self.current_r = 0       # R actual del trade
        
        # Performance tracking
        self._observers = []  # For observers
        self.trade_stats = {
            'long': defaultdict(int),
            'short': defaultdict(int),
            'total': defaultdict(int)
        }
        
        # Store equity curve
        self.equity_curve = []
        
        # Inicializar variables
        self.stop_order = None
        self.profit_order = None
        self.scale_out_order = None
        self.scale_in_order = None
        self.position_size = 0
        self.stop_price = 0
        self.scale_out_price = 0
        self.scale_in_price = 0
        self.risk_amount = 0
        self.closed_trades = []
        self.current_trade = None
        
        # Indicadores
        self.volume_ma = bt.indicators.SMA(self.data.volume, period=self.p.volume_ma)
        
    def log(self, txt, dt=None):
        """Logging function"""
        if self.p.printlog:
            dt = dt or self.data.datetime.date(0)
            print(f'{dt.isoformat()} {txt}')

    def notify_order(self, order):
        """Handle order notifications"""
        if order.status in [order.Submitted, order.Accepted]:
            return
            
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                self.entry_price = order.executed.price
                # Set initial stop for long position
                self.stop_loss = self.entry_price - self.atr[0] * self.p.atr_multiplier
                self.trailing_stop = None
                self.current_side = 'long'
            else:
                if self.current_side:  # This is an exit
                    self.log(f'SELL EXECUTED (EXIT), Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                    # Calculate trade P&L
                    if self.entry_price:
                        pnl = order.executed.price - self.entry_price if self.current_side == 'long' else self.entry_price - order.executed.price
                        self.trade_stats[self.current_side]['total_pnl'] += pnl
                        self.trade_stats['total']['total_pnl'] += pnl
                        
                        if pnl > 0:
                            self.trade_stats[self.current_side]['wins'] += 1
                            self.trade_stats['total']['wins'] += 1
                        else:
                            self.trade_stats[self.current_side]['losses'] += 1
                            self.trade_stats['total']['losses'] += 1
                            
                        self.trade_stats[self.current_side]['trades'] += 1
                        self.trade_stats['total']['trades'] += 1
                else:  # This is a short entry
                    self.log(f'SELL EXECUTED (SHORT ENTRY), Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                    self.entry_price = order.executed.price
                    # Set initial stop for short position
                    self.stop_loss = self.entry_price + self.atr[0] * self.p.atr_multiplier
                    self.trailing_stop = None
                    self.current_side = 'short'
                    
            # Reset state on position exit
            if not self.position:
                self.entry_price = None
                self.stop_loss = None
                self.trailing_stop = None
                self.bars_in_trade = 0
                self.current_side = None
                
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
            
        self.order = None

    def notify_trade(self, trade):
        """Handle trade notifications"""
        if not trade.isclosed:
            return
            
        self.log(f'OPERATION PROFIT, GROSS: {trade.pnl:.2f}, NET: {trade.pnlcomm:.2f}')

    def update_trailing_stop(self, current_price):
        """Update trailing stop if conditions are met"""
        if not self.position or not self.entry_price:
            return
            
        # Calculate current profit in terms of R
        initial_risk = self.atr[0] * self.p.atr_multiplier
        if self.current_side == 'long':
            current_r = (current_price - self.entry_price) / initial_risk
        else:
            current_r = (self.entry_price - current_price) / initial_risk
        
        # Start trailing once we have reached trailing_start R profit
        if current_r >= self.p.trailing_start:
            # Calculate dynamic trailing distance
            # Reduce trailing distance as profit increases
            excess_r = max(0, current_r - self.p.trailing_start)
            reduction_factor = min(0.8, (excess_r / 0.5) * self.p.trailing_atr_reduction)
            trail_distance = max(
                self.p.trailing_distance_min,
                self.p.trailing_distance_initial - reduction_factor
            ) * self.atr[0]
            
            if self.current_side == 'long':
                trail_level = current_price - trail_distance
                if self.trailing_stop is None or trail_level > self.trailing_stop:
                    self.trailing_stop = trail_level
                    self.stop_loss = max(self.stop_loss, self.trailing_stop)
                    if self.p.debug:
                        self.log(f'Long trailing stop updated to: {self.stop_loss:.2f} (R: {current_r:.1f}, Distance: {trail_distance:.2f})')
            else:  # short
                trail_level = current_price + trail_distance
                if self.trailing_stop is None or trail_level < self.trailing_stop:
                    self.trailing_stop = trail_level
                    self.stop_loss = min(self.stop_loss, self.trailing_stop)
                    if self.p.debug:
                        self.log(f'Short trailing stop updated to: {self.stop_loss:.2f} (R: {current_r:.1f}, Distance: {trail_distance:.2f})')

    def check_entry_filters(self):
        """Check if entry filters are satisfied"""
        signal = self.target[0]
        
        # Volume filters - ajustados
        volume_ratio = 1 + self.volume_ratio[0]
        if signal == 1:  # Long
            if volume_ratio < self.p.volume_ratio_threshold * 0.75:  # Reducido de 1.0 a 0.75
                return False
        else:  # Short
            if volume_ratio < self.p.volume_ratio_threshold * 0.2:  # Reducido de 0.25 a 0.2
                return False
            
        if signal == 1:  # Long
            if self.data.volume[0] < self.p.min_volume * 0.75:  # Reducido a 75% para longs
                return False
        else:  # Short
            if self.data.volume[0] < self.p.min_volume * 0.4:  # Reducido a 40% para shorts
                return False
            
        # Volatility filters - ampliado el rango
        if not (self.p.atr_percentile_lower * 0.75 <= self.atr_percentile[0] <= self.p.atr_percentile_upper * 1.25):
            return False
            
        # Trend filter mejorado
        if signal == 1:  # Long
            # Más permisivo: solo necesita EMA rápida > lenta y media > lenta
            if not ((self.ema_fast[0] > self.ema_slow[0]) and (self.ema_medium[0] > self.ema_slow[0])):
                return False
        elif signal == -1:  # Short
            # Mantenemos el filtro simple para shorts
            if not (self.ema_fast[0] < self.ema_slow[0]):
                return False
        else:
            return False
            
        # Signal strength filter - ligeramente más permisivo
        if abs(signal) < self.p.signal_threshold * 0.9:  # Reducido en 10%
            return False
            
        return True

    def update_position_metrics(self, current_price):
        """Actualiza métricas de la posición actual"""
        if not self.position or not self.entry_price:
            return
            
        # Calcular R actual
        initial_risk = self.atr[0] * self.p.atr_multiplier
        if self.current_side == 'long':
            self.current_r = (current_price - self.entry_price) / initial_risk
        else:
            self.current_r = (self.entry_price - current_price) / initial_risk

    def check_scale_in(self, current_price):
        """Verifica si debemos escalar la posición"""
        if not self.position or self.scale_ins >= self.p.max_scale_ins:
            return
            
        # Solo escalamos si estamos ganando y superamos el umbral de R
        if self.current_r >= self.p.scale_in_r * (self.scale_ins + 1):
            # Calcular tamaño adicional
            stop_distance = self.atr[0] * self.p.atr_multiplier
            scale_size = self.initial_size * self.p.scale_in_pct
            
            # Verificar que no excedemos el riesgo máximo
            total_risk = (self.position.size + scale_size) * stop_distance
            max_risk = self.broker.getvalue() * self.p.risk_per_trade * self.p.max_risk_multiplier
            
            if total_risk <= max_risk:
                if self.current_side == 'long':
                    self.log(f'SCALE IN {scale_size:.4f} units at {current_price:.2f} (R: {self.current_r:.1f})')
                    self.order = self.buy(size=scale_size)
                else:
                    self.log(f'SCALE IN {scale_size:.4f} units at {current_price:.2f} (R: {self.current_r:.1f})')
                    self.order = self.sell(size=scale_size)
                self.scale_ins += 1

    def check_scale_out(self, current_price):
        """Verifica si debemos tomar ganancias parciales"""
        if not self.position or self.scale_ins == 0:
            return
            
        if self.current_r >= self.p.scale_out_r:
            # Cerrar una parte de la posición
            scale_out_size = self.position.size * 0.25  # Cerramos 25% de la posición
            if self.current_side == 'long':
                self.log(f'SCALE OUT {scale_out_size:.4f} units at {current_price:.2f} (R: {self.current_r:.1f})')
                self.order = self.sell(size=scale_out_size)
            else:
                self.log(f'SCALE OUT {scale_out_size:.4f} units at {current_price:.2f} (R: {self.current_r:.1f})')
                self.order = self.buy(size=scale_out_size)
            self.scale_ins -= 1

    def next(self):
        """Main strategy logic"""
        if self.order:
            return
            
        if self.position:
            self.bars_in_trade += 1
            
        current_price = self.data.close[0]
        self.equity_curve.append(self.broker.getvalue())
        
        # Actualizar métricas de la posición
        self.update_position_metrics(current_price)
        
        # 1. Position Management
        if self.position:
            # Update trailing stop
            self.update_trailing_stop(current_price)
            
            # Check scale in/out
            self.check_scale_in(current_price)
            self.check_scale_out(current_price)
            
            # Check stops and targets
            if self.current_side == 'long':
                if current_price < self.stop_loss:
                    self.log(f'Long stop hit at {current_price:.2f}, closing position')
                    self.order = self.close()
                elif self.profit_target and current_price >= self.profit_target:
                    self.log(f'Long profit target hit at {current_price:.2f}, closing position')
                    self.order = self.close()
            elif self.current_side == 'short':
                if current_price > self.stop_loss:
                    self.log(f'Short stop hit at {current_price:.2f}, closing position')
                    self.order = self.close()
                elif self.profit_target and current_price <= self.profit_target:
                    self.log(f'Short profit target hit at {current_price:.2f}, closing position')
                    self.order = self.close()
            return
                
        # 2. Entry Logic
        if not self.position and self.check_entry_filters():
            # Calculate position size
            stop_distance = self.atr[0] * self.p.atr_multiplier
            size = calculate_position_size(
                self.broker.getvalue(),
                self.p.risk_per_trade,
                stop_distance
            )
            
            signal = self.target[0]
            
            # Long entry
            if signal > 0:
                self.log(f'BUY CREATE {size} units at {current_price:.2f}')
                self.order = self.buy(size=size)
                self.initial_size = size
                self.scale_ins = 0
                self.current_side = 'long'
                # Set profit target
                self.profit_target = current_price + (stop_distance * self.p.profit_target_r)
                # Set initial stop
                self.stop_loss = current_price - stop_distance
            
            # Short entry
            elif signal < 0:
                self.log(f'SELL CREATE {size} units at {current_price:.2f}')
                self.order = self.sell(size=size)
                self.initial_size = size
                self.scale_ins = 0
                self.current_side = 'short'
                # Set profit target
                self.profit_target = current_price - (stop_distance * self.p.profit_target_r)
                # Set initial stop
                self.stop_loss = current_price + stop_distance

        # Actualizar trailing stop si hay posición
        if self.position and self.stop_order:
            if self.position.size > 0:  # Long position
                new_stop = current_price * (1 - self.p.trail_percent)
                if new_stop > self.stop_price:
                    self.stop_price = new_stop
                    self.log(f'Long trailing stop updated to: {self.stop_price:.2f}')
                    self.cancel(self.stop_order)
                    self.stop_order = self.sell(size=abs(self.position.size), 
                                              exectype=bt.Order.Stop,
                                              price=self.stop_price)
            else:  # Short position
                new_stop = current_price * (1 + self.p.trail_percent)
                if new_stop < self.stop_price:
                    self.stop_price = new_stop
                    self.log(f'Short trailing stop updated to: {self.stop_price:.2f}')
                    self.cancel(self.stop_order)
                    self.stop_order = self.buy(size=abs(self.position.size), 
                                             exectype=bt.Order.Stop,
                                             price=self.stop_price)

    def stop(self):
        """Called when backtest is finished"""
        # Print performance statistics for both long and short
        for side in ['long', 'short', 'total']:
            trades = self.trade_stats[side]['trades']
            if trades > 0:
                win_rate = (self.trade_stats[side]['wins'] / trades) * 100
                avg_pnl = self.trade_stats[side]['total_pnl'] / trades
                print(f'\n=== {side.upper()} Performance ===')
                print(f'Total Trades: {trades}')
                print(f'Win Rate: {win_rate:.2f}%')
                print(f'Total PnL: {self.trade_stats[side]["total_pnl"]:.2f}')
                print(f'Average PnL per Trade: {avg_pnl:.2f}')
                
    def _addobserver(self, enable, observer, *args, **kwargs):
        """Add observer to the strategy"""
        if enable:
            obs = observer(*args, **kwargs)
            obs.strategy = self
            self._observers.append(obs)
            return obs
        return None 
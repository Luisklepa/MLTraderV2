import backtrader as bt
import numpy as np
# Refactor: importar utilidades centralizadas
from utils.trading_utils import (
    get_atr, get_ema, get_rsi,
    calculate_stop_loss, calculate_take_profit, calculate_position_size, print_debug
)

class BreakoutMomentumStrategy(bt.Strategy):
    params = dict(
        # Parámetros optimizados para BTCUSDT 15min
        lookback_periods=12,        # Períodos para detectar breakouts
        ema_fast=21,               # EMA rápida para tendencia
        ema_slow=50,               # EMA lenta para tendencia principal
        atr_period=14,             # ATR para volatilidad
        atr_multiplier=1.8,        # Multiplicador ATR para stop loss
        volume_multiplier=1.3,     # Multiplicador de volumen
        rr_ratio=2.2,              # Risk/Reward ratio
        rsi_period=14,             # RSI
        rsi_lower=35,              # RSI mínimo para long
        rsi_upper=65,              # RSI máximo para short
        momentum_period=10,        # Período para momentum
        min_breakout_size=0.002,   # Mínimo % de breakout (0.2%)
        max_risk_per_trade=0.02,   # Máximo 2% de riesgo por trade
        trailing_distance=1.2,     # Distancia para trailing stop
        consolidation_threshold=0.015,  # Umbral de consolidación (1.5%)
        printlog=False,
        debug=False,
    )

    def __init__(self):
        # Refactor: usar utilidades para indicadores
        self.ema_fast = get_ema(self.data, period=self.p.ema_fast)
        self.ema_slow = get_ema(self.data, period=self.p.ema_slow)
        self.atr = get_atr(self.data, period=self.p.atr_period)
        self.rsi = get_rsi(self.data, period=self.p.rsi_period)
        self.volume_sma = bt.ind.SMA(self.data.volume, period=20)
        
        # Indicadores adicionales para filtros
        self.momentum = bt.ind.Momentum(period=self.p.momentum_period)
        self.bb = bt.ind.BollingerBands(period=20, devfactor=2.0)
        
        # Variables de control
        self.order = None
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.trailing_stop = None
        self.bars_in_trade = 0
        self.last_signal = None
        
        # Métricas
        self.trade_count = 0
        self.win_count = 0
        self.total_pnl = 0
        
        # Para almacenar highs/lows
        self.highs = []
        self.lows = []

    def is_consolidating(self):
        """Detecta si el precio está en consolidación"""
        if len(self.data) < self.p.lookback_periods:
            return False
            
        recent_high = max([self.data.high[-i] for i in range(self.p.lookback_periods)])
        recent_low = min([self.data.low[-i] for i in range(self.p.lookback_periods)])
        
        range_pct = (recent_high - recent_low) / recent_low
        return range_pct < self.p.consolidation_threshold

    def calculate_dynamic_lookback(self):
        """Ajusta el lookback según la volatilidad"""
        if len(self.data) < 20:
            return self.p.lookback_periods
            
        volatility = self.atr[0] / self.data.close[0]
        
        if volatility > 0.025:  # Alta volatilidad
            return max(8, self.p.lookback_periods - 4)
        elif volatility < 0.015:  # Baja volatilidad  
            return min(20, self.p.lookback_periods + 6)
        else:
            return self.p.lookback_periods

    def check_volume_surge(self):
        """Detecta aumentos significativos de volumen"""
        if len(self.data) < 5:
            return False
            
        current_vol = self.data.volume[0]
        avg_vol = self.volume_sma[0]
        
        # Volumen debe ser mayor al promedio y mostrar aceleración
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 0
        recent_vol_avg = np.mean([self.data.volume[-i] for i in range(1, 4)])
        vol_acceleration = current_vol / recent_vol_avg if recent_vol_avg > 0 else 0
        
        return (vol_ratio > self.p.volume_multiplier and vol_acceleration > 1.2)

    def trend_strength(self):
        """Calcula la fuerza de la tendencia"""
        if len(self.data) < self.p.ema_slow:
            return 0
            
        ema_diff = (self.ema_fast[0] - self.ema_slow[0]) / self.ema_slow[0]
        price_vs_ema = (self.data.close[0] - self.ema_slow[0]) / self.ema_slow[0]
        
        return ema_diff + price_vs_ema

    def breakout_quality(self, breakout_price, reference_price, direction):
        """Evalúa la calidad del breakout"""
        breakout_size = abs(breakout_price - reference_price) / reference_price
        
        # El breakout debe ser significativo
        if breakout_size < self.p.min_breakout_size:
            return False
            
        # Verificar que no sea un falso breakout
        atr_normalized = self.atr[0] / self.data.close[0]
        
        if direction == 'long':
            # Para long, el precio debe estar claramente por encima
            return breakout_price > reference_price * (1 + atr_normalized * 0.5)
        else:
            # Para short, el precio debe estar claramente por debajo
            return breakout_price < reference_price * (1 - atr_normalized * 0.5)

    def next(self):
        # Verificar datos suficientes
        min_bars = max(self.p.lookback_periods, self.p.ema_slow, self.p.atr_period)
        if len(self.data) < min_bars:
            return

        # Actualizar contador de barras en trade
        if self.position:
            self.bars_in_trade += 1

        # Calcular lookback dinámico
        lookback = self.calculate_dynamic_lookback()
        
        # Obtener máximos y mínimos
        if len(self.data) > lookback:
            lookback_high = max([self.data.high[-i] for i in range(lookback, 0, -1)])
            lookback_low = min([self.data.low[-i] for i in range(lookback, 0, -1)])
        else:
            return

        # Condiciones de entrada
        current_price = self.data.close[0]
        
        # Detectar breakouts
        breakout_long = current_price > lookback_high
        breakout_short = current_price < lookback_low
        
        # Filtros adicionales
        trend_str = self.trend_strength()
        volume_surge = self.check_volume_surge()
        is_consolidating = self.is_consolidating()
        
        # Filtros RSI mejorados
        rsi_long_ok = self.p.rsi_lower < self.rsi[0] < 80
        rsi_short_ok = 20 < self.rsi[0] < self.p.rsi_upper
        
        # Filtro de momentum
        momentum_long = self.momentum[0] > 0
        momentum_short = self.momentum[0] < 0
        
        # Filtro Bollinger Bands (evitar trades en extremos)
        bb_middle = self.bb.mid[0]
        bb_upper = self.bb.top[0] 
        bb_lower = self.bb.bot[0]
        
        not_overbought = current_price < bb_upper * 0.98
        not_oversold = current_price > bb_lower * 1.02

        # Lógica de entrada
        if not self.position and not self.order:
            
            # Señal LONG
            if (breakout_long and 
                trend_str > 0.005 and  # Tendencia alcista fuerte
                volume_surge and
                rsi_long_ok and
                momentum_long and
                not_overbought and
                is_consolidating and  # Mejor entrada después de consolidación
                self.breakout_quality(current_price, lookback_high, 'long')):
                
                size = self.calculate_position_size()
                if size > 0:
                    self.order = self.buy(size=size)
                    self.entry_price = current_price
                    self.last_signal = 'long'
                    if self.p.debug:
                        print(f"[LONG] Entry @ {current_price:.2f}, Trend: {trend_str:.4f}")
            
            # Señal SHORT  
            elif (breakout_short and
                  trend_str < -0.005 and  # Tendencia bajista fuerte
                  volume_surge and
                  rsi_short_ok and
                  momentum_short and
                  not_oversold and
                  is_consolidating and
                  self.breakout_quality(current_price, lookback_low, 'short')):
                
                size = self.calculate_position_size()
                if size > 0:
                    self.order = self.sell(size=size)
                    self.entry_price = current_price
                    self.last_signal = 'short'
                    if self.p.debug:
                        print(f"[SHORT] Entry @ {current_price:.2f}, Trend: {trend_str:.4f}")

        # Gestión de salidas
        elif self.position:
            self.manage_exits(current_price)

    def calculate_position_size(self):
        """Calcula el tamaño de posición basado en gestión de riesgo (refactorizado)"""
        if self.atr[0] == 0:
            return 0
        account_value = self.broker.getvalue()
        risk_amount = account_value * self.p.max_risk_per_trade
        stop_distance = self.atr[0] * self.p.atr_multiplier
        # Refactor: usar utilidad centralizada
        position_size = calculate_position_size(account_value, self.p.max_risk_per_trade, stop_distance)
        max_position = account_value * 0.3 / self.data.close[0]
        return min(position_size, max_position)

    def manage_exits(self, current_price):
        """Gestión avanzada de salidas"""
        if not self.position:
            return
            
        is_long = self.position.size > 0
        
        # Calcular niveles si no existen
        if self.stop_loss is None or self.take_profit is None:
            self.calculate_exit_levels()
        
        # PnL actual
        if is_long:
            pnl_points = current_price - self.entry_price
        else:
            pnl_points = self.entry_price - current_price
            
        # Ratio de PnL vs riesgo inicial
        initial_risk = self.atr[0] * self.p.atr_multiplier
        pnl_ratio = pnl_points / initial_risk if initial_risk > 0 else 0
        
        # Trailing stop dinámico
        if pnl_ratio > 1.0:  # En ganancia > 1R
            self.update_trailing_stop(current_price, pnl_ratio)
        
        # Condiciones de salida
        exit_signal = False
        exit_reason = ""
        
        # Stop loss
        if is_long and current_price <= self.stop_loss:
            exit_signal = True
            exit_reason = "Stop Loss"
        elif not is_long and current_price >= self.stop_loss:
            exit_signal = True
            exit_reason = "Stop Loss"
        
        # Take profit
        elif is_long and current_price >= self.take_profit:
            exit_signal = True
            exit_reason = "Take Profit"
        elif not is_long and current_price <= self.take_profit:
            exit_signal = True
            exit_reason = "Take Profit"
        
        # Trailing stop
        elif self.trailing_stop is not None:
            if is_long and current_price <= self.trailing_stop:
                exit_signal = True
                exit_reason = "Trailing Stop"
            elif not is_long and current_price >= self.trailing_stop:
                exit_signal = True
                exit_reason = "Trailing Stop"
        
        # Salida por tiempo (evitar trades muy largos)
        elif self.bars_in_trade > 96:  # 24 horas en 15min
            exit_signal = True
            exit_reason = "Time Exit"
        
        # Salida por reversión de momentum
        elif pnl_ratio < -0.5 and self.bars_in_trade > 8:
            momentum_reversal = False
            if is_long and self.momentum[0] < self.momentum[-2] < 0:
                momentum_reversal = True
            elif not is_long and self.momentum[0] > self.momentum[-2] > 0:
                momentum_reversal = True
                
            if momentum_reversal:
                exit_signal = True
                exit_reason = "Momentum Reversal"
        
        # Ejecutar salida
        if exit_signal:
            self.close()
            if self.p.debug:
                print(f"[EXIT] {exit_reason} @ {current_price:.2f}, PnL Ratio: {pnl_ratio:.2f}")
            self.bars_in_trade = 0
            self.last_signal = None
            self.trailing_stop = None

    def calculate_exit_levels(self):
        """Calcula stop loss y take profit (refactorizado)"""
        if not self.position or self.entry_price is None:
            return
        atr_distance = self.atr[0] * self.p.atr_multiplier
        if self.position.size > 0:
            self.stop_loss = calculate_stop_loss(self.entry_price, self.atr[0], self.p.atr_multiplier, direction='long')
            self.take_profit = calculate_take_profit(self.entry_price, self.stop_loss, self.p.rr_ratio, direction='long')
        else:
            self.stop_loss = calculate_stop_loss(self.entry_price, self.atr[0], self.p.atr_multiplier, direction='short')
            self.take_profit = calculate_take_profit(self.entry_price, self.stop_loss, self.p.rr_ratio, direction='short')

    def update_trailing_stop(self, current_price, pnl_ratio):
        """Actualiza el trailing stop"""
        if not self.position:
            return
            
        is_long = self.position.size > 0
        trail_distance = self.atr[0] * self.p.trailing_distance
        
        if is_long:
            new_trailing = current_price - trail_distance
            if self.trailing_stop is None or new_trailing > self.trailing_stop:
                self.trailing_stop = new_trailing
                
                # Proteger breakeven cuando PnL > 1.5R
                if pnl_ratio > 1.5:
                    self.trailing_stop = max(self.trailing_stop, self.entry_price)
        else:
            new_trailing = current_price + trail_distance
            if self.trailing_stop is None or new_trailing < self.trailing_stop:
                self.trailing_stop = new_trailing
                
                # Proteger breakeven cuando PnL > 1.5R
                if pnl_ratio > 1.5:
                    self.trailing_stop = min(self.trailing_stop, self.entry_price)

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.calculate_exit_levels()
            self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            self.trade_count += 1
            if trade.pnlcomm > 0:
                self.win_count += 1
            self.total_pnl += trade.pnlcomm
            
            if self.p.printlog:
                win_rate = (self.win_count / self.trade_count) * 100
                direction = 'LONG' if trade.size > 0 else 'SHORT'
                print(f'Trade #{self.trade_count} {direction}: PnL: {trade.pnlcomm:.2f}, Win Rate: {win_rate:.1f}%')

    def stop(self):
        if self.trade_count > 0:
            win_rate = (self.win_count / self.trade_count) * 100
            avg_pnl = self.total_pnl / self.trade_count
            print(f'\n=== RESULTADOS FINALES ===')
            print(f'Total Trades: {self.trade_count}')
            print(f'Win Rate: {win_rate:.1f}%')
            print(f'Total PnL: {self.total_pnl:.2f}')
            print(f'Avg PnL per Trade: {avg_pnl:.2f}')
            print(f'Valor final: {self.broker.getvalue():.2f}')
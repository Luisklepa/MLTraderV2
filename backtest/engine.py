"""
Backtesting engine with optimization capabilities.
"""
import backtrader as bt
import pandas as pd
from typing import Dict, List, Type, Any
from dataclasses import dataclass
from config.settings import TradingConfig
import time
import plotly.io as pio

@dataclass
class BacktestResult:
    """Structured result container for backtests"""
    strategy_name: str
    final_value: float
    return_pct: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    win_rate: float
    parameters: Dict[str, Any]

class PandasData(bt.feeds.PandasData):
    """Optimized pandas data feed"""
    params = (
        ('datetime', 'open_time'),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('openinterest', None),
    )

class BacktestEngine:
    """Centralized backtesting engine with optimization"""
    
    def __init__(self, initial_cash: float = TradingConfig.INITIAL_CAPITAL):
        self.initial_cash = initial_cash
    
    def run_backtest(self, 
                    strategy_class: Type[bt.Strategy], 
                    data: pd.DataFrame,
                    strategy_params: Dict = None) -> BacktestResult:
        """Run a single backtest"""
        cerebro = bt.Cerebro()
        
        # Add strategy with parameters
        if strategy_params:
            cerebro.addstrategy(strategy_class, **strategy_params)
        else:
            cerebro.addstrategy(strategy_class)
        
        # Add data and configure
        cerebro.adddata(PandasData(dataname=data))
        cerebro.broker.set_cash(self.initial_cash)
        # Usar comisión personalizada si está definida
        commission = getattr(self, 'commission', TradingConfig.COMMISSION_RATE)
        cerebro.broker.setcommission(commission=commission)
        
        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        
        # --- INICIO DEL RESUMEN ---
        start_time = time.time()
        # Extraer info del DataFrame
        if hasattr(data, 'dataname') and hasattr(data.dataname, 'open_time'):
            df = data.dataname
        else:
            df = data
        n_bars = len(df)
        date_start = df['open_time'].iloc[0] if n_bars > 0 else 'N/A'
        date_end = df['open_time'].iloc[-1] if n_bars > 0 else 'N/A'
        symbol = getattr(df, 'symbol', getattr(TradingConfig, 'DEFAULT_SYMBOL', 'BTCUSDT'))
        timeframe = getattr(TradingConfig, 'DEFAULT_TIMEFRAME', '15m')
        # Mostrar prints solo si printlog está activado
        if strategy_params and strategy_params.get('printlog', False):
            print("\n================ BACKTEST EJECUTADO ================" )
        print(f"Símbolo: {symbol}")
        print(f"Timeframe: {timeframe}")
        print(f"Velas procesadas: {n_bars}")
        print(f"Rango de fechas: {date_start}  ->  {date_end}")
        print(f"Parámetros: {strategy_params if strategy_params else '{}'}")
        # --- FIN DEL INICIO DEL RESUMEN ---
        
        # Run backtest
        strat = cerebro.run()[0]
        
        # --- Friendly plot ---
        if strategy_params and strategy_params.get('printlog', False):
            try:
                self.plot_friendly_plotly(df, strat)
            except Exception as e:
                print(f"Error plotting friendly chart: {e}")
        
        # Extract results
        trade_analysis = strat.analyzers.trades.get_analysis()
        total_trades = trade_analysis.get('total', {}).get('closed', 0)
        won_trades = trade_analysis.get('won', {}).get('total', 0)
        
        # Calculate total return correctly
        final_value = cerebro.broker.getvalue()
        total_return = final_value - self.initial_cash
        return_pct = (total_return / self.initial_cash) * 100
        
        # --- RESUMEN FINAL ---
        elapsed = time.time() - start_time
        # Mostrar prints solo si printlog está activado
        if strategy_params and strategy_params.get('printlog', False):
            print("\n================ RESUMEN DEL BACKTEST ================" )
        print(f"Estrategia: {strategy_class.__name__}")
        print(f"Valor final: ${final_value:,.2f}")
        print(f"Retorno total: {return_pct:.2f}%")
        print(f"Sharpe Ratio: {strat.analyzers.sharpe.get_analysis().get('sharperatio', 0) or 0:.3f}")
        print(f"Max Drawdown: {strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0):.2f}%")
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {(won_trades / max(1, total_trades)) * 100:.2f}%")
        print(f"Duración del backtest: {elapsed:.2f} segundos")
        print("====================================================\n")
        # --- FIN DEL RESUMEN FINAL ---
        
        return BacktestResult(
            strategy_name=strategy_class.__name__,
            final_value=final_value,
            return_pct=return_pct,
            sharpe_ratio=strat.analyzers.sharpe.get_analysis().get('sharperatio', 0) or 0,
            max_drawdown=strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0),
            total_trades=total_trades,
            win_rate=(won_trades / max(1, total_trades)) * 100,
            parameters=strategy_params or {}
        )
    
    def optimize_strategy(self, 
                         strategy_class: Type[bt.Strategy], 
                         data: pd.DataFrame,
                         param_grid: List[Dict]) -> List[BacktestResult]:
        """Optimize strategy parameters"""
        results = []
        
        for i, params in enumerate(param_grid, 1):
            try:
                result = self.run_backtest(strategy_class, data, params)
                results.append(result)
                
                if i % 10 == 0:
                    print(f'✓ {i}/{len(param_grid)} combinations tested')
                    
            except Exception as e:
                print(f"Error in combination {i}: {e}")
                continue
        
        return results
    
    def compare_strategies(self, 
                         strategies: Dict[str, Type[bt.Strategy]], 
                         data: pd.DataFrame) -> Dict[str, BacktestResult]:
        """Compare multiple strategies on the same dataset"""
        results = {}
        for name, strategy_class in strategies.items():
            print(f"\n--- Ejecutando backtest para: {name} ---")
            try:
                result = self.run_backtest(strategy_class, data)
                results[name] = result
                print(f"✓ {name} completado")
            except Exception as e:
                print(f"Error al ejecutar {name}: {e}")
        return results 

    def plot_friendly_plotly(self, df, strat):
        """Plot price (candlesticks), equity curve, buy/sell signals in a professional Plotly chart (TradingView style)"""
        pio.renderers.default = 'browser'
        print("Entrando a plot_friendly_plotly...")
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import pandas as pd
        # --- Prepare data ---
        df = df.copy()
        df['open_time'] = pd.to_datetime(df['open_time'])
        print("DF shape:", df.shape)
        # --- Equity ---
        eq_curve = getattr(strat, 'equity_curve', [])
        eq_df = pd.DataFrame(eq_curve, columns=['datetime', 'equity'])
        eq_df['datetime'] = pd.to_datetime(eq_df['datetime'])
        print("Equity shape:", eq_df.shape)
        # --- Signals ---
        buy_signals = getattr(strat, 'buy_signals', [])
        sell_signals = getattr(strat, 'sell_signals', [])
        exit_signals = getattr(strat, 'exit_signals', [])
        buy_df = pd.DataFrame(buy_signals, columns=['datetime', 'price'])
        buy_df['datetime'] = pd.to_datetime(buy_df['datetime'])
        sell_df = pd.DataFrame(sell_signals, columns=['datetime', 'price'])
        sell_df['datetime'] = pd.to_datetime(sell_df['datetime'])
        exit_df = pd.DataFrame(exit_signals, columns=['datetime', 'price'])
        exit_df['datetime'] = pd.to_datetime(exit_df['datetime'])
        print("Buy shape:", buy_df.shape)
        print("Sell shape:", sell_df.shape)
        print("Exit shape:", exit_df.shape)
        # --- Volumen ---
        if 'volume' in df.columns:
            volume = df['volume']
        else:
            volume = None
        # --- Plotly Figure ---
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            row_heights=[0.6, 0.25, 0.15],
            vertical_spacing=0.03,
            subplot_titles=("BTCUSDT Candlestick", "Equity Curve", "Volume") if volume is not None else ("BTCUSDT Candlestick", "Equity Curve")
        )
        # --- Candlestick ---
        fig.add_trace(go.Candlestick(
            x=df['open_time'],
            open=df['open'], high=df['high'], low=df['low'], close=df['close'],
            increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
            name='Candles', showlegend=False
        ), row=1, col=1)
        # --- Buy/Sell/Exit signals con tooltips personalizados ---
        if not buy_df.empty:
            fig.add_trace(go.Scatter(
                x=buy_df['datetime'], y=buy_df['price'],
                mode='markers',
                marker=dict(symbol='triangle-up', color='lime', size=12, line=dict(color='black', width=1)),
                name='LONG',
                showlegend=True,
                hovertemplate='Señal: LONG<br>Fecha: %{x|%Y-%m-%d %H:%M}<br>Precio: %{y:.2f}<extra></extra>'
            ), row=1, col=1)
        if not sell_df.empty:
            fig.add_trace(go.Scatter(
                x=sell_df['datetime'], y=sell_df['price'],
                mode='markers',
                marker=dict(symbol='triangle-down', color='red', size=12, line=dict(color='black', width=1)),
                name='SHORT',
                showlegend=True,
                hovertemplate='Señal: SHORT<br>Fecha: %{x|%Y-%m-%d %H:%M}<br>Precio: %{y:.2f}<extra></extra>'
            ), row=1, col=1)
        if not exit_df.empty:
            fig.add_trace(go.Scatter(
                x=exit_df['datetime'], y=exit_df['price'],
                mode='markers',
                marker=dict(symbol='x', color='deepskyblue', size=14, line=dict(color='black', width=2)),
                name='EXIT',
                showlegend=True,
                hovertemplate='Señal: EXIT<br>Fecha: %{x|%Y-%m-%d %H:%M}<br>Precio: %{y:.2f}<extra></extra>'
            ), row=1, col=1)
        # --- Equity ---
        if not eq_df.empty:
            fig.add_trace(go.Scatter(
                x=eq_df['datetime'], y=eq_df['equity'],
                mode='lines',
                line=dict(color='white', width=2),
                name='Equity',
                showlegend=True
            ), row=2, col=1)
        # --- Volume coloreado según la vela ---
        if volume is not None:
            # Determinar color por barra
            colors = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(df['close'], df['open'])]
            fig.add_trace(go.Bar(
                x=df['open_time'], y=volume,
                marker_color=colors,
                name='Volume', showlegend=False,
                yaxis='y3'
            ), row=3, col=1)
        # --- Layout ---
        try:
            fig.update_layout(
                template='plotly_dark',
                title='BTCUSDT Backtest - TradingView Style',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=7, label="1w", step="day", stepmode="backward"),
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(count=3, label="3m", step="month", stepmode="backward"),
                            dict(step="all", label="All")
                        ])
                    ),
                    rangeslider=dict(visible=False),
                    type="date"
                ),
                xaxis2=dict(rangeslider=dict(visible=False)),
                xaxis3=dict(rangeslider=dict(visible=False)),
                yaxis=dict(
                    title='Precio',
                    side='right',
                    automargin=True,
                    tickformat='.2f',
                    showline=True,
                    linewidth=2,
                    linecolor='white',
                ),
                yaxis2=dict(
                    title='Equity',
                    side='right',
                    automargin=True,
                    showline=True,
                    linewidth=2,
                    linecolor='white',
                ),
                yaxis3=dict(
                    title='Volumen',
                    side='right',
                    automargin=True,
                    showline=True,
                    linewidth=2,
                    linecolor='white',
                ) if volume is not None else None,
                hovermode='x unified',
                margin=dict(l=60, r=80, t=60, b=40),
                dragmode='pan'  # Permite navegar con el click izquierdo
            )
            # NOTA: Los botones de rango solo cambian el rango visible, no la temporalidad de las velas.
        except Exception as e:
            print(f"Error en layout Plotly: {e}")
        # --- Mostrar gráfico ---
        print("Mostrando gráfico Plotly...")
        fig.show(config={
            'scrollZoom': True,
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['zoom2d', 'select2d', 'lasso2d', 'autoScale2d', 'resetScale2d']
        }) 

    def backtest_on_period(self, strategy_class, params, data_subset):
        """Ejecuta un backtest sobre un subconjunto de datos y retorna BacktestResult y analyzers (movido desde TradingApp)"""
        cerebro = bt.Cerebro()
        cerebro.addstrategy(strategy_class, **params)
        cerebro.adddata(PandasData(dataname=data_subset))
        cerebro.broker.set_cash(self.initial_cash)
        commission = getattr(self, 'commission', TradingConfig.COMMISSION_RATE)
        cerebro.broker.setcommission(commission=commission)
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        result = cerebro.run()[0]
        analyzers = result.analyzers
        trade_analysis = analyzers.trades.get_analysis()
        sharpe = analyzers.sharpe.get_analysis().get('sharperatio', 0) or 0
        max_dd = analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0)
        total_trades = trade_analysis.get('total', {}).get('closed', 0)
        won_trades = trade_analysis.get('won', {}).get('total', 0)
        win_rate = (won_trades / max(1, total_trades)) * 100
        final_value = cerebro.broker.getvalue()
        return_pct = ((final_value - self.initial_cash) / self.initial_cash) * 100
        profit_factor = self.calculate_profit_factor(trade_analysis)
        return BacktestResult(
            strategy_name=strategy_class.__name__,
            final_value=final_value,
            return_pct=return_pct,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            total_trades=total_trades,
            win_rate=win_rate,
            parameters=params
        ), profit_factor

    def calculate_profit_factor(self, trade_analysis):
        def extract_pnl(pnl):
            if isinstance(pnl, dict):
                return pnl.get('total', 0)
            return pnl if isinstance(pnl, (int, float)) else 0
        gross_win = extract_pnl(trade_analysis.get('won', {}).get('pnl', 0))
        gross_loss = abs(extract_pnl(trade_analysis.get('lost', {}).get('pnl', 0)))
        if gross_loss == 0:
            return 1.0 if gross_win == 0 else float('inf')
        return gross_win / gross_loss 

    def calculate_robust_score(self, backtest_result, profit_factor):
        # Consistente con SmartOptimizer._robust_score
        if (backtest_result.total_trades < 10 or backtest_result.max_drawdown > 25 or
            backtest_result.sharpe_ratio < 0.5 or profit_factor < 1.2):
            return -float('inf')
        score = (
            backtest_result.sharpe_ratio * 2.0 +
            profit_factor * 1.5 +
            (backtest_result.win_rate / 100.0) * 1.0 -
            (backtest_result.max_drawdown / 25.0) * 1.0
        )
        return score 
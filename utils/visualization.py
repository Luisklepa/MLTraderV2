import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class StrategyVisualizer:
    """Visualization tools for strategy analysis"""
    
    def __init__(self, df: pd.DataFrame, trades: List[Dict], initial_capital: float):
        """
        Initialize visualizer
        
        Args:
            df: DataFrame with OHLCV data and signals
            trades: List of trade dictionaries with entry/exit info
            initial_capital: Initial capital for equity curve
        """
        self.df = df
        self.trades = trades
        self.initial_capital = initial_capital
        
        # Process trades into separate long/short lists
        self.long_trades = [t for t in trades if t['side'] == 'long']
        self.short_trades = [t for t in trades if t['side'] == 'short']
        
        # Calculate trade statistics
        self._calculate_trade_stats()
        
    def _calculate_trade_stats(self):
        """Calculate various trade statistics"""
        self.stats = {
            'long': self._get_side_stats(self.long_trades),
            'short': self._get_side_stats(self.short_trades),
            'total': self._get_side_stats(self.trades)
        }
        
    def _get_side_stats(self, trades: List[Dict]) -> Dict:
        """Calculate statistics for a set of trades"""
        if not trades:
            return {
                'count': 0,
                'win_rate': 0,
                'avg_pnl': 0,
                'max_win': 0,
                'max_loss': 0,
                'profit_factor': 0,
                'avg_bars': 0
            }
            
        pnls = [t['pnl'] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        bars = [t['bars'] for t in trades]
        
        return {
            'count': len(trades),
            'win_rate': len(wins) / len(trades) * 100 if trades else 0,
            'avg_pnl': np.mean(pnls) if pnls else 0,
            'max_win': max(wins) if wins else 0,
            'max_loss': min(losses) if losses else 0,
            'profit_factor': abs(sum(wins) / sum(losses)) if losses and wins else 0,
            'avg_bars': np.mean(bars) if bars else 0
        }
        
    def plot_equity_curve(self, use_plotly: bool = True):
        """Plot equity curve with drawdown"""
        equity = self._calculate_equity_curve()
        drawdown = self._calculate_drawdown(equity)
        
        if use_plotly:
            fig = make_subplots(rows=2, cols=1, 
                              shared_xaxes=True,
                              vertical_spacing=0.03,
                              subplot_titles=('Equity Curve', 'Drawdown'),
                              row_heights=[0.7, 0.3])
            
            # Equity curve
            fig.add_trace(
                go.Scatter(x=equity.index, y=equity.values,
                          name='Equity',
                          line=dict(color='blue')),
                row=1, col=1
            )
            
            # Drawdown
            fig.add_trace(
                go.Scatter(x=drawdown.index, y=drawdown.values,
                          name='Drawdown',
                          fill='tozeroy',
                          line=dict(color='red')),
                row=2, col=1
            )
            
            fig.update_layout(
                title='Strategy Equity Curve and Drawdown',
                showlegend=True,
                height=800
            )
            
            return fig
            
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
            
            # Plot equity curve
            ax1.plot(equity.index, equity.values, label='Equity', color='blue')
            ax1.set_title('Strategy Equity Curve')
            ax1.grid(True)
            ax1.legend()
            
            # Plot drawdown
            ax2.fill_between(drawdown.index, drawdown.values, color='red', alpha=0.3)
            ax2.set_title('Drawdown')
            ax2.grid(True)
            
            plt.tight_layout()
            return fig
            
    def plot_trade_distribution(self, use_plotly: bool = True):
        """Plot distribution of trade returns"""
        long_pnls = [t['pnl'] for t in self.long_trades]
        short_pnls = [t['pnl'] for t in self.short_trades]
        
        if use_plotly:
            fig = make_subplots(rows=1, cols=2,
                              subplot_titles=('Long Trades', 'Short Trades'))
            
            # Long trades histogram
            fig.add_trace(
                go.Histogram(x=long_pnls,
                           name='Long Trades',
                           nbinsx=20,
                           marker_color='green'),
                row=1, col=1
            )
            
            # Short trades histogram
            fig.add_trace(
                go.Histogram(x=short_pnls,
                           name='Short Trades',
                           nbinsx=20,
                           marker_color='red'),
                row=1, col=2
            )
            
            fig.update_layout(
                title='Distribution of Trade Returns',
                showlegend=True,
                height=400
            )
            
            return fig
            
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Plot long trades
            ax1.hist(long_pnls, bins=20, color='green', alpha=0.6)
            ax1.set_title('Long Trades Distribution')
            ax1.grid(True)
            
            # Plot short trades
            ax2.hist(short_pnls, bins=20, color='red', alpha=0.6)
            ax2.set_title('Short Trades Distribution')
            ax2.grid(True)
            
            plt.tight_layout()
            return fig
            
    def plot_monthly_returns(self, use_plotly: bool = True):
        """Plot monthly returns heatmap"""
        # Calculate monthly returns
        equity = self._calculate_equity_curve()
        returns = equity.pct_change()
        monthly_returns = returns.resample('M').agg(lambda x: (1 + x).prod() - 1)
        monthly_returns = monthly_returns.to_frame('returns')
        monthly_returns['year'] = monthly_returns.index.year
        monthly_returns['month'] = monthly_returns.index.month
        pivot = monthly_returns.pivot_table(
            values='returns',
            index='year',
            columns='month',
            aggfunc='sum'
        )
        
        if use_plotly:
            fig = go.Figure(data=go.Heatmap(
                z=pivot.values * 100,  # Convert to percentage
                x=pivot.columns,
                y=pivot.index,
                colorscale='RdYlGn',
                text=np.round(pivot.values * 100, 1),
                texttemplate='%{text:.1f}%',
                textfont={"size": 10},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title='Monthly Returns Heatmap',
                xaxis_title='Month',
                yaxis_title='Year',
                height=400
            )
            
            return fig
            
        else:
            plt.figure(figsize=(12, 6))
            sns.heatmap(pivot * 100,  # Convert to percentage
                       annot=True,
                       fmt='.1f',
                       cmap='RdYlGn',
                       center=0)
            plt.title('Monthly Returns Heatmap')
            return plt.gcf()
            
    def _calculate_equity_curve(self) -> pd.Series:
        """Calculate equity curve from trades"""
        equity = pd.Series(index=self.df.index, data=self.initial_capital)
        
        for trade in self.trades:
            entry_idx = self.df.index.get_loc(trade['entry_time'])
            exit_idx = self.df.index.get_loc(trade['exit_time'])
            equity.iloc[exit_idx:] += trade['pnl']
            
        return equity
        
    def _calculate_drawdown(self, equity: pd.Series) -> pd.Series:
        """Calculate drawdown series"""
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max * 100
        return drawdown
        
    def generate_html_report(self) -> str:
        """Generate HTML report with all visualizations"""
        # Create Plotly figures
        equity_fig = self.plot_equity_curve(use_plotly=True)
        dist_fig = self.plot_trade_distribution(use_plotly=True)
        monthly_fig = self.plot_monthly_returns(use_plotly=True)
        
        # Convert figures to HTML
        html = f"""
        <html>
        <head>
            <title>Strategy Analysis Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <h1>Strategy Analysis Report</h1>
            
            <h2>Performance Statistics</h2>
            <table border="1">
                <tr>
                    <th>Metric</th>
                    <th>Long</th>
                    <th>Short</th>
                    <th>Total</th>
                </tr>
                <tr>
                    <td>Number of Trades</td>
                    <td>{self.stats['long']['count']}</td>
                    <td>{self.stats['short']['count']}</td>
                    <td>{self.stats['total']['count']}</td>
                </tr>
                <tr>
                    <td>Win Rate</td>
                    <td>{self.stats['long']['win_rate']:.1f}%</td>
                    <td>{self.stats['short']['win_rate']:.1f}%</td>
                    <td>{self.stats['total']['win_rate']:.1f}%</td>
                </tr>
                <tr>
                    <td>Average PnL</td>
                    <td>{self.stats['long']['avg_pnl']:.2f}</td>
                    <td>{self.stats['short']['avg_pnl']:.2f}</td>
                    <td>{self.stats['total']['avg_pnl']:.2f}</td>
                </tr>
                <tr>
                    <td>Profit Factor</td>
                    <td>{self.stats['long']['profit_factor']:.2f}</td>
                    <td>{self.stats['short']['profit_factor']:.2f}</td>
                    <td>{self.stats['total']['profit_factor']:.2f}</td>
                </tr>
            </table>
            
            <h2>Equity Curve and Drawdown</h2>
            {equity_fig.to_html(full_html=False, include_plotlyjs=False)}
            
            <h2>Trade Distribution</h2>
            {dist_fig.to_html(full_html=False, include_plotlyjs=False)}
            
            <h2>Monthly Returns</h2>
            {monthly_fig.to_html(full_html=False, include_plotlyjs=False)}
        </body>
        </html>
        """
        
        return html 
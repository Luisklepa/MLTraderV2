import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class StrategyVisualizer:
    """Class for visualizing backtest results."""
    
    def __init__(self, results):
        """
        Initialize visualizer.
        
        Args:
            results: Dictionary with backtest results
        """
        self.results = results
        self.equity_series = results['equity_series']
        self.trades = results['trades']
        
    def plot_equity_curve(self, output_file: Path):
        """Plot equity curve."""
        fig = go.Figure()
        
        # Add equity curve
        fig.add_trace(go.Scatter(
            y=self.equity_series,
            mode='lines',
            name='Equity Curve',
            line=dict(color='blue')
        ))
        
        # Update layout
        fig.update_layout(
            title='Equity Curve',
            xaxis_title='Time',
            yaxis_title='Equity',
            template='plotly_white'
        )
        
        # Save plot
        fig.write_html(str(output_file))
        
    def plot_monthly_returns(self, output_file: Path):
        """Plot monthly returns."""
        # Convert equity series to returns
        returns = self.equity_series.pct_change()
        
        # Group by month
        monthly_returns = returns.groupby(pd.Grouper(freq='M')).sum()
        
        fig = go.Figure()
        
        # Add monthly returns
        fig.add_trace(go.Bar(
            x=monthly_returns.index,
            y=monthly_returns.values * 100,
            name='Monthly Returns',
            marker_color=np.where(monthly_returns > 0, 'green', 'red')
        ))
        
        # Update layout
        fig.update_layout(
            title='Monthly Returns',
            xaxis_title='Month',
            yaxis_title='Return (%)',
            template='plotly_white'
        )
        
        # Save plot
        fig.write_html(str(output_file))
        
    def plot_trade_distribution(self, output_file: Path):
        """Plot trade distribution."""
        if not self.trades:
            return
            
        # Extract PnL from trades
        pnl = [trade['pnl'] for trade in self.trades]
        
        fig = go.Figure()
        
        # Add histogram
        fig.add_trace(go.Histogram(
            x=pnl,
            name='Trade PnL Distribution',
            nbinsx=50,
            marker_color='blue'
        ))
        
        # Update layout
        fig.update_layout(
            title='Trade PnL Distribution',
            xaxis_title='PnL',
            yaxis_title='Count',
            template='plotly_white'
        )
        
        # Save plot
        fig.write_html(str(output_file))

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

def plot_backtest_results(results: pd.DataFrame, output_file: Path) -> None:
    """Plot backtest results."""
    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Equity Curve',
            'Drawdown',
            'Daily Returns Distribution',
            'Position Size',
            'Rolling Sharpe Ratio (30D)',
            'Rolling Win Rate (30D)'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
        row_heights=[0.4, 0.3, 0.3]
    )
    
    # Plot equity curve
    fig.add_trace(
        go.Scatter(
            x=results.index,
            y=results['equity'],
            name='Equity',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    # Plot drawdown
    fig.add_trace(
        go.Scatter(
            x=results.index,
            y=results['drawdown'] * 100,
            name='Drawdown',
            fill='tozeroy',
            line=dict(color='red')
        ),
        row=1, col=2
    )
    
    # Plot daily returns distribution
    daily_returns = results['returns'].dropna()
    fig.add_trace(
        go.Histogram(
            x=daily_returns * 100,
            name='Daily Returns',
            nbinsx=50,
            marker_color='green'
        ),
        row=2, col=1
    )
    
    # Plot position size
    fig.add_trace(
        go.Scatter(
            x=results.index,
            y=results['position'],
            name='Position',
            line=dict(color='purple')
        ),
        row=2, col=2
    )
    
    # Calculate and plot rolling Sharpe ratio
    rolling_returns = results['returns'].rolling(window=30).mean()
    rolling_std = results['returns'].rolling(window=30).std()
    rolling_sharpe = np.sqrt(252) * rolling_returns / rolling_std
    
    fig.add_trace(
        go.Scatter(
            x=results.index,
            y=rolling_sharpe,
            name='Rolling Sharpe',
            line=dict(color='orange')
        ),
        row=3, col=1
    )
    
    # Calculate and plot rolling win rate
    rolling_wins = (results['returns'] > 0).rolling(window=30).mean() * 100
    
    fig.add_trace(
        go.Scatter(
            x=results.index,
            y=rolling_wins,
            name='Rolling Win Rate',
            line=dict(color='blue')
        ),
        row=3, col=2
    )
    
    # Update layout
    fig.update_layout(
        title='Backtest Results',
        showlegend=True,
        height=1200,
        width=1600,
        template='plotly_white'
    )
    
    # Update y-axes labels
    fig.update_yaxes(title_text='Equity', row=1, col=1)
    fig.update_yaxes(title_text='Drawdown (%)', row=1, col=2)
    fig.update_yaxes(title_text='Count', row=2, col=1)
    fig.update_yaxes(title_text='Position Size', row=2, col=2)
    fig.update_yaxes(title_text='Sharpe Ratio', row=3, col=1)
    fig.update_yaxes(title_text='Win Rate (%)', row=3, col=2)
    
    # Update x-axes labels
    fig.update_xaxes(title_text='Date', row=3, col=1)
    fig.update_xaxes(title_text='Date', row=3, col=2)
    fig.update_xaxes(title_text='Returns (%)', row=2, col=1)
    
    # Save plot
    fig.write_html(str(output_file))

def plot_trade_distribution(trades: List[Dict], output_file: str) -> None:
    """Create interactive plots for trade distribution analysis."""
    try:
        # Convert trades to DataFrame
        df = pd.DataFrame(trades)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'PnL Distribution',
                'Trade Duration Distribution',
                'Cumulative PnL',
                'Win Rate by Hour'
            )
        )
        
        # Plot PnL distribution
        fig.add_trace(
            go.Histogram(
                x=df['pnl'],
                name='PnL Distribution',
                nbinsx=50,
                marker_color='blue'
            ),
            row=1, col=1
        )
        
        # Plot trade duration distribution
        fig.add_trace(
            go.Histogram(
                x=df['duration'],
                name='Duration Distribution',
                nbinsx=50,
                marker_color='green'
            ),
            row=1, col=2
        )
        
        # Plot cumulative PnL
        df['cumulative_pnl'] = df['pnl'].cumsum()
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['cumulative_pnl'],
                name='Cumulative PnL',
                line=dict(color='purple', width=1)
            ),
            row=2, col=1
        )
        
        # Plot win rate by hour
        df['hour'] = pd.to_datetime(df['time']).dt.hour
        win_rate_by_hour = df.groupby('hour')['pnl'].apply(lambda x: (x > 0).mean() * 100)
        fig.add_trace(
            go.Bar(
                x=win_rate_by_hour.index,
                y=win_rate_by_hour.values,
                name='Win Rate by Hour',
                marker_color='orange'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='Trade Distribution Analysis',
            height=800,
            showlegend=True,
            template='plotly_white'
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="PnL ($)", row=1, col=1)
        fig.update_xaxes(title_text="Duration (hours)", row=1, col=2)
        fig.update_xaxes(title_text="Trade Number", row=2, col=1)
        fig.update_xaxes(title_text="Hour of Day", row=2, col=2)
        
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        fig.update_yaxes(title_text="Cumulative PnL ($)", row=2, col=1)
        fig.update_yaxes(title_text="Win Rate (%)", row=2, col=2)
        
        # Save plot
        fig.write_html(output_file)
        logger.info(f"Trade distribution plots saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error creating trade distribution plots: {str(e)}", exc_info=True)
        raise

def plot_monthly_returns(results: pd.DataFrame, output_file: str) -> None:
    """Create interactive monthly returns heatmap."""
    try:
        # Calculate monthly returns
        monthly_returns = results['returns'].resample('M').sum() * 100
        
        # Create monthly returns matrix
        returns_matrix = monthly_returns.to_frame().pivot_table(
            index=monthly_returns.index.month,
            columns=monthly_returns.index.year,
            values='returns'
        )
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=returns_matrix.values,
            x=returns_matrix.columns,
            y=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            colorscale='RdYlGn',
            center=0,
            text=np.round(returns_matrix.values, 2),
            texttemplate='%{text}%'
        ))
        
        # Update layout
        fig.update_layout(
            title='Monthly Returns Heatmap',
            xaxis_title='Year',
            yaxis_title='Month',
            height=600,
            template='plotly_white'
        )
        
        # Save plot
        fig.write_html(output_file)
        logger.info(f"Monthly returns heatmap saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error creating monthly returns heatmap: {str(e)}", exc_info=True)
        raise

def plot_risk_metrics(results: pd.DataFrame, output_file: str) -> None:
    """Create interactive plots for risk metrics."""
    try:
        # Calculate rolling metrics
        window = 252  # One trading year
        rolling_returns = results['returns'].rolling(window=window).mean() * 252 * 100
        rolling_vol = results['returns'].rolling(window=window).std() * np.sqrt(252) * 100
        rolling_sharpe = rolling_returns / rolling_vol
        rolling_drawdown = results['drawdown'].rolling(window=window).min() * 100
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Rolling Annual Returns',
                'Rolling Volatility',
                'Rolling Sharpe Ratio',
                'Rolling Maximum Drawdown'
            )
        )
        
        # Plot rolling annual returns
        fig.add_trace(
            go.Scatter(
                x=results.index,
                y=rolling_returns,
                name='Annual Returns',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
        
        # Plot rolling volatility
        fig.add_trace(
            go.Scatter(
                x=results.index,
                y=rolling_vol,
                name='Volatility',
                line=dict(color='red', width=1)
            ),
            row=1, col=2
        )
        
        # Plot rolling Sharpe ratio
        fig.add_trace(
            go.Scatter(
                x=results.index,
                y=rolling_sharpe,
                name='Sharpe Ratio',
                line=dict(color='green', width=1)
            ),
            row=2, col=1
        )
        
        # Plot rolling maximum drawdown
        fig.add_trace(
            go.Scatter(
                x=results.index,
                y=rolling_drawdown,
                name='Max Drawdown',
                line=dict(color='purple', width=1)
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='Rolling Risk Metrics',
            height=800,
            showlegend=True,
            template='plotly_white'
        )
        
        # Update axes labels
        fig.update_yaxes(title_text="Annual Returns (%)", row=1, col=1)
        fig.update_yaxes(title_text="Volatility (%)", row=1, col=2)
        fig.update_yaxes(title_text="Sharpe Ratio", row=2, col=1)
        fig.update_yaxes(title_text="Maximum Drawdown (%)", row=2, col=2)
        
        # Save plot
        fig.write_html(output_file)
        logger.info(f"Risk metrics plots saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error creating risk metrics plots: {str(e)}", exc_info=True)
        raise 
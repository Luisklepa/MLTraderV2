import pandas as pd
import numpy as np
import backtrader as bt
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import json

from strategies.ml_strategy import MLStrategy
from utils.trading_utils import calculate_position_size
from utils.visualization import StrategyVisualizer
from config.settings import TradingConfig
from utils.data_feed import MLSignalData

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for backtesting by generating ML signals.
    
    Args:
        df: Raw DataFrame with features and target
        
    Returns:
        DataFrame with OHLCV and ML signals
    """
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Convert target to signals
    df.loc[:, 'long_signal'] = (df['target'] == 1).astype(float)
    df.loc[:, 'short_signal'] = (df['target'] == -1).astype(float)
    
    # Ensure we have all required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume', 'long_signal', 'short_signal']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Convert numeric columns to float
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'long_signal', 'short_signal']
    for col in numeric_cols:
        df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop any rows with NaN values
    df = df.dropna()
    
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        print(f"[ERROR] Index is not a DatetimeIndex. Type: {type(df.index)}. Attempting to convert...")
        df.index = pd.to_datetime(df.index, errors='coerce')

    # Drop rows with invalid or missing datetimes
    invalid_dt = df.index.isna().sum()
    if invalid_dt > 0:
        print(f"[ERROR] Found {invalid_dt} rows with invalid datetimes. Dropping them.")
        df = df[~df.index.isna()]

    # Print dtype and sample of index
    print(f"[DEBUG] Index dtype: {df.index.dtype}")
    print(f"[DEBUG] Index sample: {df.index[:5]}")

    # Assert index is DatetimeIndex and not string
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.dtype == object:
        raise ValueError("[FATAL] Index is not a valid DatetimeIndex. Please check your CSV and ensure the 'datetime' column is present and parsed as datetime.")
    
    # Sort index
    df = df.sort_index()
    
    # Print min and max datetime for verification
    print(f"[prepare_data] Min datetime: {df.index.min()}, Max datetime: {df.index.max()}, Rows: {len(df)}")
    
    # Check for missing and duplicate datetimes
    missing_dt = df.index.isna().sum()
    duplicate_dt = df.index.duplicated().sum()
    print(f"[CHECK] Missing datetimes in index: {missing_dt}")
    print(f"[CHECK] Duplicate datetimes in index: {duplicate_dt}")
    if missing_dt > 0 or duplicate_dt > 0:
        raise ValueError(f"[FATAL] Found {missing_dt} missing and {duplicate_dt} duplicate datetimes in index. Please fix your data.")

    # Print first 10 index values and their integer representation
    print("[CHECK] First 10 index values:")
    print(df.index[:10])
    print("[CHECK] First 10 index values as int64:")
    print(df.index[:10].view('int64'))
    print("[CHECK] First 10 index values as datetime:")
    print(df.index[:10].to_pydatetime())
    
    return df

def run_backtest(data: pd.DataFrame,
                initial_capital: float = 100000,
                commission: float = 0.001,
                **strategy_params) -> dict:
    """
    Run backtest with ML strategy and return comprehensive metrics.
    
    Args:
        data: DataFrame with OHLCV and ML signals
        initial_capital: Initial capital for backtest
        commission: Commission rate per trade
        **strategy_params: Parameters for MLStrategy
        
    Returns:
        Dictionary with backtest results and metrics
    """
    # Create backtrader engine
    cerebro = bt.Cerebro()
    
    # Add data feed
    data_feed = MLSignalData(
        dataname=data
    )
    cerebro.adddata(data_feed)
    
    # Add strategy
    cerebro.addstrategy(MLStrategy, **strategy_params)
    
    # Set broker parameters
    cerebro.broker.setcash(initial_capital)
    cerebro.broker.setcommission(commission=commission)
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    # Run backtest
    results = cerebro.run()
    strategy = results[0]
    
    # Extract metrics
    sharpe_ratio = strategy.analyzers.sharpe.get_analysis()['sharperatio']
    returns = strategy.analyzers.returns.get_analysis()
    drawdown = strategy.analyzers.drawdown.get_analysis()
    trades = strategy.analyzers.trades.get_analysis()
    
    # Calculate additional metrics
    total_return = (cerebro.broker.getvalue() / initial_capital - 1) * 100
    equity_curve = pd.Series(strategy.equity_curve)
    
    # Prepare trade list
    trade_list = []
    # Defensive extraction for Backtrader's TradeAnalyzer output
    if isinstance(trades, dict):
        closed_trades = trades.get('closed', [])
        for trade in closed_trades:
            trade_list.append({
                'entry_date': trade.get('open_datetime', ''),
                'exit_date': trade.get('close_datetime', ''),
                'entry_price': trade.get('open_price', 0),
                'exit_price': trade.get('close_price', 0),
                'size': trade.get('size', 0),
                'pnl': trade.get('pnl', 0),
                'return_pct': (trade.get('pnl', 0) / (trade.get('open_price', 1) * abs(trade.get('size', 1)))) * 100 if trade.get('open_price') and trade.get('size') else 0,
                'direction': 'Long' if trade.get('size', 0) > 0 else 'Short'
            })
    
    # Compile results
    metrics = {
        'initial_capital': initial_capital,
        'final_capital': cerebro.broker.getvalue(),
        'total_return_pct': total_return,
        'sharpe_ratio': float(sharpe_ratio) if sharpe_ratio is not None and isinstance(sharpe_ratio, (int, float)) and not np.isnan(sharpe_ratio) else 0.0,
        'max_drawdown_pct': drawdown['max']['drawdown'] if drawdown['max']['drawdown'] else 0,
        'max_drawdown_length': drawdown['max']['len'] if drawdown['max']['len'] else 0,
        'total_trades': len(trade_list),
        'win_rate': (sum(1 for t in trade_list if t['pnl'] > 0) / len(trade_list) * 100) if trade_list else 0.0,
        'profit_factor': (sum(t['pnl'] for t in trade_list if t['pnl'] > 0) / abs(sum(t['pnl'] for t in trade_list if t['pnl'] < 0))) if trade_list and any(t['pnl'] < 0 for t in trade_list) else (float('inf') if trade_list else 0.0),
        'avg_trade_pnl': sum(t['pnl'] for t in trade_list) / len(trade_list) if trade_list else 0.0,
        'max_trade_pnl': max((t['pnl'] for t in trade_list), default=0.0),
        'min_trade_pnl': min((t['pnl'] for t in trade_list), default=0.0),
        'avg_trade_length': 0,  # No podemos calcular esto sin acceso a los datos originales
        'equity_series': equity_curve,
        'trades': trade_list,
        'strategy_params': strategy_params
    }
    
    # Print summary
    print("\nBacktest Results")
    print("-" * 40)
    print(f"Total Return: {metrics['total_return_pct']:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    if trade_list:
        print(f"Win Rate: {metrics['win_rate']:.2f}%")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Average Trade PnL: ${metrics['avg_trade_pnl']:.2f}")
        print(f"Best Trade: ${metrics['max_trade_pnl']:.2f}")
        print(f"Worst Trade: ${metrics['min_trade_pnl']:.2f}")
    else:
        print("No trades executed")
    
    return metrics

def main():
    # Set up paths
    output_dir = Path('results/walk_forward')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set initial capital
    initial_capital = 100000.0  # Starting with $100,000
    
    # Load and prepare data
    df = pd.read_csv('data/processed/ml_signals_shifted.csv', index_col='datetime', parse_dates=True)
    df = prepare_data(df)
    
    # Run backtest
    results = run_backtest(
        data=df,
        initial_capital=initial_capital
    )
    
    # Visualize results
    visualizer = StrategyVisualizer(df=df, trades=results['trades'], initial_capital=initial_capital)
    equity_fig = visualizer.plot_equity_curve()
    returns_fig = visualizer.plot_monthly_returns()
    trade_dist_fig = visualizer.plot_trade_distribution()
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save metrics
    with open(output_dir / f'metrics_{timestamp}.json', 'w') as f:
        # Convert non-serializable objects
        metrics = results.copy()
        metrics['equity_series'] = metrics['equity_series'].tolist()
        json.dump(metrics, f, indent=2)
    
    # Save plots
    equity_fig.write_html(output_dir / f'equity_curve_{timestamp}.html')
    returns_fig.write_html(output_dir / f'monthly_returns_{timestamp}.html')
    trade_dist_fig.write_html(output_dir / f'trade_distribution_{timestamp}.html')
    
    print(f"\nResults saved to: {output_dir}")

if __name__ == '__main__':
    main() 
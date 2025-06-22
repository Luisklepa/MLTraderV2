#!/usr/bin/env python
"""
Script para ejecutar un ejemplo simple de backtesting.
"""
import argparse
import logging
import yaml
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

from strategies.ml_strategy import EnhancedMLStrategy
from utils.data_feed import DataFeed
from utils.visualization import plot_backtest_results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_backtest(config_path: str, start_date: str, end_date: str, output_dir: str):
    """Run backtest with the enhanced ML strategy."""
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize strategy
    strategy = EnhancedMLStrategy(config_path)
    
    # Initialize data feed
    data_feed = DataFeed(
        symbol=f"{config['data_config']['base_symbol']}{config['data_config']['quote_symbol']}",
        timeframe=config['data_config']['timeframe']
    )
    
    # Get historical data
    df = data_feed.get_historical_data(start_date, end_date)
    logger.info(f"Loaded {len(df)} bars of data from {df.index[0]} to {df.index[-1]}")
    
    # Initialize results with correct data types
    results = pd.DataFrame(index=df.index)
    for col, dtype in [
        ('equity', float),
        ('position', float),
        ('drawdown', float),
        ('returns', float)
    ]:
        results.loc[:, col] = np.zeros(len(df), dtype=dtype)
    results.loc[:, 'equity'] = 100000.0  # Initial capital
    
    # Run backtest
    position = 0.0
    entry_price = 0.0
    trades = []
    
    for i in range(1, len(df)):
        try:
            # Get current data window
            data_window = df.iloc[:i+1]
            
            # Get signals
            signals = strategy.on_data(data_window)
            
            # Process signals
            for signal in signals:
                # Calculate PnL if we have a position
                if position != 0:
                    pnl = position * (df['close'].iloc[i] - entry_price)
                    results.loc[df.index[i], 'equity'] += pnl
                    
                    # Record trade
                    trades.append({
                        'exit_time': df.index[i],
                        'exit_price': df['close'].iloc[i],
                        'pnl': pnl,
                        'return': pnl / (abs(position) * entry_price)
                    })
                
                # Update position
                if signal['type'] == 'buy':
                    position = float(signal['size'])
                    entry_price = float(df['close'].iloc[i])
                    
                    # Record trade entry
                    trades.append({
                        'entry_time': df.index[i],
                        'entry_price': entry_price,
                        'size': position,
                        'type': 'long'
                    })
                else:  # sell
                    position = -float(signal['size'])
                    entry_price = float(df['close'].iloc[i])
                    
                    # Record trade entry
                    trades.append({
                        'entry_time': df.index[i],
                        'entry_price': entry_price,
                        'size': abs(position),
                        'type': 'short'
                    })
            
            # Update results
            results.loc[df.index[i], 'position'] = position
            results.loc[df.index[i], 'equity'] = results.loc[df.index[i-1], 'equity'] if results.loc[df.index[i], 'equity'] == 100000 else results.loc[df.index[i], 'equity']
            results.loc[df.index[i], 'returns'] = results.loc[df.index[i], 'equity'] / results.loc[df.index[i-1], 'equity'] - 1 if results.loc[df.index[i-1], 'equity'] != 0 else 0
            peak = results.loc[:df.index[i], 'equity'].max()
            results.loc[df.index[i], 'drawdown'] = (peak - results.loc[df.index[i], 'equity']) / peak if peak != 0 else 0
            
        except Exception as e:
            logger.error(f"Error at {df.index[i]}: {str(e)}")
            continue
    
    # Plot results
    plot_backtest_results(
        results,
        output_file=output_path / 'backtest_results.html'
    )
    
    # Calculate and print metrics
    total_trades = len([t for t in trades if 'pnl' in t])
    winning_trades = len([t for t in trades if 'pnl' in t and t['pnl'] > 0])
    total_pnl = sum(t['pnl'] for t in trades if 'pnl' in t)
    
    total_return = (results.loc[df.index[-1], 'equity'] / results.loc[df.index[0], 'equity'] - 1) * 100
    max_drawdown = results['drawdown'].max() * 100
    sharpe_ratio = np.sqrt(252) * results['returns'].mean() / results['returns'].std() if results['returns'].std() != 0 else 0
    
    logger.info(f"\nBacktest Results:")
    logger.info(f"Total Return: {total_return:.2f}%")
    logger.info(f"Max Drawdown: {max_drawdown:.2f}%")
    logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    logger.info(f"Total Trades: {total_trades}")
    logger.info(f"Win Rate: {(winning_trades/total_trades*100):.2f}%" if total_trades > 0 else "Win Rate: N/A")
    logger.info(f"Total PnL: ${total_pnl:.2f}")
    logger.info(f"Average Trade: ${(total_pnl/total_trades):.2f}" if total_trades > 0 else "Average Trade: N/A")

def main():
    parser = argparse.ArgumentParser(description='Run example backtest')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--start-date', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    
    args = parser.parse_args()
    
    try:
        run_backtest(
            config_path=args.config,
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=args.output_dir
        )
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main() 
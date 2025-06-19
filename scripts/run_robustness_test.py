import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import backtrader as bt
from strategies.ml_strategy import MLStrategy
from utils.data_feed import prepare_data, MLSignalData
from utils.robustness_metrics import (
    calculate_robustness_metrics,
    calculate_monte_carlo_metrics,
    calculate_regime_metrics,
    calculate_robustness_score
)
from config.robustness_config import RobustnessConfig
import json
import os
from pathlib import Path
from typing import Dict, List, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def identify_market_regimes(data: pd.DataFrame) -> pd.Series:
    """
    Identify market regimes using volatility and trend metrics.
    Returns a Series with regime labels.
    """
    # Calculate volatility (20-day rolling)
    returns = data['close'].pct_change()
    volatility = returns.rolling(20).std() * np.sqrt(252)
    
    # Calculate trend (ratio of price to 50-day MA)
    ma_50 = data['close'].rolling(50).mean()
    trend = data['close'] / ma_50
    
    # Define regimes
    regimes = pd.Series(index=data.index, dtype=str)
    
    # High volatility regimes
    high_vol = volatility > volatility.quantile(0.7)
    
    # Strong trend regimes
    strong_uptrend = trend > trend.quantile(0.7)
    strong_downtrend = trend < trend.quantile(0.3)
    
    # Assign regime labels
    regimes[high_vol & strong_uptrend] = 'volatile_bullish'
    regimes[high_vol & strong_downtrend] = 'volatile_bearish'
    regimes[high_vol & ~(strong_uptrend | strong_downtrend)] = 'volatile_sideways'
    regimes[~high_vol & strong_uptrend] = 'low_vol_bullish'
    regimes[~high_vol & strong_downtrend] = 'low_vol_bearish'
    regimes[~high_vol & ~(strong_uptrend | strong_downtrend)] = 'low_vol_sideways'
    
    return regimes

def run_backtest(data_path: str, start_date=None, end_date=None) -> Dict[str, Any]:
    """Execute backtest for a specific period"""
    # Create cerebro
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100000.0)
    
    # Prepare data
    data = prepare_data(data_path)
    if start_date and end_date:
        data = data[(data.index >= start_date) & (data.index <= end_date)]
    
    # Create data feed
    data_feed = MLSignalData(
        dataname=data,
        fromdate=data.index[0].to_pydatetime() if start_date is None else pd.Timestamp(start_date).to_pydatetime(),
        todate=data.index[-1].to_pydatetime() if end_date is None else pd.Timestamp(end_date).to_pydatetime()
    )
    
    # Add data and strategy
    cerebro.adddata(data_feed)
    cerebro.addstrategy(MLStrategy)
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    
    # Execute backtest
    try:
        results = cerebro.run()
        strategy = results[0]
        
        # Extract trade information
        trades = strategy.analyzers.trades.get_analysis()
        drawdown = strategy.analyzers.drawdown.get_analysis()
        returns = strategy.analyzers.returns.get_analysis()
        
        # Calculate metrics
        total_trades = trades.get('total', {}).get('total', 0)
        if total_trades == 0:
            return None
            
        won_trades = trades.get('won', {}).get('total', 0)
        lost_trades = trades.get('lost', {}).get('total', 0)
        win_rate = won_trades / total_trades if total_trades > 0 else 0
        
        # Extract individual trade data for Monte Carlo
        trade_list = []
        for trade in strategy.closed_trades:
            trade_list.append({
                'entry_date': trade['entry_date'],
                'exit_date': trade['exit_date'],
                'pnl': trade['pnl'],
                'return_pct': trade['return_pct'],
                'direction': trade['direction']
            })
        
        return {
            'dataset': os.path.basename(data_path),
            'period': f"{data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}",
            'total_trades': total_trades,
            'win_rate': win_rate,
            'pnl_total': returns.get('rtot', 0) * 100,  # Convert to percentage
            'avg_trade': returns.get('ravg', 0) * 100,
            'max_drawdown': drawdown.get('max', {}).get('drawdown', 0) * 100,
            'sharpe_ratio': strategy.analyzers.sharpe.get_analysis()['sharperatio'],
            'long_trades': trades.get('long', {}).get('total', 0),
            'short_trades': trades.get('short', {}).get('total', 0),
            'long_win_rate': trades.get('long', {}).get('won', 0) / trades.get('long', {}).get('total', 1),
            'short_win_rate': trades.get('short', {}).get('won', 0) / trades.get('short', {}).get('total', 1),
            'trade_list': trade_list
        }
    except Exception as e:
        print(f"Error in dataset {os.path.basename(data_path)}: {str(e)}")
        return None

def plot_robustness_analysis(results_df: pd.DataFrame, metrics: Dict[str, float], config: RobustnessConfig):
    """Generate comprehensive robustness analysis plots"""
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Returns Distribution',
            'Win Rate by Regime',
            'Trade Frequency',
            'Drawdown Distribution',
            'Long vs Short Performance',
            'Monte Carlo Simulation'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Returns distribution
    fig.add_trace(
        go.Histogram(x=results_df['pnl_total'], name='Returns', nbinsx=20),
        row=1, col=1
    )
    
    # Win rate by regime
    regime_cols = [col for col in results_df.columns if col.endswith('_win_rate')]
    regime_data = results_df[regime_cols].mean()
    fig.add_trace(
        go.Bar(x=regime_data.index, y=regime_data.values, name='Win Rate'),
        row=1, col=2
    )
    
    # Trade frequency
    fig.add_trace(
        go.Histogram(x=results_df['total_trades'], name='Trades', nbinsx=20),
        row=2, col=1
    )
    
    # Drawdown distribution
    fig.add_trace(
        go.Histogram(x=results_df['max_drawdown'], name='Drawdown', nbinsx=20),
        row=2, col=2
    )
    
    # Long vs Short performance
    fig.add_trace(
        go.Scatter(
            x=results_df['long_win_rate'],
            y=results_df['short_win_rate'],
            mode='markers',
            name='Direction Balance'
        ),
        row=3, col=1
    )
    
    # Monte Carlo simulation results
    if 'mc_results' in metrics:
        fig.add_trace(
            go.Histogram(
                x=metrics['mc_results'],
                name='Monte Carlo',
                nbinsx=30
            ),
            row=3, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=config.PLOT_SETTINGS['plot_height'],
        width=config.PLOT_SETTINGS['plot_width'],
        showlegend=True,
        template=config.PLOT_SETTINGS['template'],
        title_text='Strategy Robustness Analysis'
    )
    
    # Save plot
    output_dir = Path(config.RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_dir / 'robustness_analysis.html')

def run_robustness_analysis():
    """Execute robustness analysis across multiple datasets and periods"""
    # Load configuration
    config = RobustnessConfig()
    
    # Get list of datasets
    datasets = [f for f in os.listdir(config.DATA_DIR) if f.startswith('btcusdt_ml_dataset_win')]
    
    # Create output directory
    output_dir = Path(config.RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Store results
    all_results = []
    all_trades = []
    
    # Test each dataset
    for dataset in datasets:
        print(f"\nTesting dataset: {dataset}")
        data_path = os.path.join(config.DATA_DIR, dataset)
        
        # Load data for regime identification
        data = pd.read_csv(data_path)
        data['datetime'] = pd.to_datetime(data['datetime'])
        data.set_index('datetime', inplace=True)
        
        # Identify market regimes
        market_regimes = identify_market_regimes(data)
        
        # Full period test
        print("Running full period test...")
        result = run_backtest(data_path)
        if result:
            all_results.append(result)
            all_trades.extend(result['trade_list'])
        
        # Quarterly tests
        start_date = data.index.min()
        end_date = data.index.max()
        current_date = start_date
        
        while current_date < end_date:
            quarter_end = current_date + pd.DateOffset(months=3)
            print(f"\nTesting period: {current_date.strftime('%Y-%m-%d')} to {quarter_end.strftime('%Y-%m-%d')}")
            
            result = run_backtest(data_path, current_date, quarter_end)
            if result:
                all_results.append(result)
                all_trades.extend(result['trade_list'])
            
            current_date = quarter_end
    
    # Validate minimum requirements
    if len(all_results) < config.MIN_PERIODS:
        print(f"\nError: Insufficient test periods. Found {len(all_results)}, minimum required: {config.MIN_PERIODS}")
        return
    
    if len(all_trades) < config.MIN_TOTAL_TRADES:
        print(f"\nError: Insufficient total trades. Found {len(all_trades)}, minimum required: {config.MIN_TOTAL_TRADES}")
        return
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Calculate comprehensive metrics
    robustness_metrics = calculate_robustness_metrics(all_results)
    monte_carlo_metrics = calculate_monte_carlo_metrics(all_trades, n_simulations=config.N_SIMULATIONS)
    regime_metrics = calculate_regime_metrics(all_results, market_regimes)
    
    # Combine all metrics
    all_metrics = {
        **robustness_metrics,
        **monte_carlo_metrics,
        **regime_metrics
    }
    
    # Calculate overall robustness score
    robustness_score = config.calculate_final_score(all_metrics)
    
    # Validate results
    validations = config.validate_results(all_metrics)
    
    # Generate plots
    plot_robustness_analysis(results_df, all_metrics, config)
    
    # Save detailed results
    results_df.to_csv(output_dir / 'detailed_results.csv', index=False)
    
    # Save metrics and validations
    metrics = {
        'robustness_metrics': robustness_metrics,
        'monte_carlo_metrics': monte_carlo_metrics,
        'regime_metrics': regime_metrics,
        'robustness_score': robustness_score,
        'validations': validations
    }
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Print summary
    print("\n=== Robustness Analysis Summary ===\n")
    print(f"Total periods tested: {len(results_df)}")
    print(f"Total trades analyzed: {len(all_trades)}")
    print(f"\nOverall Robustness Score: {robustness_score:.2f}")
    
    print("\nValidation Results:")
    for criterion, passed in validations.items():
        status = "✓" if passed else "✗"
        print(f"{criterion}: {status}")
    
    print("\nKey Metrics:")
    print(f"Return Mean: {robustness_metrics['return_mean']:.2f}%")
    print(f"Return Std: {robustness_metrics['return_std']:.2f}%")
    print(f"Win Rate Mean: {robustness_metrics['win_rate_mean']:.2%}")
    print(f"Win Rate Std: {robustness_metrics['win_rate_std']:.2%}")
    
    print("\nMonte Carlo Analysis:")
    print(f"95% VaR: {monte_carlo_metrics['mc_var_95']:.2f}%")
    print(f"Probability of Positive Return: {monte_carlo_metrics['mc_positive_prob']:.2%}")
    
    print("\nRegime Analysis:")
    for metric, value in regime_metrics.items():
        if not metric.endswith('consistency'):
            print(f"{metric}: {value:.2f}")
    
    print(f"\nResults saved to: {output_dir}")

if __name__ == '__main__':
    run_robustness_analysis() 
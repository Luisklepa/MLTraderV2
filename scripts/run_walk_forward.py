import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Any, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.walk_forward import WalkForwardAnalyzer
from config.settings import TradingConfig
from config.walk_forward_config import WalkForwardConfig

def evaluate_params(args: tuple) -> Dict[str, Any]:
    """
    Evaluate a single parameter combination.
    This function is used for parallel processing.
    """
    analyzer, params, windows = args
    results = analyzer.run_analysis(params)
    
    # Calculate metrics
    test_sharpes = [r['test_metrics']['sharpe_ratio'] for r in results]
    test_returns = [r['test_metrics']['total_return_pct'] for r in results]
    test_drawdowns = [r['test_metrics']['max_drawdown'] for r in results]
    
    # Calculate robustness metrics
    train_test_metrics = {
        'sharpe_ratio': [
            (r['test_metrics']['sharpe_ratio'] / r['train_metrics']['sharpe_ratio'])
            if r['train_metrics']['sharpe_ratio'] != 0 else 0
            for r in results
        ],
        'total_return': [
            (r['test_metrics']['total_return_pct'] / r['train_metrics']['total_return_pct'])
            if r['train_metrics']['total_return_pct'] != 0 else 0
            for r in results
        ],
        'max_drawdown': [
            (r['test_metrics']['max_drawdown'] / r['train_metrics']['max_drawdown'])
            if r['train_metrics']['max_drawdown'] != 0 else 0
            for r in results
        ]
    }
    
    return {
        'params': params,
        'avg_test_sharpe': np.mean(test_sharpes),
        'avg_test_return': np.mean(test_returns),
        'avg_test_drawdown': np.mean(test_drawdowns),
        'sharpe_std': np.std(test_sharpes),
        'return_std': np.std(test_returns),
        'drawdown_std': np.std(test_drawdowns),
        'train_test_ratios': train_test_metrics,
        'window_results': results
    }

def plot_walk_forward_results(results: List[Dict[str, Any]], output_dir: Path):
    """
    Generate comprehensive walk-forward analysis plots.
    """
    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Train vs Test Sharpe Ratio',
            'Train vs Test Returns',
            'Train vs Test Drawdown',
            'Robustness Ratios',
            'Performance Distribution',
            'Parameter Sensitivity'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Extract metrics
    train_sharpes = [r['train_metrics']['sharpe_ratio'] for r in results]
    test_sharpes = [r['test_metrics']['sharpe_ratio'] for r in results]
    train_returns = [r['train_metrics']['total_return_pct'] for r in results]
    test_returns = [r['test_metrics']['total_return_pct'] for r in results]
    train_drawdowns = [r['train_metrics']['max_drawdown'] for r in results]
    test_drawdowns = [r['test_metrics']['max_drawdown'] for r in results]
    
    # Plot train vs test metrics
    fig.add_trace(
        go.Scatter(y=train_sharpes, name='Train Sharpe', mode='lines+markers'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(y=test_sharpes, name='Test Sharpe', mode='lines+markers'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(y=train_returns, name='Train Returns', mode='lines+markers'),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(y=test_returns, name='Test Returns', mode='lines+markers'),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(y=train_drawdowns, name='Train Drawdown', mode='lines+markers'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(y=test_drawdowns, name='Test Drawdown', mode='lines+markers'),
        row=2, col=1
    )
    
    # Plot robustness ratios
    ratios = [r['test_metrics']['sharpe_ratio'] / r['train_metrics']['sharpe_ratio'] 
              if r['train_metrics']['sharpe_ratio'] != 0 else 0 for r in results]
    fig.add_trace(
        go.Box(y=ratios, name='Test/Train Ratio', boxpoints='all'),
        row=2, col=2
    )
    
    # Plot performance distribution
    fig.add_trace(
        go.Histogram(x=test_returns, name='Test Returns Dist'),
        row=3, col=1
    )
    
    # Plot parameter sensitivity (for the best parameter combination)
    param_names = list(results[0]['params'].keys())
    param_values = [results[0]['params'][p] for p in param_names]
    fig.add_trace(
        go.Bar(x=param_names, y=param_values, name='Best Params'),
        row=3, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=1200,
        showlegend=True,
        title_text='Walk-Forward Analysis Results'
    )
    
    # Save plot
    fig.write_html(output_dir / 'walk_forward_analysis.html')
    fig.write_image(output_dir / 'walk_forward_analysis.png')

def main():
    # Initialize config
    config = WalkForwardConfig()
    
    # Load data
    print("Loading data...")
    DATA_PATH = 'data/processed/ml_signals_shifted.csv'
    data = pd.read_csv(DATA_PATH, index_col='datetime', parse_dates=True)
    
    # Initialize analyzer
    print("Initializing walk-forward analyzer...")
    analyzer = WalkForwardAnalyzer(
        df=data,
        train_size=config.TRAIN_SIZE,
        test_size=config.TEST_SIZE,
        gap=config.GAP_SIZE,
        expanding=config.USE_EXPANDING
    )
    
    # Generate parameter combinations
    param_combinations = []
    param_names = list(config.PARAM_GRID.keys())
    param_values = list(config.PARAM_GRID.values())
    
    def generate_combinations(current_params, idx):
        if idx == len(param_names):
            param_combinations.append(current_params.copy())
            return
        for value in param_values[idx]:
            current_params[param_names[idx]] = value
            generate_combinations(current_params, idx + 1)
    
    generate_combinations({}, 0)
    print(f"\nOptimizing parameters over {len(param_combinations)} combinations...")
    
    # Run optimization in parallel
    with ProcessPoolExecutor() as executor:
        args = [(analyzer, params, analyzer.windows) for params in param_combinations]
        all_results = list(executor.map(evaluate_params, args))
    
    # Find best parameters
    best_result = max(all_results, key=lambda x: x['avg_test_sharpe'])
    best_params = best_result['params']
    
    # Save results
    output_dir = Path(config.RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save best parameters
    with open(output_dir / f'best_params_{timestamp}.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    
    # Save all results
    results_df = pd.DataFrame([
        {
            **r['params'],
            'avg_test_sharpe': r['avg_test_sharpe'],
            'avg_test_return': r['avg_test_return'],
            'avg_test_drawdown': r['avg_test_drawdown'],
            'sharpe_std': r['sharpe_std'],
            'return_std': r['return_std'],
            'drawdown_std': r['drawdown_std']
        }
        for r in all_results
    ])
    results_df.to_csv(output_dir / f'optimization_results_{timestamp}.csv')
    
    # Generate plots
    print("\nGenerating analysis plots...")
    plot_walk_forward_results(best_result['window_results'], output_dir)
    
    # Print summary
    print("\nWalk-Forward Analysis Summary")
    print("-" * 40)
    print(f"Number of windows: {len(analyzer.windows)}")
    print(f"\nBest parameters:")
    print(json.dumps(best_params, indent=2))
    
    # Print detailed metrics
    print("\nPerformance Metrics:")
    print(f"Average Test Sharpe: {best_result['avg_test_sharpe']:.2f} (±{best_result['sharpe_std']:.2f})")
    print(f"Average Test Return: {best_result['avg_test_return']:.2f}% (±{best_result['return_std']:.2f}%)")
    print(f"Average Test Drawdown: {best_result['avg_test_drawdown']:.2f}% (±{best_result['drawdown_std']:.2f}%)")
    
    # Print robustness metrics
    print("\nRobustness Metrics:")
    for metric, ratios in best_result['train_test_ratios'].items():
        mean_ratio = np.mean(ratios)
        std_ratio = np.std(ratios)
        print(f"{metric} Test/Train Ratio: {mean_ratio:.2f} (±{std_ratio:.2f})")
    
    # Check minimum performance criteria
    windows = best_result['window_results']
    windows_above_min_trades = sum(1 for r in windows 
                                 if r['test_metrics']['total_trades'] >= config.MIN_TRADES_PER_WINDOW)
    windows_above_min_sharpe = sum(1 for r in windows 
                                 if r['test_metrics']['sharpe_ratio'] >= config.MIN_SHARPE_RATIO)
    
    print("\nPerformance Criteria:")
    print(f"Windows with sufficient trades: {windows_above_min_trades}/{len(windows)}")
    print(f"Windows above min Sharpe: {windows_above_min_sharpe}/{len(windows)}")
    
    print(f"\nResults saved to: {output_dir}")

if __name__ == '__main__':
    main() 
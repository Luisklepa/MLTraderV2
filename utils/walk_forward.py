import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scripts.run_ml_backtest import run_backtest, prepare_data

class WalkForwardAnalyzer:
    """
    Implements walk-forward analysis for strategy optimization and validation.
    Supports both fixed-size and expanding window analysis.
    """
    def __init__(self,
                df: pd.DataFrame,
                train_size: int = 252,  # 1 year of trading days
                test_size: int = 63,    # ~3 months
                gap: int = 5,           # 1 week gap
                expanding: bool = False):
        """
        Initialize the walk-forward analyzer.
        
        Args:
            df: DataFrame with features and target
            train_size: Number of bars in training window
            test_size: Number of bars in test window
            gap: Number of bars between train and test windows
            expanding: Whether to use expanding window (True) or fixed-size window (False)
        """
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Sort index
        self.df = df.sort_index()
        
        self.train_size = train_size
        self.test_size = test_size
        self.gap = gap
        self.expanding = expanding
        
        # Calculate number of windows
        total_size = len(df)
        min_size = train_size + gap + test_size
        if total_size < min_size:
            raise ValueError(f"Not enough data for walk-forward analysis. Need at least {min_size} bars.")
        
        # Calculate window start indices
        self.windows = []
        start_idx = 0
        while start_idx + min_size <= total_size:
            train_end = start_idx + train_size
            test_start = train_end + gap
            test_end = test_start + test_size
            
            if test_end > total_size:
                break
                
            self.windows.append({
                'train_start': self.df.index[start_idx],
                'train_end': self.df.index[train_end],
                'test_start': self.df.index[test_start],
                'test_end': self.df.index[test_end]
            })
            
            # Update start index
            if expanding:
                # Keep start at 0 for expanding window
                start_idx = 0
            else:
                # Move window forward by test size
                start_idx = test_end
                
    def run_analysis(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run walk-forward analysis with given parameters.
        
        Args:
            params: Dictionary of strategy parameters
            
        Returns:
            Dictionary with walk-forward results
        """
        results = []
        
        for i, window in enumerate(self.windows, 1):
            print(f"\nWindow {i}/{len(self.windows)}")
            
            # Get train/test data
            train_data = self.df[window['train_start']:window['train_end']].copy()
            test_data = self.df[window['test_start']:window['test_end']].copy()
            
            # Skip empty windows
            if train_data.empty or test_data.empty:
                print(f"[WARNING] Skipping window {i}: train_data.empty={train_data.empty}, test_data.empty={test_data.empty}")
                continue
            
            # Print periods
            print(f"Train period: {train_data.index[0]} to {train_data.index[-1]}")
            print(f"Test period: {test_data.index[0]} to {test_data.index[-1]}")
            
            # Prepare data
            train_data = prepare_data(train_data)
            test_data = prepare_data(test_data)
            
            # Run backtest on train data
            train_results = run_backtest(
                data=train_data,
                initial_capital=100000,
                commission=0.001,
                **params
            )
            
            # Run backtest on test data
            test_results = run_backtest(
                data=test_data,
                initial_capital=100000,
                commission=0.001,
                **params
            )
            
            # Store results
            results.append({
                'window': i,
                'train_start': train_data.index[0],
                'train_end': train_data.index[-1],
                'test_start': test_data.index[0],
                'test_end': test_data.index[-1],
                'train_metrics': train_results,
                'test_metrics': test_results
            })
            
        return results
        
    def optimize_parameters(self, param_grid: Dict[str, List[Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Optimize strategy parameters using walk-forward analysis.
        
        Args:
            param_grid: Dictionary mapping parameter names to lists of values
            
        Returns:
            Tuple of (best parameters, all results)
        """
        # Generate all parameter combinations
        param_combinations = []
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        def generate_combinations(current_params, idx):
            if idx == len(param_names):
                param_combinations.append(current_params.copy())
                return
                
            for value in param_values[idx]:
                current_params[param_names[idx]] = value
                generate_combinations(current_params, idx + 1)
                
        generate_combinations({}, 0)
        
        print(f"\nOptimizing parameters over {len(param_combinations)} combinations...")
        
        # Test each combination
        all_results = []
        best_sharpe = -np.inf
        best_params = None
        
        for i, params in enumerate(param_combinations, 1):
            print(f"\nTesting combination {i}/{len(param_combinations)}:")
            print(json.dumps(params, indent=2))
            
            # Run walk-forward analysis
            print("\nRunning walk-forward analysis with", len(self.windows), "windows...")
            wf_results = self.run_analysis(params)
            
            # Calculate average out-of-sample Sharpe ratio
            test_sharpes = [r['test_metrics']['sharpe_ratio'] for r in wf_results]
            avg_test_sharpe = np.mean(test_sharpes)
            
            # Store results
            result = {
                'params': params,
                'avg_test_sharpe': avg_test_sharpe,
                'window_results': wf_results
            }
            all_results.append(result)
            
            # Update best parameters
            if avg_test_sharpe > best_sharpe:
                best_sharpe = avg_test_sharpe
                best_params = params
                
        return best_params, all_results
        
    def plot_results(self, results: List[Dict[str, Any]], output_dir: Path = None):
        """
        Generate plots for walk-forward analysis results.
        
        Args:
            results: List of results from walk-forward analysis
            output_dir: Directory to save plots (optional)
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Train vs Test Sharpe Ratio',
                'Train vs Test Returns',
                'Train vs Test Win Rate',
                'Train vs Test Profit Factor'
            )
        )
        
        # Extract metrics
        train_sharpes = [r['train_metrics']['sharpe_ratio'] for r in results]
        test_sharpes = [r['test_metrics']['sharpe_ratio'] for r in results]
        train_returns = [r['train_metrics']['total_return_pct'] for r in results]
        test_returns = [r['test_metrics']['total_return_pct'] for r in results]
        train_win_rates = [r['train_metrics']['win_rate'] for r in results]
        test_win_rates = [r['test_metrics']['win_rate'] for r in results]
        train_profit_factors = [r['train_metrics']['profit_factor'] for r in results]
        test_profit_factors = [r['test_metrics']['profit_factor'] for r in results]
        
        # Add traces
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
            go.Scatter(y=train_win_rates, name='Train Win Rate', mode='lines+markers'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(y=test_win_rates, name='Test Win Rate', mode='lines+markers'),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(y=train_profit_factors, name='Train PF', mode='lines+markers'),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(y=test_profit_factors, name='Test PF', mode='lines+markers'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text='Walk-Forward Analysis Results',
            showlegend=True
        )
        
        # Save plot if output directory is provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            fig.write_html(output_dir / f'walk_forward_results_{timestamp}.html')
            
        return fig 
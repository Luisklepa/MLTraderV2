import argparse
import logging
import yaml
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta

from utils.ml_backtest import BacktestEngine
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
    
    # Initialize data feeds
    main_feed = DataFeed(
        symbol=f"{config['data_config']['base_symbol']}{config['data_config']['quote_symbol']}",
        timeframe=config['data_config']['timeframe']
    )
    
    reference_feeds = {
        asset: DataFeed(symbol=asset.replace('/', ''), timeframe=config['data_config']['timeframe'])
        for asset in config['data_config']['reference_assets']
    }
    
    # Load historical data
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    
    main_data = main_feed.get_historical_data(start, end)
    reference_data = {
        asset: feed.get_historical_data(start, end)
        for asset, feed in reference_feeds.items()
    }
    
    # Initialize backtest engine
    engine = BacktestEngine(
        strategy=strategy,
        data=main_data,
        reference_data=reference_data,
        config=config
    )
    
    # Run backtest
    logger.info("Starting backtest...")
    results = engine.run()
    
    # Save results
    results_file = output_path / f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results.to_csv(results_file)
    logger.info(f"Results saved to {results_file}")
    
    # Generate plots
    plot_backtest_results(
        results,
        output_path / f"backtest_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    )
    
    # Print summary statistics
    print("\nBacktest Summary:")
    print("-----------------")
    print(f"Total Returns: {results['returns'].sum():.2%}")
    print(f"Annualized Returns: {(1 + results['returns']).prod() ** (252/len(results)) - 1:.2%}")
    print(f"Sharpe Ratio: {results['returns'].mean() / results['returns'].std() * (252 ** 0.5):.2f}")
    print(f"Max Drawdown: {results['drawdown'].min():.2%}")
    print(f"Win Rate: {(results['returns'] > 0).mean():.2%}")
    print(f"Profit Factor: {(results['returns'][results['returns'] > 0].sum() / -results['returns'][results['returns'] < 0].sum()):.2f}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Run ML strategy backtest")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--start-date", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output-dir", type=str, default="backtest_results", help="Output directory for results")
    
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

if __name__ == "__main__":
    main() 
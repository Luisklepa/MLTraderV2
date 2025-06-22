import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import yaml
import json
from datetime import datetime
import matplotlib.pyplot as plt

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.ml_pipeline import MLPipeline, PipelineConfig

def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging configuration."""
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console.setFormatter(console_formatter)
    logger.addHandler(console)
    
    # File handler
    log_file = output_dir / f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger

def load_data(file_path: str) -> pd.DataFrame:
    """Load and preprocess data."""
    print("Loading data...")
    
    # Load data with explicit dtypes
    dtype_dict = {
        'open': np.float64,
        'high': np.float64,
        'low': np.float64,
        'close': np.float64,
        'volume': np.float64
    }
    
    df = pd.read_csv(file_path, dtype=dtype_dict)
    
    # Convert timestamp column
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    elif 'date' in df.columns:
        df['timestamp'] = pd.to_datetime(df['date'])
        df.set_index('timestamp', inplace=True)
        df.drop('date', axis=1, inplace=True, errors='ignore')
    else:
        raise ValueError("DataFrame must have a timestamp or date column")
    
    # Sort by datetime
    df.sort_index(inplace=True)
    
    # Remove duplicates
    df = df[~df.index.duplicated(keep='first')]
    
    # Forward fill missing values
    df = df.ffill()
    
    print("\nData shape:", df.shape)
    print("\nSample data:")
    print(df.head())
    
    return df

def save_results(results: dict, output_dir: Path):
    """Save results to files."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save metrics
    metrics_file = output_dir / f'metrics_{timestamp}.json'
    with open(metrics_file, 'w') as f:
        json.dump({
            'long_metrics': results['long_metrics'],
            'short_metrics': results['short_metrics']
        }, f, indent=4)
    
    # Save feature importance
    importance_file = output_dir / f'feature_importance_{timestamp}.csv'
    results['feature_importance'].to_csv(importance_file)
    
    # Save predictions
    predictions_file = output_dir / f'predictions_{timestamp}.csv'
    results['test_predictions'].to_csv(predictions_file)
    
    print(f"\nResults saved to {output_dir}")

def generate_plots(results: dict, output_dir: Path):
    """Generate analysis plots."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    importance = results['feature_importance']
    importance['mean_importance'] = importance.mean(axis=1)
    importance = importance.sort_values('mean_importance', ascending=True)
    importance.tail(20)['mean_importance'].plot(kind='barh')
    plt.title('Top 20 Features by Importance')
    plt.tight_layout()
    plt.savefig(plots_dir / f'feature_importance_{timestamp}.png')
    plt.close()
    
    # Plot prediction distribution
    plt.figure(figsize=(12, 6))
    preds = results['test_predictions']
    for side in ['long', 'short']:
        plt.hist(
            preds[f'{side}_prob'],
            bins=50,
            alpha=0.5,
            label=f'{side.capitalize()} Probability'
        )
    plt.title('Distribution of Prediction Probabilities')
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / f'prediction_dist_{timestamp}.png')
    plt.close()

def main():
    """Main execution function."""
    import argparse
    parser = argparse.ArgumentParser(description='Run ML pipeline')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--data-file', required=True, help='Path to data file')
    parser.add_argument('--output-dir', default='ml_pipeline_results', help='Output directory')
    args = parser.parse_args()
    
    # Setup output directory and logging
    output_dir = Path(args.output_dir)
    logger = setup_logging(output_dir)
    
    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = PipelineConfig(args.config)
        logger.info("Configuration loaded successfully")
        
        # Load data
        logger.info("Loading data...")
        df = load_data(args.data_file)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        
        # Initialize and run pipeline
        logger.info("Initializing pipeline...")
        pipeline = MLPipeline(config)
        logger.info("Pipeline initialized successfully")
        
        logger.info("Running pipeline...")
        results = pipeline.train(df)
        logger.info("Pipeline completed successfully")
        
        # Save results
        logger.info("Saving results...")
        save_results(results, output_dir)
        
        # Generate plots
        logger.info("Generating plots...")
        generate_plots(results, output_dir)
        
        logger.info("Pipeline execution completed successfully")
        
    except Exception as e:
        logger.error(f"Error in pipeline execution: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == '__main__':
    main() 
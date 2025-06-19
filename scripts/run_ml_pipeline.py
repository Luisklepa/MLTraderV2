import sys
from pathlib import Path
from typing import Dict

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd
import numpy as np
import logging
from utils.ml_pipeline import MLPipeline, PipelineConfig
from utils.data_feed import load_data
import yaml
import json
from datetime import datetime
import traceback
import matplotlib.pyplot as plt

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
    # Load data
    df = pd.read_csv(file_path)
    
    # Convert timestamp column
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # Ensure required columns exist
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Sort by datetime
    df.sort_index(inplace=True)
    
    # Remove duplicates
    df = df[~df.index.duplicated(keep='first')]
    
    # Forward fill missing values
    df.fillna(method='ffill', inplace=True)
    
    return df

def save_results(results: Dict):
    """Save results to files."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path('ml_pipeline_results')
    output_dir.mkdir(exist_ok=True)
    
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

def generate_analysis_plots(results: Dict):
    """Generate analysis plots."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path('ml_pipeline_results/plots')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    importance = results['feature_importance']
    importance['mean_importance'] = importance.mean(axis=1)
    importance = importance.sort_values('mean_importance', ascending=True)
    importance.tail(20)['mean_importance'].plot(kind='barh')
    plt.title('Top 20 Features by Importance')
    plt.tight_layout()
    plt.savefig(output_dir / f'feature_importance_{timestamp}.png')
    plt.close()
    
    # Plot prediction distribution
    plt.figure(figsize=(12, 6))
    preds = results['test_predictions']
    plt.hist(preds['position'], bins=50, alpha=0.5, label='Position Size')
    plt.title('Distribution of Position Sizes')
    plt.xlabel('Position Size')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f'prediction_dist_{timestamp}.png')
    plt.close()

def print_metrics(results: Dict):
    """Print detailed model metrics and analysis."""
    print("\nModel Performance Metrics:")
    print("=" * 80)
    
    # Print long model metrics
    print("\nLong Model Metrics:")
    print("-" * 40)
    metrics = results['long_metrics']
    print(f"Classification Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-score: {metrics['f1']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Positives: {metrics['true_positive']}")
    print(f"  False Positives: {metrics['false_positive']}")
    print(f"  True Negatives: {metrics['true_negative']}")
    print(f"  False Negatives: {metrics['false_negative']}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    
    # Print short model metrics
    print("\nShort Model Metrics:")
    print("-" * 40)
    metrics = results['short_metrics']
    print(f"Classification Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-score: {metrics['f1']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Positives: {metrics['true_positive']}")
    print(f"  False Positives: {metrics['false_positive']}")
    print(f"  True Negatives: {metrics['true_negative']}")
    print(f"  False Negatives: {metrics['false_negative']}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    
    # Print feature importance
    print("\nTop 20 Features by Importance:")
    print("-" * 80)
    importance = results['feature_importance']
    importance['mean_importance'] = importance.mean(axis=1)
    importance['std_importance'] = importance.std(axis=1)
    importance = importance.sort_values('mean_importance', ascending=False)
    
    print("\nRank  Feature                  Mean Imp.   Std Dev    Long Imp.  Short Imp.")
    print("-" * 80)
    for i, (feature, row) in enumerate(importance.head(20).iterrows(), 1):
        print(f"{i:4d}  {feature:22s}  {row['mean_importance']:9.4f}  {row['std_importance']:9.4f}  {row['long_importance']:9.4f}  {row['short_importance']:9.4f}")
    
    # Print prediction and position statistics
    print("\nPrediction Statistics:")
    print("-" * 80)
    preds = results['test_predictions']
    print(f"Total Predictions: {len(preds)}")
    
    # Long positions
    long_positions = preds[preds['position'] > 0]
    print("\nLong Positions:")
    print(f"  Count: {len(long_positions)} ({len(long_positions)/len(preds):.2%})")
    if len(long_positions) > 0:
        print(f"  Average Size: {long_positions['position'].mean():.4f}")
        print(f"  Max Size: {long_positions['position'].max():.4f}")
        print(f"  Min Size: {long_positions['position'].min():.4f}")
        print(f"  Average Probability: {long_positions['long_prob'].mean():.4f}")
    
    # Short positions
    short_positions = preds[preds['position'] < 0]
    print("\nShort Positions:")
    print(f"  Count: {len(short_positions)} ({len(short_positions)/len(preds):.2%})")
    if len(short_positions) > 0:
        print(f"  Average Size: {abs(short_positions['position']).mean():.4f}")
        print(f"  Max Size: {abs(short_positions['position']).max():.4f}")
        print(f"  Min Size: {abs(short_positions['position']).min():.4f}")
        print(f"  Average Probability: {short_positions['short_prob'].mean():.4f}")
    
    # No positions
    no_positions = preds[preds['position'] == 0]
    print(f"\nNo Positions: {len(no_positions)} ({len(no_positions)/len(preds):.2%})")

def main():
    """Main function to run the ML pipeline."""
    print("Starting ML pipeline execution...")
    
    # Load configuration
    config_path = 'config/ml_pipeline_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize pipeline
    pipeline = MLPipeline(config)
    
    # Load data
    data = load_data('data/btcusdt_prices.csv')
    print(f"Loaded {len(data)} rows of data")
    
    print("Initializing ML pipeline...")
    print("Running ML pipeline...")
    
    try:
        # Run pipeline
        results = pipeline.run(data)
        
        # Print results
        print_metrics(results)
        
        # Save results
        print("\nSaving results...")
        save_results(results)
        
        # Generate plots
        print("Generating analysis plots...")
        generate_analysis_plots(results)
        
        print("\nPipeline execution completed successfully!")
        
    except Exception as e:
        print(f"Pipeline execution failed: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main() 
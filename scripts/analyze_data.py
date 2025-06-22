import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import logging
import yaml

def setup_logging() -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    return logger

def load_config(config_file: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def analyze_class_distribution(df: pd.DataFrame) -> None:
    """Analyze class distribution."""
    print("\nClass Distribution:")
    print("Long signals:", df['long_target'].sum())
    print("Short signals:", df['short_target'].sum())
    print("No signals:", len(df) - df['long_target'].sum() - df['short_target'].sum())
    print("\nClass Ratios:")
    print("Long ratio: {:.2%}".format(df['long_target'].mean()))
    print("Short ratio: {:.2%}".format(df['short_target'].mean()))

def analyze_feature_correlations(df: pd.DataFrame, config: dict) -> None:
    """Analyze feature correlations."""
    features = config['model_config']['selected_features']
    
    # Calculate correlation matrix
    corr_matrix = df[features].corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(15, 12))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlations')
    plt.tight_layout()
    plt.savefig('data/analysis/feature_correlations.png')
    plt.close()
    
    # Find highly correlated features
    print("\nHighly Correlated Features (|correlation| > 0.8):")
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) > 0.8:
                print(f"{features[i]} - {features[j]}: {corr:.3f}")

def analyze_feature_importance(df: pd.DataFrame, config: dict) -> None:
    """Analyze feature importance using PCA."""
    features = config['model_config']['selected_features']
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    
    # Apply PCA
    pca = PCA()
    pca.fit(X)
    
    # Plot explained variance ratio
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
             np.cumsum(pca.explained_variance_ratio_), 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('PCA Analysis')
    plt.grid(True)
    plt.savefig('data/analysis/pca_analysis.png')
    plt.close()
    
    # Print top features by variance
    feature_var = pd.DataFrame(
        {'feature': features, 'variance': np.var(X, axis=0)})
    feature_var = feature_var.sort_values('variance', ascending=False)
    
    print("\nTop 10 Features by Variance:")
    print(feature_var.head(10))

def analyze_target_distribution(df: pd.DataFrame) -> None:
    """Analyze target distribution over time."""
    # Plot target distribution
    plt.figure(figsize=(15, 6))
    
    # Long signals
    plt.subplot(2, 1, 1)
    plt.plot(df.index, df['long_target'], label='Long Signals')
    plt.title('Long Signal Distribution Over Time')
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.legend()
    plt.grid(True)
    
    # Short signals
    plt.subplot(2, 1, 2)
    plt.plot(df.index, df['short_target'], label='Short Signals', color='red')
    plt.title('Short Signal Distribution Over Time')
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('data/analysis/target_distribution.png')
    plt.close()

def analyze_feature_distributions(df: pd.DataFrame, config: dict) -> None:
    """Analyze feature distributions."""
    features = config['model_config']['selected_features']
    
    # Create directory for feature distribution plots
    Path('data/analysis/feature_distributions').mkdir(parents=True, exist_ok=True)
    
    for feature in features:
        plt.figure(figsize=(10, 6))
        
        # Plot distribution
        sns.histplot(data=df, x=feature, hue='long_target', multiple="stack")
        plt.title(f'{feature} Distribution by Long Signal')
        plt.xlabel(feature)
        plt.ylabel('Count')
        plt.savefig(f'data/analysis/feature_distributions/{feature}_long.png')
        plt.close()
        
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x=feature, hue='short_target', multiple="stack")
        plt.title(f'{feature} Distribution by Short Signal')
        plt.xlabel(feature)
        plt.ylabel('Count')
        plt.savefig(f'data/analysis/feature_distributions/{feature}_short.png')
        plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze ML dataset')
    parser.add_argument('--data-file', required=True, help='Path to data file')
    parser.add_argument('--config', required=True, help='Path to config file')
    args = parser.parse_args()
    
    logger = setup_logging()
    
    try:
        # Create analysis directory
        logger.info("Creating analysis directory...")
        Path('data/analysis').mkdir(parents=True, exist_ok=True)
        
        # Load data and config
        logger.info("Loading data and config...")
        logger.info(f"Data file: {args.data_file}")
        logger.info(f"Config file: {args.config}")
        
        df = pd.read_csv(args.data_file)
        logger.info(f"Data shape before processing: {df.shape}")
        logger.info(f"Data columns: {df.columns.tolist()}")
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        logger.info(f"Data shape after processing: {df.shape}")
        
        config = load_config(args.config)
        logger.info("Config loaded successfully")
        
        # Run analyses
        logger.info("Analyzing class distribution...")
        analyze_class_distribution(df)
        
        logger.info("\nAnalyzing feature correlations...")
        analyze_feature_correlations(df, config)
        
        logger.info("\nAnalyzing feature importance...")
        analyze_feature_importance(df, config)
        
        logger.info("\nAnalyzing target distribution...")
        analyze_target_distribution(df)
        
        logger.info("\nAnalyzing feature distributions...")
        analyze_feature_distributions(df, config)
        
        logger.info("\nAnalysis complete. Results saved in data/analysis/")
        
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == '__main__':
    main() 
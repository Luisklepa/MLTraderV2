"""
File management utilities for handling paths and file operations.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path

class FileManager:
    """
    Manages file paths and operations based on configuration.
    """
    
    def __init__(self, config_path: str = "config/file_paths.yaml"):
        """
        Initialize file manager.
        
        Args:
            config_path: Path to file paths configuration
        """
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load file paths configuration.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dict containing configuration
        """
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get_raw_data_path(self, symbol: str) -> str:
        """
        Get path for raw price data.
        
        Args:
            symbol: Trading pair symbol (e.g., 'btc', 'eth')
            
        Returns:
            Path to price data file
        """
        filename = self.config['data']['raw']['price_files'][symbol.lower()]
        return os.path.join(self.config['data']['raw']['directory'], filename)
    
    def get_feature_path(self, pattern: Optional[str] = None) -> str:
        """
        Get path for feature data.
        
        Args:
            pattern: Optional filename pattern
            
        Returns:
            Path to feature data
        """
        if pattern is None:
            pattern = self.config['data']['features']['pattern']
        return os.path.join(self.config['data']['features']['directory'], pattern)
    
    def get_model_path(self, model_type: str, filename: str) -> str:
        """
        Get path for model file.
        
        Args:
            model_type: Type of model ('random_forest' or 'xgboost')
            filename: Model filename
            
        Returns:
            Path to model file
        """
        return os.path.join(self.config['models'][model_type]['directory'], filename)
    
    def get_result_path(self, result_type: str, filename: str) -> str:
        """
        Get path for result file.
        
        Args:
            result_type: Type of result ('feature_importance', 'plots', 'metrics')
            filename: Result filename
            
        Returns:
            Path to result file
        """
        return os.path.join(self.config['results'][result_type]['directory'], filename)
    
    def ensure_directories(self) -> None:
        """
        Ensure all required directories exist.
        """
        directories = [
            self.config['data']['raw']['directory'],
            self.config['data']['features']['directory'],
            self.config['data']['processed']['directory'],
            self.config['models']['random_forest']['directory'],
            self.config['models']['xgboost']['directory'],
            self.config['results']['feature_importance']['directory'],
            self.config['results']['plots']['directory'],
            self.config['results']['metrics']['directory'],
            self.config['cache']['directory']
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def cleanup_cache(self) -> None:
        """
        Clean up old cache files.
        """
        cache_dir = self.config['cache']['directory']
        max_size = self._parse_size(self.config['cache']['max_size'])
        ttl = self.config['cache']['ttl']
        
        # Implementation of cache cleanup logic here
        pass
    
    def _parse_size(self, size_str: str) -> int:
        """
        Parse size string (e.g., '1GB') to bytes.
        
        Args:
            size_str: Size string with unit
            
        Returns:
            Size in bytes
        """
        units = {'B': 1, 'KB': 1024, 'MB': 1024**2, 'GB': 1024**3}
        number = float(''.join(filter(str.isdigit, size_str)))
        unit = ''.join(filter(str.isalpha, size_str.upper()))
        return int(number * units[unit])

# Example usage:
"""
file_manager = FileManager()

# Get paths
btc_price_path = file_manager.get_raw_data_path('btc')
feature_path = file_manager.get_feature_path()
model_path = file_manager.get_model_path('xgboost', 'model.pkl')
result_path = file_manager.get_result_path('plots', 'equity_curve.png')

# Ensure directories exist
file_manager.ensure_directories()

# Clean up cache
file_manager.cleanup_cache()
""" 
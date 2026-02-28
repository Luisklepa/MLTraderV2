"""
Performance optimization utilities for the ML trading system.
"""

import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class PerformanceOptimizer:
    """Handles performance optimizations for data processing and calculations."""
    
    def __init__(self, config: Dict):
        """Initialize optimizer with configuration."""
        self.config = config
        self.cache_size = config.get('cache_size', 128)
        self.chunk_size = config.get('chunk_size', 10000)
        self.use_parallel = config.get('use_parallel_processing', True)
    
    @lru_cache(maxsize=128)
    def calculate_support_resistance(self, prices: Tuple[float, ...], window: int = 20) -> Tuple[float, float]:
        """Calculate support and resistance levels with caching."""
        prices_array = np.array(prices)
        pivot = (np.max(prices_array) + np.min(prices_array) + prices_array[-1]) / 3
        support = pivot - (np.max(prices_array) - np.min(prices_array))
        resistance = pivot + (np.max(prices_array) - np.min(prices_array))
        return support, resistance
    
    def optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage."""
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type == 'float64':
                df[col] = pd.to_numeric(df[col], downcast='float')
            elif col_type == 'int64':
                df[col] = pd.to_numeric(df[col], downcast='integer')
            elif col_type == 'object':
                if df[col].nunique() / len(df[col]) < 0.5:  # If low cardinality
                    df[col] = df[col].astype('category')
        
        return df
    
    def parallel_feature_calculation(self, data: pd.DataFrame, feature_funcs: List[callable]) -> pd.DataFrame:
        """Calculate features in parallel."""
        if not self.use_parallel:
            return self._sequential_feature_calculation(data, feature_funcs)
        
        df = data.copy()
        results = {}
        
        with ThreadPoolExecutor() as executor:
            future_to_func = {
                executor.submit(func, df): func.__name__ 
                for func in feature_funcs
            }
            
            for future in as_completed(future_to_func):
                func_name = future_to_func[future]
                try:
                    feature_result = future.result()
                    results.update(feature_result)
                except Exception as e:
                    logger.error(f"Error calculating {func_name}: {str(e)}")
        
        for col, values in results.items():
            df[col] = values
        
        return df
    
    def _sequential_feature_calculation(self, data: pd.DataFrame, feature_funcs: List[callable]) -> pd.DataFrame:
        """Calculate features sequentially (fallback method)."""
        df = data.copy()
        for func in feature_funcs:
            try:
                result = func(df)
                for col, values in result.items():
                    df[col] = values
            except Exception as e:
                logger.error(f"Error calculating {func.__name__}: {str(e)}")
        return df
    
    def process_large_dataset(self, file_path: str, processing_func: callable) -> pd.DataFrame:
        """Process large datasets in chunks."""
        chunks = []
        total_rows = sum(1 for _ in open(file_path)) - 1  # Subtract header
        
        with pd.read_csv(file_path, chunksize=self.chunk_size) as reader:
            for chunk_number, chunk in enumerate(reader, 1):
                logger.info(f"Processing chunk {chunk_number} ({self.chunk_size * chunk_number}/{total_rows} rows)")
                processed_chunk = processing_func(chunk)
                chunks.append(processed_chunk)
        
        return pd.concat(chunks, axis=0, ignore_index=True)
    
    def optimize_numeric_operations(self, func: callable) -> callable:
        """Decorator to optimize numeric operations."""
        def wrapper(*args, **kwargs):
            # Convert pandas objects to numpy for faster computation
            args = [arg.values if isinstance(arg, pd.Series) else arg for arg in args]
            result = func(*args, **kwargs)
            return result
        return wrapper
    
    @staticmethod
    def create_memory_profile(df: pd.DataFrame) -> Dict:
        """Create memory usage profile of DataFrame."""
        memory_usage = df.memory_usage(deep=True)
        total_memory = memory_usage.sum() / 1024**2  # Convert to MB
        
        return {
            'total_memory_mb': total_memory,
            'memory_by_column': {
                col: memory_usage[i] / 1024**2
                for i, col in enumerate(df.columns)
            },
            'dtypes': df.dtypes.to_dict()
        }
    
    def save_optimized_dataset(self, df: pd.DataFrame, file_path: str) -> None:
        """Save DataFrame in an optimized format."""
        # Optimize memory usage
        df = self.optimize_dataframe_memory(df)
        
        # Save to parquet format for better compression and faster I/O
        parquet_path = Path(file_path).with_suffix('.parquet')
        df.to_parquet(parquet_path, engine='pyarrow', compression='snappy')
        
        # Log optimization results
        original_size = df.memory_usage(deep=True).sum() / 1024**2
        logger.info(f"Optimized dataset saved to {parquet_path}")
        logger.info(f"Memory usage: {original_size:.2f} MB")
        
    def load_optimized_dataset(self, file_path: str) -> pd.DataFrame:
        """Load dataset in an optimized way."""
        parquet_path = Path(file_path).with_suffix('.parquet')
        
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
        else:
            df = pd.read_csv(file_path)
            df = self.optimize_dataframe_memory(df)
            self.save_optimized_dataset(df, file_path)
        
        return df 
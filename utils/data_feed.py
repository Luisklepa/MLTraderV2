import backtrader as bt
import pandas as pd
from datetime import datetime
import numpy as np
from typing import Optional, Union, Dict, List
from pathlib import Path
import logging
from functools import lru_cache
import mmap
import os

logger = logging.getLogger(__name__)

class MLSignalData(bt.feed.DataBase):
    """
    Custom data feed for ML signals that properly integrates with backtrader.
    """
    lines = ('open', 'high', 'low', 'close', 'volume', 'target', 'signal_strength', 'future_return',)
    
    params = (
        ('datetime', None),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('target', 'target'),
        ('signal_strength', 'signal_strength'),
        ('future_return', 'future_return'),
    )
    
    def __init__(self, df=None, **kwargs):
        """Initialize the data feed."""
        super().__init__()
        
        if df is not None:
            # Ensure we have a DataFrame
            if not isinstance(df, pd.DataFrame):
                raise ValueError("Data must be a pandas DataFrame")
                
            # Ensure we have a datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError("DataFrame must have DatetimeIndex")
            
            # Store the data
            self.df = df
            self.datetime_index = df.index
            self.open = df['open'].values
            self.high = df['high'].values
            self.low = df['low'].values
            self.close = df['close'].values
            self.volume = df['volume'].values
            self.target = df['target'].values
            self.signal_strength = df['signal_strength'].values
            self.future_return = df['future_return'].values if 'future_return' in df.columns else np.zeros_like(self.close)
            
            # Set the length of the data
            self.len = len(df)
            
    def start(self):
        """Start the data feed."""
        super().start()
        self.idx = -1
        
    def _load(self):
        """Load the next bar."""
        self.idx += 1
        if self.idx >= self.len:
            return False
            
        # Load the data
        self.lines.datetime[0] = bt.date2num(self.datetime_index[self.idx])
        self.lines.open[0] = float(self.open[self.idx])
        self.lines.high[0] = float(self.high[self.idx])
        self.lines.low[0] = float(self.low[self.idx])
        self.lines.close[0] = float(self.close[self.idx])
        self.lines.volume[0] = float(self.volume[self.idx])
        self.lines.target[0] = float(self.target[self.idx])
        self.lines.signal_strength[0] = float(self.signal_strength[self.idx])
        self.lines.future_return[0] = float(self.future_return[self.idx])
        
        return True
        
    def _gettz(self):
        """Return the timezone."""
        return None  # Assuming UTC
        
    def setenvironment(self, env):
        """Set the environment."""
        self._env = env
        
    def _getenv(self):
        """Return the environment."""
        return self._env
        
    def _setenv(self, env):
        """Set the environment."""
        self._env = env
        
    def rewind(self):
        """Rewind the data feed."""
        super().rewind()
        self.idx = -1
        
    def get_value(self, line, ago=0):
        """Get the value of a line at a specific point in time."""
        idx = self.idx - ago
        if idx < 0 or idx >= self.len:
            return None
            
        if line == 'datetime':
            return self.datetime_index[idx]
        elif line == 'open':
            return float(self.open[idx])
        elif line == 'high':
            return float(self.high[idx])
        elif line == 'low':
            return float(self.low[idx])
        elif line == 'close':
            return float(self.close[idx])
        elif line == 'volume':
            return float(self.volume[idx])
        elif line == 'target':
            return float(self.target[idx])
        elif line == 'signal_strength':
            return float(self.signal_strength[idx])
        elif line == 'future_return':
            return float(self.future_return[idx])
        else:
            return None
            
    def datetime(self, idx=0):
        """Return the datetime value at the given index."""
        return self.datetime_index[self.idx - idx]

class OptimizedDataFeed:
    """Optimized data feed for large datasets."""
    
    def __init__(self, config: Dict):
        """Initialize the data feed."""
        self.config = config
        self.cache = {}
        self.mmaps = {}
        
    def __del__(self):
        """Clean up resources."""
        for mmap_obj in self.mmaps.values():
            mmap_obj.close()
            
    @staticmethod
    def _get_optimal_chunk_size(file_size: int) -> int:
        """Calculate optimal chunk size based on file size."""
        return min(file_size, 1024 * 1024)  # 1MB chunks
        
    def _create_mmap(self, file_path: str) -> mmap.mmap:
        """Create memory-mapped file object."""
        with open(file_path, 'rb') as f:
            return mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            
    @lru_cache(maxsize=32)
    def _read_chunk(self, file_path: str, start: int, size: int) -> bytes:
        """Read a chunk of data from file."""
        mmap_obj = self.mmaps.get(file_path)
        if mmap_obj is None:
            mmap_obj = self._create_mmap(file_path)
            self.mmaps[file_path] = mmap_obj
        return mmap_obj[start:start + size]
        
    def load_data(
        self,
        file_path: str,
        columns: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load data efficiently using memory mapping.
        
        Args:
            file_path: Path to data file
            columns: List of columns to load
            start_date: Start date filter
            end_date: End date filter
            
        Returns:
            DataFrame with requested data
        """
        # Get file size
        file_size = os.path.getsize(file_path)
        chunk_size = self._get_optimal_chunk_size(file_size)
        
        # Read data in chunks
        chunks = []
        for start in range(0, file_size, chunk_size):
            chunk = self._read_chunk(file_path, start, chunk_size)
            chunks.append(chunk)
            
        # Combine chunks and create DataFrame
        data = b''.join(chunks)
        df = pd.read_csv(pd.io.common.BytesIO(data))
        
        # Apply filters
        if columns:
            df = df[columns]
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
            
        return df
        
    def preload_data(self, file_paths: List[str]):
        """Preload data into memory."""
        for file_path in file_paths:
            self._create_mmap(file_path)
            
    def clear_cache(self):
        """Clear the cache."""
        self.cache.clear()
        
    def get_latest_data(
        self,
        symbol: str,
        lookback_periods: int = 100
    ) -> pd.DataFrame:
        """
        Get latest data for a symbol.
        
        Args:
            symbol: Symbol to get data for
            lookback_periods: Number of periods to look back
            
        Returns:
            DataFrame with latest data
        """
        file_path = self.config['data_paths'].get(symbol)
        if not file_path:
            raise ValueError(f"No data path configured for symbol: {symbol}")
            
        # Load data
        df = self.load_data(file_path)
        
        # Get latest periods
        if len(df) > lookback_periods:
            df = df.iloc[-lookback_periods:]
            
        return df
        
    def update_data(
        self,
        symbol: str,
        new_data: pd.DataFrame
    ):
        """
        Update data for a symbol.
        
        Args:
            symbol: Symbol to update
            new_data: New data to append
        """
        file_path = self.config['data_paths'].get(symbol)
        if not file_path:
            raise ValueError(f"No data path configured for symbol: {symbol}")
            
        new_data.to_csv(file_path, mode='a', header=False, index=True)
        
    def get_symbols(self) -> List[str]:
        """Get list of available symbols."""
        return list(self.config['data_paths'].keys())

class DataFeed:
    """Simple data feed for testing."""
    
    def __init__(self, symbol: str, timeframe: str = '15m'):
        """Initialize the data feed."""
        self.symbol = symbol
        # Convert timeframe to pandas frequency string
        self.timeframe = timeframe.replace('m', 'min')  # Convert minutes to proper format
        
    def get_historical_data(self, start_date: Union[str, datetime], end_date: Union[str, datetime]) -> pd.DataFrame:
        """Get historical data for testing."""
        # Convert dates to datetime if they're strings
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
            
        # Generate sample dates
        dates = pd.date_range(start=start_date, end=end_date, freq=self.timeframe)
        
        # Generate sample data
        np.random.seed(42)  # For reproducibility
        n = len(dates)
        close = 50000 + np.random.normal(loc=0, scale=100, size=n).cumsum()  # Random walk starting at 50000
        
        # Generate OHLCV data with more realistic price movements
        data = pd.DataFrame({
            'open': close + np.random.normal(0, 50, n),
            'high': close + np.abs(np.random.normal(0, 100, n)),
            'low': close - np.abs(np.random.normal(0, 100, n)),
            'close': close,
            'volume': np.random.lognormal(10, 1, n),
            'target': np.random.choice([-1, 0, 1], n),  # Random signals
            'signal_strength': np.random.uniform(0, 1, n),  # Random signal strength
            'future_return': np.random.normal(0, 0.02, n)  # Random future returns
        }, index=dates)
        
        # Ensure high is highest and low is lowest
        data['high'] = data[['open', 'high', 'close']].max(axis=1)
        data['low'] = data[['open', 'low', 'close']].min(axis=1)
        
        # Ensure data types
        data = data.astype({
            'open': float,
            'high': float,
            'low': float,
            'close': float,
            'volume': float,
            'target': float,
            'signal_strength': float,
            'future_return': float
        })
        
        return data 
import backtrader as bt
import pandas as pd
from datetime import datetime
import numpy as np
from typing import Optional, Union, Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class MLSignalData(bt.feeds.PandasData):
    """Custom data feed that includes ML signal."""
    lines = ('target', 'future_return',)
    params = (
        ('datetime', None),  # El índice es el datetime
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('target', 'target'),  # Usar target directamente
        ('future_return', 'future_return'),  # Usar future_return directamente
        ('openinterest', None),
    )
    
    def __init__(self, *args, **kwargs):
        # Get the dataframe from args or kwargs
        dataname = kwargs.get('dataname', None)
        if dataname is None and len(args) > 0:
            dataname = args[0]
            args = args[1:]
            kwargs['dataname'] = dataname
        
        # If dataframe is provided, ensure it's properly formatted
        if dataname is not None:
            # Convert index to datetime if it's not already
            if not isinstance(dataname.index, pd.DatetimeIndex):
                dataname.index = pd.to_datetime(dataname.index)
            # Ensure index is sorted
            dataname = dataname.sort_index()
            kwargs['dataname'] = dataname
        
        # Initialize the parent class
        super().__init__(*args, **kwargs)
    
    def _load(self):
        """Load the next bar from the data source"""
        try:
            # Get the next row
            row = self.p.dataname.iloc[self._idx]
            
            # Load the bar values
            self.lines.datetime[0] = bt.date2num(self.p.dataname.index[self._idx].to_pydatetime())
            self.lines.open[0] = float(row['open'])
            self.lines.high[0] = float(row['high'])
            self.lines.low[0] = float(row['low'])
            self.lines.close[0] = float(row['close'])
            self.lines.volume[0] = float(row['volume'])
            self.lines.target[0] = float(row['target'])
            self.lines.future_return[0] = float(row['future_return'])
            
            # Update index
            self._idx += 1
            return True
            
        except IndexError:
            return False

    def start(self):
        super().start()
        # Verificar que las líneas estén disponibles
        print("[DEBUG] Available lines:", self.lines.getlinealiases())
        print("[DEBUG] First few values of target:", [self.lines.target[i] for i in range(5)])
        print("[DEBUG] First few values of future_return:", [self.lines.future_return[i] for i in range(5)])

def prepare_data(data_path):
    """Prepara los datos para el backtest."""
    # Cargar datos
    df = pd.read_csv(data_path)
    
    # Convertir datetime a índice
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    
    # Ordenar por fecha
    df.sort_index(inplace=True)
    
    # Convertir el retorno futuro a fuerza de señal
    df['future_return'] = df['future_return'].abs() * 100  # Convertir a porcentaje y usar valor absoluto
    
    # Verificar datos
    print("[DEBUG] Columns in DataFrame:", df.columns.tolist())
    print("[DEBUG] First few rows of target:", df['target'].head())
    print("[DEBUG] First few rows of future_return:", df['future_return'].head())
    
    # Verificar duplicados y valores faltantes
    print("\n[CHECK] Missing datetimes in index:", df.index.isnull().sum())
    print("[CHECK] Duplicate datetimes in index:", df.index.duplicated().sum())
    print("[CHECK] First 10 index values:")
    print(df.index[:10])
    print("[CHECK] First 10 index values as int64:")
    print(df.index.astype(np.int64)[:10])
    print("[CHECK] First 10 index values as datetime:")
    print([pd.Timestamp(x) for x in df.index[:10]])
    
    return df 

def load_data(
    file_path: Union[str, Path],
    datetime_column: str = 'timestamp',
    price_columns: Optional[Dict[str, str]] = None,
    dropna: bool = True
) -> pd.DataFrame:
    """
    Load and preprocess data for ML pipeline.
    
    Args:
        file_path: Path to the data file
        datetime_column: Name of datetime column
        price_columns: Mapping of required price columns
        dropna: Whether to drop rows with NaN values
        
    Returns:
        Preprocessed DataFrame
    """
    logger.info(f"Loading data from {file_path}")
    
    # Set default price columns if not provided
    if price_columns is None:
        price_columns = {
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        }
    
    # Load data
    if str(file_path).endswith('.csv'):
        df = pd.read_csv(file_path)
    elif str(file_path).endswith('.parquet'):
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    # Convert datetime
    if datetime_column in df.columns:
        df[datetime_column] = pd.to_datetime(df[datetime_column])
        df.set_index(datetime_column, inplace=True)
    else:
        raise ValueError(f"Datetime column '{datetime_column}' not found")
    
    # Verify required columns
    missing_cols = [col for col, name in price_columns.items() 
                   if name not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Sort by datetime
    df.sort_index(inplace=True)
    
    # Remove duplicates
    duplicates = df.index.duplicated()
    if duplicates.any():
        logger.warning(f"Found {duplicates.sum()} duplicate timestamps")
        df = df[~duplicates]
    
    # Handle missing values
    if dropna:
        original_len = len(df)
        df.dropna(inplace=True)
        dropped = original_len - len(df)
        if dropped > 0:
            logger.warning(f"Dropped {dropped} rows with missing values")
    
    # Add basic features
    df['returns'] = df[price_columns['close']].pct_change()
    df['log_returns'] = np.log1p(df['returns'])
    df['volume_ratio'] = df[price_columns['volume']] / df[price_columns['volume']].rolling(20).mean()
    
    # Verify data quality
    logger.info(f"Loaded {len(df)} rows of data")
    logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
    
    return df

class MLSignalData:
    """Data handler for ML signals and backtesting."""
    
    def __init__(
        self,
        data: pd.DataFrame,
        price_columns: Optional[Dict[str, str]] = None,
        target_column: Optional[str] = None
    ):
        """
        Initialize data handler.
        
        Args:
            data: Input DataFrame
            price_columns: Mapping of price column names
            target_column: Name of target column if available
        """
        self.data = data.copy()
        self.price_columns = price_columns or {
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        }
        self.target_column = target_column
        
        # Validate data
        self._validate_data()
    
    def _validate_data(self):
        """Validate data format and contents."""
        # Check index
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex")
        
        # Check required columns
        missing_cols = [col for col, name in self.price_columns.items() 
                       if name not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check target column if specified
        if self.target_column and self.target_column not in self.data.columns:
            raise ValueError(f"Target column '{self.target_column}' not found")
    
    def prepare_for_training(
        self,
        features: Optional[list] = None,
        target: Optional[str] = None
    ) -> tuple:
        """
        Prepare data for ML model training.
        
        Args:
            features: List of feature columns to use
            target: Target column name
            
        Returns:
            Tuple of (X, y) for training
        """
        if features is None:
            features = [col for col in self.data.columns 
                       if col not in [self.target_column]]
        
        if target is None:
            target = self.target_column
        
        if target not in self.data.columns:
            raise ValueError(f"Target column '{target}' not found")
        
        X = self.data[features]
        y = self.data[target]
        
        return X, y
    
    def prepare_for_backtest(
        self,
        signals: pd.Series,
        signal_threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Prepare data for backtesting.
        
        Args:
            signals: Model predictions
            signal_threshold: Threshold for generating trading signals
            
        Returns:
            DataFrame ready for backtesting
        """
        backtest_data = self.data.copy()
        
        # Convert probabilities to trading signals
        backtest_data['target'] = (signals >= signal_threshold).astype(int)
        
        # Add signal strength
        backtest_data['signal_strength'] = signals
        
        return backtest_data 
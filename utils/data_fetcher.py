"""
Data fetching utilities for the trading system.
"""
import requests
import pandas as pd
from typing import Optional, List
from config.settings import TradingConfig
import time

class BinanceDataFetcher:
    """Optimized data fetcher with caching and retry logic"""
    
    def __init__(self):
        self.config = TradingConfig()
        self._cache = {}
        
    def get_klines(self, 
                   symbol: str = TradingConfig.DEFAULT_SYMBOL,
                   interval: str = TradingConfig.DEFAULT_TIMEFRAME,
                   limit: int = TradingConfig.MAX_LOOKBACK_BARS) -> Optional[pd.DataFrame]:
        """Fetch klines data from Binance API with retry logic"""
        cache_key = (symbol, interval, limit)
        if cache_key in self._cache:
            # print(f"[CACHE] Usando datos cacheados para {symbol} {interval} {limit}")
            return self._cache[cache_key]
        # print(f"get_klines called with symbol={symbol}, interval={interval}, limit={limit}")
        t0 = time.time()
        klines = []
        remaining_limit = limit
        end_time = None
        
        while remaining_limit > 0:
            fetch_limit = min(1000, remaining_limit)
            
            for attempt in range(self.config.MAX_RETRIES):
                try:
                    # print(f"Attempt {attempt+1}/{self.config.MAX_RETRIES} for {symbol}, fetch_limit={fetch_limit}, remaining_limit={remaining_limit}")
                    params = {
                        'symbol': symbol,
                        'interval': interval,
                        'limit': fetch_limit,
                    }
                    if end_time:
                        params['endTime'] = end_time
                    
                    response = requests.get(
                        self.config.BINANCE_BASE_URL,
                        params=params,
                        timeout=self.config.TIMEOUT
                    )
                    response.raise_for_status()
                    data = response.json()
                    # print(f"Response data length: {len(data)}")
                    
                    if not data:
                        # print("No data returned from Binance API.")
                        break
                    
                    klines = data + klines
                    end_time = data[0][0] - 1
                    remaining_limit -= fetch_limit
                    
                    if len(data) < fetch_limit:
                        # print("Fetched less than fetch_limit, breaking loop.")
                        break
                    
                    break  # Success, exit retry loop
                    
                except requests.RequestException as e:
                    # print(f"Exception: {e}")
                    if attempt == self.config.MAX_RETRIES - 1:
                        # print(f"Failed to fetch data after {self.config.MAX_RETRIES} attempts: {e}")
                        return None
                    # print(f"Retry {attempt + 1}/{self.config.MAX_RETRIES} for {symbol}")
        
        if not klines:
            # print("No klines collected after all attempts.")
            return None
        
        t1 = time.time()
        # print(f"[PERF] Descarga de datos tomó {t1-t0:.2f} segundos.")
        # print(f"Returning DataFrame with {len(klines)} rows.")
        t2 = time.time()
        df = self._process_klines(klines)
        t3 = time.time()
        # print(f"[PERF] Procesamiento de DataFrame tomó {t3-t2:.2f} segundos.")
        # print(f"[PERF] Tiempo total get_klines: {t3-t0:.2f} segundos.")
        self._cache[cache_key] = df
        return df
    
    @staticmethod
    def _process_klines(klines: List) -> pd.DataFrame:
        """Process raw klines data into DataFrame"""
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        
        # Vectorized conversion
        price_cols = ['open', 'high', 'low', 'close', 'volume']
        df[price_cols] = df[price_cols].apply(pd.to_numeric, errors='coerce')
        
        # Handle NaN values
        if df[price_cols].isna().any().any():
            # print("Warning: Removing rows with NaN values")
            df = df.dropna()
        
        return df[['open_time', 'open', 'high', 'low', 'close', 'volume']] 
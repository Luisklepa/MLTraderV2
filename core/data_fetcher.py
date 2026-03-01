"""
Data fetching utilities with rate limiting and TTL cache.
"""

import logging
import time

import pandas as pd
import requests

from config.settings import TradingConfig

logger = logging.getLogger(__name__)


class _CacheEntry:
    __slots__ = ("data", "timestamp")

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.timestamp = time.monotonic()


class BinanceDataFetcher:
    """Data fetcher with TTL cache, rate limiting, and retry logic."""

    MAX_CACHE_ENTRIES = 50
    CACHE_TTL_SECONDS = 300  # 5 minutes
    MIN_REQUEST_INTERVAL = 0.1  # 100ms between requests (Binance: 1200/min)

    def __init__(self):
        self.config = TradingConfig()
        self._cache: dict[tuple, _CacheEntry] = {}
        self._last_request_time: float = 0.0

    def _rate_limit(self) -> None:
        """Enforce minimum interval between API requests."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self.MIN_REQUEST_INTERVAL:
            time.sleep(self.MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.monotonic()

    def _evict_stale_cache(self) -> None:
        """Remove expired entries and enforce max cache size."""
        now = time.monotonic()
        expired = [k for k, v in self._cache.items() if (now - v.timestamp) > self.CACHE_TTL_SECONDS]
        for k in expired:
            del self._cache[k]

        while len(self._cache) > self.MAX_CACHE_ENTRIES:
            oldest_key = min(self._cache, key=lambda k: self._cache[k].timestamp)
            del self._cache[oldest_key]

    def get_klines(
        self,
        symbol: str = TradingConfig.DEFAULT_SYMBOL,
        interval: str = TradingConfig.DEFAULT_TIMEFRAME,
        limit: int = TradingConfig.MAX_LOOKBACK_BARS,
    ) -> pd.DataFrame | None:
        """Fetch klines from Binance with caching, rate limiting, and retries."""
        cache_key = (symbol, interval, limit)

        self._evict_stale_cache()
        if cache_key in self._cache:
            return self._cache[cache_key].data.copy()

        t0 = time.time()
        klines: list = []
        remaining_limit = limit
        end_time = None

        while remaining_limit > 0:
            fetch_limit = min(1000, remaining_limit)

            for attempt in range(self.config.MAX_RETRIES):
                try:
                    self._rate_limit()

                    params = {
                        "symbol": symbol,
                        "interval": interval,
                        "limit": fetch_limit,
                    }
                    if end_time:
                        params["endTime"] = end_time

                    response = requests.get(
                        self.config.BINANCE_BASE_URL,
                        params=params,
                        timeout=self.config.TIMEOUT,
                    )
                    response.raise_for_status()
                    data = response.json()

                    if not data:
                        break

                    klines = data + klines
                    end_time = data[0][0] - 1
                    remaining_limit -= fetch_limit

                    if len(data) < fetch_limit:
                        remaining_limit = 0
                    break

                except requests.RequestException as e:
                    logger.warning(f"Attempt {attempt + 1}/{self.config.MAX_RETRIES} failed for {symbol}: {e}")
                    if attempt == self.config.MAX_RETRIES - 1:
                        logger.error(f"Failed to fetch data after {self.config.MAX_RETRIES} attempts")
                        return None
                    time.sleep(2**attempt)

        if not klines:
            return None

        logger.debug(f"Fetched {len(klines)} klines for {symbol} in {time.time() - t0:.2f}s")
        df = self._process_klines(klines)
        self._cache[cache_key] = _CacheEntry(df)
        return df

    @staticmethod
    def _process_klines(klines: list) -> pd.DataFrame:
        """Process raw klines data into a clean DataFrame."""
        df = pd.DataFrame(
            klines,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_asset_volume",
                "number_of_trades",
                "taker_buy_base_asset_volume",
                "taker_buy_quote_asset_volume",
                "ignore",
            ],
        )

        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")

        price_cols = ["open", "high", "low", "close", "volume"]
        df[price_cols] = df[price_cols].apply(pd.to_numeric, errors="coerce")

        if df[price_cols].isna().any().any():
            n_before = len(df)
            df = df.dropna(subset=price_cols)
            logger.warning(f"Dropped {n_before - len(df)} rows with NaN prices")

        return df[["open_time", "open", "high", "low", "close", "volume"]]

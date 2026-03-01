"""Tests for core/data_fetcher.py — Binance data fetcher with caching and rate limiting."""

import time
from unittest.mock import patch

import numpy as np
import pandas as pd

from core.data_fetcher import BinanceDataFetcher, _CacheEntry


class TestCacheEntry:
    def test_stores_data_and_timestamp(self):
        df = pd.DataFrame({"close": [1, 2, 3]})
        entry = _CacheEntry(df)
        assert entry.data.equals(df)
        assert isinstance(entry.timestamp, float)


class TestRateLimit:
    def test_enforces_minimum_interval(self):
        fetcher = BinanceDataFetcher()
        fetcher._last_request_time = time.monotonic()

        t0 = time.monotonic()
        fetcher._rate_limit()
        elapsed = time.monotonic() - t0

        assert elapsed >= fetcher.MIN_REQUEST_INTERVAL * 0.8

    def test_no_delay_after_interval(self):
        fetcher = BinanceDataFetcher()
        fetcher._last_request_time = time.monotonic() - 1.0  # 1 second ago

        t0 = time.monotonic()
        fetcher._rate_limit()
        elapsed = time.monotonic() - t0

        assert elapsed < 0.05


class TestCacheEviction:
    def test_evicts_expired_entries(self):
        fetcher = BinanceDataFetcher()
        df = pd.DataFrame({"close": [1]})

        old_entry = _CacheEntry(df)
        old_entry.timestamp = time.monotonic() - fetcher.CACHE_TTL_SECONDS - 1

        fetcher._cache[("BTC", "15m", 100)] = old_entry
        fetcher._evict_stale_cache()

        assert len(fetcher._cache) == 0

    def test_evicts_oldest_when_full(self):
        fetcher = BinanceDataFetcher()
        df = pd.DataFrame({"close": [1]})

        for i in range(fetcher.MAX_CACHE_ENTRIES + 5):
            entry = _CacheEntry(df)
            entry.timestamp = time.monotonic() + i * 0.001
            fetcher._cache[(f"SYM{i}", "15m", 100)] = entry

        fetcher._evict_stale_cache()
        assert len(fetcher._cache) <= fetcher.MAX_CACHE_ENTRIES


class TestProcessKlines:
    def test_processes_valid_klines(self):
        klines = [
            [
                1704067200000,
                "50000",
                "50100",
                "49900",
                "50050",
                "100",
                1704067259999,
                "5000000",
                50,
                "50",
                "2500000",
                "0",
            ],
            [
                1704067260000,
                "50050",
                "50150",
                "49950",
                "50100",
                "150",
                1704067319999,
                "7500000",
                75,
                "75",
                "3750000",
                "0",
            ],
        ]
        df = BinanceDataFetcher._process_klines(klines)
        assert len(df) == 2
        assert list(df.columns) == ["open_time", "open", "high", "low", "close", "volume"]
        assert np.issubdtype(df["close"].dtype, np.number)

    def test_drops_nan_rows(self):
        klines = [
            [
                1704067200000,
                "50000",
                "50100",
                "49900",
                "invalid",
                "100",
                1704067259999,
                "5000000",
                50,
                "50",
                "2500000",
                "0",
            ],
            [
                1704067260000,
                "50050",
                "50150",
                "49950",
                "50100",
                "150",
                1704067319999,
                "7500000",
                75,
                "75",
                "3750000",
                "0",
            ],
        ]
        df = BinanceDataFetcher._process_klines(klines)
        assert len(df) == 1


class TestGetKlines:
    @patch("core.data_fetcher.requests.get")
    def test_returns_cached_data(self, mock_get):
        fetcher = BinanceDataFetcher()
        cached_df = pd.DataFrame(
            {
                "open_time": pd.to_datetime(["2024-01-01"]),
                "open": [50000.0],
                "high": [50100.0],
                "low": [49900.0],
                "close": [50050.0],
                "volume": [100.0],
            }
        )
        fetcher._cache[("BTCUSDT", "15m", 3000)] = _CacheEntry(cached_df)

        result = fetcher.get_klines("BTCUSDT", "15m", 3000)
        assert result is not None
        assert len(result) == 1
        mock_get.assert_not_called()

    @patch("core.data_fetcher.requests.get")
    def test_returns_none_on_failure(self, mock_get):
        import requests

        mock_get.side_effect = requests.RequestException("Connection error")

        fetcher = BinanceDataFetcher()
        fetcher.config.MAX_RETRIES = 1
        result = fetcher.get_klines("BTCUSDT", "15m", 100)
        assert result is None

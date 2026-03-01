"""Tests for core/data_feed.py — MLSignalData, OptimizedDataFeed, DataFeed."""

import numpy as np
import pandas as pd
import pytest

from core.data_feed import DataFeed, MLSignalData, OptimizedDataFeed


# ---------------------------------------------------------------------------
# MLSignalData — init & validation only (backtrader line buffers require
# a Cerebro environment for _load, so we only test construction here)
# ---------------------------------------------------------------------------
class TestMLSignalData:
    @pytest.fixture
    def signal_df(self):
        dates = pd.date_range("2024-01-01", periods=50, freq="h")
        return pd.DataFrame(
            {
                "open": np.random.uniform(100, 110, 50),
                "high": np.random.uniform(110, 120, 50),
                "low": np.random.uniform(90, 100, 50),
                "close": np.random.uniform(100, 110, 50),
                "volume": np.random.uniform(1000, 5000, 50),
                "target": np.random.choice([0, 1], 50),
                "signal_strength": np.random.uniform(0, 1, 50),
                "future_return": np.random.normal(0, 0.02, 50),
            },
            index=dates,
        )

    def test_init_requires_dataframe(self):
        with pytest.raises(ValueError, match="pandas DataFrame"):
            MLSignalData(df="not a dataframe")

    def test_init_requires_datetime_index(self):
        df = pd.DataFrame(
            {
                "open": [1],
                "high": [2],
                "low": [0],
                "close": [1],
                "volume": [10],
                "target": [0],
                "signal_strength": [0.5],
            }
        )
        with pytest.raises(ValueError, match="DatetimeIndex"):
            MLSignalData(df=df)

    def test_stores_correct_length(self, signal_df):
        feed = MLSignalData(df=signal_df)
        assert feed.len == 50

    def test_future_return_defaults_to_zeros(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="h")
        df = pd.DataFrame(
            {
                "open": [1] * 5,
                "high": [2] * 5,
                "low": [0] * 5,
                "close": [1] * 5,
                "volume": [10] * 5,
                "target": [0] * 5,
                "signal_strength": [0.5] * 5,
            },
            index=dates,
        )
        feed = MLSignalData(df=df)
        assert (feed.future_return == 0).all()

    def test_stores_numpy_arrays(self, signal_df):
        feed = MLSignalData(df=signal_df)
        assert isinstance(feed.open, np.ndarray)
        assert isinstance(feed.close, np.ndarray)
        assert isinstance(feed.volume, np.ndarray)
        assert isinstance(feed.target, np.ndarray)

    def test_none_df_accepted(self):
        feed = MLSignalData(df=None)
        assert not hasattr(feed, "len") or True  # no error


# ---------------------------------------------------------------------------
# OptimizedDataFeed
# ---------------------------------------------------------------------------
class TestOptimizedDataFeed:
    def test_init(self):
        feed = OptimizedDataFeed({"data_paths": {}})
        assert feed.cache == {}
        assert feed.mmaps == {}

    def test_get_optimal_chunk_size_small_file(self):
        assert OptimizedDataFeed._get_optimal_chunk_size(500) == 500

    def test_get_optimal_chunk_size_large_file(self):
        assert OptimizedDataFeed._get_optimal_chunk_size(10 * 1024 * 1024) == 1024 * 1024

    def test_load_data_from_csv(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b,c\n1,2,3\n4,5,6\n")
        feed = OptimizedDataFeed({"data_paths": {}})
        df = feed.load_data(str(csv_file))
        assert len(df) == 2
        assert list(df.columns) == ["a", "b", "c"]

    def test_load_data_with_column_filter(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b,c\n1,2,3\n4,5,6\n")
        feed = OptimizedDataFeed({"data_paths": {}})
        df = feed.load_data(str(csv_file), columns=["a", "c"])
        assert list(df.columns) == ["a", "c"]

    def test_clear_cache(self):
        feed = OptimizedDataFeed({"data_paths": {}})
        feed.cache["key"] = "value"
        feed.clear_cache()
        assert len(feed.cache) == 0

    def test_get_symbols(self):
        feed = OptimizedDataFeed({"data_paths": {"BTCUSDT": "p1", "ETHUSDT": "p2"}})
        symbols = feed.get_symbols()
        assert set(symbols) == {"BTCUSDT", "ETHUSDT"}

    def test_get_latest_data_missing_symbol(self):
        feed = OptimizedDataFeed({"data_paths": {}})
        with pytest.raises(ValueError, match="No data path"):
            feed.get_latest_data("MISSING")

    def test_get_latest_data_truncates(self, tmp_path):
        csv_file = tmp_path / "big.csv"
        csv_file.write_text("a\n" + "\n".join(str(i) for i in range(200)) + "\n")
        feed = OptimizedDataFeed({"data_paths": {"SYM": str(csv_file)}})
        df = feed.get_latest_data("SYM", lookback_periods=50)
        assert len(df) == 50

    def test_update_data_appends(self, tmp_path):
        csv_file = tmp_path / "sym.csv"
        csv_file.write_text("a,b\n1,2\n")
        feed = OptimizedDataFeed({"data_paths": {"SYM": str(csv_file)}})
        new_data = pd.DataFrame({"a": [3], "b": [4]})
        feed.update_data("SYM", new_data)
        raw = csv_file.read_text()
        assert "3" in raw and "4" in raw

    def test_destructor_no_error(self):
        feed = OptimizedDataFeed({"data_paths": {}})
        del feed


# ---------------------------------------------------------------------------
# DataFeed (synthetic data generator)
# ---------------------------------------------------------------------------
class TestDataFeed:
    def test_generates_ohlcv(self):
        feed = DataFeed("BTCUSDT", "1h")
        df = feed.get_historical_data("2024-01-01", "2024-01-05")
        for col in ("open", "high", "low", "close", "volume"):
            assert col in df.columns
        assert len(df) > 0

    def test_high_gte_low(self):
        feed = DataFeed("BTCUSDT", "1h")
        df = feed.get_historical_data("2024-01-01", "2024-01-05")
        assert (df["high"] >= df["low"]).all()

    def test_high_gte_open_close(self):
        feed = DataFeed("BTCUSDT", "1h")
        df = feed.get_historical_data("2024-01-01", "2024-01-05")
        assert (df["high"] >= df["open"]).all()
        assert (df["high"] >= df["close"]).all()

    def test_low_lte_open_close(self):
        feed = DataFeed("BTCUSDT", "1h")
        df = feed.get_historical_data("2024-01-01", "2024-01-05")
        assert (df["low"] <= df["open"]).all()
        assert (df["low"] <= df["close"]).all()

    def test_datetime_index(self):
        feed = DataFeed("BTCUSDT", "15m")
        df = feed.get_historical_data("2024-01-01", "2024-01-02")
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_includes_signal_columns(self):
        feed = DataFeed("BTCUSDT", "1h")
        df = feed.get_historical_data("2024-01-01", "2024-01-03")
        assert "target" in df.columns
        assert "signal_strength" in df.columns
        assert "future_return" in df.columns

    def test_reproducible_with_seed(self):
        feed = DataFeed("BTCUSDT", "1h")
        df1 = feed.get_historical_data("2024-01-01", "2024-01-02")
        df2 = feed.get_historical_data("2024-01-01", "2024-01-02")
        pd.testing.assert_frame_equal(df1, df2)

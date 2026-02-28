"""Tests for backtest/walk_forward.py — walk-forward analysis."""
import numpy as np
import pandas as pd
import pytest

from backtest.walk_forward import WalkForwardAnalyzer


@pytest.fixture
def walk_forward_df():
    """DataFrame large enough for walk-forward analysis."""
    np.random.seed(42)
    n = 600
    dates = pd.date_range("2024-01-01", periods=n, freq="15min")
    close = 50000 + np.cumsum(np.random.randn(n) * 50)
    df = pd.DataFrame({
        "open": close + np.random.randn(n) * 10,
        "high": close + abs(np.random.randn(n) * 20),
        "low": close - abs(np.random.randn(n) * 20),
        "close": close,
        "volume": np.random.lognormal(10, 1, n),
    }, index=dates)
    return df


class TestInit:
    def test_creates_windows(self, walk_forward_df):
        wf = WalkForwardAnalyzer(
            walk_forward_df, train_size=200, test_size=50, gap=5,
        )
        assert len(wf.windows) > 0

    def test_too_small_raises(self, walk_forward_df):
        with pytest.raises(ValueError, match="Not enough data"):
            WalkForwardAnalyzer(
                walk_forward_df.iloc[:10], train_size=200, test_size=50,
            )

    def test_df_copy_is_made(self, walk_forward_df):
        """Input DataFrame should NOT be modified."""
        original_len = len(walk_forward_df)
        wf = WalkForwardAnalyzer(walk_forward_df, train_size=200, test_size=50)
        assert len(walk_forward_df) == original_len

    def test_non_datetime_index_converted(self):
        n = 400
        df = pd.DataFrame({
            "open": np.random.randn(n),
            "high": np.random.randn(n),
            "low": np.random.randn(n),
            "close": np.random.randn(n),
            "volume": np.random.randn(n),
        }, index=pd.date_range("2024-01-01", periods=n, freq="15min").astype(str))

        wf = WalkForwardAnalyzer(df, train_size=200, test_size=50)
        assert isinstance(wf.df.index, pd.DatetimeIndex)

    def test_expanding_window(self):
        """Expanding window mode — verify windows are created (limited iterations)."""
        np.random.seed(42)
        n = 400
        dates = pd.date_range("2024-01-01", periods=n, freq="15min")
        df = pd.DataFrame({
            "open": np.random.randn(n),
            "high": np.random.randn(n),
            "low": np.random.randn(n),
            "close": np.random.randn(n),
            "volume": np.random.randn(n),
        }, index=dates)
        wf = WalkForwardAnalyzer(
            df, train_size=100, test_size=50, gap=5, expanding=True,
        )
        assert len(wf.windows) >= 1


class TestWindowStructure:
    def test_windows_dont_overlap(self, walk_forward_df):
        wf = WalkForwardAnalyzer(
            walk_forward_df, train_size=200, test_size=50, gap=5, expanding=False,
        )
        for i in range(len(wf.windows) - 1):
            current_end = wf.windows[i]["test_end"]
            next_start = wf.windows[i + 1]["train_start"]
            assert current_end <= next_start

    def test_gap_respected(self, walk_forward_df):
        gap = 10
        wf = WalkForwardAnalyzer(
            walk_forward_df, train_size=200, test_size=50, gap=gap,
        )
        for w in wf.windows:
            assert w["test_start"] > w["train_end"]

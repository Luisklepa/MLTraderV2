"""Tests for core utility modules: trading_utils, exceptions, data_validation edge cases."""

import numpy as np
import pandas as pd
import pytest

from core.data_validation import validate_ohlcv
from core.exceptions import ConfigError, DataError, MLTraderError, ModelError, TradingError


class TestExceptionHierarchy:
    def test_all_inherit_from_base(self):
        for cls in (DataError, ModelError, TradingError, ConfigError):
            assert issubclass(cls, MLTraderError)

    def test_base_inherits_from_exception(self):
        assert issubclass(MLTraderError, Exception)

    def test_can_catch_specific(self):
        with pytest.raises(DataError):
            raise DataError("bad data")

    def test_can_catch_via_base(self):
        with pytest.raises(MLTraderError):
            raise ModelError("bad model")


class TestTradingUtils:
    def test_calculate_stop_loss_long(self):
        from core.trading_utils import calculate_stop_loss

        sl = calculate_stop_loss(100.0, 5.0, atr_mult=2.0, direction="long")
        assert sl == pytest.approx(90.0)

    def test_calculate_stop_loss_short(self):
        from core.trading_utils import calculate_stop_loss

        sl = calculate_stop_loss(100.0, 5.0, atr_mult=2.0, direction="short")
        assert sl == pytest.approx(110.0)

    def test_calculate_stop_loss_invalid_price(self):
        from core.trading_utils import calculate_stop_loss

        sl = calculate_stop_loss(-1.0, 5.0)
        assert sl == -1.0

    def test_take_profit_long(self):
        from core.trading_utils import calculate_take_profit

        tp = calculate_take_profit(100.0, 95.0, rr_ratio=2.0, direction="long")
        assert tp == pytest.approx(110.0)

    def test_take_profit_short(self):
        from core.trading_utils import calculate_take_profit

        tp = calculate_take_profit(100.0, 105.0, rr_ratio=2.0, direction="short")
        assert tp == pytest.approx(90.0)

    def test_position_size_normal(self):
        from core.trading_utils import calculate_position_size

        size = calculate_position_size(10000.0, 0.02, 100.0)
        assert size == pytest.approx(2.0)

    def test_position_size_zero_stop(self):
        from core.trading_utils import calculate_position_size

        assert calculate_position_size(10000.0, 0.02, 0) == 0.0

    def test_position_size_negative_account(self):
        from core.trading_utils import calculate_position_size

        assert calculate_position_size(-1000.0, 0.02, 100.0) == 0.0


class TestDataValidationEdgeCases:
    def test_inf_values_dropped(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="1h")
        df = pd.DataFrame(
            {
                "open": [100, np.inf, 102, 103, 104],
                "high": [101, 201, 103, 104, 105],
                "low": [99, 199, 101, 102, 103],
                "close": [100.5, 200, 102.5, 103.5, 104.5],
                "volume": [1000, 2000, 3000, 4000, 5000],
            },
            index=dates,
        )
        result = validate_ohlcv(df)
        assert len(result) == 4
        assert not np.isinf(result["open"]).any()

    def test_single_row_no_crash(self):
        dates = pd.date_range("2024-01-01", periods=1, freq="1h")
        df = pd.DataFrame(
            {
                "open": [100],
                "high": [101],
                "low": [99],
                "close": [100.5],
                "volume": [1000],
            },
            index=dates,
        )
        result = validate_ohlcv(df)
        assert len(result) == 1

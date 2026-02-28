"""Tests for config/settings.py validation."""
import pytest
import os
from config.settings import TradingConfig, RiskConfig, DataConfig, APIConfig


class TestRiskConfigValidation:
    def test_valid_defaults(self):
        rc = RiskConfig()
        assert rc.initial_capital == 10000.0
        assert rc.risk_per_trade == 0.02

    def test_invalid_risk_per_trade_zero(self):
        with pytest.raises(ValueError, match="risk_per_trade"):
            RiskConfig(risk_per_trade=0)

    def test_invalid_risk_per_trade_above_one(self):
        with pytest.raises(ValueError, match="risk_per_trade"):
            RiskConfig(risk_per_trade=1.5)

    def test_negative_capital_rejected(self):
        with pytest.raises(ValueError, match="initial_capital"):
            RiskConfig(initial_capital=-100)

    def test_invalid_max_drawdown(self):
        with pytest.raises(ValueError, match="max_drawdown"):
            RiskConfig(max_drawdown=0)


class TestDataConfigEnvVars:
    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("BINANCE_API_KEY", "test_key_123")
        dc = DataConfig()
        assert dc.binance_api_key == "test_key_123"

    def test_default_when_no_env(self, monkeypatch):
        monkeypatch.delenv("BINANCE_API_KEY", raising=False)
        dc = DataConfig()
        assert dc.binance_api_key == ""


class TestTradingConfigLegacyCompat:
    def test_legacy_attributes_synced(self):
        rc = RiskConfig(initial_capital=50000.0, risk_per_trade=0.03)
        tc = TradingConfig(risk=rc)
        assert tc.INITIAL_CAPITAL == 50000.0
        assert tc.DEFAULT_RISK_PERC == 0.03

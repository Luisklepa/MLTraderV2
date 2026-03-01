"""Tests for core/file_management.py — file path management."""

from pathlib import Path

import pytest
import yaml

from core.file_management import FileManager


@pytest.fixture
def file_config(tmp_path):
    """Create a temporary config file for FileManager."""
    config = {
        "data": {
            "raw": {
                "directory": str(tmp_path / "data" / "raw"),
                "price_files": {"btc": "btc_prices.csv", "eth": "eth_prices.csv"},
            },
            "features": {
                "directory": str(tmp_path / "data" / "features"),
                "pattern": "*.csv",
            },
            "processed": {"directory": str(tmp_path / "data" / "processed")},
        },
        "models": {
            "random_forest": {"directory": str(tmp_path / "models" / "rf")},
            "xgboost": {"directory": str(tmp_path / "models" / "xgb")},
        },
        "results": {
            "feature_importance": {"directory": str(tmp_path / "results" / "fi")},
            "plots": {"directory": str(tmp_path / "results" / "plots")},
            "metrics": {"directory": str(tmp_path / "results" / "metrics")},
        },
        "cache": {
            "directory": str(tmp_path / "cache"),
            "max_size": "1GB",
            "ttl": 3600,
        },
    }
    config_path = tmp_path / "file_paths.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return str(config_path)


class TestFileManager:
    def test_loads_config(self, file_config):
        fm = FileManager(file_config)
        assert "data" in fm.config

    def test_get_raw_data_path(self, file_config):
        fm = FileManager(file_config)
        path = fm.get_raw_data_path("btc")
        assert "btc_prices.csv" in path

    def test_get_feature_path(self, file_config):
        fm = FileManager(file_config)
        path = fm.get_feature_path()
        assert "*.csv" in path

    def test_get_model_path(self, file_config):
        fm = FileManager(file_config)
        path = fm.get_model_path("xgboost", "model.pkl")
        assert "model.pkl" in path

    def test_get_result_path(self, file_config):
        fm = FileManager(file_config)
        path = fm.get_result_path("plots", "equity.png")
        assert "equity.png" in path

    def test_ensure_directories(self, file_config):
        fm = FileManager(file_config)
        fm.ensure_directories()
        assert Path(fm.config["data"]["raw"]["directory"]).exists()
        assert Path(fm.config["models"]["xgboost"]["directory"]).exists()
        assert Path(fm.config["cache"]["directory"]).exists()

    def test_parse_size(self, file_config):
        fm = FileManager(file_config)
        assert fm._parse_size("1GB") == 1024**3
        assert fm._parse_size("512MB") == 512 * 1024**2
        assert fm._parse_size("1024KB") == 1024 * 1024

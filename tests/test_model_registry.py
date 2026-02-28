"""Tests for ml/model_registry.py — versioned model storage."""
import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier

from ml import model_registry
from core.exceptions import ModelError


def _make_dummy_model():
    """Create a real, pickleable model."""
    clf = DecisionTreeClassifier(max_depth=1, random_state=0)
    clf.fit([[0], [1]], [0, 1])
    return clf


@pytest.fixture
def registry_dir(tmp_path, monkeypatch):
    """Override the registry directory to a temp location."""
    monkeypatch.setattr(model_registry, "REGISTRY_DIR", str(tmp_path / "models"))
    return tmp_path / "models"


class TestSaveModel:
    def test_saves_model_and_metadata(self, registry_dir):
        model = _make_dummy_model()
        path = model_registry.save_model(
            model=model,
            version="v1.0",
            metrics={"f1": 0.85, "accuracy": 0.90},
            feature_names=["f1", "f2", "f3"],
        )
        model_dir = Path(path)
        assert (model_dir / "model.pkl").exists()
        assert (model_dir / "metadata.json").exists()

        with open(model_dir / "metadata.json") as f:
            meta = json.load(f)
        assert meta["version"] == "v1.0"
        assert meta["n_features"] == 3

    def test_saves_scaler(self, registry_dir):
        from sklearn.preprocessing import StandardScaler
        model = _make_dummy_model()
        scaler = StandardScaler()
        scaler.fit([[1.0], [2.0]])
        path = model_registry.save_model(
            model=model,
            version="v1.1",
            metrics={"f1": 0.8},
            feature_names=["a"],
            scaler=scaler,
        )
        assert (Path(path) / "scaler.pkl").exists()

    def test_saves_config(self, registry_dir):
        model = _make_dummy_model()
        config = {"learning_rate": 0.1, "max_depth": 5}
        path = model_registry.save_model(
            model=model,
            version="v1.2",
            metrics={},
            feature_names=["a"],
            config=config,
        )
        with open(Path(path) / "metadata.json") as f:
            meta = json.load(f)
        assert meta["config"]["learning_rate"] == 0.1


class TestLoadModel:
    def test_loads_saved_model(self, registry_dir):
        import joblib

        model_dir = registry_dir / "v2.0"
        model_dir.mkdir(parents=True)

        fake_model = {"type": "test_model"}
        joblib.dump(fake_model, model_dir / "model.pkl")

        meta = {
            "version": "v2.0",
            "created_at": "2024-01-01T00:00:00",
            "metrics": {"f1": 0.9},
            "feature_names": ["x1", "x2"],
            "n_features": 2,
            "config": {},
        }
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(meta, f)

        model, metadata, scaler = model_registry.load_model("v2.0")
        assert model == {"type": "test_model"}
        assert metadata["version"] == "v2.0"
        assert scaler is None

    def test_missing_version_raises(self, registry_dir):
        with pytest.raises(ModelError, match="not found"):
            model_registry.load_model("nonexistent")


class TestListVersions:
    def test_lists_versions(self, registry_dir):
        for v in ["v1.0", "v2.0"]:
            model_registry.save_model(
                model=_make_dummy_model(),
                version=v,
                metrics={"f1": 0.8},
                feature_names=["a"],
            )
        versions = model_registry.list_versions()
        assert len(versions) == 2
        version_names = [v["version"] for v in versions]
        assert "v1.0" in version_names
        assert "v2.0" in version_names


class TestValidateFeatures:
    def test_matching_features(self, registry_dir):
        model_registry.save_model(
            model=_make_dummy_model(),
            version="v3.0",
            metrics={},
            feature_names=["a", "b", "c"],
        )
        assert model_registry.validate_features("v3.0", ["a", "b", "c"]) is True

    def test_missing_features_returns_false(self, registry_dir):
        model_registry.save_model(
            model=_make_dummy_model(),
            version="v3.1",
            metrics={},
            feature_names=["a", "b", "c"],
        )
        assert model_registry.validate_features("v3.1", ["a", "b"]) is False

    def test_extra_features_still_valid(self, registry_dir):
        model_registry.save_model(
            model=_make_dummy_model(),
            version="v3.2",
            metrics={},
            feature_names=["a", "b"],
        )
        assert model_registry.validate_features("v3.2", ["a", "b", "extra"]) is True

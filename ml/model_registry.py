"""Versioned model registry for production ML models."""
import os
import json
import shutil
import joblib
import logging
from datetime import datetime
from typing import Any, Dict, Optional, List
from pathlib import Path

from core.exceptions import ModelError

logger = logging.getLogger(__name__)

REGISTRY_DIR = "models"
METADATA_FILE = "metadata.json"


def _ensure_registry() -> Path:
    p = Path(REGISTRY_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_model(
    model: Any,
    version: str,
    metrics: Dict[str, float],
    feature_names: List[str],
    config: Optional[Dict] = None,
    scaler: Any = None,
) -> str:
    """Save a model with metadata to the registry.

    Returns the path to the saved model directory.
    """
    registry = _ensure_registry()
    model_dir = registry / version
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.pkl"
    joblib.dump(model, model_path)

    if scaler is not None:
        joblib.dump(scaler, model_dir / "scaler.pkl")

    metadata = {
        "version": version,
        "created_at": datetime.utcnow().isoformat(),
        "metrics": metrics,
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "config": config or {},
    }
    with open(model_dir / METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Model saved to {model_dir} (v{version}, {len(feature_names)} features)")
    return str(model_dir)


def load_model(version: str) -> tuple:
    """Load a model and its metadata from the registry.

    Returns (model, metadata, scaler_or_None).
    """
    registry = _ensure_registry()
    model_dir = registry / version

    if not model_dir.exists():
        raise ModelError(f"Model version '{version}' not found in {registry}")

    model = joblib.load(model_dir / "model.pkl")

    with open(model_dir / METADATA_FILE, "r") as f:
        metadata = json.load(f)

    scaler_path = model_dir / "scaler.pkl"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    logger.info(f"Loaded model v{version} ({metadata['n_features']} features)")
    return model, metadata, scaler


def list_versions() -> List[Dict]:
    """List all model versions in the registry, newest first."""
    registry = _ensure_registry()
    versions = []

    for entry in sorted(registry.iterdir(), reverse=True):
        meta_path = entry / METADATA_FILE
        if entry.is_dir() and meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            versions.append(meta)

    return versions


def validate_features(version: str, current_features: List[str]) -> bool:
    """Check that the current feature set matches the model's expected features."""
    _, metadata, _ = load_model(version)
    expected = set(metadata["feature_names"])
    actual = set(current_features)

    missing = expected - actual
    extra = actual - expected

    if missing:
        logger.error(f"Missing features for model v{version}: {missing}")
        return False
    if extra:
        logger.warning(f"Extra features not used by model v{version}: {extra}")

    return True

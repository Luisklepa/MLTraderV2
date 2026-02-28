"""Application health check for MLTraderV2.

``check_health()`` returns a dict with component statuses, suitable for
JSON serialisation as a health endpoint or CLI diagnostic.
"""
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def check_health() -> Dict[str, Any]:
    """Run a quick health-check across core subsystems.

    Returns a dict::

        {
            "status": "healthy" | "degraded" | "unhealthy",
            "checks": { <component>: { "ok": bool, "detail": str }, ... }
        }
    """
    checks: Dict[str, Dict[str, Any]] = {}

    checks["config"] = _check_config()
    checks["data_dir"] = _check_data_dir()
    checks["models"] = _check_models()
    checks["logging"] = _check_logging()

    ok_count = sum(1 for c in checks.values() if c["ok"])
    total = len(checks)

    if ok_count == total:
        status = "healthy"
    elif ok_count >= total // 2:
        status = "degraded"
    else:
        status = "unhealthy"

    return {"status": status, "checks": checks}


def _check_config() -> Dict[str, Any]:
    try:
        from config.settings import TradingConfig
        TradingConfig()
        return {"ok": True, "detail": "Config loads successfully"}
    except Exception as e:
        return {"ok": False, "detail": str(e)}


def _check_data_dir() -> Dict[str, Any]:
    data_dir = Path("data")
    if data_dir.exists() and data_dir.is_dir():
        n_files = len(list(data_dir.rglob("*.csv")))
        return {"ok": True, "detail": f"{n_files} CSV files in data/"}
    return {"ok": False, "detail": "data/ directory not found"}


def _check_models() -> Dict[str, Any]:
    models_dir = Path("models")
    if not models_dir.exists():
        return {"ok": False, "detail": "models/ directory not found"}

    model_files = list(models_dir.rglob("*.pkl"))
    if not model_files:
        return {"ok": False, "detail": "No .pkl model files found in models/"}
    return {"ok": True, "detail": f"{len(model_files)} model files found"}


def _check_logging() -> Dict[str, Any]:
    logs_dir = Path("logs")
    if logs_dir.exists():
        return {"ok": True, "detail": "logs/ directory exists"}
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
        return {"ok": True, "detail": "logs/ directory created"}
    except OSError as e:
        return {"ok": False, "detail": f"Cannot create logs/: {e}"}

"""
Centralized logging configuration for MLTraderV2.

Call ``setup_logging()`` once at the start of every entry-point
(app.py, scripts, CLI commands) to wire up console + file logging
using settings from ``config.settings.LogConfig``.
"""

import logging
import sys
from pathlib import Path

_CONFIGURED = False


def setup_logging(
    level: str | None = None,
    fmt: str | None = None,
    file_path: str | None = None,
) -> None:
    """Initialise the root logger with console + rotating file handler.

    Parameters default to the values in ``config.settings.LogConfig``
    so callers can simply do ``setup_logging()`` with no arguments.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    from config.settings import LogConfig

    cfg = LogConfig()
    level = level or cfg.level
    fmt = fmt or cfg.format
    file_path = file_path or cfg.file_path

    log_dir = Path(file_path).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(numeric_level)

    if root.handlers:
        return

    formatter = logging.Formatter(fmt)

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(numeric_level)
    console.setFormatter(formatter)
    root.addHandler(console)

    try:
        from logging.handlers import RotatingFileHandler

        file_handler = RotatingFileHandler(
            file_path,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError:
        logging.getLogger(__name__).warning(
            "Could not create log file at %s — file logging disabled",
            file_path,
        )

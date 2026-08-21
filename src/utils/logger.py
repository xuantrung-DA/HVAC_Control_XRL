"""Consistent console and rotating-file logging for XRL-HVAC."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.utils.config import PROJECT_ROOT


LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(
    level: str | int = "INFO",
    *,
    log_directory: str | Path | None = None,
    log_filename: str = "xrl_hvac.log",
) -> logging.Logger:
    """Configure and return the project logger without duplicating handlers."""

    logger = logging.getLogger("xrl_hvac")
    resolved_level = _resolve_level(level)
    logger.setLevel(resolved_level)
    logger.propagate = False

    if getattr(logger, "_xrl_hvac_configured", False):
        for handler in logger.handlers:
            handler.setLevel(resolved_level)
        return logger

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(resolved_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    target_directory = Path(log_directory) if log_directory else PROJECT_ROOT / "outputs" / "logs"
    target_directory.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        target_directory / log_filename,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(resolved_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    setattr(logger, "_xrl_hvac_configured", True)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the project namespace."""

    return logging.getLogger(f"xrl_hvac.{name}")


def _resolve_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    resolved = logging.getLevelName(level.upper())
    if not isinstance(resolved, int):
        raise ValueError(f"Unknown logging level: {level}")
    return resolved

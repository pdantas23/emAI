"""Centralized logger configuration using loguru.

Import `log` everywhere instead of building loggers ad-hoc:
    from src.utils.logger import log
    log.info("ready")
"""

from __future__ import annotations

import sys

from loguru import logger

from config.settings import PROJECT_ROOT, settings

# Remove loguru's default stderr sink so we own the configuration.
logger.remove()

# Console sink — colorized for humans during dev.
logger.add(
    sys.stderr,
    level=settings.log_level.value,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> "
        "| <level>{level: <8}</level> "
        "| <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
        "- <level>{message}</level>"
    ),
    backtrace=False,
    diagnose=not settings.is_production,  # Hide locals in prod (LGPD/PII)
)

# File sink — rotating, structured-ish for grep/parsing.
logger.add(
    PROJECT_ROOT / "logs" / "emai.log",
    level=settings.log_level.value,
    rotation="10 MB",
    retention="14 days",
    compression="zip",
    enqueue=True,  # Thread-safe; required when running under APScheduler workers
    backtrace=False,
    diagnose=False,
)

log = logger
"""Public logger handle. Always use this, never instantiate loguru directly."""

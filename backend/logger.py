"""
Jarvis V1 — Logging Infrastructure
=====================================
Provides structured, rotating log files with separate error tracking.

Architecture decisions:
    - Two log files: jarvis.log (all levels) and error.log (ERROR+ only)
    - RotatingFileHandler prevents unbounded log growth
    - Console handler for real-time visibility during development
    - Each module gets its own named logger via get_logger(__name__)
    - Log directory is auto-created on initialization

Usage:
    from backend.logger import get_logger
    logger = get_logger(__name__)
    logger.info("System initialized")
    logger.error("Something failed", exc_info=True)
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from configs.settings import get_settings


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
_initialized = False


def _ensure_log_directory(log_dir: Path) -> None:
    """Create the log directory if it doesn't exist."""
    log_dir.mkdir(parents=True, exist_ok=True)


def initialize_logging(settings=None) -> None:
    """
    Set up the global logging configuration.
    
    This should be called once at application startup. Subsequent calls
    are no-ops to prevent duplicate handlers.
    
    Creates:
        - logs/jarvis.log  — all log messages (rotating, 5MB per file)
        - logs/error.log   — ERROR and above only (rotating, 5MB per file)
        - console output   — INFO and above for real-time visibility
    """
    global _initialized
    if _initialized:
        return

    if settings is None:
        settings = get_settings()

    log_config = settings.logging
    log_dir = settings.resolve_path(log_config.log_dir)
    _ensure_log_directory(log_dir)

    # --- Root logger configuration ---
    root_logger = logging.getLogger("jarvis")
    root_logger.setLevel(getattr(logging, log_config.level.upper(), logging.INFO))

    # Prevent propagation to root logger (avoids duplicate output)
    root_logger.propagate = False

    # Clear any existing handlers (safety for re-initialization)
    root_logger.handlers.clear()

    formatter = logging.Formatter(
        fmt=log_config.format,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # --- File handler: all logs ---
    general_handler = RotatingFileHandler(
        filename=log_dir / "jarvis.log",
        maxBytes=log_config.max_file_size,
        backupCount=log_config.backup_count,
        encoding="utf-8",
    )
    general_handler.setLevel(logging.DEBUG)
    general_handler.setFormatter(formatter)
    root_logger.addHandler(general_handler)

    # --- File handler: errors only ---
    error_handler = RotatingFileHandler(
        filename=log_dir / "error.log",
        maxBytes=log_config.max_file_size,
        backupCount=log_config.backup_count,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)

    # --- Console handler: INFO+ for real-time visibility ---
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.WARNING)  # Keep console clean; only warnings+
    console_handler.setFormatter(logging.Formatter(
        fmt="%(levelname)-8s | %(name)s | %(message)s"
    ))
    root_logger.addHandler(console_handler)

    _initialized = True
    root_logger.info("Logging initialized — log_dir=%s, level=%s", log_dir, log_config.level)


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger under the 'jarvis' namespace.
    
    Args:
        name: Module name, typically __name__. Will be prefixed with 'jarvis.'
              if not already.
    
    Returns:
        A configured logger instance.
    
    Usage:
        logger = get_logger(__name__)
        logger.info("Module loaded")
    """
    # Ensure logging is initialized
    if not _initialized:
        initialize_logging()

    # Namespace all loggers under 'jarvis' for consistent filtering
    if not name.startswith("jarvis."):
        # Convert module paths like 'backend.database' to 'jarvis.backend.database'
        logger_name = f"jarvis.{name}"
    else:
        logger_name = name

    return logging.getLogger(logger_name)

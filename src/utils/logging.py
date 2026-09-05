"""Centralised logging configuration.

Call :func:`get_logger` from any module. The first call configures the root
logger once (idempotently); subsequent calls just return a named child logger.

Log level is read from the ``LOG_LEVEL`` environment variable (default ``INFO``).
Format is a compact single line with timestamp, level, logger name and message,
which is friendly to both terminals and container log collectors.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Final

_CONFIGURED: bool = False
_DEFAULT_LEVEL: Final[str] = "INFO"
_FORMAT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATEFMT: Final[str] = "%Y-%m-%d %H:%M:%S"


def _configure_root() -> None:
    """Attach a single stderr handler to the root logger (idempotent)."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = os.environ.get("LOG_LEVEL", _DEFAULT_LEVEL).upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(fmt=_FORMAT, datefmt=_DATEFMT))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Quieten libraries that log verbosely at INFO.
    for noisy in ("rasterio", "fiona", "matplotlib", "urllib3", "botocore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger.

    Args:
        name: Logger name, conventionally ``__name__`` of the calling module.

    Returns:
        A :class:`logging.Logger` writing single-line records to stderr.
    """
    _configure_root()
    return logging.getLogger(name)

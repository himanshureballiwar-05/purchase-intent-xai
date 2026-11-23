"""Central logging setup so every module logs the same way (no print())."""

from __future__ import annotations

import logging
import os

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_CONFIGURED = False


def setup_logging(level: str | None = None) -> None:
    """Configure root logging once. Level comes from arg or ``$LOG_LEVEL`` (default INFO)."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    chosen = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(level=getattr(logging, chosen, logging.INFO), format=_FORMAT)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger (configures logging on first use)."""
    setup_logging()
    return logging.getLogger(name)

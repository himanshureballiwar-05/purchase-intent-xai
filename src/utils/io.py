"""Small helpers for saving/loading JSON and joblib artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _json_default(value: Any) -> Any:
    """Make numpy scalars/arrays JSON serialisable."""
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def save_json(obj: Any, path: str | Path) -> Path:
    """Write ``obj`` as pretty JSON, creating parent folders."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)
    logger.info("Saved JSON -> %s", path)
    return path


def load_json(path: str | Path) -> Any:
    """Read a JSON file; raises ``FileNotFoundError`` with a clear message."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_joblib(obj: Any, path: str | Path) -> Path:
    """Persist an object with joblib, creating parent folders."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    logger.info("Saved artifact -> %s", path)
    return path


def load_joblib(path: str | Path) -> Any:
    """Load a joblib artifact; raises ``FileNotFoundError`` with a clear message."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Artifact not found: {path}. Run `python -m src.models.train` first."
        )
    return joblib.load(path)

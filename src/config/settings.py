"""Typed settings loader: reads ``config.yaml`` once and exposes it as properties."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "src" / "config" / "config.yaml"


class ConfigError(RuntimeError):
    """Raised when the configuration file is missing or malformed."""


@dataclass(frozen=True)
class Settings:
    """Immutable view over the YAML configuration."""

    raw: dict[str, Any]

    def _get(self, *keys: str) -> Any:
        node: Any = self.raw
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                raise ConfigError(f"Missing config key: {'.'.join(keys)}")
            node = node[key]
        return node

    def path(self, name: str) -> Path:
        """Resolve a path from the ``paths`` section relative to the project root."""
        return PROJECT_ROOT / self._get("paths", name)

    @property
    def seed(self) -> int:
        return int(self._get("project", "random_seed"))

    @property
    def target(self) -> str:
        return str(self._get("data", "target"))

    @property
    def test_size(self) -> float:
        return float(self._get("data", "test_size"))

    @property
    def categorical_columns(self) -> list[str]:
        return list(self._get("data", "categorical_columns"))

    @property
    def id_categorical_columns(self) -> list[str]:
        return list(self._get("data", "id_categorical_columns"))

    @property
    def boolean_columns(self) -> list[str]:
        return list(self._get("data", "boolean_columns"))

    @property
    def raw_numeric_columns(self) -> list[str]:
        return list(self._get("data", "raw_numeric_columns"))

    @property
    def duration_columns(self) -> list[str]:
        return list(self._get("data", "duration_columns"))

    @property
    def count_columns(self) -> list[str]:
        return list(self._get("data", "count_columns"))

    @property
    def input_columns(self) -> list[str]:
        """All raw model-input columns, in schema order (no target)."""
        return self.raw_numeric_columns + self.categorical_columns + self.boolean_columns

    @property
    def engineered_features(self) -> list[str]:
        return list(self._get("features", "engineered"))

    @property
    def use_page_values(self) -> bool:
        return bool(self._get("features", "use_page_values"))

    def section(self, name: str) -> dict[str, Any]:
        return dict(self._get(name))


def load_settings(config_path: str | Path | None = None) -> Settings:
    """Load settings from ``config_path`` (or ``$PI_CONFIG`` or the default file)."""
    path = Path(config_path or os.environ.get("PI_CONFIG", DEFAULT_CONFIG_PATH))
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Config file is not valid YAML: {path} ({exc})") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"Config file is empty or malformed: {path}")
    return Settings(raw=raw)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached default settings (what the API and dashboard use)."""
    return load_settings()

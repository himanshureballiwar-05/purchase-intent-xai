"""Data-access layer: load + validate ``ecommerce_sessions.csv``.

Every part of the project (notebooks, training, API, dashboard) reads data
through :class:`SessionDataLoader`, so they all see exactly the same frame.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.settings import Settings, get_settings
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class DataValidationError(ValueError):
    """Raised when the dataset does not match the expected schema."""


class SessionDataLoader:
    """Load, validate and split the sessions dataset."""

    def __init__(self, settings: Settings | None = None, path: str | Path | None = None):
        self.settings = settings or get_settings()
        self.path = Path(path) if path else self.settings.path("dataset")

    def load(self) -> pd.DataFrame:
        """Read the CSV, validate the schema and return the DataFrame."""
        if not self.path.exists():
            logger.error("Dataset file missing: %s", self.path)
            raise FileNotFoundError(f"Dataset not found at {self.path}")
        try:
            df = pd.read_csv(self.path)
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
            logger.error("Dataset could not be parsed: %s", exc)
            raise DataValidationError(f"Malformed dataset file {self.path}: {exc}") from exc
        self.validate(df)
        logger.info("Loaded %d rows x %d columns from %s", df.shape[0], df.shape[1], self.path.name)
        return df

    def validate(self, df: pd.DataFrame) -> None:
        """Check required columns, target values and numeric dtypes."""
        expected = set(self.settings.input_columns) | {self.settings.target}
        missing = expected - set(df.columns)
        if missing:
            raise DataValidationError(f"Missing expected columns: {sorted(missing)}")
        if df.empty:
            raise DataValidationError("Dataset has zero rows")
        target = self.settings.target
        if not set(df[target].dropna().unique()) <= {0, 1}:
            raise DataValidationError(f"Target '{target}' must be binary (0/1)")
        bad = [
            c for c in self.settings.raw_numeric_columns if not pd.api.types.is_numeric_dtype(df[c])
        ]
        if bad:
            raise DataValidationError(f"Columns expected numeric but are not: {bad}")
        logger.info(
            "Validation OK | duplicates=%d | missing cells=%d",
            int(df.duplicated().sum()),
            int(df.isna().sum().sum()),
        )

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Stratified, seeded split -> ``X_train, X_test, y_train, y_test``."""
        X = df[self.settings.input_columns].copy()
        y = df[self.settings.target].astype(int)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=self.settings.test_size, stratify=y, random_state=self.settings.seed
        )
        logger.info(
            "Split -> train=%d (pos=%.3f) test=%d (pos=%.3f)",
            len(X_tr),
            y_tr.mean(),
            len(X_te),
            y_te.mean(),
        )
        return X_tr, X_te, y_tr, y_te


def load_dataset(settings: Settings | None = None) -> pd.DataFrame:
    """Convenience wrapper used by notebooks and the dashboard."""
    return SessionDataLoader(settings).load()

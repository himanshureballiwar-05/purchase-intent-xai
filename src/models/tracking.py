"""MLflow experiment tracking (failure-tolerant: training never dies because of tracking)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from src.config.settings import Settings, get_settings
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class Tracker:
    """Thin wrapper around MLflow so the rest of the code stays clean."""

    def __init__(self, settings: Settings | None = None, enabled: bool = True):
        self.settings = settings or get_settings()
        self.enabled = enabled
        self._mlflow = None
        if not enabled:
            return
        try:
            import mlflow

            cfg = self.settings.section("mlflow")
            mlflow.set_tracking_uri(cfg["tracking_uri"])
            mlflow.set_experiment(cfg["experiment_name"])
            self._mlflow = mlflow
        except Exception as exc:  # noqa: BLE001 - tracking must never break training
            logger.warning("MLflow disabled: %s", exc)
            self.enabled = False

    @contextmanager
    def run(self, name: str, tags: dict[str, str] | None = None) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        try:
            with self._mlflow.start_run(run_name=name, tags=tags or {}):
                yield
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLflow run '%s' failed: %s", name, exc)
            yield

    def log(
        self, params: dict[str, Any] | None = None, metrics: dict[str, float] | None = None
    ) -> None:
        if not self.enabled or self._mlflow is None:
            return
        try:
            if params:
                self._mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
            if metrics:
                self._mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLflow log failed: %s", exc)

    def log_artifact(self, path: str) -> None:
        if not self.enabled or self._mlflow is None:
            return
        try:
            self._mlflow.log_artifact(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLflow artifact failed: %s", exc)

    def log_and_register_model(self, pipeline, X_example) -> None:
        """Log the fitted pipeline with an input signature and register it."""
        if not self.enabled or self._mlflow is None:
            return
        try:
            from mlflow.models import infer_signature

            example = X_example.copy()
            int_cols = example.select_dtypes("integer").columns
            example[int_cols] = example[int_cols].astype("float64")  # avoids int-schema warning
            sig = infer_signature(example, pipeline.predict_proba(X_example)[:, 1])
            self._mlflow.sklearn.log_model(
                pipeline,
                name="model",
                signature=sig,
                serialization_format="cloudpickle",
                registered_model_name=self.settings.section("mlflow")["registered_model_name"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLflow model registration failed: %s", exc)

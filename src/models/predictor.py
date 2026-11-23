"""Serving logic shared by the FastAPI service and the Streamlit dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config.settings import Settings, get_settings
from src.explain.explainer import SessionExplainer
from src.utils.io import load_joblib, load_json
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

PIPELINE_FILE = "model_pipeline.joblib"
METADATA_FILE = "metadata.json"
BACKGROUND_FILE = "explain_background.joblib"


class Predictor:
    """Loads artifacts once; scores raw sessions and explains them."""

    def __init__(self, artifact_dir: str | Path | None = None, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.dir = Path(artifact_dir) if artifact_dir else self.settings.path("artifacts_dir")
        self.pipeline = load_joblib(self.dir / PIPELINE_FILE)
        self.metadata: dict[str, Any] = load_json(self.dir / METADATA_FILE)
        background = load_joblib(self.dir / BACKGROUND_FILE)
        self.threshold = float(self.metadata["threshold"])
        self.explainer = SessionExplainer(self.pipeline, background, self.settings)
        logger.info(
            "Predictor ready | model=%s | threshold=%.2f",
            self.metadata["model_name"],
            self.threshold,
        )

    # ------------------------------------------------------------------
    def to_frame(self, sessions: list[dict[str, Any]]) -> pd.DataFrame:
        """Raw dicts -> DataFrame in schema column order."""
        df = pd.DataFrame(sessions)
        missing = set(self.settings.input_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing input fields: {sorted(missing)}")
        return df[self.settings.input_columns]

    def confidence(self, probability: float) -> str:
        """high / medium / low = distance from the threshold, relative to the room on that side.

        A probability of 0.1% with threshold 0.14 is almost the furthest possible "no", so it is
        *high* confidence; a probability right at the threshold is *low*.
        """
        room = (1 - self.threshold) if probability >= self.threshold else self.threshold
        dist = abs(probability - self.threshold) / max(room, 1e-9)
        return "high" if dist >= 0.6 else "medium" if dist >= 0.25 else "low"

    def predict(
        self, sessions: list[dict[str, Any]], explain: bool = True, top_k: int | None = None
    ):
        """Score sessions; optionally attach the top SHAP contributors for each."""
        X = self.to_frame(sessions)
        probs = self.pipeline.predict_proba(X)[:, 1]
        results = []
        for i, p in enumerate(probs):
            item: dict[str, Any] = {
                "purchase_probability": round(float(p), 4),
                "prediction": int(p >= self.threshold),
                "threshold": self.threshold,
                "confidence": self.confidence(float(p)),
            }
            if explain:
                item["top_contributors"] = self.explainer.local_shap(X.iloc[[i]], top_k)
            results.append(item)
        return results

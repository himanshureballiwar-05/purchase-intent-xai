"""SHAP (global + local) and LIME (local) explanations for the fitted pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

from src.config.settings import Settings, get_settings
from src.preprocessing.pipeline import parent_feature, transform_features, transformed_feature_names
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _positive_class(values: Any) -> np.ndarray:
    """Normalise SHAP output of any model family to shape (n_samples, n_features)."""
    if isinstance(values, list):  # older shap: [class0, class1]
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:  # (n, features, classes)
        values = values[:, :, 1]
    return values


class SessionExplainer:
    """Explain predictions in terms of *original* behaviour features.

    The SHAP explainer is built once (``__init__``) and reused for every request.
    """

    def __init__(
        self, pipeline, background_transformed: np.ndarray, settings: Settings | None = None
    ):
        self.settings = settings or get_settings()
        self.pipeline = pipeline
        self.model = pipeline.named_steps["clf"]
        self.feature_names = transformed_feature_names(pipeline)
        self.categorical = self.settings.categorical_columns
        self.parents = [parent_feature(n, self.categorical) for n in self.feature_names]
        self.background = np.asarray(background_transformed)
        self._shap = self._build_shap()
        self._lime: LimeTabularExplainer | None = None

    # ------------------------------------------------------------------
    def _build_shap(self):
        name = type(self.model).__name__
        if name == "LogisticRegression":
            return shap.LinearExplainer(self.model, self.background)
        if name == "DecisionTreeClassifier" or name in {
            "RandomForestClassifier",
            "XGBClassifier",
            "LGBMClassifier",
            "CatBoostClassifier",
        }:
            return shap.TreeExplainer(self.model)
        return shap.Explainer(self.model.predict_proba, self.background)

    def shap_matrix(self, X_raw: pd.DataFrame) -> np.ndarray:
        """SHAP values (n, n_transformed_features) for raw input rows."""
        Xt = transform_features(self.pipeline, X_raw)
        try:
            values = self._shap.shap_values(Xt, check_additivity=False)
        except TypeError:  # LinearExplainer has no check_additivity
            values = self._shap.shap_values(Xt)
        return _positive_class(values)

    def aggregate(self, shap_row: np.ndarray) -> dict[str, float]:
        """Sum one-hot columns back into their parent feature."""
        agg: dict[str, float] = {}
        for parent, val in zip(self.parents, shap_row, strict=True):
            agg[parent] = agg.get(parent, 0.0) + float(val)
        return agg

    # ---- global ------------------------------------------------------
    def global_importance(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        """Mean |SHAP| per original feature on ``X_raw``, sorted high -> low."""
        sv = self.shap_matrix(X_raw)
        rows = [self.aggregate(r) for r in sv]
        agg = pd.DataFrame(rows)
        out = agg.abs().mean().sort_values(ascending=False).rename("mean_abs_shap").reset_index()
        return out.rename(columns={"index": "feature"})

    # ---- local -------------------------------------------------------
    def local_shap(self, row: pd.DataFrame, top_k: int | None = None) -> list[dict[str, Any]]:
        """Top contributors for a single session (row = 1-row raw DataFrame)."""
        k = top_k or int(self.settings.section("explain")["top_k"])
        agg = self.aggregate(self.shap_matrix(row)[0])
        items = sorted(agg.items(), key=lambda kv: abs(kv[1]), reverse=True)[:k]
        # raw + engineered values (e.g. TotalPages) so every contributor shows its value
        feature_values = self.pipeline.named_steps["features"].transform(row).iloc[0]
        out = []
        for feat, val in items:
            raw_val = feature_values.get(feat)
            if hasattr(raw_val, "item"):
                raw_val = raw_val.item()
            if isinstance(raw_val, float):
                raw_val = round(raw_val, 4)
            out.append(
                {
                    "feature": feat,
                    "value": raw_val,
                    "shap_value": round(float(val), 4),
                    "direction": (
                        "increases purchase intent" if val > 0 else "decreases purchase intent"
                    ),
                }
            )
        return out

    def base_value(self) -> float:
        ev = np.asarray(self._shap.expected_value).ravel()
        return float(ev[-1])

    def lime_explain(
        self, row: pd.DataFrame, num_features: int | None = None
    ) -> list[dict[str, Any]]:
        """LIME local explanation (a second opinion next to SHAP)."""
        n = num_features or int(self.settings.section("explain")["lime_num_features"])
        if self._lime is None:
            self._lime = LimeTabularExplainer(
                self.background,
                feature_names=self.feature_names,
                class_names=["no_purchase", "purchase"],
                mode="classification",
                discretize_continuous=True,
                random_state=self.settings.seed,
            )
        xt = transform_features(self.pipeline, row)[0]
        exp = self._lime.explain_instance(
            xt, self.model.predict_proba, num_features=n, num_samples=1500
        )
        return [{"rule": rule, "weight": round(float(w), 4)} for rule, w in exp.as_list()]

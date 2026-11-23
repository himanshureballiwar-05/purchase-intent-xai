"""Leak-free preprocessing: ColumnTransformer inside a (imblearn) Pipeline.

Encoding, scaling and resampling are *steps of the pipeline*, so during
cross-validation they are fitted on the training folds only.
"""

from __future__ import annotations

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler

from src.config.settings import Settings, get_settings
from src.features.engineering import ENGINEERED_BINARY, BehaviourFeatures

IMBALANCE_STRATEGIES = ("smote", "class_weight", "none")


def _log1p(X):
    return np.log1p(np.clip(X, 0, None))


def _skewed_and_bounded(settings: Settings, use_page_values: bool) -> tuple[list[str], list[str]]:
    """Split numeric columns into right-skewed (log1p + scale) and bounded (scale only)."""
    skewed = settings.count_columns + settings.duration_columns
    skewed += ["TotalPages", "TotalDuration", "AvgTimePerPage", "TimePerProductPage"]
    bounded = ["BounceRates", "ExitRates", "SpecialDay", "ProductPageShare"]
    if use_page_values:
        skewed.append("PageValues")
    return skewed, bounded


def build_preprocessor(settings: Settings | None = None, use_page_values: bool | None = None):
    """Return the ColumnTransformer that encodes categoricals and scales numerics."""
    s = settings or get_settings()
    upv = s.use_page_values if use_page_values is None else use_page_values
    skewed, bounded = _skewed_and_bounded(s, upv)
    log_scale = SkPipeline(
        [
            ("log1p", FunctionTransformer(_log1p, feature_names_out="one-to-one")),
            ("scale", RobustScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("skew", log_scale, skewed),
            ("bounded", RobustScaler(), bounded),
            # handle_unknown='ignore': unseen categories at serving time don't crash
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                s.categorical_columns,
            ),
            ("flag", "passthrough", s.boolean_columns + ENGINEERED_BINARY),
        ],
        verbose_feature_names_out=True,
    )


def build_pipeline(
    classifier: BaseEstimator,
    strategy: str = "none",
    settings: Settings | None = None,
    use_page_values: bool | None = None,
) -> Pipeline:
    """Assemble ``features -> prep -> [SMOTE] -> clf``.

    ``strategy='class_weight'`` is handled by the caller when it builds the
    classifier (class_weight / scale_pos_weight); here only SMOTE adds a step.
    """
    if strategy not in IMBALANCE_STRATEGIES:
        raise ValueError(
            f"Unknown imbalance strategy '{strategy}'. Use one of {IMBALANCE_STRATEGIES}"
        )
    s = settings or get_settings()
    steps = [
        ("features", BehaviourFeatures.from_settings(s, use_page_values)),
        ("prep", build_preprocessor(s, use_page_values)),
    ]
    if strategy == "smote":
        steps.append(("sampler", SMOTE(random_state=s.seed)))
    steps.append(("clf", classifier))
    return Pipeline(steps)


def transform_features(pipeline: Pipeline, X):
    """Apply only the fitted feature + preprocessing steps (used by explainers)."""
    return pipeline.named_steps["prep"].transform(pipeline.named_steps["features"].transform(X))


def transformed_feature_names(pipeline: Pipeline) -> list[str]:
    """Names of the model-ready columns, e.g. ``cat__Month_Nov``."""
    return list(pipeline.named_steps["prep"].get_feature_names_out())


def parent_feature(transformed_name: str, categorical_columns: list[str]) -> str:
    """Map a transformed column name back to its original behaviour feature.

    ``cat__Month_Nov`` -> ``Month``; ``skew__PageValues`` -> ``PageValues``.
    """
    _, _, rest = transformed_name.partition("__")
    for col in sorted(categorical_columns, key=len, reverse=True):
        if rest == col or rest.startswith(col + "_"):
            return col
    return rest

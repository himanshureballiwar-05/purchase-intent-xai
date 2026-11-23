"""Behavioural feature engineering as a scikit-learn transformer.

Being a transformer *inside* the Pipeline means the API applies exactly the same
feature logic at serving time (no train/serve skew).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config.settings import Settings, get_settings

ENGINEERED_NUMERIC = [
    "TotalPages",
    "TotalDuration",
    "AvgTimePerPage",
    "ProductPageShare",
    "TimePerProductPage",
]
ENGINEERED_BINARY = ["IsReturning"]


class BehaviourFeatures(BaseEstimator, TransformerMixin):
    """Clean dtypes, add engineered behaviour features, optionally drop PageValues.

    Engineered features (rationale in docs/model_documentation.md):

    * ``TotalPages``          - overall engagement depth.
    * ``TotalDuration``       - overall time on site.
    * ``AvgTimePerPage``      - attention per page (browsers vs. readers).
    * ``ProductPageShare``    - share of the visit spent on product pages (shopping intent).
    * ``TimePerProductPage``  - how carefully products are examined.
    * ``IsReturning``         - simple returning-visitor flag.
    """

    def __init__(
        self,
        categorical_columns: list[str],
        boolean_columns: list[str],
        count_columns: list[str],
        duration_columns: list[str],
        use_page_values: bool = True,
    ):
        self.categorical_columns = categorical_columns
        self.boolean_columns = boolean_columns
        self.count_columns = count_columns
        self.duration_columns = duration_columns
        self.use_page_values = use_page_values

    @classmethod
    def from_settings(cls, settings: Settings | None = None, use_page_values: bool | None = None):
        s = settings or get_settings()
        return cls(
            categorical_columns=s.categorical_columns,
            boolean_columns=s.boolean_columns,
            count_columns=s.count_columns,
            duration_columns=s.duration_columns,
            use_page_values=s.use_page_values if use_page_values is None else use_page_values,
        )

    def fit(self, X: pd.DataFrame, y=None):  # noqa: D401 - sklearn API
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        # int-coded IDs are categories -> strings so nothing treats them as quantities
        for col in self.categorical_columns:
            df[col] = df[col].astype(str)
        for col in self.boolean_columns:
            df[col] = df[col].astype(int)

        total_pages = df[self.count_columns].sum(axis=1)
        total_duration = df[self.duration_columns].sum(axis=1)
        df["TotalPages"] = total_pages
        df["TotalDuration"] = total_duration
        df["AvgTimePerPage"] = total_duration / np.maximum(total_pages, 1)
        df["ProductPageShare"] = df["ProductRelated"] / np.maximum(total_pages, 1)
        df["TimePerProductPage"] = df["ProductRelated_Duration"] / np.maximum(
            df["ProductRelated"], 1
        )
        df["IsReturning"] = (df["VisitorType"] == "Returning_Visitor").astype(int)

        if not self.use_page_values and "PageValues" in df.columns:
            df = df.drop(columns=["PageValues"])
        return df

import numpy as np
import pytest

from src.config.settings import get_settings
from src.features.engineering import BehaviourFeatures
from src.models.trainers import get_trainer
from src.preprocessing.pipeline import (
    build_pipeline,
    parent_feature,
    transformed_feature_names,
)

S = get_settings()


def test_engineered_features_added(raw_df):
    out = BehaviourFeatures.from_settings().transform(raw_df[S.input_columns].head(50))
    for col in S.engineered_features:
        assert col in out.columns
    assert (out["TotalPages"] >= 0).all()
    assert out["ProductPageShare"].between(0, 1).all()


def test_zero_pages_no_division_error(raw_df):
    row = raw_df[S.input_columns].head(1).copy()
    for col in S.count_columns + S.duration_columns:
        row[col] = 0
    out = BehaviourFeatures.from_settings().transform(row)
    assert np.isfinite(out[S.engineered_features].to_numpy(dtype=float)).all()


def test_id_columns_become_categories(raw_df):
    out = BehaviourFeatures.from_settings().transform(raw_df[S.input_columns].head(5))
    for col in S.id_categorical_columns:
        assert out[col].map(type).eq(str).all()


def test_page_values_can_be_dropped(raw_df):
    out = BehaviourFeatures.from_settings(use_page_values=False).transform(
        raw_df[S.input_columns].head(5)
    )
    assert "PageValues" not in out.columns


def test_ids_are_one_hot_encoded_not_scaled(raw_df):
    X, y = raw_df[S.input_columns].head(800), raw_df["Converted"].head(800)
    pipe = build_pipeline(get_trainer("logistic_regression").build(), "none").fit(X, y)
    names = transformed_feature_names(pipe)
    for col in S.id_categorical_columns:
        assert any(n.startswith(f"cat__{col}_") for n in names)
        assert f"skew__{col}" not in names and f"bounded__{col}" not in names


def test_unknown_category_does_not_crash(raw_df):
    X, y = raw_df[S.input_columns].head(800), raw_df["Converted"].head(800)
    pipe = build_pipeline(get_trainer("logistic_regression").build(), "none").fit(X, y)
    new = X.head(1).copy()
    new["Browser"] = 999
    assert 0 <= pipe.predict_proba(new)[0, 1] <= 1


def test_smote_lives_inside_pipeline():
    pipe = build_pipeline(get_trainer("xgboost").build("smote"), "smote")
    assert [n for n, _ in pipe.steps] == ["features", "prep", "sampler", "clf"]


def test_invalid_strategy_raises():
    with pytest.raises(ValueError):
        build_pipeline(get_trainer("xgboost").build(), "magic")


def test_parent_feature_mapping():
    cats = S.categorical_columns
    assert parent_feature("cat__Month_Nov", cats) == "Month"
    assert parent_feature("cat__TrafficType_13", cats) == "TrafficType"
    assert parent_feature("skew__PageValues", cats) == "PageValues"

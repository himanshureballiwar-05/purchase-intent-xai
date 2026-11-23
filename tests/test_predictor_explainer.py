import pandas as pd
import pytest

from src.config.settings import get_settings


def test_predict_returns_valid_result(predictor, sample_session):
    res = predictor.predict([sample_session])[0]
    assert 0 <= res["purchase_probability"] <= 1
    assert res["prediction"] in (0, 1)
    assert res["confidence"] in ("high", "medium", "low")
    assert len(res["top_contributors"]) == 5


def test_missing_fields_raise(predictor, sample_session):
    bad = dict(sample_session)
    bad.pop("PageValues")
    with pytest.raises(ValueError):
        predictor.predict([bad])


def test_batch_without_explanations(predictor, sample_session):
    out = predictor.predict([sample_session] * 3, explain=False)
    assert len(out) == 3 and "top_contributors" not in out[0]


def test_confidence_levels(predictor):
    thr = predictor.threshold
    assert predictor.confidence(thr) == "low"
    assert predictor.confidence(1.0) == "high"
    assert predictor.confidence(0.0) == "high"


def test_very_low_probability_is_high_confidence_no(predictor):
    # regression test: 0.1% with threshold ~0.14 used to be reported as "low" confidence
    assert predictor.confidence(0.001) == "high"
    assert predictor.confidence(predictor.threshold * 0.5) in ("medium", "high")


def test_shap_global_importance_sorted(predictor, raw_df):
    X = raw_df[get_settings().input_columns].head(200)
    imp = predictor.explainer.global_importance(X)
    assert imp["mean_abs_shap"].is_monotonic_decreasing
    assert set(imp["feature"]) <= set(get_settings().input_columns) | set(
        get_settings().engineered_features
    )


def test_local_shap_top_k(predictor, raw_df):
    row = raw_df[get_settings().input_columns].head(1)
    out = predictor.explainer.local_shap(row, top_k=3)
    assert len(out) == 3 and {"feature", "shap_value", "direction"} <= set(out[0])


def test_lime_returns_rules(predictor, raw_df):
    row = raw_df[get_settings().input_columns].head(1)
    out = predictor.explainer.lime_explain(row, num_features=4)
    assert len(out) == 4 and isinstance(out[0]["rule"], str)


def test_metadata_has_threshold_and_metrics(predictor):
    meta = predictor.metadata
    assert 0 < meta["threshold"] < 1
    assert "test_metrics_at_threshold" in meta and isinstance(
        pd.Series(meta["baseline"]), pd.Series
    )


def test_every_contributor_has_a_value(predictor, sample_session):
    contributors = predictor.predict([sample_session], top_k=12)[0]["top_contributors"]
    assert all(c["value"] is not None for c in contributors)

import numpy as np
import pytest

from src.models.evaluation import (
    BusinessCase,
    choose_threshold,
    classification_metrics,
    majority_baseline,
)
from src.models.trainers import TRAINERS, get_trainer


def test_six_models_registered():
    assert set(TRAINERS) == {
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "xgboost",
        "lightgbm",
        "catboost",
    }
    for name in TRAINERS:
        est = get_trainer(name).build("class_weight", 5.0)
        assert hasattr(est, "fit")


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        get_trainer("svm")


def test_perfect_predictions():
    y = np.array([0, 0, 1, 1])
    m = classification_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["recall"] == 1.0 and m["precision"] == 1.0 and m["roc_auc"] == 1.0


def test_accuracy_trap_baseline():
    y = np.array([0] * 84 + [1] * 16)
    base = majority_baseline(y)
    assert base["accuracy"] == pytest.approx(0.84)
    assert base["recall"] == 0.0


def test_break_even_probability():
    case = BusinessCase(60, 3, 0.25)
    assert case.break_even_probability == pytest.approx(0.2)


def test_threshold_follows_business_value():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.2, 3000)
    p = np.clip(y * 0.5 + rng.normal(0.25, 0.15, 3000), 0, 1)
    cheap, _ = choose_threshold(y, p, BusinessCase(60, 1, 0.5))
    costly, _ = choose_threshold(y, p, BusinessCase(60, 12, 0.5))
    assert cheap < costly  # costlier interventions need higher confidence

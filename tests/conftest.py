"""Shared fixtures: a small, fast model trained once per test session."""

from __future__ import annotations

import pytest

from src.data.loader import SessionDataLoader
from src.models.predictor import Predictor
from src.models.train import TrainingPipeline


@pytest.fixture(scope="session")
def raw_df():
    return SessionDataLoader().load()


@pytest.fixture(scope="session")
def artifact_dir(tmp_path_factory):
    """Train a tiny model into a temp dir (keeps CI independent of committed artifacts)."""
    root = tmp_path_factory.mktemp("run")
    TrainingPipeline(
        artifact_dir=root / "models",
        reports_dir=root / "reports",
        docs_dir=root / "docs",
        models=["logistic_regression", "xgboost"],
        strategies=["class_weight"],
        cv_folds=3,
        tune_top_k=0,
        max_rows=2500,
        track=False,
    ).run()
    return root / "models"


@pytest.fixture(scope="session")
def predictor(artifact_dir):
    return Predictor(artifact_dir)


@pytest.fixture()
def sample_session():
    return {
        "Administrative": 2,
        "Administrative_Duration": 60.0,
        "Informational": 0,
        "Informational_Duration": 0.0,
        "ProductRelated": 25,
        "ProductRelated_Duration": 900.0,
        "BounceRates": 0.01,
        "ExitRates": 0.03,
        "PageValues": 18.5,
        "SpecialDay": 0.0,
        "Month": "Nov",
        "OperatingSystems": 2,
        "Browser": 2,
        "Region": 1,
        "TrafficType": 2,
        "VisitorType": "Returning_Visitor",
        "Weekend": False,
    }

"""Six algorithms behind one common interface (add a model = add a class)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


class BaseTrainer(ABC):
    """Common interface: build an estimator for a given imbalance strategy."""

    name: str = "base"
    #: hyper-parameter search space, keys are prefixed with ``clf__`` by the tuner
    param_distributions: dict[str, list[Any]] = {}

    def __init__(self, seed: int = 42, n_jobs: int = 1):
        self.seed = seed
        self.n_jobs = n_jobs

    @abstractmethod
    def build(self, strategy: str = "none", pos_weight: float = 1.0) -> BaseEstimator:
        """Return an un-fitted estimator. ``pos_weight`` = n_neg / n_pos."""

    @staticmethod
    def _weighted(strategy: str) -> bool:
        return strategy == "class_weight"


class LogisticRegressionTrainer(BaseTrainer):
    name = "logistic_regression"
    param_distributions = {"C": [0.01, 0.1, 0.5, 1, 5, 10]}

    def build(self, strategy="none", pos_weight=1.0):
        return LogisticRegression(
            max_iter=2000,
            random_state=self.seed,
            class_weight="balanced" if self._weighted(strategy) else None,
        )


class DecisionTreeTrainer(BaseTrainer):
    name = "decision_tree"
    param_distributions = {"max_depth": [3, 4, 5, 6, 8], "min_samples_leaf": [10, 20, 40, 80]}

    def build(self, strategy="none", pos_weight=1.0):
        return DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=20,
            random_state=self.seed,
            class_weight="balanced" if self._weighted(strategy) else None,
        )


class RandomForestTrainer(BaseTrainer):
    name = "random_forest"
    param_distributions = {
        "n_estimators": [150, 250],
        "max_depth": [6, 10, 14, None],
        "min_samples_leaf": [2, 5, 10],
    }

    def build(self, strategy="none", pos_weight=1.0):
        return RandomForestClassifier(
            n_estimators=150,
            min_samples_leaf=5,
            n_jobs=self.n_jobs,
            random_state=self.seed,
            class_weight="balanced_subsample" if self._weighted(strategy) else None,
        )


class XGBoostTrainer(BaseTrainer):
    name = "xgboost"
    param_distributions = {
        "n_estimators": [150, 250, 400],
        "max_depth": [3, 4, 6],
        "learning_rate": [0.03, 0.06, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.7, 1.0],
    }

    def build(self, strategy="none", pos_weight=1.0):
        return XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.06,
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=self.n_jobs,
            random_state=self.seed,
            scale_pos_weight=pos_weight if self._weighted(strategy) else 1.0,
        )


class LightGBMTrainer(BaseTrainer):
    name = "lightgbm"
    param_distributions = {
        "n_estimators": [150, 250, 400],
        "num_leaves": [15, 31, 63],
        "learning_rate": [0.03, 0.06, 0.1],
        "min_child_samples": [10, 20, 40],
        "colsample_bytree": [0.7, 1.0],
    }

    def build(self, strategy="none", pos_weight=1.0):
        return LGBMClassifier(
            n_estimators=250,
            num_leaves=31,
            learning_rate=0.06,
            n_jobs=self.n_jobs,
            random_state=self.seed,
            verbose=-1,
            scale_pos_weight=pos_weight if self._weighted(strategy) else 1.0,
        )


class CatBoostTrainer(BaseTrainer):
    name = "catboost"
    param_distributions = {
        "iterations": [200, 400],
        "depth": [4, 6, 8],
        "learning_rate": [0.05, 0.1],
        "l2_leaf_reg": [1, 3, 7],
    }

    def build(self, strategy="none", pos_weight=1.0):
        return CatBoostClassifier(
            iterations=250,
            depth=6,
            learning_rate=0.08,
            random_seed=self.seed,
            thread_count=self.n_jobs,
            verbose=0,
            allow_writing_files=False,
            auto_class_weights="Balanced" if self._weighted(strategy) else None,
        )


TRAINERS: dict[str, type[BaseTrainer]] = {
    cls.name: cls
    for cls in (
        LogisticRegressionTrainer,
        DecisionTreeTrainer,
        RandomForestTrainer,
        XGBoostTrainer,
        LightGBMTrainer,
        CatBoostTrainer,
    )
}


def get_trainer(name: str, seed: int = 42, n_jobs: int = 1) -> BaseTrainer:
    """Factory used by the training pipeline."""
    if name not in TRAINERS:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(TRAINERS)}")
    return TRAINERS[name](seed=seed, n_jobs=n_jobs)

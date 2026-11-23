"""Imbalance-aware metrics and the value-based decision threshold."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

CV_SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "recall": "recall",
    "precision": "precision",
    "f1": "f1",
    "accuracy": "accuracy",
}


def classification_metrics(y_true, y_proba, threshold: float = 0.5) -> dict[str, float]:
    """Full metric suite at a given threshold. PR-AUC and recall lead on purpose."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def majority_baseline(y_true) -> dict[str, float]:
    """The 'predict nobody buys' baseline that makes accuracy a trap."""
    y_true = np.asarray(y_true)
    return {"accuracy": float(1 - y_true.mean()), "recall": 0.0, "pr_auc": float(y_true.mean())}


@dataclass(frozen=True)
class BusinessCase:
    """Economics of one intervention (discount nudge, free-shipping banner...)."""

    value_per_conversion: float
    intervention_cost: float
    success_rate: float

    @property
    def break_even_probability(self) -> float:
        """Flagging a session pays off when P(buy) * value * success > cost."""
        return self.intervention_cost / (self.value_per_conversion * self.success_rate)

    def net_value(self, tp: int, fp: int) -> float:
        """Expected profit of acting on every flagged session."""
        flagged = tp + fp
        saved = tp * self.value_per_conversion * self.success_rate
        return float(saved - flagged * self.intervention_cost)


def threshold_curve(y_true, y_proba, case: BusinessCase, n: int = 99) -> list[dict[str, float]]:
    """Net value / precision / recall for thresholds 0.01 ... 0.99."""
    y_true = np.asarray(y_true)
    rows = []
    for thr in np.linspace(0.01, 0.99, n):
        pred = (np.asarray(y_proba) >= thr).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        rows.append(
            {
                "threshold": float(thr),
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "recall": tp / (tp + fn) if tp + fn else 0.0,
                "flagged": tp + fp,
                "net_value": case.net_value(tp, fp),
            }
        )
    return rows


def choose_threshold(y_true, y_proba, case: BusinessCase) -> tuple[float, list[dict[str, float]]]:
    """Pick the threshold with the highest expected net value (not 0.5)."""
    curve = threshold_curve(y_true, y_proba, case)
    best = max(curve, key=lambda r: r["net_value"])
    return float(best["threshold"]), curve


def pr_roc_curves(y_true, y_proba) -> dict[str, list[float]]:
    """Curve points for the dashboard (down-sampled to keep files small)."""
    p, r, _ = precision_recall_curve(y_true, y_proba)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    step_pr = max(1, len(p) // 200)
    step_roc = max(1, len(fpr) // 200)
    return {
        "precision": p[::step_pr].tolist(),
        "recall": r[::step_pr].tolist(),
        "fpr": fpr[::step_roc].tolist(),
        "tpr": tpr[::step_roc].tolist(),
    }

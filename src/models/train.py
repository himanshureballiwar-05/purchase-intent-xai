"""End-to-end training pipeline.

Run:  ``python -m src.models.train``        (full)
      ``python -m src.models.train --quick`` (fast smoke run)

Stages: load -> split -> benchmark (6 models x 3 imbalance strategies, CV)
-> tune top-k -> pick best -> value-based threshold (out-of-fold)
-> fit -> test evaluation -> PageValues ablation -> SHAP/LIME -> save artifacts.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import platform
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
import sklearn  # noqa: E402
from sklearn.model_selection import (  # noqa: E402
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)

from src.config.settings import Settings, load_settings  # noqa: E402
from src.data.loader import SessionDataLoader  # noqa: E402
from src.explain.explainer import SessionExplainer  # noqa: E402
from src.explain.model_card import write_model_card  # noqa: E402
from src.models.evaluation import (  # noqa: E402
    CV_SCORING,
    BusinessCase,
    choose_threshold,
    classification_metrics,
    majority_baseline,
    pr_roc_curves,
)
from src.models.predictor import BACKGROUND_FILE, METADATA_FILE, PIPELINE_FILE  # noqa: E402
from src.models.tracking import Tracker  # noqa: E402
from src.models.trainers import get_trainer  # noqa: E402
from src.preprocessing.pipeline import build_pipeline, transform_features  # noqa: E402
from src.utils.io import save_joblib, save_json  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402

logger = get_logger(__name__)


class TrainingPipeline:
    """Orchestrates benchmarking, tuning, evaluation, explanation and saving."""

    def __init__(
        self,
        settings: Settings | None = None,
        artifact_dir: str | Path | None = None,
        reports_dir: str | Path | None = None,
        docs_dir: str | Path | None = None,
        models: list[str] | None = None,
        strategies: list[str] | None = None,
        cv_folds: int | None = None,
        tune_top_k: int | None = None,
        tune_iterations: int | None = None,
        max_rows: int | None = None,
        track: bool = True,
    ):
        self.s = settings or load_settings()
        cfg = self.s.section("training")
        self.models = models or cfg["models"]
        self.strategies = strategies or cfg["imbalance_strategies"]
        self.cv_folds = cv_folds or int(cfg["cv_folds"])
        self.tune_top_k = cfg["tune_top_k"] if tune_top_k is None else tune_top_k
        self.tune_iter = tune_iterations or int(cfg["tune_iterations"])
        self.n_jobs = int(cfg["n_jobs"])
        self.metric = cfg["selection_metric"]
        self.max_rows = max_rows
        self.artifact_dir = Path(artifact_dir) if artifact_dir else self.s.path("artifacts_dir")
        self.reports_dir = Path(reports_dir) if reports_dir else self.s.path("reports_dir")
        self.fig_dir = self.reports_dir / "figures"
        self.docs_dir = Path(docs_dir) if docs_dir else self.s.path("docs_dir")
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        biz = self.s.section("business")
        self.case = BusinessCase(
            biz["value_per_conversion"], biz["intervention_cost"], biz["intervention_success_rate"]
        )
        self.tracker = Tracker(self.s, enabled=track)
        self.skf = StratifiedKFold(self.cv_folds, shuffle=True, random_state=self.s.seed)

    # ------------------------------------------------------------------ helpers
    def _estimator(self, model: str, strategy: str, pos_weight: float, upv: bool | None = None):
        trainer = get_trainer(model, self.s.seed, self.n_jobs)
        return build_pipeline(trainer.build(strategy, pos_weight), strategy, self.s, upv)

    def _cv(self, pipe, X, y) -> dict[str, float]:
        res = cross_validate(pipe, X, y, cv=self.skf, scoring=CV_SCORING, n_jobs=self.n_jobs)
        out = {}
        for key in CV_SCORING:
            vals = res[f"test_{key}"]
            out[key] = float(np.mean(vals))
            out[f"{key}_std"] = float(np.std(vals))
        out["fit_time_s"] = float(np.mean(res["fit_time"]))
        return out

    # ------------------------------------------------------------------ stages
    def benchmark(self, X, y, pos_weight: float) -> pd.DataFrame:
        rows = []
        for model in self.models:
            for strat in self.strategies:
                t0 = time.time()
                pipe = self._estimator(model, strat, pos_weight)
                scores = self._cv(pipe, X, y)
                rows.append({"model": model, "strategy": strat, "tuned": False, **scores})
                logger.info(
                    "CV %-20s %-12s PR-AUC=%.4f recall=%.3f (%.0fs)",
                    model,
                    strat,
                    scores["pr_auc"],
                    scores["recall"],
                    time.time() - t0,
                )
                with self.tracker.run(f"{model}-{strat}", {"stage": "benchmark"}):
                    self.tracker.log({"model": model, "strategy": strat}, scores)
        return pd.DataFrame(rows)

    def tune(self, bench: pd.DataFrame, X, y, pos_weight: float) -> tuple[pd.DataFrame, dict]:
        """Tune the best-strategy version of the top-k models by PR-AUC."""
        best_each = bench.sort_values("pr_auc", ascending=False).drop_duplicates("model")
        tuned_rows, best_params = [], {}
        for _, row in best_each.head(self.tune_top_k).iterrows():
            model, strat = row["model"], row["strategy"]
            space = {f"clf__{k}": v for k, v in get_trainer(model).param_distributions.items()}
            n_combo = int(np.prod([len(v) for v in space.values()]))
            search = RandomizedSearchCV(
                self._estimator(model, strat, pos_weight),
                space,
                n_iter=min(self.tune_iter, n_combo),
                scoring="average_precision",
                cv=StratifiedKFold(3, shuffle=True, random_state=self.s.seed),
                random_state=self.s.seed,
                refit=False,
                n_jobs=self.n_jobs,
            )
            search.fit(X, y)
            pipe = self._estimator(model, strat, pos_weight).set_params(**search.best_params_)
            scores = self._cv(pipe, X, y)
            logger.info(
                "TUNED %-18s %-12s PR-AUC %.4f -> %.4f",
                model,
                strat,
                row["pr_auc"],
                scores["pr_auc"],
            )
            if scores["pr_auc"] > row["pr_auc"]:
                tuned_rows.append({"model": model, "strategy": strat, "tuned": True, **scores})
                best_params[(model, strat)] = {
                    k.replace("clf__", ""): v for k, v in search.best_params_.items()
                }
        return pd.DataFrame(tuned_rows), best_params

    def pick_best(self, bench: pd.DataFrame, tuned: pd.DataFrame) -> pd.Series:
        pool = pd.concat([bench, tuned], ignore_index=True) if len(tuned) else bench
        return pool.sort_values("pr_auc", ascending=False).iloc[0]

    # ------------------------------------------------------------------ main
    def run(self) -> dict[str, Any]:
        t_start = time.time()
        loader = SessionDataLoader(self.s)
        df = loader.load()
        if self.max_rows:
            df = df.sample(n=min(self.max_rows, len(df)), random_state=self.s.seed)
        X_tr, X_te, y_tr, y_te = loader.split(df)
        pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))

        bench = self.benchmark(X_tr, y_tr, pos_weight)
        tuned, tuned_params = (
            self.tune(bench, X_tr, y_tr, pos_weight) if self.tune_top_k else (pd.DataFrame(), {})
        )
        best = self.pick_best(bench, tuned)
        model, strat, is_tuned = best["model"], best["strategy"], bool(best["tuned"])
        params = tuned_params.get((model, strat), {}) if is_tuned else {}
        logger.info(
            "BEST: %s | %s | tuned=%s | CV PR-AUC=%.4f", model, strat, is_tuned, best["pr_auc"]
        )

        def make(upv=None):
            pipe = self._estimator(model, strat, pos_weight, upv)
            return (
                pipe.set_params(**{f"clf__{k}": v for k, v in params.items()}) if params else pipe
            )

        # threshold from OUT-OF-FOLD predictions on train (test set stays untouched)
        oof = cross_val_predict(make(), X_tr, y_tr, cv=self.skf, method="predict_proba")[:, 1]
        threshold, curve = choose_threshold(y_tr, oof, self.case)

        final = make().fit(X_tr, y_tr)
        proba_te = final.predict_proba(X_te)[:, 1]
        test_default = classification_metrics(y_te, proba_te, 0.5)
        test_tuned = classification_metrics(y_te, proba_te, threshold)
        value = {
            "chosen_threshold": self.case.net_value(test_tuned["tp"], test_tuned["fp"]),
            "default_0.5": self.case.net_value(test_default["tp"], test_default["fp"]),
            "flag_everyone": self.case.net_value(int(y_te.sum()), int((y_te == 0).sum())),
            "flag_nobody": 0.0,
        }

        ablation = self.page_values_ablation(make, X_tr, y_tr, X_te, y_te)
        explain_info = self.explain_and_plot(
            final, X_tr, X_te, y_te, proba_te, threshold, bench, tuned, curve, test_tuned
        )

        metadata = {
            "model_name": model,
            "strategy": strat,
            "tuned": is_tuned,
            "params": params,
            "threshold": threshold,
            "threshold_method": "max expected net value on out-of-fold train predictions",
            "break_even_probability": self.case.break_even_probability,
            "business_case": vars(self.case),
            "use_page_values": self.s.use_page_values,
            "input_columns": self.s.input_columns,
            "cv_metrics": {
                k: float(v)
                for k, v in best.items()
                if isinstance(v, (int, float, np.floating)) and k != "tuned"
            },
            "test_metrics_at_0.5": test_default,
            "test_metrics_at_threshold": test_tuned,
            "baseline": majority_baseline(y_te),
            "net_value_on_test": value,
            "n_train": int(len(X_tr)),
            "n_test": int(len(X_te)),
            "positive_rate": float(y_tr.mean()),
            "trained_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "seed": self.s.seed,
            "training_seconds": round(time.time() - t_start, 1),
            "ablation_page_values": ablation,
            "top_features": explain_info["top_features"],
        }

        save_joblib(final, self.artifact_dir / PIPELINE_FILE)
        save_joblib(explain_info["background"], self.artifact_dir / BACKGROUND_FILE)
        save_json(metadata, self.artifact_dir / METADATA_FILE)
        full = pd.concat([bench, tuned], ignore_index=True)
        full.to_csv(self.reports_dir / "model_comparison.csv", index=False)
        save_json(metadata, self.reports_dir / "final_metrics.json")
        save_json(curve, self.reports_dir / "threshold_curve.json")
        save_json(pr_roc_curves(y_te, proba_te), self.reports_dir / "curves.json")
        write_model_card(metadata, full, self.docs_dir / "model_card.md")

        with self.tracker.run("final-model", {"stage": "final"}):
            self.tracker.log(
                {"model": model, "strategy": strat, "threshold": threshold, **params},
                {f"test_{k}": v for k, v in test_tuned.items() if isinstance(v, float)},
            )
            self.tracker.log_artifact(str(self.artifact_dir / METADATA_FILE))
            self.tracker.log_and_register_model(final, X_te.head(5))
        logger.info(
            "DONE in %.0fs | test PR-AUC=%.4f recall@thr=%.3f",
            time.time() - t_start,
            test_tuned["pr_auc"],
            test_tuned["recall"],
        )
        return metadata

    # ------------------------------------------------------------------ ablation
    def page_values_ablation(self, make, X_tr, y_tr, X_te, y_te) -> dict[str, Any] | None:
        """Does the model still work without the checkout-proximal PageValues?"""
        if not self.s.use_page_values:
            return None
        pipe = make(False)
        cv = self._cv(pipe, X_tr, y_tr)
        p = pipe.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
        res = {"cv_pr_auc": cv["pr_auc"], "test": classification_metrics(y_te, p, 0.5)}
        save_json(res, self.reports_dir / "ablation_page_values.json")
        logger.info("Ablation (no PageValues): CV PR-AUC=%.4f", cv["pr_auc"])
        return res

    # ------------------------------------------------------------------ explain + figures
    def explain_and_plot(
        self, final, X_tr, X_te, y_te, proba_te, threshold, bench, tuned, curve, test_metrics
    ):
        size = int(self.s.section("explain")["background_size"])
        bg_rows = X_tr.sample(n=min(size, len(X_tr)), random_state=self.s.seed)
        background = transform_features(final, bg_rows)
        explainer = SessionExplainer(final, background, self.s)

        sample = X_te.sample(n=min(1000, len(X_te)), random_state=self.s.seed)
        imp = explainer.global_importance(sample)
        imp.to_csv(self.reports_dir / "global_shap.csv", index=False)

        # local examples: confident buyer, confident non-buyer, borderline
        order = pd.Series(proba_te, index=X_te.index)
        picks = {
            "likely_buyer": order.idxmax(),
            "likely_non_buyer": order.idxmin(),
            "borderline": (order - threshold).abs().idxmin(),
        }
        examples = {}
        for tag, idx in picks.items():
            row = X_te.loc[[idx]]
            examples[tag] = {
                "probability": float(order[idx]),
                "actual": int(y_te.loc[idx]),
                "input": row.iloc[0].to_dict(),
                "shap_top": explainer.local_shap(row),
                "lime_top": explainer.lime_explain(row),
            }
        save_json(examples, self.reports_dir / "local_examples.json")

        self._figures(
            imp,
            bench,
            tuned,
            curve,
            y_te,
            proba_te,
            threshold,
            test_metrics,
            final,
            sample,
            explainer,
        )
        return {
            "background": background,
            "top_features": imp.head(8).to_dict(orient="records"),
        }

    def _figures(
        self, imp, bench, tuned, curve, y_te, proba_te, threshold, tm, final, sample, explainer
    ):
        fig_dir = self.fig_dir
        # 1. global importance
        top = imp.head(12).iloc[::-1]
        plt.figure(figsize=(7, 5))
        plt.barh(top["feature"], top["mean_abs_shap"], color="#2b6cb0")
        plt.xlabel("Mean |SHAP value|")
        plt.title("Global feature importance (SHAP)")
        plt.tight_layout()
        plt.savefig(fig_dir / "shap_global_bar.png", dpi=130)
        plt.close()
        # 2. beeswarm on transformed features
        try:
            sv = explainer.shap_matrix(sample)
            Xt = transform_features(final, sample)
            shap.summary_plot(
                sv,
                Xt,
                feature_names=[n.split("__", 1)[-1] for n in explainer.feature_names],
                max_display=15,
                show=False,
            )
            plt.tight_layout()
            plt.savefig(fig_dir / "shap_beeswarm.png", dpi=130, bbox_inches="tight")
            plt.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Beeswarm plot skipped: %s", exc)
        # 3. model comparison
        pool = pd.concat([bench, tuned], ignore_index=True) if len(tuned) else bench
        best_per = (
            pool.sort_values("pr_auc", ascending=False)
            .drop_duplicates("model")
            .sort_values("pr_auc")
        )
        plt.figure(figsize=(7, 4))
        plt.barh(best_per["model"], best_per["pr_auc"], color="#38a169")
        plt.xlabel("Cross-validated PR-AUC (best strategy per model)")
        plt.title("Model comparison")
        plt.tight_layout()
        plt.savefig(fig_dir / "model_comparison.png", dpi=130)
        plt.close()
        # 4. PR curve
        from sklearn.metrics import precision_recall_curve

        p, r, _ = precision_recall_curve(y_te, proba_te)
        plt.figure(figsize=(5.5, 4.5))
        plt.plot(r, p, color="#2b6cb0", label=f"PR-AUC={tm['pr_auc']:.3f}")
        plt.axhline(float(np.mean(y_te)), ls="--", color="grey", label="baseline (class rate)")
        plt.scatter(
            [tm["recall"]],
            [tm["precision"]],
            color="red",
            zorder=5,
            label=f"threshold={threshold:.2f}",
        )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.title("Precision-Recall (test)")
        plt.tight_layout()
        plt.savefig(fig_dir / "pr_curve.png", dpi=130)
        plt.close()
        # 5. threshold vs net value
        cdf = pd.DataFrame(curve)
        plt.figure(figsize=(6, 4))
        plt.plot(cdf["threshold"], cdf["net_value"], color="#dd6b20")
        plt.axvline(threshold, ls="--", color="red", label=f"chosen={threshold:.2f}")
        plt.axvline(0.5, ls=":", color="grey", label="default 0.5")
        plt.xlabel("Decision threshold")
        plt.ylabel("Expected net value (train OOF)")
        plt.legend()
        plt.title("Threshold chosen by business value")
        plt.tight_layout()
        plt.savefig(fig_dir / "threshold_value.png", dpi=130)
        plt.close()
        # 6. confusion matrix
        cm = np.array([[tm["tn"], tm["fp"]], [tm["fn"], tm["tp"]]])
        plt.figure(figsize=(4.5, 4))
        plt.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            plt.text(j, i, int(v), ha="center", va="center", color="black", fontsize=13)
        plt.xticks([0, 1], ["No buy", "Buy"])
        plt.yticks([0, 1], ["No buy", "Buy"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion matrix @ {threshold:.2f}")
        plt.tight_layout()
        plt.savefig(fig_dir / "confusion_matrix.png", dpi=130)
        plt.close()


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train the purchase-intent model")
    ap.add_argument("--config", default=None, help="Path to a config YAML")
    ap.add_argument("--quick", action="store_true", help="Fast smoke run (fewer models/folds)")
    ap.add_argument("--no-mlflow", action="store_true", help="Disable MLflow tracking")
    ap.add_argument(
        "--no-page-values",
        action="store_true",
        help="Train the early-scoring model without PageValues",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    settings = load_settings(args.config)
    if args.no_page_values:
        raw = copy.deepcopy(settings.raw)
        raw["features"]["use_page_values"] = False
        settings = Settings(raw=raw)
    kwargs: dict[str, Any] = {}
    if args.quick:
        kwargs = dict(
            models=["logistic_regression", "random_forest", "xgboost"],
            strategies=["class_weight", "none"],
            cv_folds=3,
            tune_top_k=1,
            tune_iterations=3,
        )
    TrainingPipeline(settings, track=not args.no_mlflow, **kwargs).run()


if __name__ == "__main__":
    main()

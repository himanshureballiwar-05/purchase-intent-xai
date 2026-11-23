# 🛒 Purchase Intent Prediction using Explainable AI

An end-to-end machine-learning project that predicts whether an e-commerce visitor will make a purchase
**and explains why**, served through a REST API and an interactive dashboard.

> **Author:** Himanshu Reballiwar · Data Science Internship (Internmo) · Project ID `PROJ-ECOM-260629`
>
> **Live dashboard:** *paste your Streamlit link here* · **GitHub:** *paste your repository link here*

## Headline results (held-out test set, 2,400 sessions, 15.7% buyers)

| Metric | Threshold 0.5 | Chosen threshold (0.14) |
|---|---|---|
| **PR-AUC** | 0.863 | 0.863 |
| ROC-AUC | 0.972 | 0.972 |
| **Recall** (buyers found) | 77.2% | **95.2%** |
| Precision | 80.2% | 58.8% |
| F1 | 0.786 | 0.727 |

* **Best model:** CatBoost (tuned), selected from 6 algorithms × 3 imbalance strategies by cross-validated PR-AUC.
* **Why not accuracy?** Predicting "nobody buys" is already 84.3% accurate and finds zero buyers.
* **Threshold** is chosen by expected business value, not 0.5: it earns more than 0.5 and far more than nudging everyone (which loses money). Business numbers in `config.yaml` are *assumptions*.
* **Honest caveat:** `PageValues` drives the model. Without it PR-AUC falls from ~0.85 to ~0.53 because it is a checkout-proximal signal (100% of buyers have `PageValues > 0`). The model is best used **late in a session**; see [`docs/model_card.md`](docs/model_card.md).

## Quick start

Requires **Python 3.11+** (built and tested on 3.12).

```bash
# 1. environment
python -m venv .venv
.venv\Scripts\activate            # Windows   |   source .venv/bin/activate  (macOS/Linux)
pip install -r requirements.txt

# 2. (optional) re-train - takes ~3-4 minutes; trained artifacts are already included in models/
python -m src.models.train

# 3. run the API  ->  http://localhost:8000/docs
uvicorn src.api.main:app --reload

# 4. run the dashboard (new terminal)  ->  http://localhost:8501
streamlit run dashboard/app.py
```

**Windows shortcut:** after `pip install -r requirements.txt`, just double-click `run_dashboard.bat` (dashboard) or `run_api.bat` (API). Keep the black window open - closing it stops the server.

**Public link (no localhost):** see [`docs/deploy_streamlit_cloud.md`](docs/deploy_streamlit_cloud.md) to deploy the dashboard free on Streamlit Community Cloud.

**Everything with Docker:** `docker compose up --build` (API on :8000, dashboard on :8501).

**Experiment tracking:** `mlflow ui` (after training) -> http://localhost:5000
**Tests:** `pip install -r requirements-dev.txt` then `pytest`

> If loading the saved model gives a version/pickle error on your machine, simply re-run `python -m src.models.train`.

## What is inside

```
├── dataset/                 ecommerce_sessions.csv + data_dictionary.csv
├── src/
│   ├── config/              config.yaml (all parameters) + typed settings loader
│   ├── data/                loader + validation + stratified split
│   ├── features/            BehaviourFeatures transformer (6 engineered features)
│   ├── preprocessing/       ColumnTransformer + SMOTE inside the Pipeline
│   ├── models/              6 trainers, evaluation, training pipeline, predictor, MLflow tracking
│   ├── explain/             SHAP + LIME explainer, model-card generator
│   └── api/                 FastAPI app + Pydantic schemas
├── dashboard/app.py         Streamlit dashboard (6 sections)
├── notebooks/               01 EDA · 02 features · 03 modelling · 04 explainability (executed)
├── models/                  trained pipeline, metadata, SHAP background
├── reports/                 comparison tables, curves, SHAP values, figures
├── tests/                   pytest suite (data, features, models, API, dashboard)
├── docker/, docker-compose.yml, .github/workflows/ci.yml, deploy/aws/
└── docs/                    architecture, API, user guide, model documentation, model card, demo script
```

## How it works (one paragraph)

Raw sessions are validated and split (stratified 80/20). A scikit-learn/imbalanced-learn **Pipeline**
(`feature engineering → encoding/scaling → optional SMOTE → classifier`) guarantees that nothing is learned
from validation data during 5-fold cross-validation. Six algorithms (Logistic Regression, Decision Tree, Random Forest,
XGBoost, LightGBM, CatBoost) are benchmarked under three imbalance strategies (SMOTE, class weights, none + threshold tuning);
the best two are tuned. The decision threshold is picked from out-of-fold predictions using the business value of an
intervention; the test set is used once. SHAP (global + local) and LIME explain predictions, and the same `Predictor`
class serves both the API and the dashboard.

## API

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @examples/session.json
```
Endpoints: `GET /health`, `GET /model/info`, `POST /predict`, `POST /predict/batch`. Full reference: [`docs/api.md`](docs/api.md).

## Configuration

Everything tunable lives in [`src/config/config.yaml`](src/config/config.yaml): column groups, CV folds, models,
imbalance strategies, business assumptions, and the `features.use_page_values` switch
(`python -m src.models.train --no-page-values` trains the early-scoring variant).

## Testing & quality

* `pytest` - 56 tests: config, data validation, features (incl. division-by-zero), leakage-safe pipeline structure,
  unseen categories, metrics & threshold logic, SHAP/LIME, every API endpoint (valid, invalid, missing model),
  and a smoke test of every dashboard page.
* `ruff` + `black` enforced; type hints and docstrings throughout; structured logging (no `print`).
* GitHub Actions (`.github/workflows/ci.yml`): lint → format check → tests → Docker builds.

## PRD deliverables map

| PRD deliverable | Where |
|---|---|
| Source code (modular, configurable) | `src/`, `src/config/config.yaml` |
| Notebooks (EDA, features, modelling, XAI) | `notebooks/` |
| 6-model benchmark + imbalance strategies + tuning | `src/models/train.py`, `reports/model_comparison.csv` |
| SHAP + LIME + model card | `src/explain/`, `docs/model_card.md` |
| FastAPI service | `src/api/` |
| Streamlit dashboard (6 sections) | `dashboard/app.py` |
| Docker + compose | `docker/`, `docker-compose.yml` |
| MLflow tracking + registry | `src/models/tracking.py` |
| CI pipeline | `.github/workflows/ci.yml` |
| AWS-ready config (not deployed) | `deploy/aws/` |
| Tests | `tests/` |
| Documentation | `docs/` |
| Demo video script | `docs/demo_script.md` |

## Limitations

Single dataset snapshot (seasonality/drift), `PageValues` dependence (late-session), business-value figures are
assumptions, and explanations describe the model - not causal effects. Details in the model card.

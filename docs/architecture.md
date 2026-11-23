# Architecture

## System overview
```mermaid
flowchart LR
    A[ecommerce_sessions.csv] --> B[SessionDataLoader<br/>validate + split]
    B --> C[Pipeline<br/>BehaviourFeatures -> ColumnTransformer -> SMOTE? -> Classifier]
    C --> D[Benchmark: 6 models x 3 imbalance strategies<br/>5-fold CV]
    D --> E[Tune top-2 -> pick best by PR-AUC]
    E --> F[Value-based threshold<br/>from out-of-fold predictions]
    F --> G[(models/<br/>pipeline + metadata + SHAP background)]
    E --> H[(MLflow<br/>runs + registry)]
    G --> I[Predictor<br/>predict + SHAP + LIME]
    I --> J[FastAPI :8000<br/>/predict /health]
    I --> K[Streamlit :8501<br/>6 sections]
```

## Training pipeline (what happens in `python -m src.models.train`)
```mermaid
flowchart TD
    L[load + validate] --> S[stratified 80/20 split]
    S --> BM[benchmark: CV on train only]
    BM --> TU[tune top-k with RandomizedSearchCV]
    TU --> BEST[best config by CV PR-AUC]
    BEST --> OOF[out-of-fold probabilities on train]
    OOF --> THR[threshold = max expected net value]
    BEST --> FIT[fit on train]
    FIT --> TEST[evaluate on untouched test set]
    FIT --> ABL[PageValues ablation]
    FIT --> XAI[SHAP global + local, LIME examples]
    TEST --> SAVE[save artifacts, reports, model card, MLflow]
```

## Serving
```mermaid
sequenceDiagram
    participant Client
    participant API as FastAPI
    participant P as Predictor (loaded once)
    Client->>API: POST /predict (17 raw fields)
    API->>API: Pydantic validation (422 on bad input)
    API->>P: predict([session])
    P->>P: features -> preprocessing -> model -> probability
    P->>P: SHAP top-k contributors
    P-->>API: probability, decision, confidence, contributors
    API-->>Client: JSON
```

## Folder map
| Path | Responsibility |
|---|---|
| `src/config/` | `config.yaml` (all parameters, column lists) + typed `Settings` loader |
| `src/data/loader.py` | Read, validate, split data (single entry point for all code) |
| `src/features/engineering.py` | `BehaviourFeatures` transformer (6 engineered features, PageValues switch) |
| `src/preprocessing/pipeline.py` | ColumnTransformer + SMOTE inside the imblearn Pipeline |
| `src/models/trainers.py` | Six algorithms behind one `BaseTrainer` interface |
| `src/models/evaluation.py` | Metrics, business case, threshold selection |
| `src/models/train.py` | Orchestration: benchmark, tune, threshold, ablation, explain, save |
| `src/models/predictor.py` | Serving logic shared by API and dashboard |
| `src/models/tracking.py` | MLflow wrapper (tracking never breaks training) |
| `src/explain/` | `SessionExplainer` (SHAP + LIME), model-card generator |
| `src/api/` | FastAPI app + Pydantic schemas |
| `dashboard/app.py` | Streamlit app (6 sections) |
| `tests/` | pytest suite (config, data, features, models, explainers, API, dashboard) |
| `docker/`, `docker-compose.yml` | Container images and one-command run |
| `.github/workflows/ci.yml` | Lint + format + tests + Docker build |

## Key design decisions
1. **Everything that learns from data lives inside the Pipeline.** Encoding, scaling and SMOTE are fitted on the
   training folds only during cross-validation, so there is no leakage from validation folds.
2. **Integer-coded IDs are categories.** `OperatingSystems`, `Browser`, `Region`, `TrafficType` are one-hot encoded, never scaled.
3. **Same code for training and serving.** Feature engineering is a pipeline step, so the API cannot drift from training.
4. **Test set is touched once.** Model choice and threshold use cross-validation / out-of-fold predictions on train only.
5. **PR-AUC leads the metrics** (15.7% positives); accuracy is shown only next to the "nobody buys" baseline.
6. **Threshold from business value**, not 0.5 (see `business:` in `config.yaml`).
7. **Config over constants.** Change a column list, seed or business assumption in one YAML file.

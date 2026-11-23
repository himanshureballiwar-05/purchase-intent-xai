# User Guide - Dashboard

Start the dashboard: `streamlit run dashboard/app.py` -> http://localhost:8501
(or `docker compose up --build`). Use the sidebar to switch between the six sections.

## 1. Overview
Headline numbers (sessions, conversion rate), the model's key test metrics and the top drivers of purchase intent.
Read this first. Note that the conversion rate is only ~16%, which is why accuracy is not the headline metric.

## 2. Live prediction
1. Fill in the session form (defaults are the dataset medians).
2. Press **Predict**.
3. Read the result:
   - **Gauge** = purchase probability; the black line is the decision threshold.
   - **Decision** = likely / unlikely to buy; **Confidence** = how far the probability is from the threshold.
   - **Bar chart** = the features that pushed the prediction up (green) or down (red) - the SHAP explanation.
4. Tick **Show a second opinion with LIME** to see an independent local explanation.

> A *low confidence* warning means the probability is close to the threshold; do not trigger expensive actions on it.

## 3. Behaviour analytics
Filter by month and visitor type; see conversion by month, visitor type, product pages viewed, and how `PageValues` differs between buyers and non-buyers.

## 4. Model comparison
Pick a metric and compare the six algorithms and three imbalance strategies (5-fold CV on the training set).
Use PR-AUC for ranking quality; recall/precision here are at threshold 0.5.

## 5. Explainability
Global SHAP importance, the beeswarm plot (direction of each feature's effect) and three worked examples
(likely buyer, likely non-buyer, borderline) with both SHAP and LIME.

## 6. Model & system
Test-set metrics (at 0.5 and at the chosen threshold), confusion matrix, precision-recall curve, the business-value
threshold curve, the `PageValues` leakage check, and model/environment information.

## Troubleshooting
| Problem | Fix |
|---|---|
| "Model artifacts not found" | Run `python -m src.models.train` (about 3-4 minutes) |
| Pickle/version error when loading the model | Re-train in your own environment: `python -m src.models.train` |
| Port already in use | `streamlit run dashboard/app.py --server.port 8502` |

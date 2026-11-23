# Demo video script (3-5 minutes)

Record your screen with OBS / Windows Game Bar (Win+G) / Loom. Speak in your own words.

| Time | Show | Say (idea) |
|---|---|---|
| 0:00-0:30 | README / repo | Problem: predict which visitors will buy, and explain why. Dataset: 12,000 sessions, 15.7% buy. |
| 0:30-1:15 | Notebook 01 (EDA) | Heavy imbalance, so accuracy is misleading. Key drivers. PageValues is suspicious (100% of buyers > 0). |
| 1:15-2:00 | `model_comparison` in dashboard (section 4) | 6 models x 3 imbalance strategies. Best = CatBoost. Why PR-AUC. |
| 2:00-2:45 | Dashboard section 2 (Live prediction) | Enter a session -> probability + SHAP bars. Change PageValues to 0 and predict again. |
| 2:45-3:30 | `http://localhost:8000/docs` | Try `/predict` in Swagger; show a 422 error for a bad value. |
| 3:30-4:15 | Terminal | `docker compose up --build`, `pytest` passing, GitHub Actions badge/green run, `mlflow ui`. |
| 4:15-4:45 | Model card / dashboard section 6 | Limitations: PageValues is late-session; business numbers are assumptions. |

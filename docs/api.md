# API Reference

Base URL (local): `http://localhost:8000` - interactive docs (Swagger): `http://localhost:8000/docs`

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness + model loaded? (`degraded` if artifacts missing) |
| GET | `/model/info` | Model name, threshold, test metrics, top features |
| POST | `/predict` | Score one session (+ SHAP top contributors; `?explain=false` to skip) |
| POST | `/predict/batch` | Score 1-1000 sessions (`?explain=true` to add SHAP) |

## `POST /predict`
**Request** (all 17 fields required; same as `examples/session.json`):
```json
{
  "Administrative": 2,
  "Administrative_Duration": 60.0,
  "Informational": 0,
  "Informational_Duration": 0.0,
  "ProductRelated": 40,
  "ProductRelated_Duration": 1400.0,
  "BounceRates": 0.005,
  "ExitRates": 0.02,
  "PageValues": 35.0,
  "SpecialDay": 0.0,
  "Month": "Nov",
  "OperatingSystems": 2,
  "Browser": 2,
  "Region": 1,
  "TrafficType": 2,
  "VisitorType": "Returning_Visitor",
  "Weekend": false
}
```
**Response** (real output, top 3 of 5 contributors shown)
```json
{
  "purchase_probability": 0.8756,
  "prediction": 1,
  "threshold": 0.14,
  "confidence": "high",
  "top_contributors": [
    {
      "feature": "PageValues",
      "value": 35.0,
      "shap_value": 4.2723,
      "direction": "increases purchase intent"
    },
    {
      "feature": "ProductRelated",
      "value": 40,
      "shap_value": 0.6459,
      "direction": "increases purchase intent"
    },
    {
      "feature": "TotalPages",
      "value": 42,
      "shap_value": 0.6148,
      "direction": "increases purchase intent"
    }
  ]
}
```
`confidence` = how far the probability is from the decision threshold, relative to the room on that side (`high` / `medium` / `low`). A very low probability is therefore a high-confidence no.
`shap_value` is the feature's impact on the model's raw output (log-odds for boosted trees).

## Validation & errors
| Situation | Status |
|---|---|
| Missing field, wrong type, out-of-range value (e.g. `BounceRates: 5`), unknown `Month`/`VisitorType` | `422` with field-level details |
| Empty batch or more than 1000 sessions | `422` |
| Model artifacts not found | `503` on prediction endpoints; `/health` returns `degraded` |
| Unexpected server error | `500` `{"detail": "Internal server error"}` (stack trace only in logs) |

## Examples
```bash
curl -s http://localhost:8000/health

curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @examples/session.json

curl -s -X POST "http://localhost:8000/predict/batch" -H "Content-Type: application/json" \
     -d '{"sessions": [ ... ]}'
```
```python
import requests
r = requests.post("http://localhost:8000/predict", json=session)
print(r.json()["purchase_probability"])
```

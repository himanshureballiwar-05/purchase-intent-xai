import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.schemas import SessionFeatures
from src.config.settings import get_settings


@pytest.fixture(scope="module")
def client(artifact_dir):
    with TestClient(create_app(artifact_dir)) as c:
        yield c


def test_schema_matches_config_columns():
    assert set(SessionFeatures.model_fields) == set(get_settings().input_columns)


def test_health_ok(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] is True


def test_predict_ok(client, sample_session):
    r = client.post("/predict", json=sample_session)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["purchase_probability"] <= 1
    assert len(body["top_contributors"]) == 5


def test_predict_without_explanation(client, sample_session):
    body = client.post("/predict?explain=false", json=sample_session).json()
    assert body["top_contributors"] is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("BounceRates", 5),
        ("Administrative", -1),
        ("Month", "Smarch"),
        ("VisitorType", "Alien"),
        ("Weekend", "maybe"),
    ],
)
def test_invalid_input_rejected(client, sample_session, field, value):
    bad = dict(sample_session, **{field: value})
    assert client.post("/predict", json=bad).status_code == 422


def test_missing_field_rejected(client, sample_session):
    bad = dict(sample_session)
    bad.pop("PageValues")
    assert client.post("/predict", json=bad).status_code == 422


def test_batch(client, sample_session):
    r = client.post("/predict/batch", json={"sessions": [sample_session] * 4})
    assert r.status_code == 200 and r.json()["count"] == 4


def test_empty_batch_rejected(client):
    assert client.post("/predict/batch", json={"sessions": []}).status_code == 422


def test_model_info(client):
    assert "threshold" in client.get("/model/info").json()


def test_degraded_when_model_missing(tmp_path, sample_session):
    with TestClient(create_app(tmp_path)) as c:
        assert c.get("/health").json()["status"] == "degraded"
        assert c.post("/predict", json=sample_session).status_code == 503

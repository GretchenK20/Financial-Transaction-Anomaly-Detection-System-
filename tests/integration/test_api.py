"""
Integration tests for FastAPI endpoints.
Runs against the actual app with TestClient — no real server needed.
"""
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from api.main import app
    # Must enter as a context manager so the lifespan startup handler
    # (which loads the AE/XGBoost models) actually runs.
    with TestClient(app) as c:
        yield c


# A profile shaped like the known-fraud pattern in the ULB dataset (large
# negative V14/V12/V10/V4 — the strongest SHAP drivers for the trained model).
HIGH_RISK_TRANSACTION = {
    "transaction_id": 1,
    "v4": 4.5, "v10": -8.0, "v12": -8.0, "v14": -8.0,
    "amount": 1.0, "log_amount": 0.69, "hour_of_day": 2,
    "is_night_transaction": 1, "amount_zscore": -0.3,
}

LOW_RISK_TRANSACTION = {
    "transaction_id": 2,
    "amount": 25.0, "log_amount": 3.26, "hour_of_day": 14,
    "is_night_transaction": 0, "amount_zscore": -0.1,
}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "healthy"
    assert data["champion"] in ("xgboost", "autoencoder")


def test_champion_endpoint(client):
    r = client.get("/champion")
    assert r.status_code == 200
    data = r.json()
    assert "champion" in data
    assert "history" in data


def test_score_returns_valid_response(client):
    r = client.post("/score", json=HIGH_RISK_TRANSACTION)
    assert r.status_code == 200
    data = r.json()
    assert data["transaction_id"] == 1
    assert 0.0 <= data["risk_score"] <= 1.0
    assert isinstance(data["is_high_risk"], bool)
    assert 0.0 <= data["risk_percentile"] <= 100.0
    assert len(data["top_risk_factors"]) > 0
    assert data["explanation"] != ""


def test_high_risk_transaction_scores_higher(client):
    r_high = client.post("/score", json=HIGH_RISK_TRANSACTION)
    r_low = client.post("/score", json=LOW_RISK_TRANSACTION)
    assert r_high.status_code == 200
    assert r_low.status_code == 200
    assert r_high.json()["risk_score"] >= r_low.json()["risk_score"]


def test_batch_score(client):
    r = client.post("/batch_score", json={"transactions": [HIGH_RISK_TRANSACTION, LOW_RISK_TRANSACTION]})
    assert r.status_code == 200
    data = r.json()
    assert data["n_scored"] == 2
    assert len(data["results"]) == 2
    assert data["errors"] == []


def test_shap_factors_present(client):
    r = client.post("/score", json=HIGH_RISK_TRANSACTION)
    factors = r.json()["top_risk_factors"]
    assert all("feature" in f and "shap_value" in f for f in factors)
    assert all(f["direction"] in ("increases", "decreases") for f in factors)


def test_transaction_lookup_known_fraud(client):
    r = client.get("/transaction/243393")
    assert r.status_code == 200
    data = r.json()
    assert data["transaction_id"] == 243393
    assert data["is_high_risk"] is True


def test_transaction_lookup_not_found(client):
    r = client.get("/transaction/999999999")
    assert r.status_code == 404

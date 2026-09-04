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
    return TestClient(app)


SAMPLE_PATIENT = {
    "patient_id": "test-001",
    "age_years": 65,
    "gender_male": 0,
    "has_diabetes": 1,
    "has_hypertension": 1,
    "has_ckd": 1,
    "bmi": 32.5,
    "systolic_bp": 145,
    "total_encounters": 8,
    "emergency_count": 2,
    "composite_risk_score": 4,
}

LOW_RISK_PATIENT = {
    "patient_id": "test-002",
    "age_years": 28,
    "gender_male": 1,
    "bmi": 22.0,
    "systolic_bp": 118,
    "total_encounters": 1,
    "composite_risk_score": 0,
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
    r = client.post("/score", json=SAMPLE_PATIENT)
    assert r.status_code == 200
    data = r.json()
    assert data["patient_id"] == "test-001"
    assert 0.0 <= data["risk_score"] <= 1.0
    assert isinstance(data["is_high_risk"], bool)
    assert 0.0 <= data["risk_percentile"] <= 100.0
    assert len(data["top_risk_factors"]) > 0
    assert data["explanation"] != ""


def test_high_risk_patient_scores_higher(client):
    r_high = client.post("/score", json=SAMPLE_PATIENT)
    r_low = client.post("/score", json=LOW_RISK_PATIENT)
    assert r_high.status_code == 200
    assert r_low.status_code == 200
    assert r_high.json()["risk_score"] >= r_low.json()["risk_score"]


def test_batch_score(client):
    r = client.post("/batch_score", json={"patients": [SAMPLE_PATIENT, LOW_RISK_PATIENT]})
    assert r.status_code == 200
    data = r.json()
    assert data["n_scored"] == 2
    assert len(data["results"]) == 2
    assert data["errors"] == []


def test_shap_factors_present(client):
    r = client.post("/score", json=SAMPLE_PATIENT)
    factors = r.json()["top_risk_factors"]
    assert all("feature" in f and "shap_value" in f for f in factors)
    assert all(f["direction"] in ("increases", "decreases") for f in factors)

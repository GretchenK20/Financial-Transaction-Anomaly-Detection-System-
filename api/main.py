"""
FastAPI scoring endpoint — serves champion model with SHAP explainability.
Routes: POST /score, GET /health, GET /champion, POST /batch_score, GET /patient/{id}
"""
import numpy as np
import pandas as pd
import torch
import joblib
import shap
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DUCKDB_PATH
from models.autoencoder.train_autoencoder import (
    ClinicalAutoencoder, NUMERIC_FEATURES,
    MODEL_PATH as AE_MODEL_PATH,
    SCALER_PATH as AE_SCALER_PATH,
    MODEL_DIR as AE_DIR,
)
from models.xgboost.train_xgboost import XGB_MODEL_PATH
from models.champion_challenger import load_registry

_ae_model = None
_ae_scaler = None
_xgb_bundle = None


def _load_models():
    global _ae_model, _ae_scaler, _xgb_bundle

    # Load autoencoder — derive input_dim from saved scaler
    if AE_MODEL_PATH.exists() and AE_SCALER_PATH.exists():
        _ae_scaler = joblib.load(AE_SCALER_PATH)
        input_dim = _ae_scaler.n_features_in_
        _ae_model = ClinicalAutoencoder(input_dim=input_dim)
        _ae_model.load_state_dict(
            torch.load(AE_MODEL_PATH, map_location="cpu", weights_only=True)
        )
        _ae_model.eval()

    # Load XGBoost
    if XGB_MODEL_PATH.exists():
        _xgb_bundle = joblib.load(XGB_MODEL_PATH)


@asynccontextmanager
async def lifespan(app):
    _load_models()
    yield


app = FastAPI(
    title="Clinical Risk Scoring API",
    description=(
        "FHIR-ingested patient risk scoring with PyTorch autoencoder "
        "and XGBoost champion/challenger"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class PatientFeatures(BaseModel):
    patient_id: str
    age_years: Optional[float] = None
    gender_male: Optional[float] = None
    is_deceased: Optional[float] = 0
    has_diabetes: Optional[float] = 0
    has_hypertension: Optional[float] = 0
    has_cad: Optional[float] = 0
    has_heart_failure: Optional[float] = 0
    has_asthma_or_copd: Optional[float] = 0
    has_ckd: Optional[float] = 0
    has_depression: Optional[float] = 0
    has_anxiety: Optional[float] = 0
    has_cancer: Optional[float] = 0
    total_active_conditions: Optional[float] = 0
    bmi: Optional[float] = None
    systolic_bp: Optional[float] = None
    diastolic_bp: Optional[float] = None
    cholesterol_total: Optional[float] = None
    is_obese: Optional[float] = 0
    elevated_systolic: Optional[float] = 0
    high_cholesterol: Optional[float] = 0
    total_encounters: Optional[float] = 0
    emergency_count: Optional[float] = 0
    inpatient_count: Optional[float] = 0
    encounters_last_12m: Optional[float] = 0
    care_span_days: Optional[float] = 0
    high_ed_utilizer: Optional[float] = 0
    composite_risk_score: Optional[float] = 0


class RiskScore(BaseModel):
    patient_id: str
    champion_model: Optional[str]
    risk_score: float
    is_high_risk: bool
    risk_percentile: float
    ae_reconstruction_error: Optional[float]
    xgb_probability: Optional[float]
    top_risk_factors: list[dict]
    explanation: str


class BatchRequest(BaseModel):
    patients: list[PatientFeatures]


def _patient_to_array(patient: PatientFeatures, feature_names: list[str]) -> np.ndarray:
    vals = [
        getattr(patient, f, None) or 0.0
        for f in feature_names
    ]
    return np.array(vals, dtype=float).reshape(1, -1)


def _score_patient(patient: PatientFeatures) -> dict:
    registry = load_registry()
    champion = registry.get("champion", "xgboost")

    ae_error = None
    xgb_prob = None
    shap_values = None
    xgb_feature_names = None

    # Autoencoder — the scaler may have been fit on a subset of NUMERIC_FEATURES
    # (train_autoencoder.preprocess drops zero-variance columns), so use the
    # columns it actually knows about rather than the full feature list.
    if _ae_model is not None and _ae_scaler is not None:
        ae_feature_names = list(getattr(_ae_scaler, "feature_names_in_", NUMERIC_FEATURES))
        raw_ae = _patient_to_array(patient, ae_feature_names)
        X_ae = _ae_scaler.transform(raw_ae)
        X_t = torch.FloatTensor(X_ae)
        ae_error = float(_ae_model.reconstruction_error(X_t).item())

    # XGBoost + SHAP
    if _xgb_bundle is not None:
        model = _xgb_bundle["model"]
        scaler = _xgb_bundle["scaler"]
        xgb_feature_names = list(getattr(scaler, "feature_names_in_", NUMERIC_FEATURES))
        raw_xgb = _patient_to_array(patient, xgb_feature_names)
        X_xgb = scaler.transform(raw_xgb)
        xgb_prob = float(model.predict_proba(X_xgb)[0, 1])

        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_xgb)
        shap_values = sv[0] if sv.ndim > 1 else sv

    if _ae_model is None and _xgb_bundle is None:
        raise HTTPException(status_code=503, detail="No models loaded")

    # Primary risk score from champion
    if champion == "xgboost" and xgb_prob is not None:
        risk_score = xgb_prob
        is_high_risk = xgb_prob >= 0.5
    elif ae_error is not None:
        risk_score = min(ae_error / 0.5, 1.0)
        is_high_risk = ae_error >= 0.2110
    else:
        raise HTTPException(status_code=503, detail="Champion model not available")

    # Percentile vs reference distribution
    scores_path = AE_DIR / "anomaly_scores.parquet"
    risk_percentile = 50.0
    if scores_path.exists():
        ref = pd.read_parquet(scores_path)
        base = xgb_prob if champion == "xgboost" and xgb_prob is not None else ae_error
        risk_percentile = round(
            float(np.mean(ref["anomaly_score"] <= base) * 100), 1
        )

    # Top SHAP factors
    top_factors = []
    if shap_values is not None:
        feature_names = (xgb_feature_names or NUMERIC_FEATURES)[:len(shap_values)]
        pairs = sorted(
            zip(feature_names, shap_values),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:5]
        top_factors = [
            {
                "feature": f,
                "shap_value": round(float(v), 4),
                "direction": "increases" if v > 0 else "decreases",
            }
            for f, v in pairs
        ]

    risk_level = "high" if is_high_risk else "low"
    top_feature = (
        top_factors[0]["feature"].replace("_", " ") if top_factors
        else "overall health profile"
    )
    explanation = (
        f"Patient {patient.patient_id} has {risk_level} clinical risk "
        f"(score: {risk_score:.3f}, {risk_percentile}th percentile). "
        f"Primary driver: {top_feature}."
    )

    return {
        "patient_id": patient.patient_id,
        "champion_model": champion,
        "risk_score": round(risk_score, 4),
        "is_high_risk": is_high_risk,
        "risk_percentile": risk_percentile,
        "ae_reconstruction_error": round(ae_error, 4) if ae_error is not None else None,
        "xgb_probability": round(xgb_prob, 4) if xgb_prob is not None else None,
        "top_risk_factors": top_factors,
        "explanation": explanation,
    }


@app.get("/health")
def health():
    registry = load_registry()
    return {
        "status": "healthy",
        "champion": registry.get("champion"),
        "ae_loaded": _ae_model is not None,
        "xgb_loaded": _xgb_bundle is not None,
    }


@app.get("/champion")
def get_champion():
    return load_registry()


@app.post("/score", response_model=RiskScore)
def score(patient: PatientFeatures):
    return _score_patient(patient)


@app.post("/batch_score")
def batch_score(request: BatchRequest):
    results, errors = [], []
    for patient in request.patients:
        try:
            results.append(_score_patient(patient))
        except Exception as e:
            errors.append({"patient_id": patient.patient_id, "error": str(e)})
    return {"results": results, "errors": errors, "n_scored": len(results)}


@app.get("/patient/{patient_id}")
def score_from_db(patient_id: str):
    try:
        import duckdb
        with duckdb.connect(str(DUCKDB_PATH), read_only=True) as conn:
            row = conn.execute(
                "SELECT * FROM main_gold.fct_patient_risk_features WHERE patient_id = ?",
                [patient_id],
            ).fetchdf()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if row.empty:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")

    r = row.iloc[0].to_dict()
    patient = PatientFeatures(patient_id=patient_id, **{
        k: r.get(k) for k in PatientFeatures.model_fields if k != "patient_id"
    })
    return _score_patient(patient)

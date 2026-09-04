"""
FastAPI scoring endpoint — serves champion model with SHAP explainability.
Routes: POST /score, GET /health, GET /champion, POST /batch_score, GET /transaction/{id}
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
    FraudAutoencoder, NUMERIC_FEATURES,
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
        _ae_model = FraudAutoencoder(input_dim=input_dim)
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
    title="Financial Transaction Anomaly Detection API",
    description=(
        "284K real-world credit card transactions — PyTorch autoencoder + "
        "XGBoost champion/challenger with SHAP explainability"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class TransactionFeatures(BaseModel):
    transaction_id: int
    v1: Optional[float] = 0
    v2: Optional[float] = 0
    v3: Optional[float] = 0
    v4: Optional[float] = 0
    v5: Optional[float] = 0
    v6: Optional[float] = 0
    v7: Optional[float] = 0
    v8: Optional[float] = 0
    v9: Optional[float] = 0
    v10: Optional[float] = 0
    v11: Optional[float] = 0
    v12: Optional[float] = 0
    v13: Optional[float] = 0
    v14: Optional[float] = 0
    v15: Optional[float] = 0
    v16: Optional[float] = 0
    v17: Optional[float] = 0
    v18: Optional[float] = 0
    v19: Optional[float] = 0
    v20: Optional[float] = 0
    v21: Optional[float] = 0
    v22: Optional[float] = 0
    v23: Optional[float] = 0
    v24: Optional[float] = 0
    v25: Optional[float] = 0
    v26: Optional[float] = 0
    v27: Optional[float] = 0
    v28: Optional[float] = 0
    amount: Optional[float] = None
    log_amount: Optional[float] = None
    hour_of_day: Optional[float] = 0
    is_night_transaction: Optional[float] = 0
    amount_zscore: Optional[float] = 0


class FraudScore(BaseModel):
    transaction_id: int
    champion_model: Optional[str]
    risk_score: float
    is_high_risk: bool
    risk_percentile: float
    ae_reconstruction_error: Optional[float]
    xgb_probability: Optional[float]
    top_risk_factors: list[dict]
    explanation: str


class BatchRequest(BaseModel):
    transactions: list[TransactionFeatures]


def _transaction_to_array(txn: TransactionFeatures, feature_names: list[str]) -> np.ndarray:
    vals = [
        getattr(txn, f, None) or 0.0
        for f in feature_names
    ]
    return np.array(vals, dtype=float).reshape(1, -1)


def _score_transaction(txn: TransactionFeatures) -> dict:
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
        raw_ae = _transaction_to_array(txn, ae_feature_names)
        X_ae = _ae_scaler.transform(raw_ae)
        X_t = torch.FloatTensor(X_ae)
        ae_error = float(_ae_model.reconstruction_error(X_t).item())

    # XGBoost + SHAP
    if _xgb_bundle is not None:
        model = _xgb_bundle["model"]
        scaler = _xgb_bundle["scaler"]
        xgb_feature_names = list(getattr(scaler, "feature_names_in_", NUMERIC_FEATURES))
        raw_xgb = _transaction_to_array(txn, xgb_feature_names)
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
        else "overall transaction profile"
    )
    explanation = (
        f"Transaction {txn.transaction_id} has {risk_level} fraud risk "
        f"(score: {risk_score:.3f}, {risk_percentile}th percentile). "
        f"Primary driver: {top_feature}."
    )

    return {
        "transaction_id": txn.transaction_id,
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


@app.post("/score", response_model=FraudScore)
def score(transaction: TransactionFeatures):
    return _score_transaction(transaction)


@app.post("/batch_score")
def batch_score(request: BatchRequest):
    results, errors = [], []
    for transaction in request.transactions:
        try:
            results.append(_score_transaction(transaction))
        except Exception as e:
            errors.append({"transaction_id": transaction.transaction_id, "error": str(e)})
    return {"results": results, "errors": errors, "n_scored": len(results)}


@app.get("/transaction/{transaction_id}")
def score_from_db(transaction_id: int):
    try:
        import duckdb
        with duckdb.connect(str(DUCKDB_PATH), read_only=True) as conn:
            row = conn.execute(
                "SELECT * FROM main_gold.fct_fraud_features WHERE transaction_id = ?",
                [transaction_id],
            ).fetchdf()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if row.empty:
        raise HTTPException(status_code=404, detail=f"Transaction {transaction_id} not found")

    r = row.iloc[0].to_dict()
    transaction = TransactionFeatures(transaction_id=transaction_id, **{
        k: r.get(k) for k in TransactionFeatures.model_fields if k != "transaction_id"
    })
    return _score_transaction(transaction)

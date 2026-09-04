"""
Champion/challenger framework.
Compares autoencoder and XGBoost on key metrics, promotes winner to champion,
logs everything to MLflow, detects drift on new data.
"""
import json
import numpy as np
import pandas as pd
import torch
import joblib
from pathlib import Path
from datetime import datetime
from loguru import logger
import mlflow
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DUCKDB_PATH, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
from models.autoencoder.train_autoencoder import (
    ClinicalAutoencoder, load_features, preprocess,
    MODEL_PATH as AE_MODEL_PATH, SCALER_PATH as AE_SCALER_PATH,
    MODEL_DIR as AE_DIR, NUMERIC_FEATURES,
)
from models.xgboost.train_xgboost import (
    XGB_MODEL_PATH, load_labels, SHAP_VALUES_PATH
)

REGISTRY_PATH = Path(__file__).parent / "model_registry.json"


def load_registry() -> dict:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())
    return {"champion": None, "history": []}


def save_registry(registry: dict) -> None:
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2, default=str))


def score_autoencoder(db_path: Path) -> pd.DataFrame:
    df = load_features(db_path)
    patient_ids = df["patient_id"].values
    X, scaler, feature_cols = preprocess(df)

    model = ClinicalAutoencoder(input_dim=len(feature_cols))
    model.load_state_dict(torch.load(AE_MODEL_PATH, map_location="cpu"))
    model.eval()

    X_t = torch.FloatTensor(X)
    errors = model.reconstruction_error(X_t).numpy()
    threshold = np.percentile(errors, 95)

    return pd.DataFrame({
        "patient_id": patient_ids,
        "ae_score": errors,
        "ae_anomaly": (errors >= threshold).astype(int),
        "ae_threshold": threshold,
    })


def score_xgboost(db_path: Path) -> pd.DataFrame:
    df = load_features(db_path)
    patient_ids = df["patient_id"].values

    bundle = joblib.load(XGB_MODEL_PATH)
    model = bundle["model"]
    scaler = bundle["scaler"]

    X_raw = df[NUMERIC_FEATURES].fillna(df[NUMERIC_FEATURES].median())
    X = scaler.transform(X_raw)

    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)

    return pd.DataFrame({
        "patient_id": patient_ids,
        "xgb_score": probs,
        "xgb_anomaly": preds,
    })


def compare_and_promote(db_path: Path = DUCKDB_PATH) -> dict:
    """
    Load both models, score the full dataset, compare on F1 and AUC,
    promote the better model as champion.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    scores_path = AE_DIR / "anomaly_scores.parquet"
    if not scores_path.exists():
        raise FileNotFoundError("Run train_autoencoder.py first.")
    if not XGB_MODEL_PATH.exists():
        raise FileNotFoundError("Run train_xgboost.py first.")

    ae_scores = score_autoencoder(db_path)
    xgb_scores = score_xgboost(db_path)

    merged = ae_scores.merge(xgb_scores, on="patient_id")
    labels = load_labels(scores_path)[["patient_id", "is_anomaly"]]
    merged = merged.merge(labels, on="patient_id")

    from sklearn.metrics import f1_score, roc_auc_score
    y_true = merged["is_anomaly"].values

    ae_f1 = f1_score(y_true, merged["ae_anomaly"], zero_division=0)
    xgb_f1 = f1_score(y_true, merged["xgb_anomaly"], zero_division=0)
    ae_auc = roc_auc_score(y_true, merged["ae_score"])
    xgb_auc = roc_auc_score(y_true, merged["xgb_score"])

    comparison = {
        "autoencoder": {"f1": round(ae_f1, 4), "auc": round(ae_auc, 4)},
        "xgboost":     {"f1": round(xgb_f1, 4), "auc": round(xgb_auc, 4)},
    }

    # Promote champion based on AUC (primary) and F1 (tiebreak)
    if xgb_auc > ae_auc + 0.02 or (
        abs(xgb_auc - ae_auc) <= 0.02 and xgb_f1 > ae_f1
    ):
        champion = "xgboost"
    else:
        champion = "autoencoder"

    registry = load_registry()
    registry["champion"] = champion
    registry["history"].append({
        "timestamp": datetime.utcnow().isoformat(),
        "champion": champion,
        "metrics": comparison,
    })
    save_registry(registry)

    with mlflow.start_run(run_name="champion_challenger"):
        mlflow.log_metrics({
            "ae_f1": ae_f1, "ae_auc": ae_auc,
            "xgb_f1": xgb_f1, "xgb_auc": xgb_auc,
        })
        mlflow.log_param("champion", champion)

    logger.info(f"Champion: {champion}")
    logger.info(f"  AE  — F1: {ae_f1:.4f}  AUC: {ae_auc:.4f}")
    logger.info(f"  XGB — F1: {xgb_f1:.4f}  AUC: {xgb_auc:.4f}")

    return {"champion": champion, "comparison": comparison}


def detect_drift(
    new_scores: pd.Series,
    reference_scores: pd.Series,
    threshold: float = 0.1,
) -> dict:
    """
    PSI-based drift detection between reference and new score distributions.
    PSI < 0.1 = no drift, 0.1–0.25 = moderate, >0.25 = significant.
    """
    def psi(expected, actual, bins=10):
        breakpoints = np.linspace(0, 1, bins + 1)
        e = np.histogram(expected, bins=breakpoints)[0] / len(expected)
        a = np.histogram(actual, bins=breakpoints)[0] / len(actual)
        e = np.clip(e, 1e-6, None)
        a = np.clip(a, 1e-6, None)
        return float(np.sum((a - e) * np.log(a / e)))

    psi_value = psi(reference_scores, new_scores)
    drift_detected = psi_value > threshold

    return {
        "psi": round(psi_value, 4),
        "drift_detected": drift_detected,
        "severity": (
            "none" if psi_value < 0.1
            else "moderate" if psi_value < 0.25
            else "significant"
        ),
    }


if __name__ == "__main__":
    result = compare_and_promote()
    print(f"\nChampion: {result['champion']}")
    for model, metrics in result["comparison"].items():
        print(f"  {model}: AUC={metrics['auc']:.4f}  F1={metrics['f1']:.4f}")

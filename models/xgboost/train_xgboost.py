"""
XGBoost classifier for supervised fraud classification.
Trained on the real Class labels; compared against the autoencoder in the
champion/challenger framework.
"""
import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, average_precision_score,
)
from sklearn.preprocessing import StandardScaler
import shap
import mlflow
import mlflow.xgboost
import joblib
from pathlib import Path
from loguru import logger
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DUCKDB_PATH, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
from models.autoencoder.train_autoencoder import (
    load_features, preprocess, NUMERIC_FEATURES, MODEL_DIR as AE_DIR
)

MODEL_DIR = Path(__file__).parent
XGB_MODEL_PATH = MODEL_DIR / "xgboost_model.joblib"
SHAP_VALUES_PATH = MODEL_DIR / "shap_values.parquet"


def load_labels(scores_path: Path) -> pd.DataFrame:
    return pd.read_parquet(scores_path)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "avg_precision": average_precision_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }


def compute_segment_metrics(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    group_col: str,
) -> dict:
    """
    Compute AUC per transaction segment (e.g. amount tier) for consistency
    evaluation. Flags segments where AUC deviates >0.1 from overall — this
    dataset has no demographic attributes, so segments stand in for the
    fairness-style check some models run across demographic groups.
    """
    overall_auc = roc_auc_score(y_true, y_prob)
    results = {"overall_auc": overall_auc, "groups": {}}
    consistency_flags = []

    for group in df[group_col].dropna().unique():
        mask = df[group_col] == group
        if mask.sum() < 10 or y_true[mask].sum() == 0:
            continue
        group_auc = roc_auc_score(y_true[mask], y_prob[mask])
        gap = abs(group_auc - overall_auc)
        results["groups"][str(group)] = {
            "auc": round(group_auc, 4),
            "gap_from_overall": round(gap, 4),
            "n": int(mask.sum()),
            "flagged": gap > 0.1,
        }
        if gap > 0.1:
            consistency_flags.append(group)

    results["consistency_flags"] = consistency_flags
    return results


def train(
    db_path: Path = DUCKDB_PATH,
    scores_path: Optional[Path] = None,
    n_estimators: int = 200,
    max_depth: int = 4,
    learning_rate: float = 0.1,
) -> dict:
    scores_path = scores_path or (AE_DIR / "anomaly_scores.parquet")
    if not scores_path.exists():
        raise FileNotFoundError(
            f"Autoencoder scores not found at {scores_path}. "
            "Run train_autoencoder.py first."
        )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    df = load_features(db_path)
    labels = load_labels(scores_path)
    df = df.merge(labels[["transaction_id", "is_fraud"]], on="transaction_id", how="inner")

    # Amount tier, for segment-consistency evaluation below (not a model input)
    with duckdb.connect(str(db_path), read_only=True) as conn:
        segments = conn.execute(
            "SELECT transaction_id, amount_bucket FROM main_gold.fct_fraud_features"
        ).fetchdf()
    df = df.merge(segments, on="transaction_id", how="left")

    X_raw = df[NUMERIC_FEATURES].fillna(df[NUMERIC_FEATURES].median())
    y = df["is_fraud"].values

    demo_df = df.reset_index(drop=True)

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1),
        use_label_encoder=False,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    with mlflow.start_run(run_name="xgboost_classifier"):
        mlflow.log_params({
            "model_type": "xgboost",
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_transactions": len(df),
            "fraud_rate": float(y.mean()),
        })

        # Manual CV loop — sklearn's "roc_auc" scorer mishandles XGBClassifier's
        # predict_proba shape with this xgboost/scikit-learn combination
        # (returns the full (n, 2) proba array instead of the positive-class
        # column), so cross_val_score(..., scoring="roc_auc") silently yields NaN.
        cv_aucs = []
        for train_idx, val_idx in cv.split(X, y):
            fold_model = clone(model)
            fold_model.fit(X[train_idx], y[train_idx])
            fold_prob = fold_model.predict_proba(X[val_idx])[:, 1]
            cv_aucs.append(roc_auc_score(y[val_idx], fold_prob))
        cv_aucs = np.array(cv_aucs)
        logger.info(
            f"CV AUC: {cv_aucs.mean():.4f} ± {cv_aucs.std():.4f}"
        )
        mlflow.log_metrics({
            "cv_auc_mean": cv_aucs.mean(),
            "cv_auc_std": cv_aucs.std(),
        })

        model.fit(X, y, verbose=False)

        y_prob = model.predict_proba(X)[:, 1]
        y_pred = model.predict(X)
        metrics = compute_metrics(y, y_pred, y_prob)

        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")
        mlflow.log_metrics(metrics)

        # Save model before SHAP (so file exists even if SHAP fails)
        joblib.dump({"model": model, "scaler": scaler}, XGB_MODEL_PATH)
        try:
            mlflow.xgboost.log_model(
                model,
                name="xgboost_model",
                input_example=X[:1],
            )
        except Exception as e:
            # Known incompatibility between this xgboost/mlflow version pair
            # (_estimator_type lookup) — non-fatal, the model is already
            # persisted via joblib above and metrics/params are still tracked.
            logger.warning(f"mlflow.xgboost.log_model failed, continuing: {e}")

        # SHAP explainability
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        shap_df = pd.DataFrame(
            shap_values,
            columns=NUMERIC_FEATURES[:X.shape[1]],
        )
        shap_df["transaction_id"] = df["transaction_id"].values
        shap_df.to_parquet(SHAP_VALUES_PATH, index=False)
        mlflow.log_artifact(str(SHAP_VALUES_PATH))

        # Segment-consistency evaluation (amount tier)
        segment_results = {}
        if "amount_bucket" in demo_df.columns:
            seg = compute_segment_metrics(demo_df, y, y_prob, "amount_bucket")
            segment_results["amount_bucket"] = seg
            if seg["consistency_flags"]:
                logger.warning(
                    f"Consistency flags for amount_bucket: {seg['consistency_flags']}"
                )
            for group, stats in seg["groups"].items():
                mlflow.log_metric(
                    f"auc_amount_bucket_{group.replace(' ', '_')[:20]}",
                    stats["auc"],
                )

        logger.info("XGBoost training complete")
        return {
            "metrics": metrics,
            "cv_auc_mean": float(cv_aucs.mean()),
            "segment_results": segment_results,
            "feature_importance": dict(
                zip(
                    NUMERIC_FEATURES[:X.shape[1]],
                    model.feature_importances_.tolist(),
                )
            ),
        }


if __name__ == "__main__":
    results = train()
    print(f"\nROC-AUC:  {results['metrics']['roc_auc']:.4f}")
    print(f"F1:       {results['metrics']['f1']:.4f}")
    print(f"CV AUC:   {results['cv_auc_mean']:.4f}")

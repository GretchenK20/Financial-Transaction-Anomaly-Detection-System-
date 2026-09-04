"""
XGBoost classifier for supervised anomaly/risk scoring.
Uses autoencoder anomaly flags as labels in champion/challenger framework.
"""
import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
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


def compute_bias_metrics(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    group_col: str,
) -> dict:
    """
    Compute AUC per demographic group for fairness evaluation.
    Flags groups where AUC deviates >0.1 from overall.
    """
    overall_auc = roc_auc_score(y_true, y_prob)
    results = {"overall_auc": overall_auc, "groups": {}}
    fairness_flags = []

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
            fairness_flags.append(group)

    results["fairness_flags"] = fairness_flags
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
    df = df.merge(labels[["patient_id", "is_anomaly"]], on="patient_id", how="inner")

    X_raw = df[NUMERIC_FEATURES].fillna(df[NUMERIC_FEATURES].median())
    y = df["is_anomaly"].values

    # Keep raw df for bias analysis (with demographics)
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
            "n_patients": len(df),
            "anomaly_rate": float(y.mean()),
        })

        cv_aucs = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
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
        mlflow.xgboost.log_model(
            model,
            name="xgboost_model",
            input_example=X[:1],
        )

        # SHAP explainability
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        shap_df = pd.DataFrame(
            shap_values,
            columns=NUMERIC_FEATURES[:X.shape[1]],
        )
        shap_df["patient_id"] = df["patient_id"].values
        shap_df.to_parquet(SHAP_VALUES_PATH, index=False)
        mlflow.log_artifact(str(SHAP_VALUES_PATH))

        # Bias / fairness evaluation
        bias_results = {}
        for group_col in ["race", "gender"]:
            if group_col in demo_df.columns:
                bias = compute_bias_metrics(demo_df, y, y_prob, group_col)
                bias_results[group_col] = bias
                if bias["fairness_flags"]:
                    logger.warning(
                        f"Fairness flags for {group_col}: {bias['fairness_flags']}"
                    )
                for group, stats in bias["groups"].items():
                    mlflow.log_metric(
                        f"auc_{group_col}_{group.replace(' ', '_')[:20]}",
                        stats["auc"],
                    )

        logger.info("XGBoost training complete")
        return {
            "metrics": metrics,
            "cv_auc_mean": float(cv_aucs.mean()),
            "bias_results": bias_results,
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

"""
PyTorch autoencoder for unsupervised anomaly detection.
Trains on the gold feature mart, assigns reconstruction error as anomaly score.
"""
import duckdb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.pytorch
import shap
import joblib
from pathlib import Path
from loguru import logger
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DUCKDB_PATH, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME

NUMERIC_FEATURES = [
    *[f"v{i}" for i in range(1, 29)],
    "amount", "log_amount", "hour_of_day", "is_night_transaction", "amount_zscore",
]

MODEL_DIR = Path(__file__).parent
SCALER_PATH = MODEL_DIR / "scaler.joblib"
MODEL_PATH = MODEL_DIR / "autoencoder.pt"


class FraudAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            recon = self.forward(x)
            return torch.mean((x - recon) ** 2, dim=1)


def load_features(db_path: Path) -> pd.DataFrame:
    with duckdb.connect(str(db_path), read_only=True) as conn:
        df = conn.execute(
            f"SELECT transaction_id, class, {', '.join(NUMERIC_FEATURES)} "
            f"FROM main_gold.fct_fraud_features"
        ).fetchdf()
    logger.info(f"Loaded {len(df):,} transactions from gold layer")
    return df


def preprocess(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler, list[str]]:
    feature_df = df[NUMERIC_FEATURES].copy()
    feature_df = feature_df.fillna(feature_df.median())

    # Drop zero-variance columns
    valid_cols = [c for c in feature_df.columns if feature_df[c].std() > 0]
    feature_df = feature_df[valid_cols]

    scaler = StandardScaler()
    X = scaler.fit_transform(feature_df)
    return X, scaler, valid_cols


def train(
    db_path: Path = DUCKDB_PATH,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    latent_dim: int = 8,
    anomaly_percentile: float = 95.0,
) -> dict:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    df = load_features(db_path)
    transaction_ids = df["transaction_id"].values
    fraud_labels = df["class"].values
    X, scaler, feature_cols = preprocess(df)

    X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)

    X_train_t = torch.FloatTensor(X_train)
    X_val_t = torch.FloatTensor(X_val)
    X_all_t = torch.FloatTensor(X)

    train_loader = DataLoader(
        TensorDataset(X_train_t), batch_size=batch_size, shuffle=True
    )

    model = FraudAutoencoder(input_dim=len(feature_cols), latent_dim=latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )
    criterion = nn.MSELoss()

    with mlflow.start_run(run_name="autoencoder"):
        mlflow.log_params({
            "model_type": "autoencoder",
            "input_dim": len(feature_cols),
            "latent_dim": latent_dim,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "n_transactions": len(df),
        })

        best_val_loss = float("inf")
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for (batch,) in train_loader:
                optimizer.zero_grad()
                recon = model(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(batch)

            train_loss /= len(X_train)

            model.eval()
            with torch.no_grad():
                val_recon = model(X_val_t)
                val_loss = criterion(val_recon, X_val_t).item()

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), MODEL_PATH)

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} — "
                    f"train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}"
                )
                mlflow.log_metrics(
                    {"train_loss": train_loss, "val_loss": val_loss},
                    step=epoch,
                )

        # Load best weights
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()

        # Score all transactions
        errors = model.reconstruction_error(X_all_t).numpy()
        threshold = np.percentile(errors, anomaly_percentile)
        anomaly_flags = (errors >= threshold).astype(int)

        mlflow.log_metrics({
            "best_val_loss": best_val_loss,
            "anomaly_threshold": float(threshold),
            "anomaly_rate": float(anomaly_flags.mean()),
            "n_anomalies": int(anomaly_flags.sum()),
        })

        joblib.dump(scaler, SCALER_PATH)
        mlflow.log_artifact(str(SCALER_PATH))
        sample_input = torch.FloatTensor(X[:1])
        mlflow.pytorch.log_model(model, "autoencoder_model", input_example=sample_input.numpy())

        scores_df = pd.DataFrame({
            "transaction_id": transaction_ids,
            "reconstruction_error": errors,
            "anomaly_score": errors,
            "is_anomaly": anomaly_flags,
            "anomaly_threshold": threshold,
            "is_fraud": fraud_labels,
        })
        scores_path = MODEL_DIR / "anomaly_scores.parquet"
        scores_df.to_parquet(scores_path, index=False)

        logger.info(
            f"Training complete — threshold: {threshold:.4f}, "
            f"anomaly rate: {anomaly_flags.mean():.1%}"
        )

        return {
            "best_val_loss": best_val_loss,
            "anomaly_threshold": float(threshold),
            "anomaly_rate": float(anomaly_flags.mean()),
            "n_anomalies": int(anomaly_flags.sum()),
            "feature_cols": feature_cols,
            "scores_df": scores_df,
        }


if __name__ == "__main__":
    results = train()
    print(f"\nAnomalies detected: {results['n_anomalies']}")
    print(f"Anomaly rate: {results['anomaly_rate']:.1%}")
    print(f"Threshold: {results['anomaly_threshold']:.4f}")

"""Unit tests for model layer — runs without DuckDB or MLflow."""
import numpy as np
import torch
import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.autoencoder.train_autoencoder import ClinicalAutoencoder
from models.champion_challenger import detect_drift


def test_autoencoder_forward_shape():
    model = ClinicalAutoencoder(input_dim=20, latent_dim=8)
    x = torch.randn(16, 20)
    out = model(x)
    assert out.shape == (16, 20)


def test_autoencoder_reconstruction_error_shape():
    model = ClinicalAutoencoder(input_dim=20, latent_dim=8)
    model.eval()
    x = torch.randn(32, 20)
    errors = model.reconstruction_error(x)
    assert errors.shape == (32,)
    assert (errors >= 0).all()


def test_autoencoder_encode_shape():
    model = ClinicalAutoencoder(input_dim=20, latent_dim=8)
    x = torch.randn(16, 20)
    z = model.encode(x)
    assert z.shape == (16, 8)


def test_reconstruction_error_higher_for_noise():
    """Anomalous inputs should have higher reconstruction error than normal inputs."""
    model = ClinicalAutoencoder(input_dim=20, latent_dim=8)
    model.eval()

    normal = torch.zeros(50, 20)
    anomalous = torch.randn(50, 20) * 5

    normal_errors = model.reconstruction_error(normal).mean().item()
    anomaly_errors = model.reconstruction_error(anomalous).mean().item()
    assert anomaly_errors > normal_errors


def test_drift_detection_no_drift():
    np.random.seed(42)
    ref = np.random.beta(2, 5, 500)
    new = np.random.beta(2, 5, 500)
    result = detect_drift(pd.Series(new), pd.Series(ref))
    assert result["drift_detected"] is False
    assert result["psi"] < 0.1


def test_drift_detection_significant_drift():
    np.random.seed(42)
    ref = np.random.beta(2, 5, 500)
    new = np.random.beta(8, 2, 500)
    result = detect_drift(pd.Series(new), pd.Series(ref))
    assert result["drift_detected"] is True
    assert result["severity"] in ("moderate", "significant")


import pandas as pd

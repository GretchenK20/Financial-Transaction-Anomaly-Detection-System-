---
title: Financial Fraud Detection API
emoji: 💳
colorFrom: blue
colorTo: red
sdk: docker
dockerfile: docker/Dockerfile
app_port: 8000
pinned: false
---

# Financial Transaction Anomaly Detection System

> The YAML block above configures this repo as a [Hugging Face Space](https://huggingface.co/spaces)
> (Docker SDK) serving `api/main.py`. It's inert everywhere else (GitHub, local
> checkouts) — safe to ignore if you're not deploying there.

Production fraud-detection AI pipeline: real credit card transaction data → dbt medallion → PyTorch autoencoder / XGBoost champion-challenger → FastAPI + Docker + Kubernetes → LangChain agent → Streamlit dashboard.

Built on the ULB Machine Learning Group's **Credit Card Fraud Detection** dataset (Kaggle: `mlg-ulb/creditcardfraud`) — 284,807 real European card transactions from September 2013, with a 0.17% fraud rate (492 confirmed frauds).

** LIVE DEMO: https://jpbxjujh7jaydskfh8g6sf.streamlit.app/

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-full.txt   # full pipeline: ingestion, dbt, models, API, agent
cp .env.example .env
```

`requirements.txt` (repo root) is intentionally a separate, slim file — it's what
[Streamlit Community Cloud](https://streamlit.io/cloud) reads to build `streamlit_app.py`'s
environment, and deliberately excludes PyTorch/XGBoost/DuckDB/MLflow. If you only want to
run the dashboard (in demo mode, no local API needed): `pip install -r requirements.txt`.

## Data

Download the dataset (requires a Kaggle account/API token) and place it at `data/raw/transactions/creditcard.csv`:

```python
import kagglehub
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
```

## Run the pipeline

```bash
# 1. Ingest
python -m ingestion.transaction_loader

# 2. Transform
cd dbt_project && dbt run --profiles-dir . && dbt test --profiles-dir .

# 3. Train models
python -m models.autoencoder.train_autoencoder
python -m models.xgboost.train_xgboost
python -m models.champion_challenger

# 4. Serve
uvicorn api.main:app --reload

# 5. Dashboard
streamlit run streamlit_app.py

# 6. Agent (build the fraud-case retrieval index)
python agents/fraud_agent.py --build-index
```

## Run tests
```bash
pytest tests/ -v
```

## Project structure

```
crs/
├── ingestion/          # Transaction CSV loader → DuckDB bronze
├── dbt_project/        # Bronze/silver/gold dbt models + tests
│   └── models/
│       ├── bronze/     # stg_transactions — staging view
│       ├── silver/     # int_transaction_features — amount/time feature engineering
│       └── gold/       # fct_fraud_features (ML-ready mart)
├── models/             # PyTorch autoencoder + XGBoost champion/challenger (Layer 3)
├── api/                # FastAPI scoring endpoint (Layer 4)
├── agents/              # LangChain fraud agent (Layer 6)
├── docker/              # Dockerfile + compose
├── k8s/                # Minikube deployment manifests
├── tests/              # Unit + integration tests
├── scripts/             # Pipeline runner
└── streamlit_app.py    # Dashboard (Layer 5)
```

## Layers (build order)

| Layer | Status | Description |
|-------|--------|-------------|
| 1. Ingestion | ✅ Complete | Real 284,807-row transaction CSV → DuckDB bronze |
| 2. Transform | ✅ Complete | dbt silver/gold fraud feature mart |
| 3. Models | ✅ Complete | PyTorch autoencoder + XGBoost champion/challenger, trained on real fraud labels |
| 4. API | ✅ Complete | FastAPI scoring endpoint + SHAP explainability |
| 5. Dashboard | ✅ Complete | Streamlit + Plotly transaction scoring UI |
| 6. Agent | ✅ Complete | LangChain agent with ChromaDB fraud-case retrieval |
| 7. Infra | ✅ Complete | Docker + Kubernetes (Minikube) |

## Models

- **XGBoost** — supervised classifier trained directly on the real `Class` fraud labels. CV AUC ≈ 0.98, ROC-AUC ≈ 1.00 on the full training set.
- **Autoencoder** — unsupervised reconstruction-error anomaly detector (flags the top 5% by error), trained without labels; evaluated against the real fraud labels for comparison (AUC ≈ 0.93).
- **Champion/challenger** — the two models are compared on AUC/F1 against ground-truth fraud labels; the current champion is tracked in `models/model_registry.json` and served by the API.

## Links
- [Live API Docs](http://localhost:8000/docs)

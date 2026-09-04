# Clinical Risk Scoring System

Production clinical AI pipeline: FHIR R4 ingestion → dbt medallion → PyTorch/XGBoost champion-challenger → FastAPI + Docker + Kubernetes → LangChain agent.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Data

Download Synthea FHIR R4 sample data and unzip into `data/raw/fhir/`:
```
https://synthetichealth.github.io/synthea-sample-data/downloads/synthea_sample_data_fhir_r4_sep2019.zip
```

## Run the pipeline

```bash
# Ingest + transform (limit=50 for dev)
python scripts/pipeline.py data/raw/fhir --limit 50

# Or step by step:
python -m ingestion.bronze_loader data/raw/fhir --limit 50
cd dbt_project && dbt run --profiles-dir . && dbt test --profiles-dir .
```

## Run tests
```bash
pytest tests/ -v
```

## Project structure

```
crs/
├── ingestion/          # FHIR parser + bronze DuckDB loader
├── dbt_project/        # Bronze/silver/gold dbt models + tests
│   └── models/
│       ├── bronze/     # Staging views
│       ├── silver/     # Condition flags, vitals, encounters
│       └── gold/       # fct_patient_risk_features (ML-ready)
├── models/             # PyTorch autoencoder + XGBoost (Layer 3)
├── api/                # FastAPI scoring endpoint (Layer 4)
├── agents/             # LangChain clinical agent (Layer 5)
├── docker/             # Dockerfile + compose
├── k8s/                # Minikube deployment manifests
├── tests/              # Unit + integration tests
└── scripts/            # Pipeline runner
```

## Layers (build order)

| Layer | Status | Description |
|-------|--------|-------------|
| 1. Ingestion | ✅ Complete | FHIR R4 parser → DuckDB bronze |
| 2. Transform | ✅ Complete | dbt silver/gold feature mart |
| 3. Models | 🔲 Next | PyTorch autoencoder + XGBoost champion/challenger |
| 4. API | 🔲 | FastAPI scoring endpoint + SHAP |
| 5. Infra | 🔲 | Docker + Kubernetes (Minikube) |
| 6. Agent | 🔲 | LangChain clinical agent |

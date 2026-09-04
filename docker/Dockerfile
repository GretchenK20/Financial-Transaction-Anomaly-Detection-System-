FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies (explicit list — not requirements.txt, which is the
# slim Streamlit Community Cloud file at repo root; see requirements-full.txt)
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    torch \
    xgboost==2.0.3 \
    scikit-learn \
    shap==0.45.1 \
    numpy"<2" \
    pandas \
    duckdb \
    mlflow \
    joblib \
    loguru \
    python-dotenv

# Copy application code
COPY config.py .
COPY api/ ./api/
COPY models/ ./models/
COPY data/ ./data/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

ENV MLFLOW_TRACKING_URI=sqlite:///mlruns.db
# Points at the small, committed demo DB (all 492 real frauds + a sampled
# legit subset) rather than the full 242MB local database, which isn't
# committed to git. Override via a real DUCKDB_PATH if the full DB is
# mounted/available in your deployment.
ENV DUCKDB_PATH=data/demo_financial_anomaly.duckdb

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

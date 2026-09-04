from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

FHIR_RAW_DIR = DATA_DIR / "raw" / "fhir"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"

DUCKDB_PATH = BASE_DIR / os.getenv("DUCKDB_PATH", "data/clinical_risk.duckdb")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "clinical-risk-scoring")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))

ANOMALY_THRESHOLD = 0.17
CHAMPION_MIN_IMPROVEMENT = 0.02

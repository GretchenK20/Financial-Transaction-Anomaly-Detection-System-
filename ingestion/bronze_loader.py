"""
Bronze layer: parse all FHIR bundles and load into DuckDB raw tables.
No transformations — data lands exactly as parsed from FHIR.
"""
import duckdb
import pandas as pd
from pathlib import Path
from loguru import logger
from typing import Optional
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import FHIR_RAW_DIR, DUCKDB_PATH
from ingestion.fhir_parser import parse_bundle


DDL = {
    "bronze_patients": """
        CREATE TABLE IF NOT EXISTS bronze_patients (
            patient_id      VARCHAR PRIMARY KEY,
            given_name      VARCHAR,
            family_name     VARCHAR,
            birth_date      VARCHAR,
            age_years       INTEGER,
            gender          VARCHAR,
            race            VARCHAR,
            ethnicity       VARCHAR,
            marital_status  VARCHAR,
            city            VARCHAR,
            state           VARCHAR,
            postal_code     VARCHAR,
            deceased        BOOLEAN,
            ingested_at     TIMESTAMP DEFAULT current_timestamp
        )
    """,
    "bronze_conditions": """
        CREATE TABLE IF NOT EXISTS bronze_conditions (
            condition_id    VARCHAR PRIMARY KEY,
            patient_id      VARCHAR,
            code            VARCHAR,
            display         VARCHAR,
            system          VARCHAR,
            onset_date      DATE,
            abatement_date  DATE,
            clinical_status VARCHAR,
            category        VARCHAR,
            ingested_at     TIMESTAMP DEFAULT current_timestamp
        )
    """,
    "bronze_observations": """
        CREATE TABLE IF NOT EXISTS bronze_observations (
            observation_id  VARCHAR PRIMARY KEY,
            patient_id      VARCHAR,
            code            VARCHAR,
            display         VARCHAR,
            system          VARCHAR,
            effective_date  DATE,
            status          VARCHAR,
            value_numeric   DOUBLE,
            value_unit      VARCHAR,
            value_text      VARCHAR,
            category        VARCHAR,
            ingested_at     TIMESTAMP DEFAULT current_timestamp
        )
    """,
    "bronze_encounters": """
        CREATE TABLE IF NOT EXISTS bronze_encounters (
            encounter_id    VARCHAR PRIMARY KEY,
            patient_id      VARCHAR,
            type_code       VARCHAR,
            type_display    VARCHAR,
            class           VARCHAR,
            status          VARCHAR,
            start_date      DATE,
            end_date        DATE,
            ingested_at     TIMESTAMP DEFAULT current_timestamp
        )
    """,
    "bronze_medication_requests": """
        CREATE TABLE IF NOT EXISTS bronze_medication_requests (
            medication_id   VARCHAR PRIMARY KEY,
            patient_id      VARCHAR,
            code            VARCHAR,
            display         VARCHAR,
            status          VARCHAR,
            authored_date   DATE,
            intent          VARCHAR,
            ingested_at     TIMESTAMP DEFAULT current_timestamp
        )
    """,
}

# Maps parser key → table name, id column, and data columns (excludes ingested_at)
TABLE_META = {
    "patients": {
        "table": "bronze_patients",
        "id_col": "patient_id",
        "cols": ["patient_id","given_name","family_name","birth_date","age_years",
                 "gender","race","ethnicity","marital_status","city","state",
                 "postal_code","deceased"],
    },
    "conditions": {
        "table": "bronze_conditions",
        "id_col": "condition_id",
        "cols": ["condition_id","patient_id","code","display","system",
                 "onset_date","abatement_date","clinical_status","category"],
    },
    "observations": {
        "table": "bronze_observations",
        "id_col": "observation_id",
        "cols": ["observation_id","patient_id","code","display","system",
                 "effective_date","status","value_numeric","value_unit",
                 "value_text","category"],
    },
    "encounters": {
        "table": "bronze_encounters",
        "id_col": "encounter_id",
        "cols": ["encounter_id","patient_id","type_code","type_display",
                 "class","status","start_date","end_date"],
    },
    "medication_requests": {
        "table": "bronze_medication_requests",
        "id_col": "medication_id",
        "cols": ["medication_id","patient_id","code","display","status",
                 "authored_date","intent"],
    },
}


def init_bronze_tables(conn: duckdb.DuckDBPyConnection) -> None:
    for ddl in DDL.values():
        conn.execute(ddl)
    logger.info("Bronze tables initialized")


def load_bundles(
    fhir_dir: Optional[Path] = None,
    db_path: Optional[Path] = None,
    limit: Optional[int] = None,
) -> dict[str, int]:
    fhir_dir = fhir_dir or FHIR_RAW_DIR
    db_path = db_path or DUCKDB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)

    bundle_files = sorted(fhir_dir.glob("*.json"))
    if not bundle_files:
        raise FileNotFoundError(f"No FHIR JSON files found in {fhir_dir}")
    if limit:
        bundle_files = bundle_files[:limit]

    logger.info(f"Loading {len(bundle_files)} bundles from {fhir_dir}")

    accumulated: dict[str, list] = {k: [] for k in TABLE_META}
    for bundle_path in bundle_files:
        try:
            parsed = parse_bundle(bundle_path)
            for key in TABLE_META:
                accumulated[key].extend(parsed.get(key, []))
        except Exception as e:
            logger.error(f"Failed parsing {bundle_path.name}: {e}")

    counts = {}
    with duckdb.connect(str(db_path)) as conn:
        init_bronze_tables(conn)

        for key, meta in TABLE_META.items():
            table = meta["table"]
            id_col = meta["id_col"]
            cols = meta["cols"]

            records = accumulated[key]
            if not records:
                counts[table] = 0
                continue

            df = pd.DataFrame(records)[cols]
            df = df.drop_duplicates(subset=[id_col])

            existing_ids = conn.execute(
                f"SELECT {id_col} FROM {table}"
            ).fetchdf()[id_col].tolist()
            if existing_ids:
                df = df[~df[id_col].isin(existing_ids)]

            if not df.empty:
                col_list = ", ".join(cols)
                conn.execute(
                    f"INSERT INTO {table} ({col_list}) SELECT {col_list} FROM df"
                )

            total = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            counts[table] = total
            logger.info(f"  {table}: {total:,} rows")

    return counts


if __name__ == "__main__":
    import typer

    def main(
        fhir_dir: Path = typer.Argument(FHIR_RAW_DIR),
        db_path: Path = typer.Option(DUCKDB_PATH),
        limit: Optional[int] = typer.Option(None),
    ):
        counts = load_bundles(fhir_dir, db_path, limit)
        for table, count in counts.items():
            typer.echo(f"{table}: {count:,} rows")

    typer.run(main)

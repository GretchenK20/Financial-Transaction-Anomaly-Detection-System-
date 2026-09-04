"""
Bronze layer: load the raw credit card transaction CSV into DuckDB.
No transformations — data lands exactly as read from the source file,
aside from assigning a stable transaction_id and lower-casing columns.
"""
import duckdb
from pathlib import Path
from loguru import logger
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TRANSACTIONS_RAW_DIR, DUCKDB_PATH

V_COLUMNS = [f"V{i}" for i in range(1, 29)]

EXPECTED_COLUMNS = {"Time", *V_COLUMNS, "Amount", "Class"}
EXPECTED_ROW_COUNT = 284_807


def load_transactions(
    csv_path: Optional[Path] = None,
    db_path: Optional[Path] = None,
) -> int:
    """
    Load creditcard.csv into DuckDB as bronze_transactions.
    Returns the row count of the loaded table.
    """
    csv_path = csv_path or (TRANSACTIONS_RAW_DIR / "creditcard.csv")
    db_path = db_path or DUCKDB_PATH

    if not csv_path.exists():
        raise FileNotFoundError(f"Transaction CSV not found at {csv_path}")

    db_path.parent.mkdir(parents=True, exist_ok=True)

    v_select = ",\n                ".join(f'"V{i}" AS v{i}' for i in range(1, 29))

    with duckdb.connect(str(db_path)) as conn:
        conn.execute(
            f"""
            CREATE OR REPLACE TABLE bronze_transactions AS
            SELECT
                (row_number() OVER () - 1)::BIGINT AS transaction_id,
                "Time"::DOUBLE AS time,
                {v_select},
                "Amount"::DOUBLE AS amount,
                "Class"::INTEGER AS class,
                current_timestamp AS ingested_at
            FROM read_csv_auto(?, header=true)
            """,
            [str(csv_path)],
        )
        count = conn.execute("SELECT COUNT(*) FROM bronze_transactions").fetchone()[0]

    logger.info(f"bronze_transactions: {count:,} rows loaded from {csv_path}")
    return count


def validate_load(db_path: Optional[Path] = None) -> dict:
    """
    Validate the loaded bronze_transactions table:
      - row count matches the known dataset size
      - all expected source columns are present (case-insensitive)
    """
    db_path = db_path or DUCKDB_PATH

    with duckdb.connect(str(db_path), read_only=True) as conn:
        row_count = conn.execute("SELECT COUNT(*) FROM bronze_transactions").fetchone()[0]
        columns = {
            row[0]
            for row in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'bronze_transactions'"
            ).fetchall()
        }

    lower_expected = {c.lower() for c in EXPECTED_COLUMNS}
    missing = lower_expected - columns

    result = {
        "row_count": row_count,
        "row_count_ok": row_count == EXPECTED_ROW_COUNT,
        "missing_columns": sorted(missing),
        "columns_ok": not missing,
    }

    if not result["row_count_ok"]:
        logger.error(
            f"Row count mismatch: expected {EXPECTED_ROW_COUNT:,}, got {row_count:,}"
        )
    if missing:
        logger.error(f"Missing expected columns: {sorted(missing)}")
    if result["row_count_ok"] and result["columns_ok"]:
        logger.info("Validation passed: row count and schema match expectations")

    return result


if __name__ == "__main__":
    import typer

    def main(
        csv_path: Path = typer.Argument(TRANSACTIONS_RAW_DIR / "creditcard.csv"),
        db_path: Path = typer.Option(DUCKDB_PATH),
    ):
        count = load_transactions(csv_path, db_path)
        typer.echo(f"bronze_transactions: {count:,} rows")

        result = validate_load(db_path)
        typer.echo(f"Row count OK: {result['row_count_ok']}")
        typer.echo(f"Columns OK: {result['columns_ok']}")
        if not (result["row_count_ok"] and result["columns_ok"]):
            raise typer.Exit(code=1)

    typer.run(main)

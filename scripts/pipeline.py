"""
Main pipeline runner — executes ingestion → dbt → model training in sequence.
Run: python scripts/pipeline.py
"""
import subprocess
import sys
from pathlib import Path
import typer

sys.path.insert(0, str(Path(__file__).parent.parent))
from ingestion.transaction_loader import load_transactions, validate_load
from config import TRANSACTIONS_RAW_DIR, DUCKDB_PATH

app = typer.Typer()


def run_dbt(duckdb_path: Path) -> None:
    result = subprocess.run(
        ["dbt", "run", "--profiles-dir", "."],
        cwd=Path(__file__).parent.parent / "dbt_project",
        env={**__import__("os").environ, "DUCKDB_PATH": str(duckdb_path)},
        capture_output=False,
    )
    if result.returncode != 0:
        raise RuntimeError("dbt run failed")

    result = subprocess.run(
        ["dbt", "test", "--profiles-dir", "."],
        cwd=Path(__file__).parent.parent / "dbt_project",
        env={**__import__("os").environ, "DUCKDB_PATH": str(duckdb_path)},
        capture_output=False,
    )
    if result.returncode != 0:
        raise RuntimeError("dbt test failed")


@app.command()
def main(
    csv_path: Path = typer.Argument(TRANSACTIONS_RAW_DIR / "creditcard.csv", help="Path to creditcard.csv"),
    db_path: Path = typer.Option(DUCKDB_PATH),
    skip_dbt: bool = typer.Option(False),
    skip_training: bool = typer.Option(False),
):
    typer.echo("── Step 1: Bronze ingestion ──")
    count = load_transactions(csv_path=csv_path, db_path=db_path)
    typer.echo(f"  bronze_transactions: {count:,} rows")
    result = validate_load(db_path=db_path)
    if not (result["row_count_ok"] and result["columns_ok"]):
        raise RuntimeError(f"Ingestion validation failed: {result}")

    if not skip_dbt:
        typer.echo("── Step 2: dbt transform ──")
        run_dbt(db_path)

    if not skip_training:
        typer.echo("── Step 3: Model training ──")
        typer.echo("  Run: python -m models.autoencoder.train_autoencoder")
        typer.echo("  Run: python -m models.xgboost.train_xgboost")
        typer.echo("  Run: python -m models.champion_challenger")


if __name__ == "__main__":
    app()

"""
Main pipeline runner — executes ingestion → dbt → model training in sequence.
Run: python scripts/pipeline.py --fhir-dir data/raw/fhir
"""
import subprocess
import sys
from pathlib import Path
import typer

sys.path.insert(0, str(Path(__file__).parent.parent))
from ingestion.bronze_loader import load_bundles
from config import FHIR_RAW_DIR, DUCKDB_PATH

app = typer.App()


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
    fhir_dir: Path = typer.Argument(FHIR_RAW_DIR, help="Path to FHIR JSON files"),
    db_path: Path = typer.Option(DUCKDB_PATH),
    limit: int = typer.Option(None, help="Limit bundles (for dev)"),
    skip_dbt: bool = typer.Option(False),
    skip_training: bool = typer.Option(False),
):
    typer.echo("── Step 1: Bronze ingestion ──")
    counts = load_bundles(fhir_dir=fhir_dir, db_path=db_path, limit=limit)
    for table, count in counts.items():
        typer.echo(f"  {table}: {count:,} rows")

    if not skip_dbt:
        typer.echo("── Step 2: dbt transform ──")
        run_dbt(db_path)

    if not skip_training:
        typer.echo("── Step 3: Model training ──")
        typer.echo("  Run: python models/train.py")


if __name__ == "__main__":
    app()

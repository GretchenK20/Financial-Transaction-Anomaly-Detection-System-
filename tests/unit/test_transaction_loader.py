"""Unit tests for the transaction loader — runs against a small synthetic CSV."""
import duckdb
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ingestion.transaction_loader import load_transactions, validate_load


SYNTHETIC_CSV_HEADER = "Time," + ",".join(f"V{i}" for i in range(1, 29)) + ",Amount,Class"


def _synthetic_row(time, amount, cls):
    values = [str(time)] + ["0.1"] * 28 + [str(amount), str(cls)]
    return ",".join(values)


@pytest.fixture
def csv_file(tmp_path):
    rows = [SYNTHETIC_CSV_HEADER]
    for i in range(10):
        rows.append(_synthetic_row(time=i * 60, amount=10.0 + i, cls=1 if i == 0 else 0))
    p = tmp_path / "creditcard.csv"
    p.write_text("\n".join(rows))
    return p


def test_load_transactions_row_count(tmp_path, csv_file):
    db_path = tmp_path / "test.duckdb"
    count = load_transactions(csv_path=csv_file, db_path=db_path)
    assert count == 10


def test_load_transactions_columns(tmp_path, csv_file):
    db_path = tmp_path / "test.duckdb"
    load_transactions(csv_path=csv_file, db_path=db_path)
    with duckdb.connect(str(db_path)) as conn:
        cols = {
            r[0] for r in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'bronze_transactions'"
            ).fetchall()
        }
    assert {"transaction_id", "time", "v1", "v28", "amount", "class"} <= cols


def test_load_transactions_id_is_sequential(tmp_path, csv_file):
    db_path = tmp_path / "test.duckdb"
    load_transactions(csv_path=csv_file, db_path=db_path)
    with duckdb.connect(str(db_path)) as conn:
        ids = conn.execute(
            "SELECT transaction_id FROM bronze_transactions ORDER BY transaction_id"
        ).fetchdf()["transaction_id"].tolist()
    assert ids == list(range(10))


def test_validate_load_detects_wrong_row_count(tmp_path, csv_file):
    db_path = tmp_path / "test.duckdb"
    load_transactions(csv_path=csv_file, db_path=db_path)
    result = validate_load(db_path=db_path)
    assert result["row_count"] == 10
    assert result["row_count_ok"] is False  # doesn't match the real dataset's 284,807
    assert result["columns_ok"] is True


def test_missing_csv_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_transactions(csv_path=tmp_path / "nope.csv", db_path=tmp_path / "test.duckdb")

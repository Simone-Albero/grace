"""Stage clean on a dataset split over several CSVs."""

import sqlite3

import polars as pl
import pytest

from tests.conftest import make_workspace


def _split(rows: pl.DataFrame, folder):
    """Write rows over three CSVs that differ as the CIC-IDS2018 days do.

    part_0 holds the columns as generated; part_1 has them in reverse
    order and an extra one; part_2 repeats its header twice in the
    middle. Returns the glob of the three.
    """
    folder.mkdir()
    thirds = [rows.slice(0, 4000), rows.slice(4000, 4000), rows.slice(8000)]
    thirds[0].write_csv(folder / "part_0.csv")
    reversed_cols = thirds[1].select(reversed(rows.columns))
    reversed_cols.with_columns(pl.lit("x").alias("extra")).write_csv(
        folder / "part_1.csv"
    )
    lines = thirds[2].write_csv().splitlines()
    header = lines[0]
    lines = [*lines[:1001], header, *lines[1001:2001], header, *lines[2001:]]
    (folder / "part_2.csv").write_text("\n".join(lines) + "\n")
    return folder / "part_*.csv"


def test_clean_reads_a_glob_as_the_concatenation_of_its_files(
    tmp_path, synthetic_csv, done
):
    rows = pl.read_csv(synthetic_csv, infer_schema=False)
    pattern = _split(rows, tmp_path / "raw")
    workspace = make_workspace(tmp_path, pattern)
    workspace.run("clean")

    split = pl.read_parquet(workspace.output_dir / "data.parquet")
    whole = pl.read_parquet(done.output_dir / "data.parquet")
    assert split.equals(whole)
    for name in ("classes.json", "unparsable.parquet"):
        assert (workspace.output_dir / name).read_bytes() == (
            done.output_dir / name
        ).read_bytes()
    with sqlite3.connect(workspace.db_path) as conn:
        sources = conn.execute("SELECT file, n_read, n_header FROM sources").fetchall()
        (n_raw,) = conn.execute("SELECT n_raw FROM clean_summary").fetchone()
    assert sources == [
        ("part_0.csv", 4000, 0),
        ("part_1.csv", 4000, 0),
        ("part_2.csv", rows.height - 8000, 2),
    ]
    assert n_raw == rows.height


def test_clean_names_the_file_that_lacks_a_column(tmp_path, synthetic_csv):
    rows = pl.read_csv(synthetic_csv, infer_schema=False)
    folder = tmp_path / "raw"
    folder.mkdir()
    rows.write_csv(folder / "a.csv")
    rows.drop("num_1").write_csv(folder / "b.csv")
    workspace = make_workspace(tmp_path, folder / "*.csv")
    with pytest.raises(ValueError, match=r"missing from .*b\.csv: \['num_1'\]"):
        workspace.run("clean")


def test_clean_fails_when_the_glob_matches_no_file(tmp_path, synthetic_csv):
    workspace = make_workspace(tmp_path, tmp_path / "nothing_*.csv")
    with pytest.raises(FileNotFoundError, match="no CSV matches"):
        workspace.run("clean")

"""Every stage end to end on the small synthetic dataset."""

import json
import sqlite3

import numpy as np
import polars as pl
import pytest
from conftest import N_DUPLICATES, STAGE_ORDER, Workspace, make_workspace

from src.complexity.measure import MEASURES
from src.data import DUPLICATES, LABEL


def _rows(workspace: Workspace, sql: str) -> list[tuple]:
    with sqlite3.connect(workspace.db_path) as conn:
        return conn.execute(sql).fetchall()


def _read(workspace: Workspace, name: str) -> pl.DataFrame:
    return pl.read_parquet(workspace.output_dir / name)


def test_every_run_is_ok(done):
    runs = _rows(done, "SELECT stage, status FROM runs ORDER BY id")
    assert [r[0] for r in runs] == list(STAGE_ORDER)
    assert {r[1] for r in runs} == {"ok"}


def test_clean_drops_the_rare_class_and_the_constant_column_only(done):
    classes = json.loads((done.output_dir / "classes.json").read_text())
    assert "class_10" not in classes and len(classes) == 9
    dropped = _rows(done, "SELECT label FROM dropped_classes")
    assert dropped == [("class_10",)]
    assert _rows(done, "SELECT column FROM dropped_columns") == [("flat",)]
    data = _read(done, "data.parquet")
    n_raw, n_duplicate, n_rare, n_kept = _rows(
        done, "SELECT n_raw, n_duplicate, n_rare, n_kept FROM clean_summary"
    )[0]
    assert n_duplicate == N_DUPLICATES
    assert data.height == n_kept == n_raw - n_duplicate - n_rare
    counts = dict(_rows(done, "SELECT label, n_rows FROM class_counts"))
    per_code = dict(data[LABEL].value_counts().iter_rows())
    assert counts == {name: per_code[code] for code, name in enumerate(classes)}
    (n_read,) = _rows(done, "SELECT sum(n_read) FROM class_counts")[0]
    assert data[DUPLICATES].sum() == n_read
    numbers = data.select(pl.selectors.starts_with("num_"))
    assert numbers.select(pl.any_horizontal(pl.all().is_infinite().any())).item()
    assert numbers.null_count().sum_horizontal().item() > 0
    assert "flat" not in data.columns
    assert data["cat_1"].dtype == pl.String
    assert data.columns[-1] == LABEL
    assert set(data[LABEL].unique().to_list()) == set(range(9))


def test_clean_keeps_one_copy_of_each_row_and_counts_them(done):
    data = _read(done, "data.parquet")
    assert data[DUPLICATES].dtype == pl.Int64
    assert set(data[DUPLICATES].unique().to_list()) == {1, 2, 3}
    features = data.drop(DUPLICATES, LABEL)
    assert not data.drop(DUPLICATES).is_duplicated().any()
    # Row 51 under another class: same features, two rows.
    assert features.is_duplicated().sum() == 2
    (kind,) = _rows(done, f"SELECT kind FROM features WHERE column = '{DUPLICATES}'")[0]
    assert kind == "num"


def test_assignments_cover_every_row_and_keep_its_class(done):
    assignments = _read(done, "assignments.parquet").sort("row")
    regions = _read(done, "regions.parquet")
    data = _read(done, "data.parquet")
    assert assignments["row"].to_list() == list(range(data.height))
    region_class = dict(zip(regions["region"].to_list(), regions["class"].to_list()))
    labels = data[LABEL].to_list()
    assert [region_class[r] for r in assignments["region"].to_list()] == labels
    assert regions["n_rows"].sum() == data.height


def test_complexity_has_every_measure_of_every_population(done):
    table = _read(done, "complexity.parquet")
    regions = _read(done, "regions.parquet")
    region_rows = table.filter(pl.col("scope") == "region")
    assert region_rows.height == regions.height * len(MEASURES)
    assert table.filter(pl.col("scope") == "class").height == 9 * len(MEASURES)


def test_complexity_refuses_regions_of_other_data(copy_of_done):
    data_file = copy_of_done.output_dir / "data.parquet"
    data = pl.read_parquet(data_file)
    shuffled = np.random.default_rng(0).permutation(data[LABEL].to_numpy())
    data.with_columns(pl.Series(LABEL, shuffled)).write_parquet(data_file)
    with pytest.raises(ValueError, match="run regions again"):
        copy_of_done.run("complexity")


def test_a_stage_without_its_input_fails_and_is_logged(tmp_path, synthetic_csv):
    workspace = make_workspace(tmp_path, synthetic_csv)
    with pytest.raises(FileNotFoundError, match="run clean first"):
        workspace.run("regions")
    assert _rows(workspace, "SELECT stage, status FROM runs") == [("regions", "failed")]


def test_clean_refuses_rows_without_a_label(tmp_path, synthetic_csv):
    csv = tmp_path / "unlabelled.csv"
    rows = pl.read_csv(synthetic_csv, infer_schema=False)
    unlabelled = pl.when(pl.int_range(pl.len()) < 2).then(None).otherwise("label")
    rows.with_columns(unlabelled.alias("label")).write_csv(csv)
    workspace = make_workspace(tmp_path, csv)
    with pytest.raises(ValueError, match="2 rows .* have no label"):
        workspace.run("clean")

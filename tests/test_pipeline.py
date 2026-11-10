"""Every stage end to end on the small synthetic dataset."""

import json
import sqlite3

import numpy as np
import polars as pl
import pytest
from conftest import N_DUPLICATES, STAGE_ORDER, Workspace, make_workspace

from src.complexity.measure import MEASURES
from src.config import load_config
from src.data import DUPLICATES, LABEL
from src.prepare import MISSING, OTHER
from stages.paths import read_points


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


def test_prepare_has_one_finite_row_per_data_row(done):
    data = _read(done, "data.parquet")
    prepared = _read(done, "prepared.parquet")
    assert prepared.height == data.height
    assert prepared[LABEL].to_list() == data[LABEL].to_list()
    features = prepared.drop(LABEL)
    assert set(features.dtypes) == {pl.Float64}
    assert np.isfinite(features.to_numpy()).all()
    numbers = [c for c in data.columns if c.startswith("num_")] + [DUPLICATES]
    assert features.columns[: len(numbers) + 1] == [*numbers, MISSING]
    numeric = data.select(numbers).to_numpy().astype(float)
    n_missing = (~np.isfinite(numeric)).sum(axis=1)
    assert features[MISSING].to_list() == n_missing.astype(float).tolist()
    assert n_missing.sum() > 0
    slots = features.columns[len(numbers) + 1 :]
    assert {slot.split("=")[0] for slot in slots} == {"cat_1", "cat_2"}
    # Every row sets exactly one slot of each categorical column, all to
    # the column's weight.
    for column in ("cat_1", "cat_2"):
        block = features.select(pl.selectors.starts_with(f"{column}=")).to_numpy()
        assert (block > 0).sum(axis=1).tolist() == [1] * data.height
        assert len(np.unique(block[block > 0])) == 1


def test_prepare_logs_every_column(done):
    prepared = _read(done, "prepared.parquet").drop(LABEL)
    logged = _rows(
        done,
        "SELECT column, source, kind, mean, std, value, n_rows, weight "
        "FROM prepared_columns",
    )
    assert [row[0] for row in logged] == prepared.columns
    height = prepared.height
    for column, source, kind, mean, std, value, n_rows, weight in logged:
        if kind == "num":
            assert column == source and std is not None and mean is not None
        elif kind == "missing":
            assert column == source == MISSING
            assert n_rows == int((prepared[MISSING] > 0).sum())
        else:
            assert column == f"{source}={value}"
            assert n_rows == int((prepared[column] > 0).sum())
            assert prepared[column].max() == pytest.approx(weight)
    for source in ("cat_1", "cat_2"):
        counts = [r[6] for r in logged if r[1] == source]
        assert sum(counts) == height
    others = {row[1] for row in logged if row[5] == OTHER}
    assert others == {"cat_2"}
    n_rows, n_num, n_cat, n_columns, n_missing = _rows(
        done,
        "SELECT n_rows, n_num, n_cat, n_columns, n_missing FROM prepare_summary",
    )[0]
    assert (n_rows, n_cat, n_columns) == (height, 2, len(prepared.columns))
    assert n_missing == prepared[MISSING].sum()
    assert n_num == sum(row[2] == "num" for row in logged)


def test_the_points_leave_missing_out(done):
    cfg = load_config(str(done.dataset), method=str(done.method))
    points, labels = read_points(cfg)
    prepared = _read(done, "prepared.parquet")
    without = prepared.drop(MISSING, LABEL)
    assert points.shape == (prepared.height, without.width)
    np.testing.assert_array_equal(points, without.to_numpy())
    assert labels.tolist() == prepared[LABEL].to_list()


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
    prepared_file = copy_of_done.output_dir / "prepared.parquet"
    prepared = pl.read_parquet(prepared_file)
    shuffled = np.random.default_rng(0).permutation(prepared[LABEL].to_numpy())
    prepared.with_columns(pl.Series(LABEL, shuffled)).write_parquet(prepared_file)
    with pytest.raises(ValueError, match="run regions again"):
        copy_of_done.run("complexity")


def test_a_stage_without_its_input_fails_and_is_logged(tmp_path, synthetic_csv):
    workspace = make_workspace(tmp_path, synthetic_csv)
    with pytest.raises(FileNotFoundError, match="run clean first"):
        workspace.run("prepare")
    workspace.run("clean")
    with pytest.raises(FileNotFoundError, match="run prepare first"):
        workspace.run("regions")
    assert _rows(workspace, "SELECT stage, status FROM runs") == [
        ("prepare", "failed"),
        ("clean", "ok"),
        ("regions", "failed"),
    ]


def test_clean_refuses_rows_without_a_label(tmp_path, synthetic_csv):
    csv = tmp_path / "unlabelled.csv"
    rows = pl.read_csv(synthetic_csv, infer_schema=False)
    unlabelled = pl.when(pl.int_range(pl.len()) < 2).then(None).otherwise("label")
    rows.with_columns(unlabelled.alias("label")).write_csv(csv)
    workspace = make_workspace(tmp_path, csv)
    with pytest.raises(ValueError, match="2 rows .* have no label"):
        workspace.run("clean")

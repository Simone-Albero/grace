"""Every stage end to end on the small synthetic dataset."""

import json
import sqlite3

import numpy as np
import polars as pl
import pytest
from conftest import MODEL, STAGE_ORDER, Workspace, make_workspace

from main import main
from src.complexity.measure import MEASURES
from src.prepare import LABEL


def _rows(workspace: Workspace, sql: str) -> list[tuple]:
    with sqlite3.connect(workspace.db_path) as conn:
        return conn.execute(sql).fetchall()


def _read(workspace: Workspace, name: str) -> pl.DataFrame:
    return pl.read_parquet(workspace.output_dir / name)


def test_every_run_is_ok(done):
    runs = _rows(done, "SELECT stage, status, model FROM runs ORDER BY id")
    assert [r[0] for r in runs] == list(STAGE_ORDER)
    assert {r[1] for r in runs} == {"ok"}
    assert [r[2] for r in runs] == [None] * 3 + [MODEL] * 3


def test_split_drops_the_rare_class_and_prepares_every_split(done):
    classes = json.loads((done.output_dir / "classes.json").read_text())
    assert "class_10" not in classes and len(classes) == 9
    dropped = _rows(done, "SELECT label FROM dropped_classes")
    assert dropped == [("class_10",)]
    sizes = {
        name: _read(done, f"{name}.parquet").height for name in ("train", "val", "test")
    }
    (n_kept,) = _rows(done, "SELECT n_kept FROM split_summary")[0]
    assert sum(sizes.values()) == n_kept
    train = _read(done, "train.parquet")
    assert train.columns[-1] == LABEL
    assert set(train[LABEL].unique().to_list()) == set(range(9))


def test_assignments_cover_every_row_and_train_keeps_its_class(done):
    assignments = _read(done, "assignments.parquet")
    regions = _read(done, "regions.parquet")
    for name in ("train", "val", "test"):
        rows = assignments.filter(pl.col("split") == name).sort("row")
        assert rows["row"].to_list() == list(
            range(_read(done, f"{name}.parquet").height)
        )
        assert set(rows["region"].to_list()) <= set(regions["region"].to_list())
    train = assignments.filter(pl.col("split") == "train").sort("row")
    region_class = dict(zip(regions["region"].to_list(), regions["class"].to_list()))
    labels = _read(done, "train.parquet")[LABEL].to_list()
    assert [region_class[r] for r in train["region"].to_list()] == labels
    assert regions["n_test"].sum() == _read(done, "test.parquet").height


def test_complexity_has_every_measure_of_every_population(done):
    table = _read(done, "complexity.parquet")
    regions = _read(done, "regions.parquet")
    region_rows = table.filter(pl.col("scope") == "region")
    assert region_rows.height == regions.height * len(MEASURES)
    assert table.filter(pl.col("scope") == "class").height == 9 * len(MEASURES)


def test_predictions_line_up_with_the_splits(done):
    for name in ("val", "test"):
        pred = _read(done, f"pred_{MODEL}_{name}.parquet")
        split = _read(done, f"{name}.parquet")
        assert pred["row"].to_list() == list(range(split.height))
        assert pred["label"].to_list() == split[LABEL].to_list()
        proba = pred.select(pl.selectors.starts_with("proba_")).to_numpy()
        assert proba.shape[1] == 9
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
        np.testing.assert_array_equal(proba.argmax(axis=1), pred["pred"].to_numpy())


def test_failures_count_every_test_row(done):
    failures = _read(done, f"failures_{MODEL}.parquet")
    regions = _read(done, "regions.parquet")
    assert failures["region"].to_list() == regions["region"].to_list()
    test = _read(done, f"pred_{MODEL}_test.parquet")
    assert failures["n_eval"].sum() == test.height
    assert failures["n_error"].sum() == (test["label"] != test["pred"]).sum()
    (n_used,) = _rows(done, "SELECT n_used FROM failure_stats")[0]
    assert n_used == failures["used"].sum() > 0


def test_regress_estimates_every_used_region(done):
    estimates = _read(done, f"regress_{MODEL}_estimates.parquet")
    failures = _read(done, f"failures_{MODEL}.parquet").filter(pl.col("used"))
    assert sorted(estimates["region"].to_list()) == sorted(failures["region"].to_list())
    contributions = _read(done, f"regress_{MODEL}_contributions.parquet")
    assert set(contributions["region"].to_list()) == set(estimates["region"].to_list())
    scores = dict(_rows(done, "SELECT predictor, spearman FROM regress_scores"))
    assert set(scores) == {"rf", "mcp", "atc"}
    assert scores["rf"] > 0


def test_regions_never_read_the_test_labels(copy_of_done):
    before = _read(copy_of_done, "assignments.parquet")
    test_file = copy_of_done.output_dir / "test.parquet"
    test = pl.read_parquet(test_file)
    shuffled = np.random.default_rng(0).permutation(test[LABEL].to_numpy())
    test.with_columns(pl.Series(LABEL, shuffled)).write_parquet(test_file)
    copy_of_done.run("regions")
    after = _read(copy_of_done, "assignments.parquet")
    assert before.equals(after)


def test_a_stage_without_its_input_fails_and_is_logged(tmp_path, synthetic_csv):
    workspace = make_workspace(tmp_path, synthetic_csv)
    with pytest.raises(FileNotFoundError, match="run split first"):
        workspace.run("train")
    assert _rows(workspace, "SELECT stage, status FROM runs") == [("train", "failed")]


def test_model_stages_need_a_model(done):
    with pytest.raises(SystemExit):
        main(["--stage", "train", "--dataset", str(done.dataset)])


def test_failures_refuse_predictions_of_another_split(copy_of_done):
    test_file = copy_of_done.output_dir / "test.parquet"
    test = pl.read_parquet(test_file)
    shuffled = np.random.default_rng(0).permutation(test[LABEL].to_numpy())
    test.with_columns(pl.Series(LABEL, shuffled)).write_parquet(test_file)
    with pytest.raises(ValueError, match="run train again"):
        copy_of_done.run("failures")


def test_regress_refuses_descriptors_of_other_regions(copy_of_done):
    complexity_file = copy_of_done.output_dir / "complexity.parquet"
    complexity = pl.read_parquet(complexity_file)
    complexity.filter(pl.col("id") != 0).write_parquet(complexity_file)
    with pytest.raises(ValueError, match="run complexity and failures again"):
        copy_of_done.run("regress")

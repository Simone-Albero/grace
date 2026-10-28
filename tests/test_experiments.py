"""dashboard/experiments.py and artifacts.py on toy data."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dashboard import artifacts, queries
from dashboard.experiments import (
    build,
    compare_rows,
    config_key,
    grouped_values,
    summarize,
)
from pipeline.runner import STAGES


def _run(run_id: int, seed: int, *, classifier: str = "rf", **extra) -> dict:
    """Build a run as `queries.runs` lists it."""
    config = {
        "groups": {"data": "d", "classifier": classifier},
        "seed": seed,
        "root": f"root{run_id}",
        "runtime": {"n_jobs": run_id},
        "figure_format": "pdf",
        **extra,
    }
    stages = [
        {"stage": "classify", "execution_id": 10 * seed, "cached": 0, "status": "done"}
    ]
    return {"id": run_id, "status": "done", "config": config, "stages": stages}


def test_stage_order_is_the_runners() -> None:
    """Check that the dashboard's stage order is the runner's."""
    assert list(queries.STAGE_ORDER) == list(STAGES)


def test_seeds_group_and_the_latest_run_represents_a_seed() -> None:
    """Check that runs differing by seed, root or runtime group."""
    runs = [_run(1, 1), _run(2, 2), _run(3, 1)]
    metrics = [
        (1, "classify", 10, "accuracy", 0.5),
        (2, "classify", 20, "accuracy", 0.7),
        (3, "classify", 30, "accuracy", 0.9),
    ]
    (experiment,) = build(runs, metrics)
    assert [seed.seed for seed in experiment.seeds] == [1, 2]
    first = experiment.seeds[0]
    assert first.run["id"] == 3 and [r["id"] for r in first.earlier] == [1]
    assert first.metrics["classify.accuracy"] == 0.9


def test_a_different_config_is_another_experiment_with_a_variant() -> None:
    """Check that a differing nested key shows as the variant."""
    runs = [_run(1, 1, loss={"gamma": 1}), _run(2, 1, loss={"gamma": 2})]
    experiments = build(runs, [])
    assert len(experiments) == 2
    assert {e.variant for e in experiments} == {"loss.gamma=1", "loss.gamma=2"}
    assert config_key(runs[0])[0] != config_key(runs[1])[0]
    (alone,) = build([_run(1, 1)], [])
    assert alone.variant == ""


def test_summarize_skips_missing_values() -> None:
    """Check the mean, the sample std and the count."""
    assert summarize([1.0, 3.0, None, float("nan")]) == (2.0, pytest.approx(2**0.5), 2)
    assert summarize([4.0]) == (4.0, None, 1)
    assert summarize([None]) == (None, None, 0)


def test_an_execution_counts_once() -> None:
    """Check that rows sharing the metric's execution count once."""
    rows = compare_rows(
        build(
            [_run(1, 1, loss={"gamma": 1}), _run(2, 1, loss={"gamma": 2})],
            [
                (1, "classify", 10, "accuracy", 0.5),
                (2, "classify", 10, "accuracy", 0.5),
            ],
        )
    )
    grouped = grouped_values(rows, "classify.accuracy", "classifier", None)
    assert grouped == {"": {"rf": [0.5]}}
    other = compare_rows(
        build(
            [_run(1, 1), _run(2, 2)],
            [
                (1, "classify", 10, "accuracy", 0.5),
                (2, "classify", 20, "accuracy", 0.7),
            ],
        )
    )
    assert grouped_values(other, "classify.accuracy", "classifier", "data") == {
        "d": {"rf": [0.5, 0.7]}
    }


def test_confusion_pairs_rows_by_position(tmp_path: Path) -> None:
    """Check that predictions sort by row before meeting the labels."""
    prepare, classify = tmp_path / "prepare", tmp_path / "classify"
    prepare.mkdir(), classify.mkdir()
    pd.DataFrame({"label": [0, 0, 1, 1]}).to_parquet(prepare / "test.parquet")
    predictions = pd.DataFrame(
        {
            "split": ["val", "test", "test", "test", "test"],
            "row": [0, 3, 1, 0, 2],
            "y_pred": [1, 0, 0, 0, 1],
        }
    )
    predictions.to_parquet(classify / "predictions.parquet")
    counts = artifacts.confusion(prepare, classify, 2)
    # Rows 0..3 predict 0, 0, 1, 0 against the labels 0, 0, 1, 1.
    assert np.array_equal(counts, [[2, 0], [1, 1]])
    assert artifacts.confusion(prepare, tmp_path / "missing", 2) is None
    pd.DataFrame({"label": [0, 1]}).to_parquet(prepare / "test.parquet")
    assert artifacts.confusion(prepare, classify, 2) is None


def _write_train_rows(
    tmp_path: Path, points: list[list[float]], leaves: list[int]
) -> tuple[Path, Path]:
    """Write space's z and tree's rows, shuffled, plus a val row."""
    space, tree = tmp_path / "space", tmp_path / "tree"
    space.mkdir(exist_ok=True)
    tree.mkdir(exist_ok=True)
    order = np.random.default_rng(0).permutation(len(points))
    columns = {
        f"z_{i}": [points[row][i] for row in order] for i in range(len(points[0]))
    }
    z = pd.DataFrame({"split": "train", "row": order, **columns})
    val_row = pd.DataFrame(
        {"split": ["val"], "row": [0]} | {name: [99.0] for name in columns}
    )
    pd.concat([z, val_row]).to_parquet(space / "z.parquet")
    rows = pd.DataFrame(
        {"split": "train", "row": order, "leaf": [leaves[row] for row in order]}
    )
    val_leaf = pd.DataFrame({"split": ["val"], "row": [0], "leaf": [1]})
    pd.concat([rows, val_leaf]).to_parquet(tree / "rows.parquet")
    return space, tree


def test_class_plane_centres_each_leaf_and_sizes_it_by_its_spread(
    tmp_path: Path,
) -> None:
    """Check a class's PCA plane, its circles and what stays out."""
    # Leaves 1 and 2 are the class: they split along z_0, and each
    # spreads over z_1. Leaf 3 belongs to another class, far away.
    points = [[-2, 1], [-2, -1], [2, 1], [2, -1], [100, 100], [100, 100]]
    space, tree = _write_train_rows(tmp_path, points, [1, 1, 2, 2, 3, 3])
    plane = artifacts.class_plane(space, tree, [1, 2, 7])
    # z_0 has variance 4 and z_1 has 1; the sign of a component is free.
    assert plane.shares == pytest.approx((0.8, 0.2))
    assert set(plane.circles) == {1, 2}
    (x1, y1, r1), (x2, y2, r2) = plane.circles[1], plane.circles[2]
    assert (abs(x1), abs(x2)) == pytest.approx((2, 2))
    assert x1 == pytest.approx(-x2)
    assert (y1, y2) == pytest.approx((0, 0))
    assert (r1, r2) == pytest.approx((1, 1))


def test_class_plane_reads_again_a_folder_rewritten_in_place(
    tmp_path: Path,
) -> None:
    """Check that a rewritten file is not served from the cache."""
    space, tree = _write_train_rows(tmp_path, [[0.0], [2.0]], [1, 1])
    before = artifacts.class_plane(space, tree, [1]).circles[1]
    _write_train_rows(tmp_path, [[0.0], [20.0]], [1, 1])
    after = artifacts.class_plane(space, tree, [1]).circles[1]
    assert (before[2], after[2]) == pytest.approx((1, 10))


def test_class_plane_with_one_axis_and_without_rows(tmp_path: Path) -> None:
    """Check one axis, a class no train row reaches, and bad files."""
    space, tree = _write_train_rows(
        tmp_path, [[0.0], [2.0], [4.0], [6.0]], [1, 1, 2, 2]
    )
    plane = artifacts.class_plane(space, tree, [1, 2])
    assert plane.shares == pytest.approx((1.0, 0.0))
    assert [y for _, y, _ in plane.circles.values()] == [0, 0]
    assert artifacts.class_plane(space, tree, [5]) == artifacts.ClassPlane({}, (0, 0))
    assert artifacts.class_plane(space, tmp_path / "missing", [1]) is None
    short = tmp_path / "short"
    short.mkdir()
    pd.DataFrame({"split": ["train"], "row": [0], "leaf": [1]}).to_parquet(
        short / "rows.parquet"
    )
    assert artifacts.class_plane(space, short, [1]) is None


def test_an_infinite_metric_is_not_a_value() -> None:
    """Check that an infinity cannot reach the statistics or the JSON."""
    (experiment,) = build(
        [_run(1, 1)], [(1, "evaluate", 10, "tree_mse_ratio", float("inf"))]
    )
    assert experiment.seeds[0].metrics["evaluate.tree_mse_ratio"] is None

"""dashboard/experiments.py and artifacts.py on toy data."""

from collections.abc import Sequence
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


def _write_grown_rows(
    tmp_path: Path,
    points: Sequence[Sequence[float]],
    leaves: Sequence[int],
    disagreement: Sequence[float] | None = None,
) -> tuple[Path, Path]:
    """Write space's z and tree's rows, shuffled, plus a test row."""
    if disagreement is None:
        disagreement = [0.0] * len(points)
    space, tree = tmp_path / "space", tmp_path / "tree"
    space.mkdir(exist_ok=True)
    tree.mkdir(exist_ok=True)
    order = np.random.default_rng(0).permutation(len(points))
    columns = {
        f"z_{i}": [points[row][i] for row in order] for i in range(len(points[0]))
    }
    z = pd.DataFrame({"split": "val", "row": order, **columns})
    test_row = pd.DataFrame(
        {"split": ["test"], "row": [0]} | {name: [99.0] for name in columns}
    )
    pd.concat([z, test_row]).to_parquet(space / "z.parquet")
    rows = pd.DataFrame(
        {
            "split": "val",
            "row": order,
            "grow": True,
            "leaf": [leaves[row] for row in order],
            "d": [disagreement[row] for row in order],
        }
    )
    test_leaf = pd.DataFrame(
        {"split": ["test"], "row": [0], "grow": [False], "leaf": [1], "d": [0.5]}
    )
    pd.concat([rows, test_leaf]).to_parquet(tree / "rows.parquet")
    return space, tree


def test_class_plane_places_the_sampled_rows_of_each_leaf(tmp_path: Path) -> None:
    """Check a class's PCA plane, its points and what stays out."""
    # Leaves 1 and 2 are the class: they split along z_0, and each
    # spreads over z_1. Leaf 3 belongs to another class, far away.
    points = [[-2, 1], [-2, -1], [2, 1], [2, -1], [100, 100], [100, 100]]
    space, tree = _write_grown_rows(tmp_path, points, [1, 1, 2, 2, 3, 3])
    plane = artifacts.class_plane(space, tree, [1, 2, 7])
    # z_0 has variance 4 and z_1 has 1; the sign of a component is free.
    assert plane.shares == pytest.approx((0.8, 0.2))
    assert sorted(plane.leaves.tolist()) == [1, 1, 2, 2]
    assert np.abs(plane.points) == pytest.approx(np.array([[2, 1]] * 4))
    left = plane.points[plane.leaves == 1][:, 0]
    right = plane.points[plane.leaves == 2][:, 0]
    assert left.tolist() == pytest.approx([-right[0]] * 2)


def test_class_plane_adds_the_standardized_disagreement_as_an_axis(
    tmp_path: Path,
) -> None:
    """Check `s_d` joins z in the plane, as the tree grew on it."""
    # z barely varies, while d is 0.2 or 0.8: mean 0.5, std 0.3, and a
    # scale of 3 puts s_d at -3 or 3, the axis the plane has to find.
    points = [[0.0], [0.1], [0.0], [0.1]]
    space, tree = _write_grown_rows(
        tmp_path, points, [1, 1, 2, 2], disagreement=[0.2, 0.2, 0.8, 0.8]
    )
    axis = artifacts.DisagreementAxis(mean=0.5, std=0.3, scale=3.0)
    plane = artifacts.class_plane(space, tree, [1, 2], disagreement=axis)
    assert plane.shares[0] > 0.99
    assert np.abs(plane.points[:, 0]) == pytest.approx([3.0] * 4, abs=0.1)
    assert plane.leaves[plane.points[:, 0] > 0].tolist() in ([1, 1], [2, 2])
    # Without it the plane is that of z alone, and the same cached read.
    alone = artifacts.class_plane(space, tree, [1, 2])
    assert alone.shares == pytest.approx((1.0, 0.0))
    assert np.abs(alone.points[:, 0]).max() == pytest.approx(0.05)


def test_class_plane_draws_the_same_count_from_every_leaf_within_its_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check that a big leaf does not crowd out a small one."""
    points = [[float(i), 0.0] for i in range(7)]
    space, tree = _write_grown_rows(tmp_path, points, [1, 1, 1, 1, 1, 2, 2])
    monkeypatch.setattr(artifacts, "_MAX_POINTS", 4)
    plane = artifacts.class_plane(space, tree, [1, 2])
    assert np.bincount(plane.leaves).tolist() == [0, 2, 2]
    assert artifacts.class_plane(space, tree, [1, 2]).points.tolist() == (
        plane.points.tolist()
    )
    assert artifacts._points_per_leaf([3, 10, 10], 12) == 4
    assert artifacts._points_per_leaf([3, 10, 10], 1000) == 10
    assert artifacts._points_per_leaf([3, 10, 10], 1) == 1


def test_class_plane_reads_again_a_folder_rewritten_in_place(
    tmp_path: Path,
) -> None:
    """Check that a rewritten file is not served from the cache."""
    space, tree = _write_grown_rows(tmp_path, [[0.0], [2.0]], [1, 1])
    before = artifacts.class_plane(space, tree, [1]).points
    _write_grown_rows(tmp_path, [[0.0], [20.0]], [1, 1])
    after = artifacts.class_plane(space, tree, [1]).points
    assert (np.abs(before).max(), np.abs(after).max()) == pytest.approx((1, 10))


def test_class_plane_with_one_axis_and_without_rows(tmp_path: Path) -> None:
    """Check one axis, a class no grown row reaches, and bad files."""
    space, tree = _write_grown_rows(
        tmp_path, [[0.0], [2.0], [4.0], [6.0]], [1, 1, 2, 2]
    )
    plane = artifacts.class_plane(space, tree, [1, 2])
    assert plane.shares == pytest.approx((1.0, 0.0))
    assert plane.points[:, 1].tolist() == [0, 0, 0, 0]
    empty = artifacts.class_plane(space, tree, [5])
    assert len(empty.leaves) == 0 and empty.points.shape == (0, 2)
    assert artifacts.class_plane(space, tmp_path / "missing", [1]) is None
    short = tmp_path / "short"
    short.mkdir()
    pd.DataFrame(
        {"split": ["val"], "row": [0], "grow": [True], "leaf": [1], "d": [0.0]}
    ).to_parquet(short / "rows.parquet")
    assert artifacts.class_plane(space, short, [1]) is None


def test_an_infinite_metric_is_not_a_value() -> None:
    """Check that an infinity cannot reach the statistics or the JSON."""
    (experiment,) = build(
        [_run(1, 1)], [(1, "evaluate", 10, "tree_mse_ratio", float("inf"))]
    )
    assert experiment.seeds[0].metrics["evaluate.tree_mse_ratio"] is None

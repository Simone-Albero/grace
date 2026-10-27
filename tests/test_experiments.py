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
        "regressor": True,
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


def test_an_infinite_metric_is_not_a_value() -> None:
    """Check that an infinity cannot reach the statistics or the JSON."""
    (experiment,) = build(
        [_run(1, 1)], [(1, "evaluate", 10, "tree_mse_ratio", float("inf"))]
    )
    assert experiment.seeds[0].metrics["evaluate.tree_mse_ratio"] is None

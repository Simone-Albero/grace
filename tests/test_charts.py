"""dashboard/charts.py's pure figure-building functions, on toy data."""

import numpy as np
import pytest

from dashboard.charts import (
    confusion_heatmap,
    group_bars,
    predictor_bars,
    shrinkage_scatter,
)


def test_confusion_heatmap_normalizes_each_row_and_keeps_counts() -> None:
    """Check that a row's shares sum to 1, an empty row stays 0."""
    counts = np.array([[3, 1], [0, 0]])
    trace = confusion_heatmap(counts, ["a", "b"])["data"][0]
    assert trace["z"] == [[0.75, 0.25], [0.0, 0.0]]
    assert trace["customdata"] == [[3, 1], [0, 0]]


def test_group_bars_use_a_sample_std_and_hover_the_count() -> None:
    """Check the whisker (0 for one value) and the n of each bar."""
    figure = group_bars(
        [("t", "#000000", {"x": [1.0, 3.0], "y": [5.0]})], x_title="x", y_title="y"
    )
    trace = figure["data"][0]
    assert trace["y"] == [2.0, 5.0]
    assert trace["error_y"]["array"] == [pytest.approx(2**0.5), 0.0]
    assert trace["text"] == ["n = 2", "n = 1"]


def test_predictor_bars_keep_the_predictors_with_a_value() -> None:
    """Check a predictor without the score is left out, in order."""
    scores = {
        "gbm": {"mse_ratio": 0.8},
        "tree": {"mse_ratio": 0.9},
        "binned": {"mse_ratio": None},
    }
    trace = predictor_bars(scores, "mse_ratio")["data"][0]
    assert trace["x"] == ["Tree", "GBM"] and trace["y"] == [0.9, 0.8]


def test_shrinkage_scatter_drops_a_leaf_without_val_rows_and_links_the_rest() -> None:
    """Check only leaves with a val rate are drawn, each linked."""
    leaves = [
        {"node_id": 3, "class": "a", "n_val": 40, "val_rate": 0.1, "q": 0.12},
        {"node_id": 4, "class": "a", "n_val": 0, "val_rate": None, "q": 0.2},
    ]
    points = shrinkage_scatter(leaves, run_id=7)["data"][1]
    assert points["x"] == [0.1] and points["y"] == [0.12]
    assert points["customdata"] == ["/runs/7/leaves/3"]

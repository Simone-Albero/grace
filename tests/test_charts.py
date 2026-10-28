"""dashboard/charts.py's pure figure-building functions, on toy data."""

import re
from pathlib import Path

import numpy as np
import pytest

from dashboard.artifacts import ClassPlane
from dashboard.charts import (
    _FILL_STEPS,
    NEUTRAL,
    confusion_heatmap,
    group_bars,
    leaf_map,
    leaf_size_bars,
    predictor_bars,
    shrinkage_scatter,
    split_bars,
    tree_diagram,
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


def _node(
    node_id: int, parent_id: int | None, depth: int, status: str, **overrides: float
) -> dict:
    return {
        "node_id": node_id,
        "parent_id": parent_id,
        "depth": depth,
        "status": status,
        "n_train": 10,
        "oof_rate": 0.1,
        "purity": 0.9,
        "n_val": 4,
        "val_rate": None,
        "n_test": None,
        "test_rate": None,
        "q": 0.2,
    } | overrides


def test_tree_diagram_centers_parents_and_links_only_leaves() -> None:
    """Check leaves in depth-first order, a parent over its children."""
    nodes = [
        _node(0, None, 0, "internal"),
        _node(1, 0, 1, "internal"),
        _node(2, 0, 1, "leaf"),
        _node(3, 1, 2, "leaf"),
        _node(4, 1, 2, "leaf"),
    ]
    edges, marks = tree_diagram(nodes, color="purity", run_id=7)["data"]
    assert marks["x"] == [1.25, 0.5, 2, 0, 1]
    assert marks["customdata"] == [
        {"node": 0},
        {"node": 1},
        "/runs/7/leaves/2",
        "/runs/7/leaves/3",
        "/runs/7/leaves/4",
    ]
    assert len(edges["x"]) == 3 * 4
    # An unknown count reads as a dash, never as None or 0.
    assert "purity 0.9000" in marks["text"][2]
    assert "val 4 · val rate –" in marks["text"][2]
    assert "test – · test rate –" in marks["text"][2]


def test_tree_diagram_fades_the_nodes_outside_the_focus_and_tags_the_splits() -> None:
    """Check the opacity, and what a click on a node carries."""
    nodes = [
        _node(0, None, 0, "internal"),
        _node(1, 0, 1, "internal"),
        _node(2, 0, 1, "leaf"),
        _node(3, 1, 2, "leaf"),
        _node(4, 1, 2, "leaf"),
    ]
    marks = tree_diagram(nodes, color="purity", run_id=7, focus={1, 3, 4})["data"][1]
    assert marks["marker"]["opacity"] == [0.3, 1, 0.3, 1, 1]
    assert marks["customdata"][:2] == [{"node": 0}, {"node": 1}]
    assert marks["customdata"][2] == "/runs/7/leaves/2"
    plain = tree_diagram(nodes, color="purity", run_id=7)["data"][1]
    assert plain["marker"]["opacity"] == [1] * 5


def _class_nodes() -> list[dict]:
    """A root over three leaves, two of them with a val rate."""
    return [
        _node(0, None, 0, "internal", oof_rate=0.2, purity=0.8, val_rate=0.3),
        _node(1, 0, 1, "leaf", oof_rate=0.4, purity=0.6, val_rate=0.5),
        _node(2, 0, 1, "leaf", oof_rate=0.1, purity=0.9, val_rate=0.1),
        _node(3, 0, 1, "leaf", oof_rate=0.0, purity=1.0, val_rate=None),
    ]


def test_tree_diagram_colors_by_the_chosen_measure_on_the_scale_of_the_leaves() -> None:
    """Check the color values, the bar title and the scale ends."""
    nodes = _class_nodes()
    marker = tree_diagram(nodes, color="val_rate", run_id=1)["data"][1]["marker"]
    assert marker["color"] == [0.3, 0.5, 0.1, None]
    assert marker["colorbar"]["title"] == "val rate"
    assert (marker["cmin"], marker["cmax"]) == (0, 0.5)
    # A purity is best at 1, so its scale runs from the worst leaf
    # up to it.
    marker = tree_diagram(nodes, color="purity", run_id=1)["data"][1]["marker"]
    assert (marker["cmin"], marker["cmax"]) == (0.6, 1.0)
    # With no rate to show, the scale is still a range.
    unknown = [_node(0, None, 0, "leaf", test_rate=None)]
    marker = tree_diagram(unknown, color="test_rate", run_id=1)["data"][1]["marker"]
    assert (marker["cmin"], marker["cmax"]) == (0.0, 1.0)


def test_leaf_map_draws_the_largest_circle_first_and_shares_the_axis_scale() -> None:
    """Check the circles, their fills, the centres and the axes."""
    plane = ClassPlane(
        {1: (0.0, 0.0, 2.0), 2: (3.0, 1.0, 3.0), 3: (-1.0, -1.0, 0.5)}, (0.75, 0.2)
    )
    figure = leaf_map(_class_nodes(), plane, color="val_rate", run_id=9)
    shapes = figure["layout"]["shapes"]
    # Radius 3, 2, 0.5: leaf 2, then leaf 1, then leaf 3.
    assert [shape["x1"] - shape["x0"] for shape in shapes] == [6, 4, 1]
    assert shapes[0]["y0"] == -2 and shapes[0]["y1"] == 4
    # val rate 0.1 is the lowest of the scale 0..0.5, 0.5 the
    # highest, an unknown one has no step.
    assert [shape["fillcolor"] for shape in shapes] == [
        _FILL_STEPS[1],
        _FILL_STEPS[-1],
        NEUTRAL,
    ]
    (centres,) = figure["data"]
    # The internal node has no circle, and no marker.
    assert centres["customdata"] == [
        "/runs/9/leaves/2",
        "/runs/9/leaves/1",
        "/runs/9/leaves/3",
    ]
    assert centres["x"] == [3.0, 0.0, -1.0]
    assert centres["marker"]["color"] == [0.1, 0.5, None]
    assert (centres["marker"]["cmin"], centres["marker"]["cmax"]) == (0, 0.5)
    layout = figure["layout"]
    assert layout["yaxis"]["scaleanchor"] == "x"
    assert layout["xaxis"]["title"] == "PC 1 · 75% of the variance"
    assert layout["yaxis"]["title"] == "PC 2 · 20% of the variance"


def test_dark_theme_swaps_each_fill_step_with_its_mirror() -> None:
    """Check that dashboard.js knows every fill step of charts.py."""
    script = Path(__file__).parents[1] / "dashboard" / "static" / "dashboard.js"
    swaps = dict(re.findall(r'"(#[0-9a-f]{6})": "(#[0-9a-f]{6})"', script.read_text()))
    for step, mirror in zip(_FILL_STEPS, reversed(_FILL_STEPS), strict=True):
        assert swaps[step] == mirror


def test_leaf_size_bars_bin_by_powers_of_two_and_skip_an_absent_split() -> None:
    """Check the 0, 1 and 2-3 bins, and test left out without counts."""
    leaves = [
        {"n_train": 0, "n_val": 1, "n_test": None},
        {"n_train": 3, "n_val": 2, "n_test": None},
        {"n_train": 2, "n_val": 0, "n_test": None},
    ]
    figure = leaf_size_bars(leaves)
    by_split = {trace["name"]: trace for trace in figure["data"]}
    assert list(by_split) == ["train", "val"]
    assert by_split["train"]["x"] == ["0", "1", "2–3"]
    assert by_split["train"]["y"] == [1, 0, 2]
    assert by_split["val"]["y"] == [1, 1, 1]


def test_split_bars_use_a_log_axis_with_the_largest_class_first() -> None:
    """Check the log axis, the order and the share of the split."""
    classes = [
        {"name": "rare", "n_train": 2, "n_val": 1, "n_test": 1},
        {"name": "big", "n_train": 198, "n_val": 99, "n_test": 99},
    ]
    figure = split_bars(classes)
    assert figure["layout"]["xaxis"]["type"] == "log"
    train = figure["data"][0]
    assert train["y"] == ["big", "rare"] and train["x"] == [198, 2]
    assert train["customdata"] == [0.99, 0.01]

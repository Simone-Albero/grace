"""dashboard/charts.py's pure figure-building functions, on toy data."""

import re
from pathlib import Path

import numpy as np
import pytest

from dashboard.artifacts import ClassPlane
from dashboard.charts import (
    _DIVERGING,
    _DIVERGING_NEUTRAL,
    NEUTRAL,
    _color_scale,
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


def test_shrinkage_scatter_links_only_the_leaves_with_estimate_rows() -> None:
    """Check only leaves with an estimate rate are drawn, and linked."""
    leaves = [
        {"node_id": 3, "class": "a", "n_estimate": 40, "estimate_rate": 0.1, "q": 0.12},
        {"node_id": 4, "class": "a", "n_estimate": 0, "estimate_rate": None, "q": 0.2},
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
        "n_grow": 10,
        "grow_rate": 0.1,
        "purity": 0.9,
        "n_estimate": 4,
        "estimate_rate": None,
        "n_val": 14,
        "val_rate": 0.1,
        "n_test": None,
        "test_rate": None,
        "q": 0.2,
        "pred_rate": 0.2,
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
    # A row per split, with its rows and rate; an unknown count reads as
    # a dash, never as None or 0.
    leaf_text = marks["text"][2].split("<br>")
    assert leaf_text == [
        "node 2 · leaf",
        "val: 14 rows · rate 0.1000 (grow 10, estimate 4)",
        "test: – rows · rate –",
        "purity (grow) 0.9000",
        "predicted rate 0.2000",
    ]
    assert "predicted rate, leaves' mean 0.2000" in marks["text"][0]


def test_tree_diagram_tags_the_splits_for_the_zoom() -> None:
    """Check what a click on a split or on a leaf carries."""
    nodes = [
        _node(0, None, 0, "internal"),
        _node(1, 0, 1, "internal"),
        _node(2, 0, 1, "leaf"),
    ]
    marks = tree_diagram(nodes, color="purity", run_id=7)["data"][1]
    assert marks["customdata"] == [{"node": 0}, {"node": 1}, "/runs/7/leaves/2"]
    assert "opacity" not in marks["marker"]


def _class_nodes() -> list[dict]:
    """A root over three leaves, two of them with a val rate."""
    return [
        _node(0, None, 0, "internal", val_rate=0.3, purity=0.8, pred_rate=0.25),
        _node(1, 0, 1, "leaf", val_rate=0.5, purity=0.6, pred_rate=0.4),
        _node(2, 0, 1, "leaf", val_rate=0.1, purity=0.9, pred_rate=0.2),
        _node(3, 0, 1, "leaf", val_rate=None, purity=1.0, pred_rate=0.05),
    ]


def test_tree_diagram_colors_by_the_chosen_measure_around_the_rate_of_the_top() -> None:
    """Check the colors, the bar and a scale centred on the root."""
    nodes = _class_nodes()
    marker = tree_diagram(nodes, color="val_rate", run_id=1)["data"][1]["marker"]
    # The root's val rate is the middle; the two leaves that have one
    # lie beyond the 2nd and 98th percentiles of themselves, so they
    # take the ends.
    assert marker["color"] == [0, 1, -1, None]
    assert (marker["cmin"], marker["cmax"]) == (-1, 1)
    assert [color for _, color in marker["colorscale"]] == _DIVERGING
    bar = marker["colorbar"]
    assert bar["title"] == "val rate"
    assert bar["tickvals"] == [-1, -0.5, 0, 0.5, 1]
    assert bar["ticktext"][0] == "0.1080" and bar["ticktext"][2] == "top 0.3000"
    assert bar["ticktext"][1] == "0.2520" and bar["ticktext"][4] == "0.4920"
    # A purity sits on the scale of its complement: its worst leaf is
    # the red one, above the middle, and its bar reads in purities.
    marker = tree_diagram(nodes, color="purity", run_id=1)["data"][1]["marker"]
    assert 0 < marker["color"][1] < 1 and marker["color"][2:] == [-1, -1]
    assert -1 < marker["color"][0] < 0
    assert marker["colorbar"]["tickvals"] == [-1, -0.5, 0, 0.5, 1]
    assert marker["colorbar"]["ticktext"][0] == "0.8920"
    assert marker["colorbar"]["ticktext"][2] == "top 0.7000"
    assert marker["colorbar"]["ticktext"][4] == "0.5080"
    # With no rate to show, the scale is still a range.
    unknown = [_node(0, None, 0, "leaf", val_rate=None, test_rate=None)]
    marker = tree_diagram(unknown, color="test_rate", run_id=1)["data"][1]["marker"]
    assert marker["color"] == [None]
    assert marker["colorbar"]["ticktext"][2] == "top 0.5000"


def test_each_measure_colors_the_nodes_differently_on_the_same_scale() -> None:
    """Check that a change of measure moves the colors, not the scale."""
    nodes = _class_nodes()
    nodes[1]["test_rate"] = 0.1
    nodes[2]["test_rate"] = 0.45
    markers = {
        color: tree_diagram(nodes, color=color, run_id=1)["data"][1]["marker"]
        for color in ("pred_rate", "val_rate", "test_rate", "purity")
    }
    colors = [tuple(marker["color"]) for marker in markers.values()]
    assert len(set(colors)) == len(colors)
    rate_bars = [markers[color]["colorbar"]["ticktext"] for color in markers]
    assert rate_bars[0] == rate_bars[1] == rate_bars[2] != rate_bars[3]


def test_color_scale_keeps_its_middle_inside_and_its_ends_within_0_and_1() -> None:
    """Check the margin around a root past every leaf, and the clamp."""
    nodes = [
        _node(0, None, 0, "internal", val_rate=0.05),
        _node(1, 0, 1, "leaf", val_rate=0.2),
        _node(2, 0, 1, "leaf", val_rate=0.2),
    ]
    low, middle, high = _color_scale(nodes)
    assert 0 <= low < middle < high
    # A root on the edge of 0..1 leaves the scale inside it, and still draws.
    edge = [
        _node(0, None, 0, "internal", val_rate=0.0, purity=1.0),
        _node(1, 0, 1, "leaf", val_rate=0.0, purity=1.0),
        _node(2, 0, 1, "leaf", val_rate=0.2, purity=0.8),
    ]
    assert _color_scale(edge)[0] == 0.0
    top = [_node(0, None, 0, "internal", val_rate=1.0), *edge[1:]]
    assert _color_scale(top)[2] == 1.0
    for color in ("val_rate", "purity", "pred_rate"):
        marker = tree_diagram(edge, color=color, run_id=1)["data"][1]["marker"]
        assert all(-1 <= at <= 1 for at in marker["color"])
        assert not any(text.startswith("-") for text in marker["colorbar"]["ticktext"])


def test_a_class_with_one_val_rate_does_not_saturate_the_other_measures() -> None:
    """Check a predicted rate near a rate of 0 reads as near the middle."""
    nodes = [
        _node(0, None, 0, "internal", val_rate=0.0, pred_rate=0.002),
        _node(1, 0, 1, "leaf", val_rate=0.0, pred_rate=0.002),
        _node(2, 0, 1, "leaf", val_rate=0.0, pred_rate=0.2),
    ]
    marker = tree_diagram(nodes, color="pred_rate", run_id=1)["data"][1]["marker"]
    assert 0 < marker["color"][1] < 0.5 and marker["color"][2] == 1
    assert marker["colorbar"]["ticktext"][4] == "0.0100"


def test_leaf_map_draws_a_point_per_row_with_the_extremes_on_top() -> None:
    """Check the points, their colors and order, and the axes."""
    nodes = _class_nodes() + [_node(4, 0, 1, "leaf", val_rate=0.35)]
    nodes[1]["val_rate"] = 0.6
    plane = ClassPlane(
        np.array([1, 2, 2, 3, 4]),
        np.array([[0, 0], [3, 1], [3, 2], [-1, -1], [5, 5]], dtype=float),
        (0.75, 0.2),
    )
    figure = leaf_map(nodes, plane, color="val_rate", run_id=9)
    unknown, points = figure["data"]
    assert points["type"] == "scattergl"
    # Leaf 3 has no rate: neutral, beneath. Then, closest to the middle
    # (0.3) first, leaf 4, then leaves 1 and 2 at the ends of the scale.
    assert unknown["customdata"] == ["/runs/9/leaves/3"]
    assert unknown["marker"]["color"] == NEUTRAL and unknown["x"] == [-1.0]
    assert points["customdata"] == [f"/runs/9/leaves/{leaf}" for leaf in (4, 1, 2, 2)]
    assert points["x"] == [5.0, 0.0, 3.0, 3.0]
    colors = points["marker"]["color"]
    assert 0 < colors[0] < 1 and colors[1:] == [1, -1, -1]
    assert "leaf 2" in points["text"][2]
    layout = figure["layout"]
    assert layout["yaxis"]["scaleanchor"] == "x"
    # The view frames the bulk of the points: the 1st to 99th percentile
    # of x, widened by a fifth; a single value has no range to frame.
    low, high = layout["xaxis"]["range"]
    assert low < -0.9 < 4.9 < high and high - low < 20
    one_axis = ClassPlane(np.array([1, 1]), np.array([[0.0, 0.0], [2.0, 0.0]]), (1, 0))
    assert (
        "range"
        not in leaf_map(nodes, one_axis, color="val_rate", run_id=9)["layout"]["yaxis"]
    )
    assert layout["xaxis"]["title"] == "PC 1 · 75% of the variance"
    assert layout["yaxis"]["title"] == "PC 2 · 20% of the variance"


def test_dark_theme_keeps_the_diverging_order_and_darkens_its_middle() -> None:
    """Check that dashboard.js knows each diverging step."""
    script = Path(__file__).parents[1] / "dashboard" / "static" / "dashboard.js"
    text = script.read_text()
    swaps = dict(re.findall(r'"(#[0-9a-f]{6})": "(#[0-9a-f]{6})"', text))
    assert all(color in swaps for color in _DIVERGING)
    assert f'DIVERGING_NEUTRAL = "{_DIVERGING_NEUTRAL}"' in text


def test_leaf_size_bars_bin_by_powers_of_two_and_skip_an_absent_subset() -> None:
    """Check the 0, 1 and 2-3 bins, and test left out without counts."""
    leaves = [
        {"n_grow": 0, "n_estimate": 1, "n_test": None},
        {"n_grow": 3, "n_estimate": 2, "n_test": None},
        {"n_grow": 2, "n_estimate": 0, "n_test": None},
    ]
    figure = leaf_size_bars(leaves)
    by_subset = {trace["name"]: trace for trace in figure["data"]}
    assert list(by_subset) == ["grow", "estimate"]
    assert by_subset["grow"]["x"] == ["0", "1", "2–3"]
    assert by_subset["grow"]["y"] == [1, 0, 2]
    assert by_subset["estimate"]["y"] == [1, 1, 1]


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

"""dashboard/charts.py's pure figure-building functions, on toy data."""

import numpy as np
import pytest

from dashboard.charts import (
    _mean_points,
    class_rate_bars,
    class_rate_gap,
    confusion_heatmap,
    group_bars,
    group_rate_bins,
    partition_class_curves,
    partition_curves,
    partition_val_rates,
    partition_val_test_scatter,
    predicted_vs_observed,
    rate_ecdf,
    region_rate_bins,
    val_test_scatter,
)


def test_mean_points_drops_a_row_with_no_y_and_bounds_a_group() -> None:
    """Check that a None y is dropped and a group's bounds are kept.

    A key left with no surviving row does not appear at all; a group
    of several rows keeps its y mean between y_min and y_max, and a
    group of one row has y_min == y_max == its own y.
    """
    rows = [
        {"level": 2, "x": 0.0, "y": 9.0},
        {"level": 0, "x": 10.0, "y": 1.0},
        {"level": 0, "x": 20.0, "y": None},  # dropped: no y.
        {"level": 1, "x": 0.0, "y": None},  # the whole group is dropped.
        {"level": 1, "x": 0.0, "y": None},
        {"level": 3, "x": 100.0, "y": 2.0},
        {"level": 3, "x": 100.0, "y": 4.0},
        {"level": 3, "x": 100.0, "y": 6.0},
    ]
    out = _mean_points(rows, "level", "x", "y")
    assert out["key"] == [0, 2, 3]  # level 1 is gone; sorted ascending.
    assert out["x"] == [10.0, 0.0, 100.0]
    assert out["y"] == pytest.approx([1.0, 9.0, 4.0])
    assert out["y_min"] == [1.0, 9.0, 2.0]
    assert out["y_max"] == [1.0, 9.0, 6.0]  # level 0 and 2: one row each.


def _partition_row(
    arm: str, rule: str, level: int, n_groups: int, skill: float
) -> dict:
    return {
        "arm": arm,
        "rule": rule,
        "level": level,
        "n_groups": n_groups,
        "val_median": 0.1,
        "skill": skill,
    }


_PARTITION_ROWS = [
    _partition_row("intra", "level", 0, 50, 0.20),
    _partition_row("intra", "level", 1, 20, 0.30),
    _partition_row("intra", "selected", -1, 20, 0.30),
    # null draws twice per level, so n_groups and skill both vary.
    _partition_row("null", "level", 0, 48, 0.05),
    _partition_row("null", "level", 0, 52, 0.07),
    _partition_row("null", "selected", -1, 50, 0.06),
    # confidence is a bin arm: operating points only, no level sweep.
    _partition_row("confidence", "fixed15", -1, 15, 0.15),
]


def test_partition_curves_bands_only_the_arm_whose_draws_disagree() -> None:
    """Check that only the arm whose draws disagree gets a band."""
    names = {
        trace["name"]
        for trace in partition_curves(_PARTITION_ROWS, metric="skill", x="n_groups")[
            "data"
        ]
    }
    assert "null (min-max)" in names
    assert "intra (min-max)" not in names


def test_partition_curves_draws_a_bin_arm_as_markers_only() -> None:
    """Check that a bin arm, with no level sweep, draws markers only."""
    names = {
        trace["name"]
        for trace in partition_curves(_PARTITION_ROWS, metric="skill", x="n_groups")[
            "data"
        ]
    }
    assert "confidence operating point" in names
    assert "confidence" not in names
    assert "confidence (min-max)" not in names


def test_partition_curves_x_axis_is_log_only_for_n_groups() -> None:
    """Check that the log x axis applies to n_groups, not val_median."""
    by_groups = partition_curves(_PARTITION_ROWS, metric="skill", x="n_groups")
    by_median = partition_curves(_PARTITION_ROWS, metric="skill", x="val_median")
    assert by_groups["layout"]["xaxis"]["type"] == "log"
    assert by_median["layout"]["xaxis"]["type"] == "linear"


def test_partition_class_curves_names_classes_and_falls_back_to_id() -> None:
    """Check that a class name is used when given, else its id."""
    rows = [
        {"class_id": 0, "level": 0, "n_clusters": 4, "loss": 1.0, "selected": 0},
        {"class_id": 0, "level": 1, "n_clusters": 2, "loss": 0.5, "selected": 1},
        {"class_id": 1, "level": 0, "n_clusters": 4, "loss": 2.0, "selected": 1},
    ]
    figure = partition_class_curves(rows, class_names={0: "benign"})
    names = {trace["name"] for trace in figure["data"] if trace["name"] != "selected"}
    assert names == {"benign", "1"}


# In the query's own order (`ORDER BY arm, rule, group_id`), so a test
# that only matched this order by coincidence would not survive a
# reordering of the rows.
_GROUP_ROWS = [
    {
        # An empty group: no val row landed here.
        "arm": "confidence",
        "rule": "fixed15",
        "group_id": 1,
        "n_val": 0,
        "val_rate": None,
        "test_rate": 0.4,
    },
    {
        # A non-empty confidence group, so the arm still gets a box
        # and a scatter trace, just not from the empty group above.
        "arm": "confidence",
        "rule": "cbrt",
        "group_id": 2,
        "n_val": 5,
        "val_rate": 0.5,
        "test_rate": 0.45,
    },
    {
        # A group with no test row: the scatter must drop it too.
        "arm": "inter",
        "rule": "selected",
        "group_id": 0,
        "n_val": 8,
        "val_rate": 0.2,
        "test_rate": None,
    },
    {
        "arm": "intra",
        "rule": "selected",
        "group_id": 0,
        "n_val": 20,
        "val_rate": 0.3,
        "test_rate": 0.25,
    },
    {
        "arm": "null",
        "rule": "selected",
        "group_id": 0,
        "n_val": 10,
        "val_rate": 0.1,
        "test_rate": 0.2,
    },
]


def test_partition_val_rates_drops_an_empty_group_and_keeps_arm_order() -> None:
    """Check a None val_rate is dropped and boxes follow ARM_COLORS.

    The rows arrive in the query's alphabetical-by-arm order, which
    disagrees with ARM_COLORS for every pair here, so only a real
    re-sort by ARM_COLORS, not the input order, passes this.
    """
    figure = partition_val_rates(_GROUP_ROWS)
    for trace in figure["data"]:
        assert None not in trace["y"]
    names = [trace["name"] for trace in figure["data"]]
    assert "confidence (fixed15)" not in names
    assert names == ["intra", "inter", "null", "confidence (cbrt)"]


def test_partition_val_test_scatter_drops_a_group_missing_either_rate() -> None:
    """Check that a group missing either rate adds no point."""
    figure = partition_val_test_scatter(_GROUP_ROWS)
    for trace in figure["data"]:
        if trace["name"] != "val = test":
            assert None not in trace["x"] and None not in trace["y"]
    plotted = {trace["name"] for trace in figure["data"]} - {"val = test"}
    assert "inter" not in plotted  # its only group has no test_rate.
    confidence = next(
        trace for trace in figure["data"] if trace["name"] == "confidence"
    )
    assert len(confidence["x"]) == 1
    assert all("group 1," not in text for text in confidence["text"])


def _region(region_id: int, **values) -> dict:
    """Build a region row with the columns the failure charts read."""
    row = {
        "region_id": region_id,
        "class": "a",
        "n_val": 10,
        "n_test": 20,
        "n_val_error": 1,
        "n_test_error": 4,
        "val_rate": 0.1,
        "test_rate": 0.2,
        "used": 1,
        "regressor": None,
        "val": None,
        "mcp": None,
        "atc": None,
    }
    return row | values


def test_confusion_heatmap_normalizes_each_row_and_keeps_counts() -> None:
    """Check that a row's shares sum to 1, an empty row stays 0."""
    counts = np.array([[3, 1], [0, 0]])
    trace = confusion_heatmap(counts, ["a", "b"])["data"][0]
    assert trace["z"] == [[0.75, 0.25], [0.0, 0.0]]
    assert trace["customdata"] == [[3, 1], [0, 0]]


def test_val_test_scatter_line_ends_at_the_data() -> None:
    """Check that the y = x line does not stretch the axes to 1."""
    figure = val_test_scatter(
        [_region(0, val_rate=0.2, test_rate=0.3), _region(1, test_rate=None)],
        run_id=1,
    )
    line, points = figure["data"]
    assert line["x"][1] == pytest.approx(0.3 * 1.05)
    assert points["x"] == [0.2] and points["customdata"] == ["/runs/1/regions/0"]


def test_class_rate_bars_weigh_regions_by_their_rows() -> None:
    """Check that a class's rate is its errors over its rows."""
    rows = [
        _region(0, n_test=10, n_test_error=1),
        _region(1, n_test=30, n_test_error=9),
    ]
    test = next(t for t in class_rate_bars(rows)["data"] if t["name"] == "test")
    assert test["y"] == [pytest.approx(10 / 40)]


def test_rate_ecdf_ends_at_one() -> None:
    """Check that each split's ECDF is sorted and reaches 1."""
    rows = [_region(0, val_rate=0.3), _region(1, val_rate=0.1, test_rate=None)]
    val = next(t for t in rate_ecdf(rows)["data"] if t["name"] == "val")
    assert val["x"] == [0.1, 0.3] and val["y"] == [0.5, 1.0]


def test_predicted_vs_observed_has_a_panel_per_predictor_with_values() -> None:
    """Check that only predictors with a value get a panel."""
    rows = [_region(0, val=0.1, mcp=0.4), _region(1, val=0.2, used=0, mcp=0.9)]
    figure = predicted_vs_observed(rows, run_id=1)
    assert {t["name"] for t in figure["data"] if t["name"] != "y = x"} == {
        "Val rate",
        "MCP",
    }
    assert figure["layout"]["yaxis2"]["matches"] == "y"
    # The unused region is not scored.
    val = next(t for t in figure["data"] if t["name"] == "Val rate")
    assert val["x"] == [0.1]
    empty = predicted_vs_observed([_region(0)], run_id=1)
    assert empty["data"] == [] and empty["layout"]["annotations"]


def test_group_bars_use_a_sample_std_and_hover_the_count() -> None:
    """Check the whisker (0 for one value) and the n of each bar."""
    figure = group_bars(
        [("t", "#000000", {"x": [1.0, 3.0], "y": [5.0]})], x_title="x", y_title="y"
    )
    trace = figure["data"][0]
    assert trace["y"] == [2.0, 5.0]
    assert trace["error_y"]["array"] == [pytest.approx(2**0.5), 0.0]
    assert trace["text"] == ["n = 2", "n = 1"]


def test_region_rate_bins_give_zero_a_bin_and_keep_empty_bins() -> None:
    """Check the bin edges, counts, mean sizes and empty bins.

    A rate of exactly 0 has its own bin, a rate on an edge falls in the
    bin it closes, a region with no rate is left out, and a bin nobody
    reaches stays on the axis with no mean.
    """
    rows = [
        {"val_rate": 0.0, "n_val": 10, "test_rate": 1.0, "n_test": 4},
        {"val_rate": 0.05, "n_val": 20, "test_rate": None, "n_test": 0},
        {"val_rate": 0.04, "n_val": 40, "test_rate": 0.5, "n_test": 6},
        {"val_rate": None, "n_val": 0, "test_rate": 0.3, "n_test": 8},
    ]
    val_counts, test_counts = region_rate_bins(rows, measure="count")["data"]
    assert val_counts["x"][:3] == ["0", "0–5%", "5–10%"]
    assert val_counts["y"] == [1, 2, 0, 0, 0, 0, 0]
    val_sizes = region_rate_bins(rows, measure="size")["data"][0]
    assert val_sizes["y"] == [10, 30, None, None, None, None, None]
    # 0.3 closes 20–30%, 0.5 closes 30–50%, 1.0 falls in 50–100%.
    assert test_counts["y"] == [0, 0, 0, 0, 1, 1, 1]


def test_group_rate_bins_read_each_arm_at_its_operating_rule() -> None:
    """Check a bin arm at `cbrt` and a geometric arm at `selected`."""
    rows = [
        {"arm": "intra", "rule": "selected", "val_rate": 0.0, "n_val": 5},
        {"arm": "intra", "rule": "level", "val_rate": 0.9, "n_val": 5},
        {"arm": "confidence", "rule": "cbrt", "val_rate": 0.6, "n_val": 7},
        {"arm": "confidence", "rule": "fixed15", "val_rate": 0.0, "n_val": 7},
    ]
    traces = group_rate_bins(rows, measure="count")["data"]
    counts_by_arm = {trace["name"]: trace["y"] for trace in traces}
    assert counts_by_arm["intra"] == [1, 0, 0, 0, 0, 0, 0]
    assert counts_by_arm["confidence"] == [0, 0, 0, 0, 0, 0, 1]
    assert "null" not in counts_by_arm


def test_class_rate_gap_is_val_minus_test_and_skips_a_region_without_both() -> None:
    """Check the gap's sign, and that a region missing a rate is out."""
    rows = [
        {
            "class": "a",
            "region_id": 1,
            "val_rate": 0.5,
            "test_rate": 0.25,
            "n_val": 4,
            "n_test": 4,
        },
        {
            "class": "a",
            "region_id": 2,
            "val_rate": None,
            "test_rate": 0.1,
            "n_val": 0,
            "n_test": 3,
        },
    ]
    trace = class_rate_gap(rows, run_id=7)["data"][0]
    assert trace["x"] == ["a"]
    assert trace["y"] == [0.25]
    assert trace["customdata"] == ["/runs/7/regions/1"]

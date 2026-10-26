"""dashboard/charts.py's pure figure-building functions, on toy data."""

import pytest

from dashboard.charts import (
    _ARM_COLORS,
    _mean_points,
    partition_class_curves,
    partition_curves,
    partition_val_rates,
    partition_val_test_scatter,
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
    """Check a None val_rate is dropped and boxes follow _ARM_COLORS.

    The rows arrive in the query's alphabetical-by-arm order, which
    disagrees with _ARM_COLORS for every pair here, so only a real
    re-sort by _ARM_COLORS, not the input order, passes this.
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

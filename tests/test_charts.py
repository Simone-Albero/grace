"""dashboard/charts.py's pure figure-building functions, on toy data."""

import numpy as np
import pytest

from dashboard.charts import (
    class_rate_bars,
    class_rate_gap,
    confusion_heatmap,
    group_bars,
    predicted_vs_observed,
    rate_ecdf,
    region_rate_bins,
    val_test_scatter,
)


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
        [_region(0, val_rate=0.2, test_rate=0.3), _region(1, test_rate=None)]
    )
    line, points = figure["data"]
    assert line["x"][1] == pytest.approx(0.3 * 1.05)
    assert points["x"] == [0.2]


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
    figure = predicted_vs_observed(rows)
    assert {t["name"] for t in figure["data"] if t["name"] != "y = x"} == {
        "Val rate",
        "MCP",
    }
    assert figure["layout"]["yaxis2"]["matches"] == "y"
    # The unused region is not scored.
    val = next(t for t in figure["data"] if t["name"] == "Val rate")
    assert val["x"] == [0.1]
    empty = predicted_vs_observed([_region(0)])
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
    trace = class_rate_gap(rows)["data"][0]
    assert trace["x"] == ["a"]
    assert trace["y"] == [0.25]

"""Plotly figures of the dashboard, as JSON-ready dicts.

A figure carries its structure and its light-theme colors; the page's
`dashboard.js` adds the margins, fonts and grid colors, and swaps the
colors below for their dark-theme steps.
"""

__all__ = [
    "ARM_COLORS",
    "NEUTRAL",
    "OPERATING_RULES",
    "PREDICTOR_COLORS",
    "SERIES",
    "arm_bars",
    "class_rate_bars",
    "class_score_bars",
    "confusion_heatmap",
    "grid_search",
    "group_bars",
    "importance_bars",
    "loss_curve",
    "partition_class_curves",
    "partition_curves",
    "partition_val_rates",
    "partition_val_test_scatter",
    "predicted_vs_observed",
    "predictor_bars",
    "rate_ecdf",
    "region_scatter",
    "region_size_box",
    "regions_per_class",
    "seed_strip",
    "split_bars",
    "val_test_scatter",
]

import math
import statistics
from collections.abc import Mapping, Sequence

import numpy as np

from dashboard.queries import PREDICTORS

_HEIGHT = 380
# The categorical slots, in their fixed order; a ninth entity is
# neutral.
SERIES = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
]
NEUTRAL = "#898781"
# Tokens that `dashboard.js` swaps in the dark theme.
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_ON_DARK_FILL = "#ffffff"
_ON_LIGHT_FILL = "#0b0b0b"
_SEQUENTIAL = [[0, "#cde2fb"], [0.5, "#3987e5"], [1, "#0d366b"]]
# An entity keeps its color in every figure.
_SPLIT_COLORS = {"train": SERIES[0], "val": SERIES[1], "test": SERIES[2]}
_SCORE_COLORS = {"precision": SERIES[0], "recall": SERIES[1], "f1": SERIES[2]}
PREDICTOR_COLORS = dict(zip(PREDICTORS, SERIES))
# One fixed hue per arm, so an arm keeps its color in every figure of
# the partitions panel.
ARM_COLORS = dict(
    zip(["intra", "inter", "null", "confidence", "predicted_class"], SERIES)
)
# The marker of each operating point: a geometric arm's `selected`,
# or a bin arm's rule.
_POINT_SYMBOLS = {
    "selected": "star",
    "fixed15": "circle",
    "cbrt": "diamond",
    "per150": "square",
}
# The rule each arm shows when a figure has room for one point per arm.
OPERATING_RULES = {
    "intra": "selected",
    "inter": "selected",
    "null": "selected",
    "confidence": "cbrt",
    "predicted_class": "cbrt",
}
# Most classes that get a numbered cell or their own color.
_MAX_LABELLED = 15
_MAX_COLORED = len(SERIES)
# The feature importances drawn, the largest ones.
_TOP_IMPORTANCES = 20


def _stdev(values: Sequence[float]) -> float:
    """Return the sample standard deviation, 0 for a single value."""
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _empty(message: str) -> dict[str, object]:
    """Draw a figure that only says why there is nothing to draw."""
    note = {
        "text": message,
        "showarrow": False,
        "xref": "paper",
        "yref": "paper",
        "x": 0.5,
        "y": 0.5,
    }
    layout = {
        "height": 160,
        "xaxis": {"visible": False},
        "yaxis": {"visible": False},
        "annotations": [note],
    }
    return {"data": [], "layout": layout}


def split_bars(classes: Sequence[Mapping]) -> dict[str, object]:
    """Stack each class's rows by split, one bar per class.

    Args:
        classes: The `queries.classes` rows.

    Returns:
        A horizontal stacked-bar figure.
    """
    names = [row["name"] for row in classes]
    traces = [
        {
            "type": "bar",
            "orientation": "h",
            "name": split,
            "y": names,
            "x": [row[f"n_{split}"] for row in classes],
            "marker": {"color": color, "line": {"width": 2, "color": _SURFACE}},
            "hovertemplate": f"%{{y}} · {split}: %{{x:,}}<extra></extra>",
        }
        for split, color in _SPLIT_COLORS.items()
    ]
    layout = {
        "barmode": "stack",
        "height": max(260, 36 * len(classes) + 120),
        "xaxis": {"title": "rows"},
        "yaxis": {"autorange": "reversed", "type": "category"},
    }
    return {"data": traces, "layout": layout}


def class_score_bars(classes: Sequence[Mapping]) -> dict[str, object]:
    """Draw the test precision, recall and F1 of every class.

    Args:
        classes: The `queries.class_metrics` rows.

    Returns:
        A grouped-bar figure.
    """
    names = [row["name"] for row in classes]
    traces = [
        {
            "type": "bar",
            "name": score,
            "x": names,
            "y": [row[score] for row in classes],
            "marker": {"color": color},
        }
        for score, color in _SCORE_COLORS.items()
    ]
    layout = {
        "barmode": "group",
        "height": _HEIGHT,
        "xaxis": {"type": "category"},
        "yaxis": {"range": [0, 1.02]},
    }
    return {"data": traces, "layout": layout}


def confusion_heatmap(counts: np.ndarray, names: Sequence[str]) -> dict[str, object]:
    """Draw the test confusion matrix, each row as a share of its class.

    Args:
        counts: The true-by-predicted counts of `artifacts.confusion`.
        names: The class names, in id order.

    Returns:
        A heatmap figure; the cells hold their share when the classes
        are few, the counts show on hover.
    """
    totals = counts.sum(axis=1, keepdims=True)
    share = np.divide(counts, totals, out=np.zeros(counts.shape), where=totals > 0)
    trace = {
        "type": "heatmap",
        "x": list(names),
        "y": list(names),
        "z": share.round(4).tolist(),
        "customdata": counts.tolist(),
        "zmin": 0,
        "zmax": 1,
        "colorscale": _SEQUENTIAL,
        "colorbar": {"title": "% of true", "thickness": 12},
        "xgap": 2,
        "ygap": 2,
        "hovertemplate": (
            "true %{y}<br>predicted %{x}<br>%{z:.1%} (%{customdata:,} rows)"
            "<extra></extra>"
        ),
    }
    notes = []
    if len(names) <= _MAX_LABELLED:
        for i, row in enumerate(share):
            for j, value in enumerate(row):
                dark = value > 0.5
                notes.append(
                    {
                        "x": names[j],
                        "y": names[i],
                        "text": f"{value * 100:.0f}" if value >= 0.005 else "",
                        "showarrow": False,
                        "font": {
                            "size": 10,
                            "color": _ON_DARK_FILL if dark else _ON_LIGHT_FILL,
                        },
                    }
                )
    layout = {
        "height": max(380, 38 * len(names) + 140),
        "xaxis": {"title": "predicted", "type": "category"},
        "yaxis": {"title": "true", "type": "category", "autorange": "reversed"},
        "annotations": notes,
    }
    return {"data": [trace], "layout": layout}


def grid_search(candidates: Sequence[Mapping]) -> dict[str, object]:
    """Draw the cross-validated score of every grid-search candidate.

    Args:
        candidates: The `queries.classifier_grid` rows.

    Returns:
        A figure with one marker per candidate, its std as an error
        bar, the best one apart.
    """
    ranked = sorted(
        candidates,
        key=lambda row: -math.inf if row["mean_score"] is None else row["mean_score"],
    )
    traces = []
    for best, name, color in ((0, "candidate", SERIES[0]), (1, "best", SERIES[1])):
        rows = [row for row in ranked if row["best"] == best]
        traces.append(
            {
                "type": "scatter",
                "mode": "markers",
                "name": name,
                "x": [row["candidate"] for row in rows],
                "y": [row["mean_score"] for row in rows],
                "error_y": {
                    "type": "data",
                    "array": [row["std_score"] or 0 for row in rows],
                },
                "marker": {"color": color, "size": 9},
                "text": [str(row["params"]) for row in rows],
                "hovertemplate": (
                    "candidate %{x}<br>%{y:.4f}<br>%{text}<extra></extra>"
                ),
            }
        )
    layout = {
        "height": _HEIGHT,
        "xaxis": {"title": "candidate", "type": "category"},
        "yaxis": {"title": "mean CV score"},
    }
    return {"data": traces, "layout": layout}


def loss_curve(history: Sequence[Mapping]) -> dict[str, object]:
    """Draw a DL classifier's training loss step by step.

    Args:
        history: The `queries.history` rows.

    Returns:
        A line figure, thinned to about two thousand points.
    """
    stride = max(1, len(history) // 2000)
    rows = history[::stride]
    trace = {
        "type": "scatter",
        "mode": "lines",
        "name": "training loss",
        "x": [row["step"] for row in rows],
        "y": [row["loss"] for row in rows],
        "line": {"color": SERIES[0], "width": 2},
    }
    layout = {
        "height": _HEIGHT,
        "xaxis": {"title": "step"},
        "yaxis": {"title": "loss"},
        "showlegend": False,
    }
    return {"data": [trace], "layout": layout}


def regions_per_class(clusterings: Sequence[Mapping]) -> dict[str, object]:
    """Draw how many regions each class was divided into.

    Args:
        clusterings: The `queries.class_clusterings` rows.

    Returns:
        A bar figure.
    """
    trace = {
        "type": "bar",
        "x": [row["name"] for row in clusterings],
        "y": [row["n_clusters"] for row in clusterings],
        "marker": {"color": SERIES[0]},
    }
    layout = {
        "height": _HEIGHT,
        "xaxis": {"type": "category"},
        "yaxis": {"title": "regions"},
        "showlegend": False,
    }
    return {"data": [trace], "layout": layout}


def region_size_box(rows: Sequence[Mapping]) -> dict[str, object]:
    """Box the train rows of the regions of each class.

    Args:
        rows: The region rows.

    Returns:
        A box figure, every region a point.
    """
    kept = [row for row in rows if row["n_train"] > 0]
    trace = {
        "type": "box",
        "x": [row["class"] for row in kept],
        "y": [row["n_train"] for row in kept],
        "text": [f"region {row['region_id']}" for row in kept],
        "boxpoints": "all",
        "jitter": 0.5,
        "pointpos": 0,
        "marker": {"color": SERIES[0], "size": 4, "opacity": 0.6},
        "line": {"width": 1.5},
        "fillcolor": "rgba(0,0,0,0)",
    }
    layout = {
        "height": _HEIGHT,
        "xaxis": {"type": "category"},
        "yaxis": {"title": "train rows per region", "rangemode": "tozero"},
        "showlegend": False,
    }
    return {"data": [trace], "layout": layout}


def _rate_points(rows: Sequence[Mapping]) -> list[Mapping]:
    """Return the regions that have a val and a test failure rate."""
    return [
        row
        for row in rows
        if row["val_rate"] is not None and row["test_rate"] is not None
    ]


def _identity_line(top: float = 1) -> dict[str, object]:
    """Build the y = x reference line, from 0 to `top`."""
    return {
        "type": "scatter",
        "mode": "lines",
        "name": "y = x",
        "x": [0, top],
        "y": [0, top],
        "line": {"dash": "dot", "color": NEUTRAL},
        "hoverinfo": "skip",
    }


def val_test_scatter(rows: Sequence[Mapping], *, run_id: int) -> dict[str, object]:
    """Draw each region's test failure rate against its val rate.

    A marker's area grows with the region's test rows, so a rate the
    test barely covers reads as barely visible.

    Args:
        rows: The region rows.
        run_id: The run, for the link of each marker to its region.

    Returns:
        A figure with the `y = x` line and one marker per region.
    """
    points = _rate_points(rows)
    largest = max((row["n_test"] for row in points), default=1) or 1
    trace = {
        "type": "scatter",
        "mode": "markers",
        "name": "region",
        "x": [row["val_rate"] for row in points],
        "y": [row["test_rate"] for row in points],
        "marker": {
            "color": SERIES[0],
            "opacity": 0.7,
            "size": [4 + 18 * (row["n_test"] / largest) ** 0.5 for row in points],
            "line": {"width": 1, "color": _SURFACE},
        },
        "text": [
            f"region {row['region_id']} · {row['class']}<br>"
            f"n_val {row['n_val']:,} · n_test {row['n_test']:,}"
            for row in points
        ],
        "customdata": [f"/runs/{run_id}/regions/{row['region_id']}" for row in points],
        "hovertemplate": "%{text}<br>val %{x:.3f} · test %{y:.3f}<extra></extra>",
    }
    layout = {
        "height": _HEIGHT,
        "xaxis": {"title": "val failure rate"},
        "yaxis": {"title": "test failure rate"},
        "showlegend": False,
    }
    top = max(
        [row["val_rate"] for row in points] + [row["test_rate"] for row in points],
        default=1,
    )
    return {"data": [_identity_line(top * 1.05), trace], "layout": layout}


def class_rate_bars(rows: Sequence[Mapping]) -> dict[str, object]:
    """Draw each class's failure rate on val and on test.

    The rate is the class's errors over its rows, so a region weighs by
    its size.

    Args:
        rows: The region rows.

    Returns:
        A grouped-bar figure.
    """
    names = list(dict.fromkeys(row["class"] for row in rows))
    traces = []
    for split in ("val", "test"):
        rates = []
        for name in names:
            members = [row for row in rows if row["class"] == name]
            seen = sum(row[f"n_{split}"] for row in members)
            errors = sum(row[f"n_{split}_error"] or 0 for row in members)
            rates.append(errors / seen if seen else None)
        traces.append(
            {
                "type": "bar",
                "name": split,
                "x": names,
                "y": rates,
                "marker": {"color": _SPLIT_COLORS[split]},
            }
        )
    layout = {
        "barmode": "group",
        "height": _HEIGHT,
        "xaxis": {"type": "category"},
        "yaxis": {"title": "failure rate"},
    }
    return {"data": traces, "layout": layout}


def rate_ecdf(rows: Sequence[Mapping]) -> dict[str, object]:
    """Draw the distribution of the regions' failure rates per split.

    Args:
        rows: The region rows.

    Returns:
        A step figure: for each rate, the share of regions at or below.
    """
    traces = []
    for split in ("val", "test"):
        rates = sorted(
            row[f"{split}_rate"] for row in rows if row[f"{split}_rate"] is not None
        )
        traces.append(
            {
                "type": "scatter",
                "mode": "lines",
                "name": split,
                "x": rates,
                "y": [(i + 1) / len(rates) for i in range(len(rates))],
                "line": {"shape": "hv", "color": _SPLIT_COLORS[split], "width": 2},
            }
        )
    layout = {
        "height": _HEIGHT,
        "xaxis": {"title": "region failure rate"},
        "yaxis": {"title": "share of regions", "range": [0, 1.02]},
    }
    return {"data": traces, "layout": layout}


def predictor_bars(
    scores: Mapping[str, Mapping[str, float | None]], score: str
) -> dict[str, object]:
    """Draw one score of every predictor, against the test rates.

    Args:
        scores: Each predictor's scores by name, as the metrics hold
            them.
        score: The score name, e.g. "spearman".

    Returns:
        A bar figure with the predictors that have a value.
    """
    shown = [key for key in PREDICTORS if scores.get(key, {}).get(score) is not None]
    trace = {
        "type": "bar",
        "x": [PREDICTORS[key] for key in shown],
        "y": [scores[key][score] for key in shown],
        "marker": {"color": [PREDICTOR_COLORS[key] for key in shown]},
        "hovertemplate": f"%{{x}}<br>{score} %{{y:.4f}}<extra></extra>",
    }
    layout = {
        "height": 300,
        "xaxis": {"type": "category"},
        "yaxis": {"title": score},
        "showlegend": False,
    }
    return {"data": [trace], "layout": layout}


def predicted_vs_observed(rows: Sequence[Mapping], *, run_id: int) -> dict[str, object]:
    """Draw each predictor's rate against the test rate, one panel each.

    Args:
        rows: The region rows.
        run_id: The run, for the link of each marker to its region.

    Returns:
        A figure of side-by-side panels over the used regions, those
        that have a prediction and a test rate; the panels share both
        axes and the `y = x` line.
    """
    scored = [row for row in rows if row["used"] == 1 and row["test_rate"] is not None]
    panels = [key for key in PREDICTORS if any(row[key] is not None for row in scored)]
    if not panels:
        return _empty("No predictor has a prediction for a region with test rows.")
    gap = 0.04
    width = (1 - gap * (len(panels) - 1)) / len(panels)
    traces, layout, notes = [], {"height": 360, "showlegend": False}, []
    for i, key in enumerate(panels, start=1):
        suffix = "" if i == 1 else str(i)
        points = [row for row in scored if row[key] is not None]
        domain = [(i - 1) * (width + gap), (i - 1) * (width + gap) + width]
        layout[f"xaxis{suffix}"] = {
            "domain": domain,
            "range": [0, 1],
            "title": f"{PREDICTORS[key]} estimate",
            "anchor": f"y{suffix}",
        }
        layout[f"yaxis{suffix}"] = {
            "range": [0, 1],
            "anchor": f"x{suffix}",
            **({"title": "test failure rate"} if i == 1 else {"matches": "y"}),
        }
        notes.append(
            {
                "text": PREDICTORS[key],
                "showarrow": False,
                "xref": "paper",
                "yref": "paper",
                "x": domain[0],
                "y": 1.08,
                "xanchor": "left",
            }
        )
        traces.append(
            {**_identity_line(), "xaxis": f"x{suffix}", "yaxis": f"y{suffix}"}
        )
        traces.append(
            {
                "type": "scatter",
                "mode": "markers",
                "name": PREDICTORS[key],
                "xaxis": f"x{suffix}",
                "yaxis": f"y{suffix}",
                "x": [row[key] for row in points],
                "y": [row["test_rate"] for row in points],
                "marker": {"color": PREDICTOR_COLORS[key], "size": 7, "opacity": 0.7},
                "text": [
                    f"region {row['region_id']} · {row['class']}" for row in points
                ],
                "customdata": [
                    f"/runs/{run_id}/regions/{row['region_id']}" for row in points
                ],
                "hovertemplate": (
                    "%{text}<br>estimate %{x:.3f} · test %{y:.3f}" "<extra></extra>"
                ),
            }
        )
    layout["annotations"] = notes
    return {"data": traces, "layout": layout}


def importance_bars(rows: Sequence[Mapping]) -> dict[str, object]:
    """Draw the regressor's largest feature importances.

    Args:
        rows: The `queries.importances` rows, largest first.

    Returns:
        A horizontal bar figure of the twenty largest, the largest on
        top.
    """
    kept = list(rows[:_TOP_IMPORTANCES])
    trace = {
        "type": "bar",
        "orientation": "h",
        "y": [row["feature"] for row in kept],
        "x": [row["importance"] for row in kept],
        "marker": {"color": SERIES[0]},
    }
    layout = {
        "height": max(260, 24 * len(kept) + 100),
        "xaxis": {"title": "mean importance"},
        "yaxis": {"autorange": "reversed", "type": "category"},
        "showlegend": False,
        "margin": {"l": 220},
    }
    return {"data": [trace], "layout": layout}


def arm_bars(points: Sequence[Mapping], metric: str) -> dict[str, object]:
    """Draw one metric at each arm's operating point of a run.

    Args:
        points: The `queries.operating_points` rows of one run.
        metric: A column of them, e.g. "skill".

    Returns:
        A bar figure, one bar per arm with a value at the rule
        `OPERATING_RULES` gives it.
    """
    shown = [
        row
        for row in points
        if OPERATING_RULES.get(row["arm"]) == row["rule"] and row[metric] is not None
    ]
    shown.sort(key=lambda row: list(ARM_COLORS).index(row["arm"]))
    trace = {
        "type": "bar",
        "x": [f"{row['arm']} ({row['rule']})" for row in shown],
        "y": [row[metric] for row in shown],
        "marker": {"color": [ARM_COLORS[row["arm"]] for row in shown]},
        "hovertemplate": f"%{{x}}<br>{metric} %{{y:.4f}}<extra></extra>",
    }
    layout = {
        "height": 300,
        "xaxis": {"type": "category"},
        "yaxis": {"title": metric},
        "showlegend": False,
    }
    return {"data": [trace], "layout": layout}


def seed_strip(
    series: Mapping[str, Sequence[tuple[int, float | None]]], y_title: str
) -> dict[str, object]:
    """Draw each seed's value of some metrics, with their mean.

    Args:
        series: Per label of the x axis, the `(seed, value)` pairs.
        y_title: The y axis title.

    Returns:
        A figure with a marker per seed and a dash at the mean.
    """
    labels, values, hover, means = [], [], [], []
    for label, pairs in series.items():
        kept = [(seed, value) for seed, value in pairs if value is not None]
        for seed, value in kept:
            labels.append(label)
            values.append(value)
            hover.append(f"seed {seed}")
        means.append(statistics.mean(value for _, value in kept) if kept else None)
    traces = [
        {
            "type": "scatter",
            "mode": "markers",
            "name": "seed",
            "x": labels,
            "y": values,
            "text": hover,
            "marker": {"color": SERIES[0], "size": 9, "opacity": 0.8},
            "hovertemplate": "%{text}<br>%{y:.4f}<extra></extra>",
        },
        {
            "type": "scatter",
            "mode": "markers",
            "name": "mean",
            "x": list(series),
            "y": means,
            "marker": {
                "symbol": "line-ew",
                "size": 26,
                "color": _INK,
                "line": {"width": 3, "color": _INK},
            },
            "hovertemplate": "mean %{y:.4f}<extra></extra>",
        },
    ]
    layout = {
        "height": 300,
        "xaxis": {
            "type": "category",
            "categoryorder": "array",
            "categoryarray": list(series),
        },
        "yaxis": {"title": y_title},
    }
    return {"data": traces, "layout": layout}


def group_bars(
    traces: Sequence[tuple[str, str, Mapping[str, Sequence[float]]]],
    *,
    x_title: str,
    y_title: str,
) -> dict[str, object]:
    """Draw the mean of some values per x, with its std as an error bar.

    Args:
        traces: One `(name, color, values by x)` per trace.
        x_title: The x axis title.
        y_title: The y axis title.

    Returns:
        A grouped-bar figure with ascending categories; the hover shows
        how many values each bar averages.
    """
    data = []
    for name, color, groups in traces:
        groups = dict(sorted(groups.items()))
        data.append(
            {
                "type": "bar",
                "name": name,
                "x": list(groups),
                "y": [statistics.mean(values) for values in groups.values()],
                "error_y": {
                    "type": "data",
                    "array": [_stdev(values) for values in groups.values()],
                },
                "marker": {"color": color},
                "text": [f"n = {len(values)}" for values in groups.values()],
                "textposition": "none",
                "hovertemplate": (
                    f"%{{x}}<br>%{{y:.4f}} (%{{text}})<extra>{name}</extra>"
                ),
            }
        )
    layout = {
        "barmode": "group",
        "height": _HEIGHT,
        "xaxis": {
            "title": x_title,
            "type": "category",
            "categoryorder": "category ascending",
        },
        "yaxis": {"title": y_title},
    }
    return {"data": data, "layout": layout}


def region_scatter(
    rows: Sequence[Mapping], *, x: str, y: str, run_id: int
) -> dict[str, object]:
    """Draw `y` against `x` per region, a color per class if few.

    Args:
        rows: The region rows.
        x: The region column of the x axis.
        y: The region column of the y axis.
        run_id: The run, for the link of each marker to its region.

    Returns:
        A figure; regions without a value of `x` or `y` are left out.
        With more classes than categorical colors every region has the
        first color and the class shows on hover.
    """
    names = sorted({row["class"] for row in rows})
    colored = len(names) <= _MAX_COLORED
    traces = []
    for i, name in enumerate(names if colored else [""]):
        points = [
            row
            for row in rows
            if (not colored or row["class"] == name)
            and row[x] is not None
            and row[y] is not None
        ]
        traces.append(
            {
                "type": "scatter",
                "mode": "markers",
                "name": name,
                "showlegend": colored,
                "x": [row[x] for row in points],
                "y": [row[y] for row in points],
                "text": [
                    f"region {row['region_id']} · {row['class']}" for row in points
                ],
                "customdata": [
                    f"/runs/{run_id}/regions/{row['region_id']}" for row in points
                ],
                "marker": {"color": SERIES[i], "size": 7, "opacity": 0.75},
                "hovertemplate": "%{text}<br>%{x:.4g} · %{y:.4g}<extra></extra>",
            }
        )
    layout = {"height": _HEIGHT, "xaxis": {"title": x}, "yaxis": {"title": y}}
    return {"data": traces, "layout": layout}


def _fade(color: str) -> str:
    """Return a hex color as translucent rgba, for a band fill."""
    red, green, blue = (int(color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({red}, {green}, {blue}, 0.15)"


def _mean_points(rows: Sequence[dict], key: str, x: str, y: str) -> dict[str, list]:
    """Average `x` and `y` over the rows of each value of `key`.

    A row whose `y` is None is left out, and so is a value of `key`
    that only such rows hold.

    Returns:
        The lists `key`, `x`, `y`, `y_min` and `y_max`, one item per
        value of `key`, sorted by it; `y_min` and `y_max` bound the
        `y` values averaged.
    """
    groups: dict[object, list[dict]] = {}
    for row in rows:
        if row[y] is not None:
            groups.setdefault(row[key], []).append(row)
    groups = dict(sorted(groups.items()))
    return {
        "key": list(groups),
        "x": [statistics.mean(row[x] for row in group) for group in groups.values()],
        "y": [statistics.mean(row[y] for row in group) for group in groups.values()],
        "y_min": [min(row[y] for row in group) for group in groups.values()],
        "y_max": [max(row[y] for row in group) for group in groups.values()],
    }


def _sweep_traces(arm: str, color: str, sweep: dict[str, list], x: str) -> list[dict]:
    """Build an arm's sweep line, plus its band where draws vary.

    The band only draws for `x == "n_groups"`: `level` orders it, and
    `n_groups` falls with `level`, but `val_median` has no such order,
    so tracing it out and back could fold the fill on itself.
    """
    if not sweep["key"]:
        return []
    traces = []
    if x == "n_groups" and sweep["y_min"] != sweep["y_max"]:
        traces.append(
            {
                "type": "scatter",
                "mode": "lines",
                "name": f"{arm} (min-max)",
                "legendgroup": arm,
                "x": sweep["x"] + sweep["x"][::-1],
                "y": sweep["y_max"] + sweep["y_min"][::-1],
                "fill": "toself",
                "fillcolor": _fade(color),
                "line": {"width": 0},
                "hoverinfo": "skip",
                "showlegend": False,
            }
        )
    traces.append(
        {
            "type": "scatter",
            "mode": "lines+markers",
            "name": arm,
            "legendgroup": arm,
            "x": sweep["x"],
            "y": sweep["y"],
            "line": {"color": color},
        }
    )
    return traces


def _operating_point_trace(
    arm: str, color: str, points: dict[str, list]
) -> dict | None:
    """Build an arm's operating-point markers, or None if empty."""
    if not points["key"]:
        return None
    return {
        "type": "scatter",
        "mode": "markers",
        "name": f"{arm} operating point",
        "legendgroup": arm,
        "x": points["x"],
        "y": points["y"],
        "marker": {
            "color": color,
            "size": 12,
            "symbol": [_POINT_SYMBOLS[rule] for rule in points["key"]],
            "line": {"width": 1, "color": _INK},
        },
        "text": points["key"],
        "hoverinfo": "text+x+y+name",
    }


def partition_curves(rows: Sequence[dict], *, metric: str, x: str) -> dict[str, object]:
    """Draw `metric` against `x`, one line per geometric arm.

    `intra` and `inter` draw their level sweep as one line; `null`
    draws the mean of its draws with a min-max band. Every arm also
    gets a marker per operating point (`selected`, or a bin rule),
    averaged over draws the same way the sweep is.

    Args:
        rows: The partitions rows of one execution.
        metric: The partitions column of the y axis.
        x: The partitions column of the x axis: `n_groups`, on a log
            scale, or `val_median`.

    Returns:
        A Plotly figure; rows without a value of `metric` are left
        out, a bin arm (`confidence`, `predicted_class`) draws markers
        only, since it sweeps no level, and the band only draws for
        `x == "n_groups"`.
    """
    traces = []
    for arm, color in ARM_COLORS.items():
        arm_rows = [row for row in rows if row["arm"] == arm]
        sweep = _mean_points(
            [row for row in arm_rows if row["rule"] == "level"], "level", x, metric
        )
        traces += _sweep_traces(arm, color, sweep, x)
        points = _mean_points(
            [row for row in arm_rows if row["rule"] != "level"], "rule", x, metric
        )
        operating_point = _operating_point_trace(arm, color, points)
        if operating_point is not None:
            traces.append(operating_point)
    layout = {
        "height": _HEIGHT,
        "xaxis": {"title": x, "type": "log" if x == "n_groups" else "linear"},
        "yaxis": {"title": metric},
    }
    return {"data": traces, "layout": layout}


def partition_class_curves(
    rows: Sequence[dict], *, class_names: Mapping[int, str] | None = None
) -> dict[str, object]:
    """Draw intra's granularity loss against level, one line per class.

    Args:
        rows: The partition_class_levels rows of one execution.
        class_names: Each class's name, as the regions page shows it;
            a class missing from it, or no mapping at all, falls back
            to its id.

    Returns:
        A Plotly figure; each class's selected level gets a star.
    """
    class_names = class_names or {}
    traces = []
    for class_id in sorted({row["class_id"] for row in rows}):
        class_rows = sorted(
            (row for row in rows if row["class_id"] == class_id),
            key=lambda row: row["level"],
        )
        traces.append(
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": class_names.get(class_id, str(class_id)),
                "x": [row["level"] for row in class_rows],
                "y": [row["loss"] for row in class_rows],
            }
        )
    selected = [row for row in rows if row["selected"]]
    if selected:
        traces.append(
            {
                "type": "scatter",
                "mode": "markers",
                "name": "selected",
                "x": [row["level"] for row in selected],
                "y": [row["loss"] for row in selected],
                "marker": {"symbol": "star", "size": 14, "color": _INK},
            }
        )
    layout = {
        "height": _HEIGHT,
        "xaxis": {"title": "level", "dtick": 1},
        "yaxis": {"title": "loss"},
    }
    return {"data": traces, "layout": layout}


def partition_val_rates(rows: Sequence[dict]) -> dict[str, object]:
    """Box the val failure rate of every operating point's groups.

    Args:
        rows: The partition_groups rows of one execution.

    Returns:
        A Plotly figure with one box per arm and rule; a group without
        a val rate is left out.
    """
    rates_by_arm: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        if row["val_rate"] is not None:
            by_rule = rates_by_arm.setdefault(row["arm"], {})
            by_rule.setdefault(row["rule"], []).append(row["val_rate"])
    traces = []
    for arm in ARM_COLORS:
        for rule, values in sorted(rates_by_arm.get(arm, {}).items()):
            name = arm if rule == "selected" else f"{arm} ({rule})"
            traces.append(
                {
                    "type": "box",
                    "name": name,
                    "y": values,
                    "marker": {"color": ARM_COLORS[arm]},
                }
            )
    layout = {"height": _HEIGHT, "yaxis": {"title": "val_rate"}, "showlegend": False}
    return {"data": traces, "layout": layout}


def partition_val_test_scatter(rows: Sequence[dict]) -> dict[str, object]:
    """Draw each operating-point group's test rate against its val rate.

    A marker's size grows with its group's val row count, so a group
    the val estimate barely covers reads as barely visible.

    Args:
        rows: The partition_groups rows of one execution.

    Returns:
        A Plotly figure: the `val = test` line, then one trace per
        arm; a group without a val or a test rate is left out.
    """
    scored = [
        row
        for row in rows
        if row["val_rate"] is not None and row["test_rate"] is not None
    ]
    max_n_val = max((row["n_val"] for row in scored), default=1)
    traces = [
        {
            "type": "scatter",
            "mode": "lines",
            "name": "val = test",
            "x": [0, 1],
            "y": [0, 1],
            "line": {"dash": "dot", "color": NEUTRAL},
            "hoverinfo": "skip",
        }
    ]
    for arm, color in ARM_COLORS.items():
        points = [row for row in scored if row["arm"] == arm]
        if points:
            traces.append(
                {
                    "type": "scatter",
                    "mode": "markers",
                    "name": arm,
                    "x": [row["val_rate"] for row in points],
                    "y": [row["test_rate"] for row in points],
                    "marker": {
                        "color": color,
                        "size": [
                            4 + 20 * (row["n_val"] / max_n_val) ** 0.5 for row in points
                        ],
                        "line": {"width": 1, "color": _SURFACE},
                    },
                    "text": [
                        f"{row['rule']} group {row['group_id']}, n_val={row['n_val']}"
                        for row in points
                    ],
                    "hoverinfo": "text+x+y",
                }
            )
    layout = {
        "height": _HEIGHT,
        "xaxis": {"title": "val_rate"},
        "yaxis": {"title": "test_rate"},
    }
    return {"data": traces, "layout": layout}

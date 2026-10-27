"""Plotly figures of the dashboard, as JSON-ready dicts.

A figure carries its structure and its light-theme colors; the page's
`dashboard.js` adds the margins, fonts and grid colors, and swaps the
colors below for their dark-theme steps.
"""

__all__ = [
    "NEUTRAL",
    "PREDICTOR_COLORS",
    "SERIES",
    "class_rate_bars",
    "class_rate_gap",
    "class_score_bars",
    "confusion_heatmap",
    "grid_search",
    "group_bars",
    "importance_bars",
    "loss_curve",
    "predicted_vs_observed",
    "predictor_bars",
    "rate_ecdf",
    "region_rate_bins",
    "region_size_box",
    "regions_per_class",
    "seed_strip",
    "split_bars",
    "val_test_scatter",
]

import bisect
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
# The upper edges of the failure-rate bins, after a bin of its own for a
# rate of exactly 0: a good classifier leaves most regions error-free.
_RATE_EDGES = [0.05, 0.1, 0.2, 0.3, 0.5]
_RATE_BINS = ["0", "0–5%", "5–10%", "10–20%", "20–30%", "30–50%", "50–100%"]
# Most classes that get a numbered cell.
_MAX_LABELLED = 15
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


def val_test_scatter(rows: Sequence[Mapping]) -> dict[str, object]:
    """Draw each region's test failure rate against its val rate.

    A marker's area grows with the region's test rows, so a rate the
    test barely covers reads as barely visible.

    Args:
        rows: The region rows.

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


def _rate_bin(rate: float) -> str:
    """Return the label of the failure-rate bin of `rate`."""
    if rate == 0:
        return _RATE_BINS[0]
    return _RATE_BINS[1 + bisect.bisect_left(_RATE_EDGES, rate)]


def _rate_bin_bars(
    traces: Sequence[tuple[str, str, Sequence[tuple[float | None, int]]]],
    *,
    measure: str,
) -> dict[str, object]:
    """Draw how a set of clusters spreads over the failure-rate bins.

    Args:
        traces: One `(name, color, clusters)` per trace, a cluster being
            its `(failure rate, rows)`; a cluster without a rate is left
            out.
        measure: "count" for the clusters in each bin, "size" for their
            mean rows.

    Returns:
        A grouped-bar figure with every bin on the x axis, empty ones
        included; the hover shows both measures.
    """
    data = []
    for name, color, clusters in traces:
        sizes: dict[str, list[int]] = {label: [] for label in _RATE_BINS}
        for rate, n_rows in clusters:
            if rate is not None and not math.isnan(rate):
                sizes[_rate_bin(rate)].append(n_rows)
        counts = [len(sizes[label]) for label in _RATE_BINS]
        means = [
            statistics.mean(sizes[label]) if sizes[label] else None
            for label in _RATE_BINS
        ]
        data.append(
            {
                "type": "bar",
                "name": name,
                "x": _RATE_BINS,
                "y": counts if measure == "count" else means,
                "customdata": [[count, mean] for count, mean in zip(counts, means)],
                "marker": {"color": color},
                "hovertemplate": (
                    "%{x}<br>%{customdata[0]} in the bin · mean "
                    f"%{{customdata[1]:,.1f}} rows<extra>{name}</extra>"
                ),
            }
        )
    layout = {
        "barmode": "group",
        "height": _HEIGHT,
        "xaxis": {
            "title": "failure rate",
            "type": "category",
            "categoryorder": "array",
            "categoryarray": _RATE_BINS,
        },
        "yaxis": {
            "title": "clusters" if measure == "count" else "mean rows per cluster"
        },
    }
    return {"data": data, "layout": layout}


def region_rate_bins(rows: Sequence[Mapping], *, measure: str) -> dict[str, object]:
    """Bin the regions by failure rate, on val and on test.

    Args:
        rows: The region rows.
        measure: "count" or "size", as `_rate_bin_bars` takes it; a
            region's size is its rows in the split of the rate.

    Returns:
        A grouped-bar figure with a trace per split.
    """
    traces = [
        (
            split,
            _SPLIT_COLORS[split],
            [(row[f"{split}_rate"], row[f"n_{split}"]) for row in rows],
        )
        for split in ("val", "test")
    ]
    return _rate_bin_bars(traces, measure=measure)


def class_rate_gap(rows: Sequence[Mapping]) -> dict[str, object]:
    """Box each class's gap between the val and the test failure rate.

    Args:
        rows: The region rows.

    Returns:
        A box figure of `val_rate - test_rate`, every region a point,
        with a line at 0: a point above it is a region whose val rate
        overshot the test one.
    """
    points = _rate_points(rows)
    trace = {
        "type": "box",
        "x": [row["class"] for row in points],
        "y": [row["val_rate"] - row["test_rate"] for row in points],
        "text": [
            f"region {row['region_id']}<br>"
            f"n_val {row['n_val']:,} · n_test {row['n_test']:,}"
            for row in points
        ],
        "boxpoints": "all",
        "jitter": 0.5,
        "pointpos": 0,
        "marker": {"color": SERIES[0], "size": 4, "opacity": 0.6},
        "line": {"width": 1.5},
        "fillcolor": "rgba(0,0,0,0)",
        "hovertemplate": "%{text}<br>val − test %{y:.3f}<extra></extra>",
    }
    layout = {
        "height": _HEIGHT,
        "xaxis": {"type": "category"},
        "yaxis": {"title": "val − test failure rate", "zeroline": False},
        "shapes": [
            {
                "type": "line",
                "xref": "paper",
                "x0": 0,
                "x1": 1,
                "y0": 0,
                "y1": 0,
                "line": {"dash": "dot", "color": NEUTRAL},
            }
        ],
        "showlegend": False,
    }
    return {"data": [trace], "layout": layout}


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


def predicted_vs_observed(rows: Sequence[Mapping]) -> dict[str, object]:
    """Draw each predictor's rate against the test rate, one panel each.

    Args:
        rows: The region rows.

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

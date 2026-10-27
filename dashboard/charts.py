"""Plotly figures of the dashboard, as JSON-ready dicts.

A figure carries its structure and its light-theme colors; the page's
`dashboard.js` adds the margins, fonts and grid colors, and swaps the
colors below for their dark-theme steps.
"""

__all__ = [
    "NEUTRAL",
    "PREDICTOR_COLORS",
    "SERIES",
    "class_score_bars",
    "confusion_heatmap",
    "grid_search",
    "group_bars",
    "importance_bars",
    "loss_curve",
    "predictor_bars",
    "seed_strip",
    "shrinkage_scatter",
    "split_bars",
    "test_scatter",
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
# Most classes that get a numbered cell.
_MAX_LABELLED = 15
# The feature importances drawn, the largest ones.
_TOP_IMPORTANCES = 20


def _stdev(values: Sequence[float]) -> float:
    """Return the sample standard deviation, 0 for a single value."""
    return statistics.stdev(values) if len(values) > 1 else 0.0


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


def _leaf_scatter(
    leaves: Sequence[Mapping],
    *,
    x_key: str,
    y_key: str,
    size_key: str,
    x_title: str,
    y_title: str,
    run_id: int,
) -> dict[str, object]:
    """Draw leaves as markers sized by a count, linked to their page."""
    kept = [
        leaf for leaf in leaves if leaf[x_key] is not None and leaf[y_key] is not None
    ]
    biggest = max((leaf[size_key] for leaf in kept), default=1) or 1
    trace = {
        "type": "scatter",
        "mode": "markers",
        "x": [leaf[x_key] for leaf in kept],
        "y": [leaf[y_key] for leaf in kept],
        "text": [
            f"leaf {leaf['node_id']} · {leaf['class']} · {leaf[size_key]} rows"
            for leaf in kept
        ],
        "customdata": [f"/runs/{run_id}/leaves/{leaf['node_id']}" for leaf in kept],
        "marker": {
            "color": SERIES[0],
            "opacity": 0.7,
            "size": [6 + 18 * (leaf[size_key] / biggest) ** 0.5 for leaf in kept],
        },
        "hovertemplate": "%{text}<br>%{x:.3f} · %{y:.3f}<extra></extra>",
    }
    identity = {
        "type": "scatter",
        "mode": "lines",
        "x": [0, 1],
        "y": [0, 1],
        "line": {"color": NEUTRAL, "dash": "dot"},
        "hoverinfo": "skip",
    }
    layout = {
        "height": _HEIGHT,
        "showlegend": False,
        "xaxis": {"title": x_title},
        "yaxis": {"title": y_title},
    }
    return {"data": [identity, trace], "layout": layout}


def shrinkage_scatter(leaves: Sequence[Mapping], *, run_id: int) -> dict[str, object]:
    """Draw each leaf's estimate against its raw val failure rate.

    Args:
        leaves: The `queries.leaves` rows.
        run_id: The run, for the link of each marker to its leaf.

    Returns:
        A figure of the leaves with val rows, sized by them; the points
        off the diagonal are the ones the shrinkage moved.
    """
    return _leaf_scatter(
        leaves,
        x_key="val_rate",
        y_key="q",
        size_key="n_val",
        x_title="val failure rate of the leaf",
        y_title="estimate q",
        run_id=run_id,
    )


def test_scatter(leaves: Sequence[Mapping], *, run_id: int) -> dict[str, object]:
    """Draw each leaf's test failure rate against its estimate.

    Args:
        leaves: The `queries.leaves` rows.
        run_id: The run, for the link of each marker to its leaf.

    Returns:
        A figure of the leaves with test rows, sized by them.
    """
    return _leaf_scatter(
        leaves,
        x_key="q",
        y_key="test_rate",
        size_key="n_test",
        x_title="estimate q",
        y_title="test failure rate of the leaf",
        run_id=run_id,
    )

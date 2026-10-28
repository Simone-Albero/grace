"""Plotly figures of the dashboard, as JSON-ready dicts.

A figure carries its structure and its light-theme colors; the page's
`dashboard.js` adds the margins, fonts and grid colors, and swaps the
colors below for their dark-theme steps.
"""

__all__ = [
    "NEUTRAL",
    "PREDICTOR_COLORS",
    "SERIES",
    "TREE_COLORS",
    "class_score_bars",
    "confusion_heatmap",
    "grid_search",
    "group_bars",
    "leaf_map",
    "leaf_size_bars",
    "loss_curve",
    "predictor_bars",
    "seed_strip",
    "shrinkage_scatter",
    "split_bars",
    "test_scatter",
    "tree_diagram",
]

import math
import statistics
from collections.abc import Mapping, Sequence

import numpy as np

from dashboard.artifacts import ClassPlane
from dashboard.queries import PREDICTORS, SCORES

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
_INK = "#0b0b0b"
_ON_DARK_FILL = "#ffffff"
_ON_LIGHT_FILL = "#0b0b0b"
_SEQUENTIAL = [[0, "#cde2fb"], [0.5, "#3987e5"], [1, "#0d366b"]]
# A shape cannot take its color from a colorscale, and in the dark theme
# the colorscales are reversed, so the circles of `leaf_map` take one of
# these, sampled from `_SEQUENTIAL`. `dashboard.js` swaps step i with
# step 7 - i, and a test keeps the two lists equal.
_FILL_STEPS = [
    "#bad7f8",
    "#96c0f3",
    "#70a9ed",
    "#4c92e8",
    "#347dd6",
    "#2869b7",
    "#1e5499",
    "#12407a",
]
# What colors the nodes of a class's tree, by the key of the node.
TREE_COLORS = {
    "purity": "purity",
    "oof_rate": "train rate (out-of-fold)",
    "val_rate": "val rate",
    "test_rate": "test rate",
}
# An entity keeps its color in every figure.
_SPLIT_COLORS = {"train": SERIES[0], "val": SERIES[1], "test": SERIES[2]}
_SCORE_COLORS = {"precision": SERIES[0], "recall": SERIES[1], "f1": SERIES[2]}
PREDICTOR_COLORS = dict(zip(PREDICTORS, SERIES))
# Most classes that get a numbered cell.
_MAX_LABELLED = 15


def _stdev(values: Sequence[float]) -> float:
    """Return the sample standard deviation, 0 for a single value."""
    return statistics.stdev(values) if len(values) > 1 else 0.0


def split_bars(classes: Sequence[Mapping]) -> dict[str, object]:
    """Draw each class's rows per split, on a log axis.

    A log axis keeps a class thousands of times smaller than the largest
    visible, and grouped bars stay honest on it where stacked ones would
    not.

    Args:
        classes: The `queries.classes` rows.

    Returns:
        A horizontal grouped-bar figure, the largest class on top; the
        hover gives the rows and the class's share of the split.
    """
    ranked = sorted(
        classes, key=lambda row: -(row["n_train"] + row["n_val"] + row["n_test"])
    )
    names = [row["name"] for row in ranked]
    traces = []
    for split, color in _SPLIT_COLORS.items():
        counts = [row[f"n_{split}"] for row in ranked]
        total = sum(counts) or 1
        traces.append(
            {
                "type": "bar",
                "orientation": "h",
                "name": split,
                "y": names,
                "x": counts,
                "customdata": [count / total for count in counts],
                "marker": {"color": color},
                "hovertemplate": (
                    f"%{{y}} · {split}: %{{x:,}} rows (%{{customdata:.2%}} of "
                    f"{split})<extra></extra>"
                ),
            }
        )
    layout = {
        "barmode": "group",
        "height": max(260, 54 * len(ranked) + 120),
        "xaxis": {"title": "rows (log scale)", "type": "log"},
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
        score: The score name, e.g. "auroc".

    Returns:
        A bar figure with the predictors that have a value.
    """
    shown = [key for key in PREDICTORS if scores.get(key, {}).get(score) is not None]
    trace = {
        "type": "bar",
        "x": [PREDICTORS[key] for key in shown],
        "y": [scores[key][score] for key in shown],
        "marker": {"color": [PREDICTOR_COLORS[key] for key in shown]},
        "hovertemplate": f"%{{x}}<br>{SCORES[score]} %{{y:.4f}}<extra></extra>",
    }
    layout = {
        "height": 300,
        "xaxis": {"type": "category"},
        "yaxis": {"title": SCORES[score]},
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


def _size_label(bin_: int) -> str:
    """Name a power-of-two size bin: 0, 1, 2–3, 4–7, ..."""
    return str(bin_) if bin_ < 2 else f"{2 ** (bin_ - 1)}–{2**bin_ - 1}"


def leaf_size_bars(leaves: Sequence[Mapping]) -> dict[str, object]:
    """Count the leaves by their rows per split, in power-of-two bins.

    Args:
        leaves: The `queries.leaves` rows.

    Returns:
        A grouped-bar figure with a series per split; test is left out
        when no leaf has a test count. The 0 bin holds the leaves that
        no row of the split reaches.
    """
    counts = {
        split: [leaf[f"n_{split}"] for leaf in leaves if leaf[f"n_{split}"] is not None]
        for split in _SPLIT_COLORS
    }
    counts = {split: values for split, values in counts.items() if values}
    n_bins = (
        max((max(values).bit_length() for values in counts.values()), default=0) + 1
    )
    labels = [_size_label(bin_) for bin_ in range(n_bins)]
    traces = []
    for split, values in counts.items():
        per_bin = [0] * n_bins
        for value in values:
            per_bin[value.bit_length()] += 1
        traces.append(
            {
                "type": "bar",
                "name": split,
                "x": labels,
                "y": per_bin,
                "marker": {"color": _SPLIT_COLORS[split]},
                "hovertemplate": (
                    f"{split} · %{{x}} rows: %{{y:,}} leaves<extra></extra>"
                ),
            }
        )
    layout = {
        "barmode": "group",
        "height": _HEIGHT,
        "xaxis": {"title": "rows in the leaf", "type": "category"},
        "yaxis": {"title": "leaves"},
    }
    return {"data": traces, "layout": layout}


def _format_count(value: int | None) -> str:
    """Format a row count, or a dash when it is unknown."""
    return "–" if value is None else f"{value:,}"


def _format_rate(value: float | None) -> str:
    """Format a rate, or a dash when it is unknown."""
    return "–" if value is None else f"{value:.4f}"


def _tree_hover(node: Mapping) -> str:
    """Format the hover of a tree node: its counts and rates."""
    kind = "leaf" if node["status"] == "leaf" else "internal"
    lines = [
        f"node {node['node_id']} · {kind}",
        (
            f"train {_format_count(node['n_train'])} · "
            f"oof rate {_format_rate(node['oof_rate'])} · "
            f"purity {_format_rate(node['purity'])}"
        ),
        (
            f"val {_format_count(node['n_val'])} · "
            f"val rate {_format_rate(node['val_rate'])}"
        ),
        (
            f"test {_format_count(node['n_test'])} · "
            f"test rate {_format_rate(node['test_rate'])}"
        ),
    ]
    if kind == "leaf":
        lines.append(f"q {_format_rate(node['q'])}")
    return "<br>".join(lines)


def _tree_positions(nodes: Sequence[Mapping]) -> dict[int, float]:
    """Place each node on the x axis, keyed by its id.

    The leaves take 0, 1, 2, ... in depth-first order, and a parent
    sits at the mean of its children.
    """
    children: dict[int, list[int]] = {}
    roots = []
    for node in sorted(nodes, key=lambda node: node["node_id"]):
        if node["parent_id"] is None:
            roots.append(node["node_id"])
        else:
            children.setdefault(node["parent_id"], []).append(node["node_id"])
    positions: dict[int, float] = {}
    next_leaf = 0

    def place(node_id: int) -> None:
        nonlocal next_leaf
        below = children.get(node_id, [])
        for child in below:
            place(child)
        if below:
            positions[node_id] = sum(positions[child] for child in below) / len(below)
        else:
            positions[node_id] = next_leaf
            next_leaf += 1

    for root in roots:
        place(root)
    return positions


def _color_range(nodes: Sequence[Mapping], color: str) -> tuple[float, float]:
    """Return the ends of the color scale of a class's tree and map.

    The ends come from the leaves, since a node's rate pools the rows
    of the leaves below it and so stays within theirs; the tree and the
    map of a class share one scale. A rate goes from 0 to its largest,
    the purity from its smallest to 1.
    """
    values = [
        node[color]
        for node in nodes
        if node["status"] == "leaf" and node[color] is not None
    ]
    if not values:
        return 0.0, 1.0
    if color == "purity":
        low, high = min(values), 1.0
    else:
        low, high = 0.0, max(values)
    if high - low < 1e-9:
        return (high - 1, high) if color == "purity" else (low, low + 1)
    return low, high


def _fill_step(value: float | None, low: float, high: float) -> str:
    """Pick the fill step of a value on a scale; neutral if unknown."""
    if value is None:
        return NEUTRAL
    step = int((value - low) / (high - low) * len(_FILL_STEPS))
    return _FILL_STEPS[min(max(step, 0), len(_FILL_STEPS) - 1)]


def tree_diagram(
    nodes: Sequence[Mapping], *, color: str, run_id: int
) -> dict[str, object]:
    """Draw the tree of one predicted class, root on top.

    Args:
        nodes: The `queries.class_tree` rows, the kept nodes of the
            class.
        color: The key of `TREE_COLORS` that colors the nodes.
        run_id: The run, for the link of each leaf to its page.

    Returns:
        A node-link figure: a marker per node sized by its train rows
        and colored by `color`, squares for the leaves; the hover gives
        the counts and rates of train, val and test.
    """
    positions = _tree_positions(nodes)
    by_id = {node["node_id"]: node for node in nodes}
    edges_x: list[float | None] = []
    edges_y: list[float | None] = []
    for node in nodes:
        parent = by_id.get(node["parent_id"])
        if parent is not None:
            edges_x += [positions[parent["node_id"]], positions[node["node_id"]], None]
            edges_y += [parent["depth"], node["depth"], None]
    biggest = max((node["n_train"] for node in nodes), default=1) or 1
    low, high = _color_range(nodes, color)
    marks = {
        "type": "scatter",
        "mode": "markers",
        "x": [positions[node["node_id"]] for node in nodes],
        "y": [node["depth"] for node in nodes],
        "text": [_tree_hover(node) for node in nodes],
        "customdata": [
            (
                f"/runs/{run_id}/leaves/{node['node_id']}"
                if node["status"] == "leaf"
                else None
            )
            for node in nodes
        ],
        "marker": {
            "symbol": [
                "square" if node["status"] == "leaf" else "circle" for node in nodes
            ],
            "size": [6 + 18 * (node["n_train"] / biggest) ** 0.5 for node in nodes],
            "color": [node[color] for node in nodes],
            "colorscale": _SEQUENTIAL,
            "cmin": low,
            "cmax": high,
            "colorbar": {"title": TREE_COLORS[color], "thickness": 12},
            "line": {"width": 1, "color": NEUTRAL},
        },
        "hovertemplate": "%{text}<extra></extra>",
    }
    edges = {
        "type": "scatter",
        "mode": "lines",
        "x": edges_x,
        "y": edges_y,
        "line": {"color": NEUTRAL, "width": 1},
        "hoverinfo": "skip",
    }
    max_depth = max((node["depth"] for node in nodes), default=0)
    layout = {
        "height": max(380, 46 * (max_depth + 1) + 120),
        "showlegend": False,
        "xaxis": {"visible": False},
        "yaxis": {"title": "depth", "autorange": "reversed", "dtick": 1},
    }
    return {"data": [edges, marks], "layout": layout}


def leaf_map(
    nodes: Sequence[Mapping], plane: ClassPlane, *, color: str, run_id: int
) -> dict[str, object]:
    """Draw the leaves of one predicted class as circles in a plane.

    Args:
        nodes: The `queries.class_tree` rows, the kept nodes of the
            class.
        plane: The circles of the leaves, from `artifacts.class_plane`.
        color: The key of `TREE_COLORS` that colors the leaves, on the
            scale of `tree_diagram`.
        run_id: The run, for the link of each leaf to its page.

    Returns:
        A figure with a circle per leaf that has train rows, centred on
        their mean with their RMS distance as radius, and a marker on
        each centre; the hover is the tree's. The axes share a scale so
        the circles stay round.
    """
    low, high = _color_range(nodes, color)
    leaves = [node for node in nodes if node["node_id"] in plane.circles]
    # The largest first, so that the small circles stay visible.
    leaves.sort(key=lambda node: -plane.circles[node["node_id"]][2])
    shapes = []
    for node in leaves:
        x, y, radius = plane.circles[node["node_id"]]
        shapes.append(
            {
                "type": "circle",
                "xref": "x",
                "yref": "y",
                "x0": x - radius,
                "x1": x + radius,
                "y0": y - radius,
                "y1": y + radius,
                "fillcolor": _fill_step(node[color], low, high),
                "opacity": 0.45,
                "line": {"color": NEUTRAL, "width": 1},
                "layer": "below",
            }
        )
    centres = {
        "type": "scatter",
        "mode": "markers",
        "x": [plane.circles[node["node_id"]][0] for node in leaves],
        "y": [plane.circles[node["node_id"]][1] for node in leaves],
        "text": [_tree_hover(node) for node in leaves],
        "customdata": [f"/runs/{run_id}/leaves/{node['node_id']}" for node in leaves],
        "marker": {
            "size": 7,
            "color": [node[color] for node in leaves],
            "colorscale": _SEQUENTIAL,
            "cmin": low,
            "cmax": high,
            "colorbar": {"title": TREE_COLORS[color], "thickness": 12},
            "line": {"width": 1, "color": NEUTRAL},
        },
        "hovertemplate": "%{text}<extra></extra>",
    }
    first, second = plane.shares
    layout = {
        "height": 520,
        "showlegend": False,
        "shapes": shapes,
        "xaxis": {"title": f"PC 1 · {first:.0%} of the variance"},
        "yaxis": {
            "title": f"PC 2 · {second:.0%} of the variance",
            "scaleanchor": "x",
            "scaleratio": 1,
        },
    }
    return {"data": [centres], "layout": layout}

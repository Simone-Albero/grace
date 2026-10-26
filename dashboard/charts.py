"""Plotly figures of the dashboard, as JSON-ready dicts."""

__all__ = [
    "metric_bars",
    "partition_class_curves",
    "partition_curves",
    "partition_val_rates",
    "partition_val_test_scatter",
    "predictor_bars",
    "region_scatter",
]

import statistics
from collections.abc import Mapping, Sequence

from dashboard.queries import PREDICTORS

_HEIGHT = 380
# One fixed hue per arm, so an arm keeps its color in every figure of
# the partitions panel.
_ARM_COLORS = {
    "intra": "#2a78d6",
    "inter": "#eb6834",
    "null": "#1baf7a",
    "confidence": "#eda100",
    "predicted_class": "#e87ba4",
}
# The marker of each operating point: a geometric arm's `selected`,
# or a bin arm's rule.
_POINT_SYMBOLS = {
    "selected": "star",
    "fixed15": "circle",
    "cbrt": "diamond",
    "per150": "square",
}


def _groups(rows: Sequence[dict], key: str, value: str) -> dict[str, list[float]]:
    """Group the non-null `value` of the rows by the string of `key`.

    Returns:
        The groups, sorted by key.
    """
    groups: dict[str, list[float]] = {}
    for row in rows:
        if row[value] is not None:
            groups.setdefault(str(row[key]), []).append(row[value])
    return dict(sorted(groups.items()))


def _bars(name: str, groups: Mapping[str, Sequence[float]]) -> dict[str, object]:
    """Build a bar trace: group mean, std error bar, size on hover."""
    values = list(groups.values())
    return {
        "type": "bar",
        "name": name,
        "x": list(groups),
        "y": [statistics.mean(group) for group in values],
        "error_y": {
            "type": "data",
            "array": [statistics.pstdev(group) for group in values],
        },
        "text": [f"n = {len(group)}" for group in values],
        "hoverinfo": "text+y+name",
    }


def _layout(x: str, y: str) -> dict[str, object]:
    """Build the grouped-bar layout, with ascending categories."""
    return {
        "barmode": "group",
        "height": _HEIGHT,
        "xaxis": {
            "title": x,
            "type": "category",
            "categoryorder": "category ascending",
        },
        "yaxis": {"title": y},
    }


def metric_bars(
    rows: Sequence[dict], *, metric: str, x: str, color: str | None
) -> dict[str, object]:
    """Draw the mean `metric` per value of `x`, one trace per `color`.

    Args:
        rows: The results rows.
        metric: The results column to average.
        x: The results column of the x axis.
        color: The results column that splits the bars into traces;
            `None` draws one trace.

    Returns:
        A Plotly figure.
    """
    if color is None:
        traces = [_bars(metric, _groups(rows, x, metric))]
    else:
        traces = []
        for value in sorted({str(row[color]) for row in rows}):
            subset = [row for row in rows if str(row[color]) == value]
            traces.append(_bars(f"{color}={value}", _groups(subset, x, metric)))
    return {"data": traces, "layout": _layout(x, metric)}


def predictor_bars(rows: Sequence[dict], *, score: str, x: str) -> dict[str, object]:
    """Draw the mean RF regressor, val, MCP and ATC `score` per `x`.

    Args:
        rows: The results rows.
        score: The score name, e.g. "spearman".
        x: The results column of the x axis.

    Returns:
        A Plotly figure with one trace per predictor that has a value in
        `rows`.
    """
    traces = []
    for key, (label, _) in PREDICTORS.items():
        groups = _groups(rows, x, f"{key}_{score}")
        if groups:
            traces.append(_bars(label, groups))
    return {"data": traces, "layout": _layout(x, score)}


def region_scatter(
    rows: Sequence[dict], *, x: str, y: str, run_id: int
) -> dict[str, object]:
    """Draw `y` against `x` per region, one marker trace per class.

    Args:
        rows: The region rows.
        x: The region column of the x axis.
        y: The region column of the y axis.
        run_id: The run, for the link of each marker to its region.

    Returns:
        A Plotly figure; regions without a value of `x` or `y` are
        left out.
    """
    traces = []
    for name in sorted({row["class"] for row in rows}):
        points = [
            row
            for row in rows
            if row["class"] == name and row[x] is not None and row[y] is not None
        ]
        traces.append(
            {
                "type": "scatter",
                "mode": "markers",
                "name": name,
                "x": [row[x] for row in points],
                "y": [row[y] for row in points],
                "text": [f"region {row['region_id']}" for row in points],
                "customdata": [
                    f"/runs/{run_id}/regions/{row['region_id']}" for row in points
                ],
            }
        )
    layout = {
        "height": _HEIGHT,
        "xaxis": {"title": x},
        "yaxis": {"title": y},
        "hovermode": "closest",
    }
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
            "line": {"width": 1, "color": "black"},
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
    for arm, color in _ARM_COLORS.items():
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
                "marker": {"symbol": "star", "size": 14, "color": "black"},
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
    for arm in _ARM_COLORS:
        for rule, values in sorted(rates_by_arm.get(arm, {}).items()):
            name = arm if rule == "selected" else f"{arm} ({rule})"
            traces.append(
                {
                    "type": "box",
                    "name": name,
                    "y": values,
                    "marker": {"color": _ARM_COLORS[arm]},
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
            "line": {"dash": "dot", "color": "gray"},
            "hoverinfo": "skip",
        }
    ]
    for arm, color in _ARM_COLORS.items():
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
                        "line": {"width": 1, "color": "white"},
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

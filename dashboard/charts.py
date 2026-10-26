"""Plotly figures of the dashboard, as JSON-ready dicts."""

__all__ = ["metric_bars", "predictor_bars", "region_scatter"]

import statistics
from collections.abc import Mapping, Sequence

from dashboard.queries import PREDICTORS

_HEIGHT = 380


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

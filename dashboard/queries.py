"""Read-only queries: db tables and parquet files to JSON-able dicts.

Every function raises LookupError("run <stage> first") when the stage
output is missing (no db, table, run or file). The db is opened
read-only (file:...?mode=ro). The run shown is the latest one with
status "ok" for the stage and dataset. Every "run" value is
{id, started, finished}. Non-finite numbers become None (JSON has no
NaN).
"""

from __future__ import annotations

__all__ = [
    "MAP_MAX_POINTS",
    "MAP_PER_LEAF",
    "clean",
    "prepare",
    "quality",
    "region_map",
    "regions",
]

import dataclasses
import json
import math
import sqlite3
from collections import defaultdict
from collections.abc import Mapping
from contextlib import closing
from typing import Any

import numpy as np
import polars as pl
from sklearn.decomposition import PCA

from src.config import DatasetConfig
from stages.paths import (
    assignments_file,
    need,
    prepared_file,
    read_classes,
    read_points,
)

MAP_PER_LEAF = 2000  # most rows of one region in a node's map
MAP_MAX_POINTS = 20000  # most rows of a node's map


def clean(cfg: DatasetConfig) -> dict[str, Any]:
    """Return the last clean run.

    Args:
        cfg: The dataset.

    Returns:
        run; summary {n_raw, n_duplicate, n_rare, n_kept, n_classes,
        n_constant}; classes [{label, n_read, n_rows}] sorted by label;
        dropped_classes [{label, n_rows}]; dropped_columns [column];
        features {num: [...], cat: [...]}.
    """
    with closing(_connect(cfg)) as conn:
        run = _latest_run(conn, "clean", cfg)
        summary = _rows(conn, "clean", "clean_summary", run)[0]
        features = _rows(conn, "clean", "features", run)
        return {
            "run": _public(run),
            "summary": summary,
            "classes": sorted(
                _rows(conn, "clean", "class_counts", run),
                key=lambda r: r["label"],
            ),
            "dropped_classes": _rows(
                conn, "clean", "dropped_classes", run, missing_ok=True
            ),
            "dropped_columns": [
                row["column"]
                for row in _rows(conn, "clean", "dropped_columns", run, missing_ok=True)
            ],
            "features": {
                kind: [f["column"] for f in features if f["kind"] == kind]
                for kind in ("num", "cat")
            },
        }


def prepare(cfg: DatasetConfig) -> dict[str, Any]:
    """Return the last prepare run.

    Args:
        cfg: The dataset.

    Returns:
        run; summary {n_rows, n_num, n_cat, n_columns, n_missing,
        n_missing_rows}; columns
        [{column, source, kind, mean, std, value, n_rows, weight}] in column
        order.
    """
    with closing(_connect(cfg)) as conn:
        run = _latest_run(conn, "prepare", cfg)
        return {
            "run": _public(run),
            "summary": _rows(conn, "prepare", "prepare_summary", run)[0],
            "columns": _rows(conn, "prepare", "prepared_columns", run),
        }


def regions(cfg: DatasetConfig) -> dict[str, Any]:
    """Return the last regions run.

    Args:
        cfg: The dataset.

    Returns:
        run; stats {n_regions, distance, split_separation,
        overlap_separation, n_zones}; classes [{class, n_rows, n_read,
        n_regions, n_pieces, n_voted, split_separation, stability}];
        candidates [{class, threshold, n_regions}] (the curve); regions
        [{region, class, n_rows, n_read, stability}] (stability null
        when it was not measured); tree [{node, class, parent,
        depth, region, n_rows, n_read}] in pre-order per class; zones
        [{node, zone, parent, depth, separation, region_a, region_b,
        region, class, n_classes, n_regions, n_rows, n_read}] zone by
        zone in pre-order; overlaps [{zone, region_a, class_a, region_b,
        class_b, distance, spread_a, spread_b, separation}]; both empty
        when no regions overlap or the run is older than the zones.
    """
    with closing(_connect(cfg)) as conn:
        run = _latest_run(conn, "regions", cfg)
        classes = _rows(conn, "regions", "class_clusterings", run)
        candidates = _rows(conn, "regions", "granularity_candidates", run)
        return {
            "run": _public(run),
            "stats": _rows(conn, "regions", "region_stats", run)[0],
            "classes": classes,
            "candidates": candidates,
            "regions": _rows(conn, "regions", "regions", run),
            "tree": _rows(conn, "regions", "region_tree", run),
            "zones": _rows(conn, "regions", "overlap_zones", run, missing_ok=True),
            "overlaps": _rows(conn, "regions", "region_overlaps", run, missing_ok=True),
        }


def quality(cfg: DatasetConfig) -> dict[str, Any]:
    """Return the last quality_report run.

    Args:
        cfg: The dataset.

    Returns:
        run; stats (the quality_stats row, shortcut_cols and highlight
        parsed from JSON); classes, regions and pairs: the rows of
        quality_classes, quality_regions and quality_pairs, pairs with
        reasons as a list.
    """
    with closing(_connect(cfg)) as conn:
        run = _latest_run(conn, "quality_report", cfg)
        stats = _rows(conn, "quality_report", "quality_stats", run)[0]
        stats["shortcut_cols"] = json.loads(stats["shortcut_cols"])
        stats["highlight"] = json.loads(stats["highlight"])
        pairs = _rows(conn, "quality_report", "quality_pairs", run)
        for pair in pairs:
            pair["reasons"] = pair["reasons"].split(",")
        return {
            "run": _public(run),
            "stats": stats,
            "classes": _rows(conn, "quality_report", "quality_classes", run),
            "regions": _rows(conn, "quality_report", "quality_regions", run),
            "pairs": pairs,
        }


def region_map(cfg: DatasetConfig, node: int) -> dict[str, Any]:
    """Return the rows of one node of the region trees in their own plane.

    The rows of the regions below node (assignments.parquet of the last
    regions run's tree) are sampled at random (seed cfg.seed): at most
    per_leaf = min(MAP_PER_LEAF, max(1, MAP_MAX_POINTS // regions))
    of each region. A PCA fitted on the sample, in the space the regions
    were built in (graph.space_points of the prepared file with the
    distance of that run's method, not of cfg's), gives their first two
    components. The prepared points are cached per file and mtime.

    Args:
        cfg: The dataset.
        node: A node id of region_tree.

    Returns:
        node; shares [s1, s2] (the variance of each component, 0 when
        it does not exist); points [[pc1, pc2, region]], pc rounded to 4
        decimals.

    Raises:
        LookupError: When regions has not run or node is not a node of
            its tree.
    """
    return _map(cfg, "region_tree", node)


def overlap_map(cfg: DatasetConfig, node: int) -> dict[str, Any]:
    """Return the rows of one node of the overlap zones in their own plane.

    As region_map, with the regions below node in overlap_zones.

    Args:
        cfg: The dataset.
        node: A node id of overlap_zones.

    Returns:
        node; shares [s1, s2]; points [[pc1, pc2, region]].

    Raises:
        LookupError: When regions has not run or node is not a node of
            its overlap zones.
    """
    return _map(cfg, "overlap_zones", node)


def _map(cfg: DatasetConfig, table: str, node: int) -> dict[str, Any]:
    """Return region_map of the regions below node in table, a tree of the last regions run."""
    with closing(_connect(cfg)) as conn:
        run = _latest_run(conn, "regions", cfg)
        (config,) = conn.execute(
            "SELECT config FROM runs WHERE id = ?", (run["id"],)
        ).fetchone()
        tree = _rows(conn, "regions", table, run)
    children = defaultdict(list)
    for row in tree:
        children[row["parent"]].append(row)
    by_node = {row["node"]: row for row in tree}
    if node not in by_node:
        raise LookupError(f"no node {node} in {table}")
    leaves, stack = [], [by_node[node]]
    while stack:
        row = stack.pop()
        if row["region"] is not None:
            leaves.append(row["region"])
        stack += children[row["node"]]
    region = pl.read_parquet(need(assignments_file(cfg), "regions"))
    region = region.sort("row")["region"].to_numpy()
    rows = np.flatnonzero(np.isin(region, leaves))
    rows = rows[np.argsort(region[rows], kind="stable")]
    bounds = np.flatnonzero(np.diff(region[rows])) + 1
    per_leaf = min(MAP_PER_LEAF, max(1, MAP_MAX_POINTS // len(leaves)))
    rng = np.random.default_rng(cfg.seed)
    take = np.sort(
        np.concatenate(
            [rng.permutation(group)[:per_leaf] for group in np.split(rows, bounds)]
        )
    )
    distance = json.loads(config)["method"]["distance"]
    points = _space(cfg, distance)[take]
    plane = np.zeros((len(points), 2))
    shares = [0.0, 0.0]
    if len(points) >= 2:
        pca = PCA(n_components=min(2, *points.shape), random_state=cfg.seed)
        plane[:, : pca.n_components] = pca.fit_transform(points)
        shares[: pca.n_components] = [float(s) for s in pca.explained_variance_ratio_]
    return {
        "node": node,
        "shares": shares,
        "points": [
            [round(float(a), 4), round(float(b), 4), int(r)]
            for (a, b), r in zip(plane, region[take])
        ],
    }


def _finite(value: Any) -> float | None:
    """Return a finite float, or None (JSON has no NaN)."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _json_value(value: Any) -> Any:
    """Return floats through _finite, other values unchanged."""
    return _finite(value) if isinstance(value, float) else value


def _connect(cfg: DatasetConfig) -> sqlite3.Connection:
    if not cfg.db_path.exists():
        raise LookupError("run clean first")
    conn = sqlite3.connect(f"file:{cfg.db_path.resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _latest_run(
    conn: sqlite3.Connection, stage: str, cfg: DatasetConfig
) -> dict[str, Any]:
    """Return the last ok run of a stage on the dataset."""
    row = conn.execute(
        "SELECT id, started, finished FROM runs WHERE stage = ? AND dataset = ? "
        "AND status = 'ok' ORDER BY id DESC LIMIT 1",
        (stage, cfg.name),
    ).fetchone()
    if row is None:
        raise LookupError(f"run {stage} first")
    return dict(row)


def _public(run: Mapping[str, Any]) -> dict[str, Any]:
    return {k: run[k] for k in ("id", "started", "finished")}


def _rows(
    conn: sqlite3.Connection,
    stage: str,
    table: str,
    run: Mapping[str, Any],
    missing_ok: bool = False,
) -> list[dict[str, Any]]:
    """Return the rows of a table logged by a run, without run_id.

    A missing table means the stage never logged it: LookupError, or no
    rows when missing_ok, for a table that db.log_table skips when a
    run has nothing to log in it.
    """
    try:
        rows = conn.execute(
            f'SELECT * FROM "{table}" WHERE run_id = ?', (run["id"],)
        ).fetchall()
    except sqlite3.OperationalError as e:
        if missing_ok and "no such table" in str(e):
            return []
        raise LookupError(f"run {stage} first") from e
    return [{k: _json_value(r[k]) for k in r.keys() if k != "run_id"} for r in rows]


def _classes(cfg: DatasetConfig) -> list[str]:
    try:
        return read_classes(cfg)
    except FileNotFoundError as e:
        raise LookupError(str(e)) from e


_SPACE: dict[tuple, np.ndarray] = {}  # the points of the last prepared files read


def _space(cfg: DatasetConfig, distance: str) -> np.ndarray:
    """Return paths.read_points under distance, cached per file and mtime."""
    path = need(prepared_file(cfg), "prepare")
    key = (str(path), path.stat().st_mtime_ns, distance)
    if key not in _SPACE:
        method = dataclasses.replace(cfg.method, distance=distance)
        points, _ = read_points(dataclasses.replace(cfg, method=method))
        if len(_SPACE) >= 2:
            _SPACE.clear()
        _SPACE[key] = points
    return _SPACE[key]

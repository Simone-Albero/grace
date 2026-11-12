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
    "CONFLICT_GROUPS",
    "MAP_MAX_POINTS",
    "MAP_PER_CELL",
    "cell_map",
    "cells",
    "clean",
    "prepare",
    "quality",
    "quality_map",
    "quality_region",
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
from src.data import feature_cols
from stages.paths import (
    assignments_file,
    data_file,
    need,
    prepared_file,
    read_classes,
    read_points,
)

MAP_PER_CELL = 2000  # most rows of one cell in a map
MAP_MAX_POINTS = 20000  # most rows of a map
CONFLICT_GROUPS = 10  # most label conflict groups in a region's detail


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


def cells(cfg: DatasetConfig) -> dict[str, Any]:
    """Return the last cells run.

    Args:
        cfg: The dataset.

    Returns:
        run; stats (the cell_stats row: n_cells, distance, max_rows,
        min_rows, max_fit_samples); classes [{class, n_rows, n_read,
        n_cells}]; cells [{cell, class, n_rows, n_read}]; tree [{node,
        class, parent, depth, cell, n_rows, n_read}] in pre-order per
        class.
    """
    with closing(_connect(cfg)) as conn:
        run = _latest_run(conn, "cells", cfg)
        return {
            "run": _public(run),
            "stats": _rows(conn, "cells", "cell_stats", run)[0],
            "classes": _rows(conn, "cells", "class_cells", run),
            "cells": _rows(conn, "cells", "cells", run),
            "tree": _rows(conn, "cells", "cell_tree", run),
        }


def cell_map(cfg: DatasetConfig, node: int) -> dict[str, Any]:
    """Return the rows of one node of the cell trees in their own plane.

    The rows of the cells below node (assignments.parquet of the last
    cells run) are sampled at random (seed cfg.seed): at most per_cell
    = min(MAP_PER_CELL, max(1, MAP_MAX_POINTS // cells)) of each cell.
    A PCA fitted on the sample, in the space the cells were cut in
    (graph.space_points of the prepared file with the distance of that
    run's method, not of cfg's), gives their first two components. The
    prepared points are cached per file and mtime.

    Args:
        cfg: The dataset.
        node: A node id of cell_tree.

    Returns:
        node; shares [s1, s2] (the variance of each component, 0 when
        it does not exist); points [[pc1, pc2, cell]], pc rounded to 4
        decimals; sampled [[cell, rows, taken]], by cell: its rows and
        how many of them are in points, so that a point stands for
        rows / taken rows of its cell.

    Raises:
        LookupError: When cells has not run or node is not a node of
            its tree.
    """
    with closing(_connect(cfg)) as conn:
        run = _latest_run(conn, "cells", cfg)
        tree = _rows(conn, "cells", "cell_tree", run)
    children = defaultdict(list)
    for row in tree:
        children[row["parent"]].append(row)
    by_node = {row["node"]: row for row in tree}
    if node not in by_node:
        raise LookupError(f"no node {node} in cell_tree")
    below, stack = [], [by_node[node]]
    while stack:
        row = stack.pop()
        if row["cell"] is not None:
            below.append(row["cell"])
        stack += children[row["node"]]
    return {"node": node, **_map_cells(cfg, sorted(below))}


def quality(cfg: DatasetConfig) -> dict[str, Any]:
    """Return the last quality_report run.

    Args:
        cfg: The dataset.

    Returns:
        run; stats (the quality_stats row); classes (quality_classes);
        regions (quality_regions, defects and the class lists as
        lists); missing (cell_missing summed by region, column and
        kind: [{region, column, kind, n_rows, n_read}], none when no
        cell is null or invalid).
    """
    with closing(_connect(cfg)) as conn:
        run = _latest_run(conn, "quality_report", cfg)
        return {
            "run": _public(run),
            "stats": _rows(conn, "quality_report", "quality_stats", run)[0],
            "classes": _rows(conn, "quality_report", "quality_classes", run),
            "regions": [
                _listed(r)
                for r in _rows(conn, "quality_report", "quality_regions", run)
            ],
            "missing": _summed_missing(conn, run, ("region",)),
        }


def quality_region(cfg: DatasetConfig, region: int) -> dict[str, Any]:
    """Return the detail of one region in the last quality_report run.

    Args:
        cfg: The dataset.
        region: A region id of quality_regions.

    Returns:
        region (its quality_regions row, as quality's); cells (its
        quality_cells rows, the lists as lists); pairs (the quality_pairs
        rows with one of its cells, confusion_without_a and _b as
        lists); missing (its cell_missing rows summed by column and
        kind); n_groups (the label conflict groups with a row in it) and
        conflicts: the CONFLICT_GROUPS of them with the most CSV rows,
        heaviest first (ties to the lower group), each {group, n_read,
        rows: [{class, cell, region, n_read}] heaviest first, values:
        {column: value}} with values the features of the data file
        shared by its rows (null for a null, NaN or infinite one).

    Raises:
        LookupError: When quality_report has not run or region is not
            one of its regions.
    """
    with closing(_connect(cfg)) as conn:
        run = _latest_run(conn, "quality_report", cfg)
        rows = _where(conn, "quality_regions", run, "region = ?", region)
        if not rows:
            raise LookupError(f"no region {region} in quality_regions")
        cells = _where(conn, "quality_cells", run, "region = ?", region)
        pairs = _where(
            conn, "quality_pairs", run, "region_a = ? OR region_b = ?", region, region
        )
        missing = _summed_missing(conn, run, (), "region = ?", region)
        groups = _where(
            conn,
            "label_conflict_groups",
            run,
            '"group" IN (SELECT "group" FROM label_conflict_groups '
            "WHERE run_id = ? AND region = ?)",
            run["id"],
            region,
        )
    by_group: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in groups:
        by_group[row["group"]].append(row)
    heaviest = sorted(
        by_group.items(), key=lambda g: (-sum(r["n_read"] for r in g[1]), g[0])
    )[:CONFLICT_GROUPS]
    values = _feature_values(cfg, [members[0]["row"] for _, members in heaviest])
    return {
        "region": _listed(rows[0]),
        "cells": [_listed(c) for c in sorted(cells, key=lambda c: c["cell"])],
        "pairs": [_pair_row(p) for p in pairs],
        "missing": missing,
        "n_groups": len(by_group),
        "conflicts": [
            {
                "group": group,
                "n_read": sum(r["n_read"] for r in members),
                "rows": [
                    {k: r[k] for k in ("class", "cell", "region", "n_read")}
                    for r in sorted(members, key=lambda r: -r["n_read"])
                ],
                "values": values[members[0]["row"]],
            }
            for group, members in heaviest
        ],
    }


def quality_map(cfg: DatasetConfig, region: int) -> dict[str, Any]:
    """Return the rows of a region and of its rivals in their own plane.

    As cell_map, with the cells of the region and every rival cell of
    their pairs in the last quality_report run (quality_pairs).

    Args:
        cfg: The dataset.
        region: A region id of quality_regions.

    Returns:
        region; cells (its cells, then the rival cells, each by id);
        shares [s1, s2]; points [[pc1, pc2, cell]]; sampled [[cell,
        rows, taken]].

    Raises:
        LookupError: When quality_report or cells has not run, or
            region is not one of its regions.
    """
    with closing(_connect(cfg)) as conn:
        run = _latest_run(conn, "quality_report", cfg)
        own = _where(conn, "quality_cells", run, "region = ?", region)
        if not own:
            raise LookupError(f"no region {region} in quality_regions")
        pairs = _where(
            conn, "quality_pairs", run, "region_a = ? OR region_b = ?", region, region
        )
    mine = sorted(c["cell"] for c in own)
    paired = {p["cell_a"] for p in pairs} | {p["cell_b"] for p in pairs}
    shown = [*mine, *sorted(paired - set(mine))]
    return {"region": region, "cells": shown, **_map_cells(cfg, shown)}


def _listed(row: dict[str, Any]) -> dict[str, Any]:
    """Return a quality_regions or quality_cells row with its comma lists as lists."""
    out = dict(row)
    for key in ("defects", "overlap_with", "shortcut_with", "conflict_with"):
        out[key] = row[key].split(",") if row[key] else []
    return out


def _pair_row(row: dict[str, Any]) -> dict[str, Any]:
    """Return a quality_pairs row with its confusion without columns as lists."""
    out = dict(row)
    for side in ("a", "b"):
        key = f"confusion_without_{side}"
        out[key] = json.loads(row[key]) if row[key] else []
    return out


def _summed_missing(
    conn: sqlite3.Connection,
    run: Mapping[str, Any],
    by: tuple[str, ...],
    condition: str = "1",
    *params: Any,
) -> list[dict[str, Any]]:
    """Return the cell_missing rows that meet condition, summed by `by`, column and kind."""
    keys = ", ".join([*by, '"column"', "kind"])
    try:
        rows = conn.execute(
            f"SELECT {keys}, sum(n_rows) AS n_rows, sum(n_read) AS n_read "
            f"FROM cell_missing WHERE run_id = ? AND ({condition}) "
            f"GROUP BY {keys} ORDER BY {keys}",
            (run["id"], *params),
        ).fetchall()
    except sqlite3.OperationalError as e:
        if "no such table" in str(e):
            return []
        raise
    return [{k: _json_value(r[k]) for k in r.keys()} for r in rows]


def _map_cells(cfg: DatasetConfig, shown: list[int]) -> dict[str, Any]:
    """Return shares, points and sampled of cell_map for the rows of the cells shown."""
    with closing(_connect(cfg)) as conn:
        run = _latest_run(conn, "cells", cfg)
        (config,) = conn.execute(
            "SELECT config FROM runs WHERE id = ?", (run["id"],)
        ).fetchone()
    cell = pl.read_parquet(need(assignments_file(cfg), "cells"))
    cell = cell.sort("row")["cell"].to_numpy()
    rows = np.flatnonzero(np.isin(cell, shown))
    rows = rows[np.argsort(cell[rows], kind="stable")]
    bounds = np.flatnonzero(np.diff(cell[rows])) + 1
    per_cell = min(MAP_PER_CELL, max(1, MAP_MAX_POINTS // len(shown)))
    rng = np.random.default_rng(cfg.seed)
    groups = [group for group in np.split(rows, bounds) if len(group)]
    take = np.sort(
        np.concatenate(
            [rng.permutation(group)[:per_cell] for group in groups] or [rows]
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
        "shares": shares,
        "points": [
            [round(float(a), 4), round(float(b), 4), int(c)]
            for (a, b), c in zip(plane, cell[take])
        ],
        "sampled": [[int(cell[g[0]]), len(g), min(len(g), per_cell)] for g in groups],
    }


def _feature_values(cfg: DatasetConfig, rows: list[int]) -> dict[int, dict[str, Any]]:
    """Return the features of some rows of the data file, num then cat, by row."""
    path = need(data_file(cfg), "clean")
    present = feature_cols(
        pl.read_parquet_schema(path).names(), cfg.num_cols, cfg.cat_cols
    )
    columns = [*present.num, *present.cat]
    found = (
        pl.scan_parquet(path)
        .with_row_index("__row__")
        .filter(pl.col("__row__").is_in(rows))
        .select("__row__", *columns)
        .collect()
    )
    return {
        int(r["__row__"]): {c: _json_value(r[c]) for c in columns}
        for r in found.iter_rows(named=True)
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


def _where(
    conn: sqlite3.Connection,
    table: str,
    run: Mapping[str, Any],
    condition: str,
    *params: Any,
) -> list[dict[str, Any]]:
    """Return the rows of a table logged by a run that meet condition, without run_id.

    A missing table gives no rows: db.log_table skips an empty one.
    """
    try:
        rows = conn.execute(
            f'SELECT * FROM "{table}" WHERE run_id = ? AND ({condition})',
            (run["id"], *params),
        ).fetchall()
    except sqlite3.OperationalError as e:
        if "no such table" in str(e):
            return []
        raise
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

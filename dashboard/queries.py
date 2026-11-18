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
    "HYPERPARAMETERS",
    "MAP_MAX_POINTS",
    "MAP_PER_CELL",
    "cell_map",
    "cells",
    "clean",
    "prepare",
    "quality",
    "quality_map",
    "quality_profile",
    "quality_region",
]

import dataclasses
import datetime as dt
import json
import math
import sqlite3
from collections import defaultdict
from collections.abc import Mapping
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.decomposition import PCA

from src.config import DatasetConfig
from src.data import feature_cols
from src.provenance import KEYS, TIME, concentration, time_window
from src.quality import decisive_rows
from src.report import Verdicts, judge
from stages.paths import (
    assignments_file,
    cell_tree_file,
    data_file,
    need,
    prepared_file,
    provenance_file,
    read_classes,
    read_points,
    read_weights,
)

MAP_PER_CELL = 2000  # most rows of one cell in a map
MAP_MAX_POINTS = 20000  # most rows of a map
CONFLICT_GROUPS = 10  # most label conflict groups in a region's detail
# the settings each stage's tab shows, by stage, as its run's config set them
HYPERPARAMETERS = {
    "clean": ("min_class_count",),
    "prepare": ("top_n", "cat_cost"),
    "cells": ("distance", "min_rows", "max_fit_samples", "seed"),
    "quality_report": (
        "tau",
        "rival_rows",
        "max_rows_per_cell",
        "max_group",
        "time_share",
        "confidence",
    ),
}


def clean(cfg: DatasetConfig) -> dict[str, Any]:
    """Return the last clean run.

    Args:
        cfg: The dataset.

    Returns:
        run; params (_params); summary {n_raw, n_duplicate, n_rare,
        n_kept, n_classes, n_constant}; classes [{label, n_read, n_rows}] sorted by label;
        dropped_classes [{label, n_rows}]; dropped_columns [column];
        features {num: [...], cat: [...]}; provenance [{key, column,
        n_read, n_empty, n_unparsable, n_values}] (the provenance_columns
        rows, none when the dataset names no provenance).
    """
    with closing(_connect(cfg)) as conn:
        run = _latest_run(conn, "clean", cfg)
        summary = _rows(conn, "clean", "clean_summary", run)[0]
        features = _rows(conn, "clean", "features", run)
        return {
            "run": _public(run),
            "params": _params(conn, run, "clean"),
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
            "provenance": _rows(
                conn, "clean", "provenance_columns", run, missing_ok=True
            ),
        }


def prepare(cfg: DatasetConfig) -> dict[str, Any]:
    """Return the last prepare run.

    Args:
        cfg: The dataset.

    Returns:
        run; params (_params); summary {n_rows, n_num, n_cat, n_columns,
        n_missing, n_missing_rows}; columns
        [{column, source, kind, mean, std, value, n_rows, weight}] in column
        order.
    """
    with closing(_connect(cfg)) as conn:
        run = _latest_run(conn, "prepare", cfg)
        return {
            "run": _public(run),
            "params": _params(conn, run, "prepare"),
            "summary": _rows(conn, "prepare", "prepare_summary", run)[0],
            "columns": _rows(conn, "prepare", "prepared_columns", run),
        }


def cells(cfg: DatasetConfig, tau: float | None = None) -> dict[str, Any]:
    """Return the last cells run.

    Args:
        cfg: The dataset.
        tau: The threshold the defects of the cells are judged at, in
            (0, 0.5]; None: the last quality_report run's own.

    Returns:
        run; params (_params); stats (the cell_stats row: n_cells,
        distance, min_rows, max_fit_samples); classes [{class, n_rows, n_read,
        n_cells}]; cells [{cell, class, n_rows, n_read, region,
        defects}], region and defects (a list) those of the last
        quality_report run after this cells run, judged at tau (_judged),
        null without one; tree [{node, class, parent, depth, cell, n_rows, n_read}] in
        pre-order per class.
    """
    with closing(_connect(cfg)) as conn:
        run = _latest_run(conn, "cells", cfg)
        judged = _cell_defects(conn, cfg, run, tau)
        return {
            "run": _public(run),
            "params": _params(conn, run, "cells"),
            "stats": _rows(conn, "cells", "cell_stats", run)[0],
            "classes": _rows(conn, "cells", "class_cells", run),
            "cells": [
                {**c, **judged.get(c["cell"], {"region": None, "defects": None})}
                for c in _rows(conn, "cells", "cells", run)
            ],
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


def quality(cfg: DatasetConfig, tau: float | None = None) -> dict[str, Any]:
    """Return the last quality_report run, judged at tau.

    Args:
        cfg: The dataset.
        tau: The threshold, in (0, 0.5]; None: the run's own. What the
            run measured does not depend on it (src.report.judge).

    Returns:
        run; params (_params: the run's own settings, its tau too);
        stats (the quality_stats row, its tau the one judged at, at the
        run's confidence; n_regions, regions_, undecided_ and share_
        judged at it; those of atypical_provenance null when the dataset
        names no provenance, and so the classes'); classes
        (quality_classes); regions (quality_regions, defects and the
        class and key lists as lists); missing
        (cell_missing summed by region, column and kind: [{region,
        column, kind, n_rows, n_read}], none when no cell is null or
        invalid).
    """
    with closing(_connect(cfg)) as conn:
        run = _latest_run(conn, "quality_report", cfg)
        judged = _judged(cfg, conn, run, tau)
        return {
            "run": _public(run),
            "params": _params(conn, run, "quality_report"),
            "stats": {
                **_rows(conn, "quality_report", "quality_stats", run)[0],
                **judged.stats,
            },
            "classes": judged.classes,
            "regions": [_listed(r) for r in judged.regions],
            "missing": _missing_by_region(conn, run, judged),
        }


def quality_region(
    cfg: DatasetConfig, region: int, tau: float | None = None
) -> dict[str, Any]:
    """Return the detail of one region in the last quality_report run, judged at tau.

    Args:
        cfg: The dataset.
        region: A region id of quality_regions, at tau.
        tau: The threshold, in (0, 0.5]; None: the run's own.

    Returns:
        region (its quality_regions row, as quality's); cells (its
        quality_cells rows, the lists as lists, with the provenance
        columns when the dataset names some); rivals (the quality_rivals
        rows of its cells, one per cell and rival class, rival_cells,
        rival_regions and confusion_without as lists); missing (its
        cell_missing rows summed by column and
        kind); n_groups (the label conflict groups with a row in it) and
        conflicts: the CONFLICT_GROUPS of them with the most CSV rows,
        heaviest first (ties to the lower group), each {group, n_read,
        rows: [{class, cell, region, n_read}] heaviest first, values:
        {column: value}} with values the features of the data file
        shared by its rows (null for a null, NaN or infinite one);
        provenance and time (_provenance_of its rows).

    Raises:
        LookupError: When quality_report has not run or region is not
            one of its regions at tau.
    """
    with closing(_connect(cfg)) as conn:
        run = _latest_run(conn, "quality_report", cfg)
        judged = _judged(cfg, conn, run, tau)
        rows = [r for r in judged.regions if r["region"] == region]
        if not rows:
            raise LookupError(f"no region {region} in quality_regions")
        cells = _cells_of(judged, region)
        rivals = _rivals_of(judged, region)
        missing = [
            {k: v for k, v in m.items() if k != "region"}
            for m in _missing_by_region(conn, run, judged)
            if m["region"] == region
        ]
        marks = ", ".join("?" * len(cells))
        groups = _where(
            conn,
            "label_conflict_groups",
            run,
            '"group" IN (SELECT "group" FROM label_conflict_groups '
            f"WHERE run_id = ? AND cell IN ({marks}))",
            run["id"],
            *[c["cell"] for c in cells],
        )
        where = {c["cell"]: c["region"] for c in judged.cells}
        groups = [{**g, "region": where[g["cell"]]} for g in groups]
        share = _params(conn, run, "quality_report")["time_share"]
    by_group: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in groups:
        by_group[row["group"]].append(row)
    heaviest = sorted(
        by_group.items(), key=lambda g: (-sum(r["n_read"] for r in g[1]), g[0])
    )[:CONFLICT_GROUPS]
    values = _feature_values(cfg, [members[0]["row"] for _, members in heaviest])
    (own,) = _rows_of_cells(cfg, [[c["cell"] for c in cells]])
    return {
        **_provenance_of(cfg, own, share),
        "region": _listed(rows[0]),
        "cells": [_listed(c) for c in cells],
        "rivals": [_rival_row(r) for r in rivals],
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


def quality_map(
    cfg: DatasetConfig, region: int, tau: float | None = None
) -> dict[str, Any]:
    """Return the rows of a region and of its rival cells in their own plane.

    As cell_map, with the cells of the region and every rival cell of
    them in the last quality_report run (quality_rivals), judged at
    tau.

    Args:
        cfg: The dataset.
        region: A region id of quality_regions, at tau.
        tau: The threshold, in (0, 0.5]; None: the run's own.

    Returns:
        region; cells (its cells, then the rival cells, each by id);
        shares [s1, s2]; points [[pc1, pc2, cell]]; sampled [[cell,
        rows, taken]].

    Raises:
        LookupError: When quality_report or cells has not run, or
            region is not one of its regions.
    """
    with closing(_connect(cfg)) as conn:
        judged = _judged(cfg, conn, _latest_run(conn, "quality_report", cfg), tau)
    own = _cells_of(judged, region)
    if not own:
        raise LookupError(f"no region {region} in quality_regions")
    mine = sorted(c["cell"] for c in own)
    rival = {
        o for r in _rivals_of(judged, region) for o in json.loads(r["rival_cells"])
    }
    shown = [*mine, *sorted(rival - set(mine))]
    return {"region": region, "cells": shown, **_map_cells(cfg, shown)}


def quality_profile(
    cfg: DatasetConfig, region: int, tau: float | None = None
) -> dict[str, Any]:
    """Return the mean features of a region and of each of its rival classes.

    Over the rows of the region (its cells in the last quality_report
    run judged at tau, assignments.parquet) and, per rival class, of
    the rival cells of that class of its cells (quality_rivals), each
    row weighted by its copies in the CSV: the mean of every numeric
    column of the prepared file (logged and standardized: a difference
    of means is in standard deviations of the dataset, alike for every
    column), and the most frequent value of every categorical column of
    the data file (ties to the first in text order; null for a null).

    Args:
        cfg: The dataset.
        region: A region id of quality_regions, at tau.
        tau: The threshold, in (0, 0.5]; None: the run's own.

    Returns:
        region; columns (the numeric columns, in data file order); own
        {means: [mean per column], top: {categorical column: value}};
        rivals [{class, cells (its rival cells), means, top}] by class
        name.

    Raises:
        LookupError: When quality_report has not run, or region is not
            one of its regions.
    """
    with closing(_connect(cfg)) as conn:
        judged = _judged(cfg, conn, _latest_run(conn, "quality_report", cfg), tau)
    if not any(r["region"] == region for r in judged.regions):
        raise LookupError(f"no region {region} in quality_regions")
    by_class: dict[str, set[int]] = defaultdict(set)
    for r in _rivals_of(judged, region):
        by_class[r["rival_class"]] |= set(json.loads(r["rival_cells"]))
    names_of = sorted(by_class)
    groups = [[c["cell"] for c in _cells_of(judged, region)]]
    groups += [sorted(by_class[k]) for k in names_of]
    rows = _rows_of_cells(cfg, groups)
    every = np.unique(np.concatenate(rows))
    path = need(prepared_file(cfg), "prepare")
    names = feature_cols(
        pl.read_parquet_schema(path).names(), cfg.num_cols, cfg.cat_cols
    ).num
    numbers = _rows_at(path, every, names).to_numpy().reshape(len(every), len(names))
    data = need(data_file(cfg), "clean")
    cat = feature_cols(
        pl.read_parquet_schema(data).names(), cfg.num_cols, cfg.cat_cols
    ).cat
    texts = _rows_at(data, every, cat).select(pl.all().cast(pl.String))
    weights = read_weights(cfg)

    def profile(own: np.ndarray) -> dict[str, Any]:
        at, w = np.searchsorted(every, own), weights[own]
        mean = (numbers[at] * w[:, None]).sum(0) / w.sum() if len(own) else []
        return {
            "means": [_finite(v) for v in mean],
            "top": {c: _weighted_mode(texts[c].to_numpy()[at], w) for c in cat},
        }

    return {
        "region": region,
        "columns": names,
        "own": profile(rows[0]),
        "rivals": [
            {"class": k, "cells": cells, **profile(own)}
            for k, cells, own in zip(names_of, groups[1:], rows[1:])
        ],
    }


def _rows_at(path: Path, rows: np.ndarray, columns: list[str]) -> pl.DataFrame:
    """Return some columns of some rows of a parquet file, the rows in order."""
    return (
        pl.scan_parquet(path)
        .with_row_index("__row__")
        .join(
            pl.LazyFrame({"__row__": pl.Series(rows, dtype=pl.UInt32)}),
            on="__row__",
            how="semi",
        )
        .sort("__row__")
        .select(columns)
        .collect()
    )


def _weighted_mode(values: np.ndarray, weights: np.ndarray) -> str | None:
    """Return the value of most weight, ties to the first in text order, a null last."""
    if not len(values):
        return None
    votes = (
        pl.DataFrame(
            {"v": values, "w": weights}, schema={"v": pl.String, "w": pl.Float64}
        )
        .group_by("v")
        .agg(pl.col("w").sum())
        .sort(["w", "v"], descending=[True, False], nulls_last=True)
    )
    return votes["v"][0]


def _params(
    conn: sqlite3.Connection, run: Mapping[str, Any], stage: str
) -> dict[str, Any]:
    """Return HYPERPARAMETERS[stage] as the config of run set them.

    A name is looked up in the dataset (min_class_count), then in the
    method (distance), then in its groups (method.quality.tau); one the
    config lacks is left out. min_rows is no key of the config: it is
    what the run's tau and confidence ask for (src.config.MethodConfig).
    """
    (text,) = conn.execute(
        "SELECT config FROM runs WHERE id = ?", (run["id"],)
    ).fetchone()
    config = json.loads(text)
    method = config.get("method", {})
    places = [config, method, *(v for v in method.values() if isinstance(v, dict))]
    quality = method.get("quality", {})
    if {"tau", "confidence"} <= set(quality):
        rows = 2 * decisive_rows(quality["tau"], quality["confidence"])
        places.insert(0, {"min_rows": rows})
    found = {}
    for name in HYPERPARAMETERS[stage]:
        place = next((p for p in places if name in p), None)
        if place is not None:
            found[name] = place[name]
    return found


def _rows_of_cells(cfg: DatasetConfig, groups: list[list[int]]) -> list[np.ndarray]:
    """Return the rows of the data file of each group of cells, sorted (assignments.parquet)."""
    assigned = pl.read_parquet(
        need(assignments_file(cfg), "cells"), columns=["row", "cell"]
    )
    row, cell = assigned["row"].to_numpy(), assigned["cell"].to_numpy()
    return [np.sort(row[np.isin(cell, cells)]) for cells in groups]


def _cells_of(judged: Verdicts, region: int) -> list[dict[str, Any]]:
    """Return the cells of a region, by cell."""
    return [c for c in judged.cells if c["region"] == region]


def _rivals_of(judged: Verdicts, region: int) -> list[dict[str, Any]]:
    """Return the quality_rivals rows of the cells of a region."""
    return [r for r in judged.rivals if r["region"] == region]


def _provenance_of(
    cfg: DatasetConfig, rows: np.ndarray, share: float
) -> dict[str, Any]:
    """Return where the CSV rows of some rows of the data file come from.

    Over the CSV rows that are copies of rows (provenance.parquet), an
    empty value left out.

    Args:
        cfg: The dataset.
        rows: Rows of the data file.
        share: quality.time_share, the share of the CSV rows the time
            window holds (provenance.time_window).

    Returns:
        provenance: per key of provenance.KEYS the dataset names, in
        order, {key, n_values (the distinct values), top, top_share (its
        share of the rows with a value; provenance.concentration),
        repeat (the mean CSV rows of a value among those of more than
        one, null when none is)}; time: {time_from, time_to,
        time_window} when it names a time, else null. No key without a
        provenance.parquet.
    """
    path = provenance_file(cfg)
    if cfg.provenance is None or not path.exists():
        return {"provenance": [], "time": None}
    origin = (
        pl.scan_parquet(path)
        .join(
            pl.LazyFrame({"row": pl.Series(rows, dtype=pl.Int64)}), on="row", how="semi"
        )
        .collect()
    )
    group = np.zeros(origin.height, dtype=np.int64)
    found = []
    for key in KEYS:
        if key not in origin.columns:
            continue
        top = concentration(group, origin[key], 1).row(0, named=True)
        counts = origin[key].drop_nulls().value_counts(name="n")["n"]
        repeated = counts.filter(counts > 1)
        found.append(
            {
                "key": key,
                "n_values": top["n_values"],
                "top": top["top"],
                "top_share": top["top_share"],
                "repeat": float(repeated.mean()) if len(repeated) else None,
            }
        )
    time = None
    if TIME in origin.columns:
        window = time_window(group, origin[TIME], 1, share).row(0, named=True)
        time = {
            k: v.isoformat() if isinstance(v, dt.datetime) else _json_value(v)
            for k, v in window.items()
        }
    return {"provenance": found, "time": time}


_LISTS = (
    "defects",
    "undecided",
    "conflict_with",
    "label_with",
    "overlap_with",
    "shortcut_with",
    "provenance_with",
)  # the comma lists of quality_cells and quality_regions


def _listed(row: dict[str, Any]) -> dict[str, Any]:
    """Return a quality_regions or quality_cells row with its comma lists as lists."""
    out = dict(row)
    for key in _LISTS:
        if key in row:
            out[key] = row[key].split(",") if row[key] else []
    return out


def _cell_defects(
    conn: sqlite3.Connection,
    cfg: DatasetConfig,
    cells_run: Mapping[str, Any],
    tau: float | None,
) -> dict[int, dict[str, Any]]:
    """Return the region and defects of every cell, by cell, as quality_report judges them at tau.

    From the last ok quality_report run, when it ran after cells_run (an
    older one judged other cells); none otherwise.
    """
    try:
        run = _latest_run(conn, "quality_report", cfg)
    except LookupError:
        return {}
    if run["id"] < cells_run["id"]:
        return {}
    return {
        row["cell"]: {"region": row["region"], "defects": row["defects"]}
        for row in map(_listed, _judged(cfg, conn, run, tau).cells)
    }


_JUDGED: dict[tuple, Verdicts] = {}  # the last verdicts judged, per run, tau and tree


def _judged(
    cfg: DatasetConfig,
    conn: sqlite3.Connection,
    run: Mapping[str, Any],
    tau: float | None,
) -> Verdicts:
    """Return the verdicts of a quality_report run at tau (src.report.judge).

    What the run measured is read from quality_cells and quality_rivals,
    the cells' tree from cell_tree.parquet, as the stage did. Cached per
    database, run, tau and tree file.

    Args:
        cfg: The dataset.
        conn: The database, of the run.
        run: The quality_report run.
        tau: The threshold, in (0, 0.5]; None: the run's own.

    Raises:
        LookupError: When the cells tree is missing, the cells were cut
            again after the run (its cells are not those of the tree),
            or the run is of an older quality_report (no clear_rows,
            pairs of cells instead of rival classes, or no confidence).
    """
    if _latest_run(conn, "cells", cfg)["id"] > run["id"]:
        raise LookupError("run quality_report again: the cells were cut after it")
    params = _params(conn, run, "quality_report")
    if "confidence" not in params:
        raise LookupError("run quality_report again: it judged without intervals")
    tau = params["tau"] if tau is None else tau
    path = need(cell_tree_file(cfg), "cells")
    key = (str(cfg.db_path.resolve()), run["id"], tau, path.stat().st_mtime_ns)
    if key not in _JUDGED:
        cells = sorted(
            _rows(conn, "quality_report", "quality_cells", run),
            key=lambda c: c["cell"],
        )
        if cells and "clear_rows" not in cells[0]:
            raise LookupError("run quality_report again: it did not log clear_rows")
        if _where(conn, "quality_pairs", run, "1"):
            raise LookupError("run quality_report again: it measured pairs of cells")
        rivals = _where(conn, "quality_rivals", run, "1")
        if len(_JUDGED) >= 8:
            _JUDGED.clear()
        _JUDGED[key] = judge(
            cells,
            rivals,
            pl.read_parquet(path).sort("node"),
            _classes(cfg),
            tau,
            params["confidence"],
        )
    return _JUDGED[key]


def _missing_by_region(
    conn: sqlite3.Connection, run: Mapping[str, Any], judged: Verdicts
) -> list[dict[str, Any]]:
    """Return cell_missing summed by the region of the cell, column and kind, in that order."""
    region = {c["cell"]: c["region"] for c in judged.cells}
    sums: dict[tuple, list] = {}
    for row in _summed_missing(conn, run, ("cell",)):
        key = (region[row["cell"]], row["column"], row["kind"])
        total = sums.setdefault(key, [0, 0.0])
        total[0] += row["n_rows"]
        total[1] += row["n_read"]
    return [
        {"region": r, "column": c, "kind": k, "n_rows": n, "n_read": w}
        for (r, c, k), (n, w) in sorted(sums.items())
    ]


def _rival_row(row: dict[str, Any]) -> dict[str, Any]:
    """Return a quality_rivals row with its JSON lists as lists."""
    out = dict(row)
    for key in ("rival_cells", "rival_regions", "confusion_without"):
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

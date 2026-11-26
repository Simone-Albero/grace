"""Stage quality_report: the measures of every cell and of every node of the cut, then the verdicts of the regions."""

from __future__ import annotations

__all__ = ["run"]

import json
import sqlite3
from collections import defaultdict

import numpy as np
import polars as pl

from src import db
from src.config import DatasetConfig
from src.data import DUPLICATES, KINDS, LABEL
from src.provenance import KEYS, TIME, concentration, key_coincidence, time_nearness
from src.quality import (
    DEFECTS,
    branch_spread,
    column_values,
    cut_shares,
    invalid_values,
    label_conflicts,
    nearest_rivals,
    node_ranges,
    node_sums,
    pair_conflicts,
    rival_shares,
)
from src.report import COUNTS, judge
from stages.paths import (
    assignments_file,
    cell_tree_file,
    cells_file,
    need,
    prepared_columns,
    read_classes,
    read_data,
    read_points,
    read_provenance,
    read_unparsable,
    read_weights,
)


def run(cfg: DatasetConfig, conn: sqlite3.Connection, run_id: int) -> None:
    """Measure every cell and every node of the cut, and judge the regions at quality.tau.

    Reads the data file (the features as read, DUPLICATES), the cells
    stage clean could not read as numbers (paths.read_unparsable), the
    points (paths.read_points, as stage cells), cells.parquet,
    assignments.parquet and cell_tree.parquet, which must be of the same
    rows ("run cells again" otherwise). With q = method.quality. Order:
    quality.nearest_rivals on every row -> quality.rival_shares: per
    cell and class of its rows' nearest rival rows (a rival class), its
    rival cells and the share of its rows confused with it -> per node
    of the cut, quality.branch_spread, its rival class (the most
    frequent class of its rows' nearest rival rows, ties to the lower
    code) and, per original column (quality.column_values of its point
    columns), quality.cut_shares on quality.node_ranges ->
    quality.invalid_values -> quality.label_conflicts on every feature
    -> quality.pair_conflicts; with cfg.provenance, on provenance.parquet
    (paths.read_provenance: every row of the CSV, with the row of the
    data file it is a copy of; "run clean again" when they disagree),
    per key provenance.key_coincidence and, with a time,
    provenance.time_nearness, each with the nearest rival row and, the
    control, the nearest own row -> src.report.judge at q.tau and
    q.confidence, the one place they are applied -> with
    cfg.provenance, provenance.concentration of each key over the CSV
    rows of each region.

    Logs:
    - quality_rivals: one row per cell and rival class: cell, region,
      class, rival_class, rival_cells and rival_regions (JSON lists: its
      rival cells of that class and their regions), rival_share (the
      share of the cell's distinct rows whose nearest row in the dataset
      is of the rival class, a tie one half).
    - quality_branches: node, class, parent, cell as in
      cell_tree.parquet, then n_rows and n_read below the node, variance
      of its rows, cut_class (its rival class, null for none), cut_share
      (the least share over the columns of the cut class's distinct
      rows inside the node's values), n_cut_rows (the distinct rows of
      the cut class) and cut_columns (JSON list of [column, share] of
      the columns under one half, the lowest first).
    - quality_cells: cell, region, class, n_rows, n_read, n_duplicates
      (n_read - n_rows), conflict_rows, conflict_read, conflict_with
      (the classes its conflicting rows are shared with, joined by
      ","), missing_rows and missing_read (rows with a null or invalid
      value), per kind of data.KINDS <kind>_rows; with cfg.provenance,
      per configured key of provenance.KEYS <key>_same, <key>_own and
      <key>_chance (the sums over its rows of
      provenance.key_coincidence) and <key>_n (its rows summed, known
      for the row and both neighbours), and with a time time_near,
      time_near_own and time_n (the same of provenance.time_nearness).
      Measures only: the verdicts are the regions'.
    - quality_regions: src.report.judge's, and with cfg.provenance per
      key <key>_n_values, <key>_top and <key>_top_share
      (provenance.concentration over its CSV rows).
    - quality_classes, cell_missing (cell, region, column, kind,
      n_rows, n_read: the null and invalid cells; no row without one),
      label_conflict_groups (group, row, class, cell, region, n_read),
      quality_stats (one row: tau, confidence, n_rivals (the rows of
      quality_rivals), n_groups, n_cells, then src.report.judge's).

    n_rows counts distinct rows, n_read rows of the CSV.
    """
    q = cfg.method.quality
    classes = read_classes(cfg)
    cells = pl.read_parquet(need(cells_file(cfg), "cells")).sort("cell")
    assignments = pl.read_parquet(need(assignments_file(cfg), "cells"))
    tree = pl.read_parquet(need(cell_tree_file(cfg), "cells")).sort("node")
    cell = assignments.sort("row")["cell"].to_numpy()
    cell_class = cells["class"].to_numpy()
    m = len(cell_class)
    weights = read_weights(cfg)
    points, labels = read_points(cfg)
    if len(cell) != len(labels) or not np.array_equal(cell_class[cell], labels):
        raise ValueError("the cells were cut on other data: run cells again")
    rival_row, confused, own_row = nearest_rivals(points, labels)
    rivals = rival_shares(cell, rival_row, confused, cell_class)
    print(
        f"[quality_report] nearest rival rows of {len(cell)} rows found: "
        f"{len(rivals)} cells against a rival class",
        flush=True,
    )
    n_read = np.bincount(cell, weights=weights, minlength=m)
    branches = _branches(cfg, tree, points, labels, cell, n_read, rival_row, classes)
    del points

    data = read_data(cfg)
    features = data.drop(DUPLICATES, LABEL)
    invalid = invalid_values(features, read_unparsable(cfg))
    groups = label_conflicts(features, labels)
    conflicts = pair_conflicts(groups, labels, cell, weights)
    counts = _counts(cell, m, weights, groups, invalid)
    conflict_with = defaultdict(set)
    for a, b in conflicts:
        conflict_with[a].add(int(cell_class[b]))
        conflict_with[b].add(int(cell_class[a]))

    origin = None
    measured = [{} for _ in range(m)]
    if cfg.provenance is not None:
        origin, measured = _provenance(cfg, cell, labels, weights, rival_row, own_row)
    rival_rows = [_rival_row(rival, cell_class, classes) for rival in rivals]
    cell_rows = [
        _cell_row(c, cell_class, classes, counts, conflict_with, measured[c])
        for c in range(m)
    ]
    judged = judge(cell_rows, rival_rows, branches, classes, q.tau, q.confidence)
    region = np.array([row["region"] for row in judged.cells], dtype=np.int64)
    regions = judged.regions
    if origin is not None:
        regions = _hosts(origin, cell, region, regions)
    found = {d: judged.stats[f"regions_{d}"] for d in DEFECTS}
    print(
        f"[quality_report] {len(rivals)} cells against a rival class, "
        f"{int(groups.max()) + 1} label conflicts; "
        f"{len(judged.cells)} cells merged into {len(regions)} regions; "
        f"regions by defect: {found}",
        flush=True,
    )
    db.log_table(conn, run_id, "quality_rivals", judged.rivals)
    db.log_table(conn, run_id, "quality_branches", branches.to_dicts())
    db.log_table(conn, run_id, "quality_cells", judged.cells)
    db.log_table(conn, run_id, "quality_regions", regions)
    db.log_table(conn, run_id, "quality_classes", judged.classes)
    db.log_table(
        conn, run_id, "cell_missing", _missing_rows(invalid, cell, region, weights)
    )
    db.log_table(
        conn,
        run_id,
        "label_conflict_groups",
        [
            {
                "group": int(groups[row]),
                "row": int(row),
                "class": classes[labels[row]],
                "cell": int(cell[row]),
                "region": int(region[cell[row]]),
                "n_read": float(weights[row]),
            }
            for row in np.flatnonzero(groups >= 0)
        ],
    )
    stats = {k: v for k, v in judged.stats.items() if k not in ("tau", "confidence")}
    db.log_table(
        conn,
        run_id,
        "quality_stats",
        [
            {
                "tau": q.tau,
                "confidence": q.confidence,
                "n_rivals": len(rivals),
                "n_groups": int(groups.max()) + 1,
                "n_cells": len(judged.cells),
                **stats,
            }
        ],
    )


def _branches(
    cfg: DatasetConfig,
    tree: pl.DataFrame,
    points: np.ndarray,
    labels: np.ndarray,
    cell: np.ndarray,
    n_read: np.ndarray,
    rival_row: np.ndarray,
    classes: list[str],
) -> pl.DataFrame:
    """Return the quality_branches rows: the cells' tree with the rows, the spread and the cut of every node.

    The rival class of a node is the most frequent class of its rows'
    nearest rival rows (ties to the lower code); its cut, per original
    column, quality.cut_shares against it.
    """
    at = {v: i for i, v in enumerate(tree["node"].to_list())}
    parent = np.array([at.get(v, -1) for v in tree["parent"].to_list()])
    node_cell = tree["cell"].fill_null(-1).to_numpy()
    variance, n_rows = branch_spread(points, cell, parent, node_cell)
    m = int(node_cell.max()) + 1
    read = node_sums(parent, node_cell, n_read)
    has_rival = rival_row >= 0
    by_class = np.zeros((m, len(classes)), dtype=np.int64)
    np.add.at(by_class, (cell[has_rival], labels[rival_row[has_rival]]), 1)
    near = node_sums(parent, node_cell, by_class)
    rival = np.where(near.sum(axis=1) > 0, near.argmax(axis=1), -1)
    names = prepared_columns(cfg)
    columns = list(dict.fromkeys(names))
    owner = np.array([columns.index(name) for name in names], dtype=np.int64)
    shares = np.empty((len(parent), len(columns)))
    for c, column in enumerate(columns):
        numeric = column in cfg.num_cols
        values = column_values(points[:, owner == c], numeric)
        ranges = node_ranges(values, cell, parent, node_cell, numeric)
        shares[:, c] = cut_shares(ranges, rival, values, labels, numeric)
    print(f"[quality_report] cuts on {len(columns)} columns measured", flush=True)
    n_class = np.bincount(labels, minlength=len(classes))
    has_cut = rival >= 0
    least = np.full(len(parent), np.nan)
    least[has_cut] = shares[has_cut].min(axis=1)
    cut_columns = [
        (
            json.dumps(
                [
                    [columns[c], float(shares[v, c])]
                    for c in np.argsort(shares[v], kind="stable")
                    if shares[v, c] < 0.5
                ]
            )
            if has_cut[v]
            else None
        )
        for v in range(len(parent))
    ]
    return tree.select("node", "class", "parent", "cell").with_columns(
        n_rows=pl.Series(n_rows.astype(np.int64)),
        n_read=pl.Series(read.astype(np.float64)),
        variance=pl.Series(variance),
        cut_class=pl.Series(
            [classes[k] if k >= 0 else None for k in rival], dtype=pl.String
        ),
        cut_share=pl.Series(least).fill_nan(None),
        n_cut_rows=pl.Series(np.where(has_cut, n_class[np.maximum(rival, 0)], 0)),
        cut_columns=pl.Series(cut_columns, dtype=pl.String),
    )


def _counts(cell, m, weights, groups, invalid) -> dict[str, np.ndarray]:
    """Return per cell its distinct and CSV rows, conflicting rows and null or invalid rows."""
    conflicting = groups >= 0
    hit = np.unique(invalid["row"].to_numpy())
    counts = {
        "n_rows": np.bincount(cell, minlength=m),
        "n_read": np.bincount(cell, weights=weights, minlength=m),
        "conflict_rows": np.bincount(cell[conflicting], minlength=m),
        "conflict_read": np.bincount(
            cell[conflicting], weights=weights[conflicting], minlength=m
        ),
        "missing_rows": np.bincount(cell[hit], minlength=m),
        "missing_read": np.bincount(cell[hit], weights=weights[hit], minlength=m),
    }
    for kind in KINDS:
        rows = np.unique(invalid.filter(pl.col("kind") == kind)["row"].to_numpy())
        counts[f"{kind}_rows"] = np.bincount(cell[rows], minlength=m)
    return counts


def _rival_row(rival, cell_class, classes) -> dict:
    """Return the quality_rivals row of a cell and a rival class, without the regions (src.report.judge)."""
    own, k, others, share = rival
    return {
        "cell": own,
        "class": classes[cell_class[own]],
        "rival_class": classes[k],
        "rival_cells": json.dumps(others),
        "rival_share": share,
    }


def _cell_row(c, cell_class, classes, counts, conflict_with, measured) -> dict:
    """Return the quality_cells row of a cell, without its region (src.report.judge)."""
    own = {key: counts[key][c] for key in COUNTS}
    return {
        "cell": c,
        "class": classes[cell_class[c]],
        "n_rows": int(own["n_rows"]),
        "n_read": float(own["n_read"]),
        "n_duplicates": float(own["n_read"] - own["n_rows"]),
        "conflict_rows": int(own["conflict_rows"]),
        "conflict_read": float(own["conflict_read"]),
        "conflict_with": _names(conflict_with[c], classes),
        "missing_rows": int(own["missing_rows"]),
        "missing_read": float(own["missing_read"]),
        **{f"{kind}_rows": int(own[f"{kind}_rows"]) for kind in KINDS},
        **measured,
    }


def _names(codes, classes: list[str]) -> str | None:
    """Return the names of class codes in code order joined by ",", None for none."""
    return ",".join(classes[c] for c in sorted(codes)) or None


def _provenance(
    cfg: DatasetConfig,
    cell: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    rival_row: np.ndarray,
    own_row: np.ndarray,
) -> tuple[pl.DataFrame, list[dict]]:
    """Return the provenance of the CSV rows and, per cell, the provenance columns of its quality_cells row.

    Args:
        cfg: The dataset config, with a provenance.
        cell: The cell of every row of the data file.
        labels: The class code of every row of the data file.
        weights: The copies of every row of the data file in the CSV.
        rival_row: The nearest rival row of every row (-1 for none).
        own_row: The nearest own row of every row (-1 for none).

    Returns:
        provenance.parquet (paths.read_provenance), and one dict per
        cell, in cell order: per key of provenance.KEYS in the file,
        <key>_same, <key>_own, <key>_chance and <key>_n
        (provenance.key_coincidence summed over the cell's rows known
        for the row and both neighbours); with a time, time_near,
        time_near_own and time_n (provenance.time_nearness, the same).

    Raises:
        ValueError: When provenance.parquet is not of the rows of the
            data file ("run clean again").
    """
    origin = read_provenance(cfg)
    row = origin["row"].to_numpy()
    m = int(cell.max()) + 1
    if not np.array_equal(np.bincount(row, minlength=len(cell)), weights):
        raise ValueError("the provenance is of other rows: run clean again")
    sums: dict[str, np.ndarray] = {}

    def summed(name: str, known: np.ndarray, value: np.ndarray | None) -> None:
        added = None if value is None else value[known]
        sums[name] = np.bincount(cell[known], weights=added, minlength=m)

    for key in KEYS:
        if key not in origin.columns:
            continue
        same, own, chance = key_coincidence(
            row, origin[key], labels, rival_row, own_row
        )
        known = ~np.isnan(same) & ~np.isnan(own)
        summed(f"{key}_same", known, same)
        summed(f"{key}_own", known, own)
        summed(f"{key}_chance", known, chance)
        summed(f"{key}_n", known, None)
    if TIME in origin.columns:
        times = origin[TIME].to_numpy()
        near = time_nearness(row, times, labels, rival_row, own=False)
        near_own = time_nearness(row, times, labels, own_row, own=True)
        known = ~np.isnan(near) & ~np.isnan(near_own)
        summed("time_near", known, near)
        summed("time_near_own", known, near_own)
        summed("time_n", known, None)
    print(f"[quality_report] provenance of {len(row)} CSV rows measured", flush=True)
    per_cell = [
        {
            name: int(v[c]) if name.endswith("_n") else float(v[c])
            for name, v in sums.items()
        }
        for c in range(m)
    ]
    return origin, per_cell


def _hosts(
    origin: pl.DataFrame, cell: np.ndarray, region: np.ndarray, regions: list[dict]
) -> list[dict]:
    """Return the quality_regions rows with the concentration of each key's values over their CSV rows.

    Per key of provenance.KEYS in origin: <key>_n_values, <key>_top and
    <key>_top_share (provenance.concentration).
    """
    order = {r["region"]: i for i, r in enumerate(regions)}
    of_cell = np.array([order[int(r)] for r in region], dtype=np.int64)
    group = of_cell[cell[origin["row"].to_numpy()]]
    out = [dict(r) for r in regions]
    for key in KEYS:
        if key not in origin.columns:
            continue
        found = concentration(group, origin[key], len(regions))
        for i, values in enumerate(found.iter_rows(named=True)):
            out[i].update({f"{key}_{name}": v for name, v in values.items()})
    return out


def _missing_rows(
    invalid: pl.DataFrame, cell: np.ndarray, region: np.ndarray, weights: np.ndarray
) -> list[dict]:
    """Return the cell_missing rows: the null and invalid cells per cell, column and kind."""
    rows = invalid["row"].to_numpy()
    found = invalid.with_columns(
        cell=pl.Series(cell[rows]),
        region=pl.Series(region[cell[rows]]),
        weight=pl.Series(weights[rows]),
    )
    return (
        found.group_by("cell", "region", "column", "kind")
        .agg(n_rows=pl.len(), n_read=pl.col("weight").sum())
        .sort("cell", "column", "kind")
        .to_dicts()
    )

"""Stage cells: each class of the data cut into cells."""

from __future__ import annotations

__all__ = ["run"]

import sqlite3

import numpy as np
import polars as pl

from src import db
from src.cells import CellTree, class_cells
from src.config import DatasetConfig
from stages.paths import (
    assignments_file,
    cell_tree_file,
    cells_file,
    read_classes,
    read_points,
    read_weights,
)


def run(cfg: DatasetConfig, conn: sqlite3.Connection, run_id: int) -> None:
    """Cut each class into cells.

    Reads the points of the prepared file (paths.read_points: the
    numbers scaled for distance, the categorical slots, not MISSING;
    every distance from here on is euclidean on them) and the weight of
    every row (paths.read_weights: its copies in the CSV). Per class,
    ascending: cells.class_cells with method.cells (max_rows, min_rows,
    max_fit_samples) and seed -> cell id = cell + offset, offset += the
    class's cells; node id = node + node offset.

    Writes to output_dir: cells.parquet (cell, class code, n_rows,
    n_read), assignments.parquet (row, cell for every row),
    cell_tree.parquet (node, class code, parent (-1 at a root), cell
    (null at an internal node): the cut of every class, nodes in
    pre-order class by class).
    Logs: cells (cell, class name, n_rows, n_read), cell_tree (node,
    class name, parent, depth (0 at a root), cell, n_rows and n_read of
    the cells below), class_cells (class, n_rows, n_read, n_cells),
    cell_stats (one row: n_cells, distance, max_rows, min_rows,
    max_fit_samples). n_rows counts distinct rows, n_read rows of the
    CSV (the summed weights).
    """
    limits = cfg.method.cells
    points, labels = read_points(cfg)
    weights = read_weights(cfg)
    classes = read_classes(cfg)
    cell = np.empty(len(labels), dtype=np.int64)
    cell_rows, tree_rows, class_rows, tree_files = [], [], [], []
    offset = node_offset = 0
    for code, name in enumerate(classes):
        rows = np.flatnonzero(labels == code)
        own, tree = class_cells(
            points[rows],
            weights[rows],
            limits.max_rows,
            limits.min_rows,
            limits.max_fit_samples,
            cfg.seed,
        )
        k = int(own.max()) + 1
        sizes = np.bincount(own, minlength=k)
        reads = np.bincount(own, weights=weights[rows], minlength=k)
        cell[rows] = own + offset
        cell_rows += [
            {
                "cell": offset + c,
                "class": name,
                "n_rows": int(sizes[c]),
                "n_read": float(reads[c]),
            }
            for c in range(k)
        ]
        tree_rows += _tree_rows(tree, sizes, reads, name, offset, node_offset)
        tree_files.append(
            pl.DataFrame(
                {
                    "node": node_offset + np.arange(len(tree.parent)),
                    "class": np.full(len(tree.parent), code),
                    "parent": np.where(tree.parent >= 0, tree.parent + node_offset, -1),
                    "cell": pl.Series(
                        [int(c) + offset if c >= 0 else None for c in tree.cell],
                        dtype=pl.Int64,
                    ),
                }
            )
        )
        class_rows.append(
            {
                "class": name,
                "n_rows": len(rows),
                "n_read": float(weights[rows].sum()),
                "n_cells": k,
            }
        )
        print(
            f"[cells] {name}: {len(rows)} rows -> {k} cells "
            f"(rows {sizes.min()}-{int(np.median(sizes))}-{sizes.max()})",
            flush=True,
        )
        offset += k
        node_offset += len(tree.parent)

    pl.DataFrame(
        {
            "cell": [r["cell"] for r in cell_rows],
            "class": [classes.index(r["class"]) for r in cell_rows],
            "n_rows": [r["n_rows"] for r in cell_rows],
            "n_read": [r["n_read"] for r in cell_rows],
        }
    ).write_parquet(cells_file(cfg))
    pl.DataFrame({"row": np.arange(len(cell)), "cell": cell}).write_parquet(
        assignments_file(cfg)
    )
    pl.concat(tree_files).write_parquet(cell_tree_file(cfg))
    print(f"[cells] saved {offset} cells", flush=True)

    db.log_table(conn, run_id, "cells", cell_rows)
    db.log_table(conn, run_id, "cell_tree", tree_rows)
    db.log_table(conn, run_id, "class_cells", class_rows)
    db.log_table(
        conn,
        run_id,
        "cell_stats",
        [
            {
                "n_cells": offset,
                "distance": cfg.method.distance,
                "max_rows": limits.max_rows,
                "min_rows": limits.min_rows,
                "max_fit_samples": limits.max_fit_samples,
            }
        ],
    )


def _tree_rows(
    tree: CellTree,
    sizes: np.ndarray,
    reads: np.ndarray,
    name: str,
    cell_offset: int,
    node_offset: int,
) -> list[dict]:
    """Return the cell_tree rows of one class: each node counts the rows of the cells below it."""
    k = len(tree.parent)
    leaf = tree.cell >= 0
    n_rows, n_read = np.zeros(k), np.zeros(k)
    n_rows[leaf] = sizes[tree.cell[leaf]]
    n_read[leaf] = reads[tree.cell[leaf]]
    # Pre-order: walking back adds each finished node into its parent.
    for node in range(k - 1, 0, -1):
        n_rows[tree.parent[node]] += n_rows[node]
        n_read[tree.parent[node]] += n_read[node]
    depth = np.zeros(k, dtype=np.int64)
    for node in range(1, k):
        depth[node] = depth[tree.parent[node]] + 1
    return [
        {
            "node": node_offset + node,
            "class": name,
            "parent": node_offset + int(parent) if parent >= 0 else -1,
            "depth": int(depth[node]),
            "cell": cell_offset + int(cell) if cell >= 0 else None,
            "n_rows": int(n_rows[node]),
            "n_read": float(n_read[node]),
        }
        for node, (parent, cell) in enumerate(zip(tree.parent, tree.cell))
    ]

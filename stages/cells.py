"""Stage cells: each class of the data cut into cells."""

from __future__ import annotations

__all__ = ["run"]

import sqlite3

import numpy as np
import polars as pl

from src import db
from src.cells import CellTree, class_cells, nearest_rows
from src.config import DatasetConfig
from src.graph import sample_rows
from stages.paths import (
    assignments_file,
    cell_tree_file,
    cells_file,
    read_classes,
    read_points,
    read_weights,
)

_JOIN_ROWS = 200_000  # rows not sampled joined to their nearest sampled row at once


def run(cfg: DatasetConfig, conn: sqlite3.Connection, run_id: int) -> None:
    """Read each class on a uniform sample of its rows, and cut it into cells.

    Reads the points (paths.read_points, on disk: the numbers and the
    categorical slots; every distance from here on is euclidean on
    them) and the copies of every row in the CSV (paths.read_weights),
    only to count n_read: no row weighs more than another in the cut.
    Per class, ascending: graph.sample_rows of its rows
    (method.cells.max_sample_rows, seed: the rows read, the first of one
    random order of the class, every row of a class with no more and
    with None) -> cells.class_cells of the rows read with
    method.min_rows (what tau and the confidence ask for) -> every other
    row of the class joins the cell of its nearest row read
    (cells.nearest_rows, _JOIN_ROWS at a time) -> cell id = cell +
    offset, offset += the class's cells; node id = node + node offset.

    Writes to output_dir: cells.parquet (cell, class code, n_rows,
    n_read, n_sampled), assignments.parquet (row, cell, sampled for
    every row), cell_tree.parquet (node, class code, parent (-1 at a
    root), cell (null at an internal node): the cut of every class,
    nodes in pre-order class by class).
    Logs: cells (cell, class name, n_rows, n_read, n_sampled), cell_tree
    (node, class name, parent, depth (0 at a root), cell, n_rows,
    n_read and n_sampled of the cells below), class_cells (class,
    n_rows, n_read, n_sampled, n_cells), cell_stats (one row: n_cells,
    min_rows, max_sample_rows). n_rows counts distinct rows, n_read rows
    of the CSV (the summed copies), n_sampled the rows read.
    """
    limits = cfg.method.cells
    points, labels = read_points(cfg)
    weights = read_weights(cfg)
    classes = read_classes(cfg)
    cell = np.empty(len(labels), dtype=np.int64)
    sampled = np.zeros(len(labels), dtype=bool)
    cell_rows, tree_rows, class_rows, tree_files = [], [], [], []
    offset = node_offset = 0
    for code, name in enumerate(classes):
        rows = np.flatnonzero(labels == code)
        read = rows[sample_rows(len(rows), limits.max_sample_rows, cfg.seed)]
        X = np.asarray(points[read], dtype=np.float64)
        own, tree = class_cells(X, cfg.method.min_rows)
        cell[read] = own + offset
        sampled[read] = True
        others = rows[~np.isin(rows, read)]
        for start in range(0, len(others), _JOIN_ROWS):
            block = others[start : start + _JOIN_ROWS]
            near = nearest_rows(np.asarray(points[block], dtype=np.float64), X)
            cell[block] = own[near] + offset
        k = int(own.max()) + 1
        of_class = cell[rows] - offset
        sizes = np.bincount(of_class, minlength=k)
        reads = np.bincount(of_class, weights=weights[rows], minlength=k)
        n_sampled = np.bincount(own, minlength=k)
        cell_rows += [
            {
                "cell": offset + c,
                "class": name,
                "n_rows": int(sizes[c]),
                "n_read": float(reads[c]),
                "n_sampled": int(n_sampled[c]),
            }
            for c in range(k)
        ]
        tree_rows += _tree_rows(
            tree, sizes, reads, n_sampled, name, offset, node_offset
        )
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
                "n_sampled": len(read),
                "n_cells": k,
            }
        )
        print(
            f"[cells] {name}: {len(rows)} rows, {len(read)} read -> {k} cells "
            f"(rows read {n_sampled.min()}-{int(np.median(n_sampled))}-"
            f"{n_sampled.max()})",
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
            "n_sampled": [r["n_sampled"] for r in cell_rows],
        }
    ).write_parquet(cells_file(cfg))
    pl.DataFrame(
        {"row": np.arange(len(cell)), "cell": cell, "sampled": sampled}
    ).write_parquet(assignments_file(cfg))
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
                "min_rows": cfg.method.min_rows,
                "max_sample_rows": limits.max_sample_rows,
            }
        ],
    )


def _tree_rows(
    tree: CellTree,
    sizes: np.ndarray,
    reads: np.ndarray,
    n_sampled: np.ndarray,
    name: str,
    cell_offset: int,
    node_offset: int,
) -> list[dict]:
    """Return the cell_tree rows of one class: each node counts the rows of the cells below it."""
    k = len(tree.parent)
    leaf = tree.cell >= 0
    n_rows, n_read, n_read_rows = np.zeros(k), np.zeros(k), np.zeros(k)
    n_rows[leaf] = sizes[tree.cell[leaf]]
    n_read[leaf] = reads[tree.cell[leaf]]
    n_read_rows[leaf] = n_sampled[tree.cell[leaf]]
    # Pre-order: walking back adds each finished node into its parent.
    for node in range(k - 1, 0, -1):
        n_rows[tree.parent[node]] += n_rows[node]
        n_read[tree.parent[node]] += n_read[node]
        n_read_rows[tree.parent[node]] += n_read_rows[node]
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
            "n_sampled": int(n_read_rows[node]),
        }
        for node, (parent, cell) in enumerate(zip(tree.parent, tree.cell))
    ]

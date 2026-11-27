"""Stage prepare: the cleaned rows as the features distances are measured on."""

from __future__ import annotations

__all__ = ["run"]

import json
import math
import sqlite3

import numpy as np
import pyarrow.parquet as pq

from src import db
from src.config import DatasetConfig
from src.data import feature_cols
from src.prepare import (
    MISSING,
    NULL,
    OTHER,
    column_names,
    column_sources,
    fit_preparation,
    transform,
)
from stages.paths import data_blocks, data_file, need, points_columns_file, points_file


def run(cfg: DatasetConfig, conn: sqlite3.Connection, run_id: int) -> None:
    """Prepare the cleaned rows for the clustering.

    Reads the data file twice, a block at a time (paths.data_blocks), so
    the rows are never held whole. Order: data.feature_cols of its
    columns (the kept num_cols numeric, the kept cat_cols categorical;
    DUPLICATES is a weight, read by stage cells) ->
    prepare.fit_preparation on every row (top_n) -> prepare.transform,
    block by block.

    Writes to output_dir: points.npy (prepare.transform's points,
    float32, shape (n, d); row i is row i of the data file) and
    points.json ({"columns": prepare.column_names, "sources":
    prepare.column_sources}).
    Logs: prepare_summary (one row: n_rows, n_num, n_cat, n_columns
    (of the points), n_missing (numbers set to 0), n_missing_rows (rows
    with one)), prepared_columns (column, source, kind "num" | "missing"
    | "cat", mean and std of the signed log for num, value, n_rows (rows
    with the slot active) and weight (the active slot's value, the same
    for every slot of a column) for cat, n_rows (rows with a missing
    number) for missing; null where not applying).
    """
    names = pq.ParquetFile(need(data_file(cfg), "clean")).schema_arrow.names
    n = pq.ParquetFile(data_file(cfg)).metadata.num_rows
    features = feature_cols(names, cfg.num_cols, cfg.cat_cols)
    read = [*features.num, *features.cat]
    print(
        f"[prepare] {len(features.num)} numeric and {len(features.cat)} "
        f"categorical columns, top_n={cfg.top_n}",
        flush=True,
    )
    preparation = fit_preparation(
        data_blocks(cfg, read), features.num, features.cat, top_n=cfg.top_n
    )
    columns = column_names(preparation)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    points = np.lib.format.open_memmap(
        points_file(cfg), mode="w+", dtype=np.float32, shape=(n, len(columns))
    )
    n_missing = n_missing_rows = start = 0
    for block in data_blocks(cfg, read):
        block_points, missing = transform(preparation, block)
        points[start : start + block.height] = block_points
        n_missing += int(missing.sum())
        n_missing_rows += int((missing > 0).sum())
        start += block.height
    points.flush()
    del points
    points_columns_file(cfg).write_text(
        json.dumps({"columns": columns, "sources": column_sources(preparation)})
    )
    print(
        f"[prepare] saved {n} rows x {len(columns)} columns, "
        f"{n_missing} missing or non-finite numbers set to 0",
        flush=True,
    )

    scaler = preparation.scaler
    column_rows = [
        {
            "column": column,
            "source": column,
            "kind": "num",
            "mean": _number(mean),
            "std": _number(math.sqrt(var)),
            "value": None,
            "n_rows": None,
            "weight": None,
        }
        for column, mean, var in zip(preparation.num_cols, scaler.mean_, scaler.var_)
    ]
    column_rows.append(
        {
            "column": MISSING,
            "source": MISSING,
            "kind": "missing",
            "mean": None,
            "std": None,
            "value": None,
            "n_rows": n_missing_rows,
            "weight": None,
        }
    )
    for encoder, weight in zip(preparation.encoders, preparation.weights):
        values = [NULL if v is None else v for v in encoder.values_]
        values += [OTHER] * encoder.has_other_
        for slot, value, rows in zip(
            encoder.slot_names(), values, encoder.slot_counts_
        ):
            column_rows.append(
                {
                    "column": slot,
                    "source": encoder.column,
                    "kind": "cat",
                    "mean": None,
                    "std": None,
                    "value": value,
                    "n_rows": rows,
                    "weight": weight,
                }
            )
    db.log_table(
        conn,
        run_id,
        "prepare_summary",
        [
            {
                "n_rows": n,
                "n_num": len(features.num),
                "n_cat": len(features.cat),
                "n_columns": len(columns),
                "n_missing": n_missing,
                "n_missing_rows": n_missing_rows,
            }
        ],
    )
    db.log_table(conn, run_id, "prepared_columns", column_rows)


def _number(value: float) -> float | None:
    """Return value as a float, None when not finite."""
    return float(value) if math.isfinite(value) else None

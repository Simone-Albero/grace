"""Stage prepare: the cleaned rows as the features distances are measured on."""

from __future__ import annotations

__all__ = ["run"]

import math
import sqlite3

from src import db
from src.config import DatasetConfig
from src.data import LABEL, feature_cols
from src.prepare import (
    MISSING,
    NULL,
    OTHER,
    column_names,
    fit_preparation,
    transform,
)
from stages.paths import prepared_file, read_data


def run(cfg: DatasetConfig, conn: sqlite3.Connection, run_id: int) -> None:
    """Prepare the cleaned rows for the clustering.

    Reads the data file. Order: data.feature_cols of its columns (the
    kept num_cols and DUPLICATES numeric, the kept cat_cols
    categorical) -> prepare.fit_preparation on every row (top_n,
    method.prepare.cat_cost) -> prepare.transform.

    Writes to output_dir: prepared.parquet (the columns of
    prepare.column_names, Float64, MISSING included, then LABEL as in
    the data file; row i is row i of the data file).
    Logs: prepare_summary (one row: n_rows, n_num, n_cat, n_columns,
    n_missing (numbers set to 0), n_missing_rows (rows with one)),
    prepared_columns (column, source, kind "num" | "missing" | "cat",
    mean and std of the signed log for num, value, n_rows (rows with
    the slot active) and weight (the active slot's value, the same for
    every slot of a column) for cat, n_rows (rows with a missing number)
    for missing; null where not applying).
    """
    data = read_data(cfg)
    features = feature_cols(data.columns, cfg.num_cols, cfg.cat_cols)
    print(
        f"[prepare] {len(features.num)} numeric and {len(features.cat)} "
        f"categorical columns, top_n={cfg.top_n}",
        flush=True,
    )
    preparation = fit_preparation(
        data,
        features.num,
        features.cat,
        top_n=cfg.top_n,
        cat_cost=cfg.method.prepare.cat_cost,
    )
    prepared = transform(preparation, data)
    names = column_names(preparation)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    prepared.with_columns(data[LABEL]).write_parquet(prepared_file(cfg))
    print(
        f"[prepare] saved {prepared.height} rows x {len(names)} columns, "
        f"{int(prepared[MISSING].sum())} missing or non-finite numbers set to 0",
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
    missing = prepared[MISSING]
    column_rows.append(
        {
            "column": MISSING,
            "source": MISSING,
            "kind": "missing",
            "mean": None,
            "std": None,
            "value": None,
            "n_rows": int((missing > 0).sum()),
            "weight": None,
        }
    )
    for encoder, weight in zip(preparation.encoders, preparation.weights):
        values = [NULL if v is None else v for v in encoder.values_]
        values += [OTHER] * encoder.has_other_
        codes = encoder.transform(data[encoder.column])
        # Slot i is code i + 1; the "other" slot, last, is code 0.
        slot_codes = [*range(1, len(encoder.values_) + 1), 0]
        for slot, value, code in zip(encoder.slot_names(), values, slot_codes):
            column_rows.append(
                {
                    "column": slot,
                    "source": encoder.column,
                    "kind": "cat",
                    "mean": None,
                    "std": None,
                    "value": value,
                    "n_rows": int((codes == code).sum()),
                    "weight": weight,
                }
            )
    db.log_table(
        conn,
        run_id,
        "prepare_summary",
        [
            {
                "n_rows": prepared.height,
                "n_num": len(features.num),
                "n_cat": len(features.cat),
                "n_columns": len(names),
                "n_missing": int(missing.sum()),
                "n_missing_rows": int((missing > 0).sum()),
            }
        ],
    )
    db.log_table(conn, run_id, "prepared_columns", column_rows)


def _number(value: float) -> float | None:
    """Return value as a float, None when not finite."""
    return float(value) if math.isfinite(value) else None

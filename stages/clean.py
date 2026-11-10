"""Stage clean: read the raw rows, drop duplicates, rare classes and constant columns."""

from __future__ import annotations

__all__ = ["run"]

import json
import sqlite3

import polars as pl

from src import db
from src.config import DatasetConfig
from src.data import (
    LABEL,
    constant_columns,
    drop_duplicates,
    drop_rare_classes,
    feature_cols,
)
from stages.paths import classes_file, data_file


def _read(cfg: DatasetConfig) -> pl.DataFrame:
    columns = [*cfg.num_cols, *cfg.cat_cols, cfg.label_col]
    scan = pl.scan_csv(cfg.path, infer_schema=False)
    missing = sorted(set(columns) - set(scan.collect_schema().names()))
    if missing:
        raise ValueError(f"columns missing from {cfg.path}: {missing}")
    numeric = pl.col(cfg.num_cols).cast(pl.Float64, strict=False)
    return scan.select(columns).with_columns(numeric).collect()


def run(cfg: DatasetConfig, conn: sqlite3.Connection, run_id: int) -> None:
    """Read the CSV, drop duplicates, rare classes and constant columns.

    Order: read num_cols (Float64; a value that is not a number becomes
    null) and cat_cols and label_col (text) of cfg.path ->
    data.drop_duplicates (one copy of each row equal on every column,
    the label included, with DUPLICATES = its copies, 1 when unique) ->
    data.drop_rare_classes(min_class_count), on the distinct rows ->
    data.constant_columns of num_cols and cat_cols, dropped -> classes
    sorted, label_col replaced by its code in LABEL. No row is dropped
    for a missing or non-finite value, and nothing is scaled or encoded:
    the region space is fitted by stage regions.

    Writes to output_dir: data.parquet (the kept num_cols as read, then
    DUPLICATES, a numeric feature, then the kept cat_cols as read, then
    LABEL as class code; row order = the CSV's order of first copies among the kept
    rows, so a row's position is its id everywhere downstream),
    classes.json (the names, indexed by code).
    Logs: clean_summary (one row: n_raw, n_duplicate (copies dropped),
    n_rare, n_kept, n_classes, n_constant; n_kept = n_raw -
    n_duplicate - n_rare), dropped_classes (label, n_rows: distinct),
    dropped_columns (column), class_counts (label, n_read, n_rows:
    rows read and kept), features (column, kind "num" | "cat": the kept
    ones, DUPLICATES included).

    Raises:
        ValueError: When a configured column is missing from the CSV, a
            row has no label, or no class survives.
    """
    print(f"[clean] reading {cfg.path}", flush=True)
    raw = _read(cfg)
    n_unlabelled = raw[cfg.label_col].null_count()
    if n_unlabelled:
        raise ValueError(f"{n_unlabelled} rows of {cfg.path} have no {cfg.label_col}")
    distinct = drop_duplicates(raw)
    kept, rare_classes = drop_rare_classes(distinct, cfg.label_col, cfg.min_class_count)
    if kept.is_empty():
        raise ValueError(f"no class has min_class_count={cfg.min_class_count} rows")
    constant_cols = constant_columns(kept, [*cfg.num_cols, *cfg.cat_cols])
    features = feature_cols(
        [column for column in kept.columns if column not in constant_cols],
        cfg.num_cols,
        cfg.cat_cols,
    )
    n_duplicate = raw.height - distinct.height
    n_rare = distinct.height - kept.height
    print(
        f"[clean] {raw.height} rows, {n_duplicate} duplicates, {n_rare} in "
        f"{len(rare_classes)} rare classes, {len(constant_cols)} constant columns",
        flush=True,
    )
    classes = sorted(kept[cfg.label_col].unique().to_list())
    codes = pl.col(cfg.label_col).replace_strict(
        {name: code for code, name in enumerate(classes)}, return_dtype=pl.Int64
    )
    data = kept.select(*features.num, *features.cat, codes.alias(LABEL))

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    data.write_parquet(data_file(cfg))
    classes_file(cfg).write_text(json.dumps(classes))
    print(f"[clean] saved {data.height} rows", flush=True)

    raw_counts = dict(raw[cfg.label_col].value_counts().iter_rows())
    distinct_counts = dict(distinct[cfg.label_col].value_counts().iter_rows())
    db.log_table(
        conn,
        run_id,
        "clean_summary",
        [
            {
                "n_raw": raw.height,
                "n_duplicate": n_duplicate,
                "n_rare": n_rare,
                "n_kept": kept.height,
                "n_classes": len(classes),
                "n_constant": len(constant_cols),
            }
        ],
    )
    db.log_table(
        conn,
        run_id,
        "dropped_classes",
        [{"label": label, "n_rows": distinct_counts[label]} for label in rare_classes],
    )
    db.log_table(
        conn,
        run_id,
        "dropped_columns",
        [{"column": column} for column in constant_cols],
    )
    db.log_table(
        conn,
        run_id,
        "class_counts",
        [
            {
                "label": label,
                "n_read": raw_counts[label],
                "n_rows": distinct_counts[label],
            }
            for label in classes
        ],
    )
    db.log_table(
        conn,
        run_id,
        "features",
        [{"column": column, "kind": "num"} for column in features.num]
        + [{"column": column, "kind": "cat"} for column in features.cat],
    )

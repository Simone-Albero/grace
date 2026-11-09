"""Stage split: clean and split the raw rows."""

from __future__ import annotations

__all__ = ["run"]

import json
import sqlite3

import polars as pl

from src import db
from src.config import DatasetConfig
from src.data import (
    LABEL,
    SPLITS,
    constant_columns,
    drop_rare_classes,
    stratified_split,
)
from src.preprocessing import feature_cols
from stages.paths import classes_file, split_file


def _read(cfg: DatasetConfig) -> pl.DataFrame:
    columns = [*cfg.num_cols, *cfg.cat_cols, cfg.label_col]
    scan = pl.scan_csv(cfg.path, infer_schema=False)
    missing = sorted(set(columns) - set(scan.collect_schema().names()))
    if missing:
        raise ValueError(f"columns missing from {cfg.path}: {missing}")
    numeric = pl.col(cfg.num_cols).cast(pl.Float64, strict=False)
    return scan.select(columns).with_columns(numeric).collect()


def run(cfg: DatasetConfig, conn: sqlite3.Connection, run_id: int) -> None:
    """Read the CSV, drop rare classes and constant columns, split.

    Order: read num_cols (Float64; a value that is not a number becomes
    null) and cat_cols and label_col (text) of cfg.path ->
    data.drop_rare_classes(min_class_count) -> data.constant_columns of
    num_cols and cat_cols, dropped -> classes sorted, label_col replaced
    by its code in LABEL -> data.stratified_split(cfg.split, seed). No
    row is dropped for a missing or non-finite value, and nothing is
    scaled or encoded: the classifier pipeline and the region space are
    fitted on train by their own stages.

    Writes to output_dir: {train,val,test}.parquet (the kept num_cols
    and cat_cols as read, then LABEL as class code; row order = the
    split's order, so a row's position is its id everywhere
    downstream), classes.json (the names, indexed by code).
    Logs: split_summary (one row: n_raw, n_rare, n_kept, n_classes,
    n_constant), dropped_classes (label, n_rows), dropped_columns
    (column), split_class_counts (label, train, val, test), features
    (column, kind "num" | "cat": the kept ones).

    Raises:
        ValueError: When a configured column is missing from the CSV, a
            row has no label, or no class survives.
    """
    print(f"[split] reading {cfg.path}", flush=True)
    raw = _read(cfg)
    n_unlabelled = raw[cfg.label_col].null_count()
    if n_unlabelled:
        raise ValueError(f"{n_unlabelled} rows of {cfg.path} have no {cfg.label_col}")
    kept, rare_classes = drop_rare_classes(raw, cfg.label_col, cfg.min_class_count)
    if kept.is_empty():
        raise ValueError(f"no class has min_class_count={cfg.min_class_count} rows")
    constant_cols = constant_columns(kept, [*cfg.num_cols, *cfg.cat_cols])
    features = feature_cols(
        [column for column in kept.columns if column not in constant_cols],
        cfg.num_cols,
        cfg.cat_cols,
    )
    print(
        f"[split] {raw.height} rows, {raw.height - kept.height} in "
        f"{len(rare_classes)} rare classes, {len(constant_cols)} constant columns",
        flush=True,
    )
    classes = sorted(kept[cfg.label_col].unique().to_list())
    codes = pl.col(cfg.label_col).replace_strict(
        {name: code for code, name in enumerate(classes)}, return_dtype=pl.Int64
    )
    coded = kept.select(*features.num, *features.cat, codes.alias(LABEL))
    shares = {"train": cfg.split.train, "val": cfg.split.val, "test": cfg.split.test}
    parts = stratified_split(coded, LABEL, shares, cfg.seed)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name in SPLITS:
        parts[name].write_parquet(split_file(cfg, name))
        counts[name] = dict(parts[name][LABEL].value_counts().iter_rows())
        print(f"[split] saved {name}: {parts[name].height} rows", flush=True)
    classes_file(cfg).write_text(json.dumps(classes))

    raw_counts = dict(raw[cfg.label_col].value_counts().iter_rows())
    db.log_table(
        conn,
        run_id,
        "split_summary",
        [
            {
                "n_raw": raw.height,
                "n_rare": raw.height - kept.height,
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
        [{"label": label, "n_rows": raw_counts[label]} for label in rare_classes],
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
        "split_class_counts",
        [
            {"label": label, **{split: counts[split].get(code, 0) for split in SPLITS}}
            for code, label in enumerate(classes)
        ],
    )
    db.log_table(
        conn,
        run_id,
        "features",
        [{"column": column, "kind": "num"} for column in features.num]
        + [{"column": column, "kind": "cat"} for column in features.cat],
    )

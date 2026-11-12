"""Stage clean: read the raw rows, drop duplicates, rare classes and constant columns."""

from __future__ import annotations

__all__ = ["run"]

import json
import sqlite3

import polars as pl

from src import db
from src.config import DatasetConfig
from src.data import (
    DUPLICATES,
    KINDS,
    LABEL,
    UNPARSABLE,
    constant_columns,
    count_kinds,
    csv_files,
    drop_duplicates,
    drop_rare_classes,
    feature_cols,
    header_rows,
    read_numbers,
    unparsable_cells,
)
from stages.paths import classes_file, data_file, unparsable_file

_HEADER = "__header__"


def _read(cfg: DatasetConfig) -> tuple[pl.DataFrame, list[dict[str, object]]]:
    """Read the configured columns of every CSV of cfg.path, in file order.

    Each file is selected on the configured columns by name, so files
    of one dataset may differ in their other columns and in their
    order; the header lines a file repeats inside are dropped. The
    numbers are cast before the rows are collected, so the text of a
    large dataset is never held in memory.

    Returns:
        The rows (as data.read_numbers returns them) and, per file, its
        name, n_read (its rows kept) and n_header (its repeated header
        lines dropped).
    """
    columns = [*cfg.num_cols, *cfg.cat_cols, cfg.label_col]
    parts, sources = [], []
    for file in csv_files(cfg.path):
        scan = pl.scan_csv(file, infer_schema=False)
        missing = sorted(set(columns) - set(scan.collect_schema().names()))
        if missing:
            raise ValueError(f"columns missing from {file}: {missing}")
        flagged = scan.select(columns).with_columns(header_rows(columns).alias(_HEADER))
        rows = read_numbers(flagged, cfg.num_cols).collect(engine="streaming")
        n_header = int(rows[_HEADER].sum())
        if n_header:
            rows = rows.filter(~pl.col(_HEADER))
        parts.append(rows.drop(_HEADER))
        sources.append(
            {"file": file.name, "n_read": parts[-1].height, "n_header": n_header}
        )
        print(
            f"[clean] {file.name}: {parts[-1].height} rows, {n_header} repeated headers",
            flush=True,
        )
    return pl.concat(parts, how="vertical", rechunk=False), sources


def run(cfg: DatasetConfig, conn: sqlite3.Connection, run_id: int) -> None:
    """Read the CSV, drop duplicates, rare classes and constant columns.

    Order: read num_cols, cat_cols and label_col of each CSV of cfg.path
    (data.csv_files: one file or a glob pattern, in sorted order) as
    text, without the header lines a file repeats (data.header_rows) ->
    data.read_numbers (num_cols as Float64; a text that is not a number
    becomes null, marked in UNPARSABLE) -> data.count_kinds ->
    data.drop_duplicates (one copy of each row equal on every column,
    the label and UNPARSABLE included, with DUPLICATES = its copies, 1
    when unique) ->
    data.drop_rare_classes(min_class_count), on the distinct rows ->
    data.constant_columns of num_cols and cat_cols, dropped -> classes
    sorted, label_col replaced by its code in LABEL ->
    data.unparsable_cells of the kept rows and columns. No row is
    dropped for a missing or non-finite value, and nothing is scaled or
    encoded: stage prepare does it.

    Writes to output_dir: data.parquet (the kept num_cols as read, then
    DUPLICATES, the weight of the row and no feature, then the kept
    cat_cols as read, then LABEL as class code; row order = the CSV's
    order of first copies among the kept rows, so a row's position is
    its id everywhere downstream),
    classes.json (the names, indexed by code), unparsable.parquet (row of
    data.parquet, column: the cells whose text was not a number).
    Logs: clean_summary (one row: n_raw, n_duplicate (copies dropped),
    n_rare, n_kept, n_classes, n_constant; n_kept = n_raw -
    n_duplicate - n_rare; then n_empty, n_unparsable, n_nan, n_inf:
    the cells of the CSV rows read, by kind), dropped_classes (label, n_rows: distinct),
    dropped_columns (column), class_counts (label, n_read, n_rows:
    rows read and kept), features (column, kind "num" | "cat": the kept
    ones), sources (file, n_read: its rows read, n_header: its repeated
    header lines dropped; one row per CSV, in reading order).

    Raises:
        FileNotFoundError: When cfg.path matches no file.
        ValueError: When a configured column is missing from a CSV, a
            row has no label, or no class survives.
    """
    print(f"[clean] reading {cfg.path}", flush=True)
    raw, sources = _read(cfg)
    n_unlabelled = raw[cfg.label_col].null_count()
    if n_unlabelled:
        raise ValueError(f"{n_unlabelled} rows of {cfg.path} have no {cfg.label_col}")
    kinds = count_kinds(raw, cfg.num_cols, cfg.cat_cols)
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
    data = kept.select(*features.num, DUPLICATES, *features.cat, codes.alias(LABEL))

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    data.write_parquet(data_file(cfg))
    unparsable_cells(kept[UNPARSABLE], [*features.num, *features.cat]).write_parquet(
        unparsable_file(cfg)
    )
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
                **{f"n_{kind}": kinds[kind] for kind in KINDS},
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
    db.log_table(conn, run_id, "sources", sources)
    db.log_table(
        conn,
        run_id,
        "features",
        [{"column": column, "kind": "num"} for column in features.num]
        + [{"column": column, "kind": "cat"} for column in features.cat],
    )

"""Stage clean: read the raw rows, drop duplicates, rare classes and constant columns."""

from __future__ import annotations

__all__ = ["run"]

import json
import sqlite3
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from src import db
from src.config import DatasetConfig
from src.data import (
    DUPLICATES,
    KEY,
    KINDS,
    LABEL,
    UNPARSABLE,
    cap_classes,
    constant_columns,
    count_kinds,
    csv_files,
    feature_cols,
    group_copies,
    header_rows,
    rare_classes,
    read_numbers,
    row_key,
    unparsable_cells,
)
from src.provenance import TIME, read_provenance
from stages.paths import classes_file, data_file, provenance_file, unparsable_file

_HEADER = "__header__"
_ORIGIN = "__provenance__"  # prefix of the provenance keys while the rows are read
_UNREAD_TIME = "__unread_time__"  # a time that time_format does not read
_POSITION = "__position__"  # the position of a row among the rows of its file


@dataclass(frozen=True)
class _Index:
    """What the first reading keeps of the rows read: never the rows.

    Attributes:
        keys: Per row read, its data.row_key, uint64, shape (n, 2).
        labels: Per row read, its class (Categorical).
        kinds: The null and invalid cells of the rows read, by kind.
        sources: Per file, its name, n_read (its rows read) and n_header
            (its repeated header lines dropped).
    """

    keys: np.ndarray
    labels: pl.Series
    kinds: dict[str, int]
    sources: list[dict[str, object]]


def _origin_exprs(cfg: DatasetConfig) -> list[pl.Expr]:
    """The provenance of a row (provenance.read_provenance, every key named
    _ORIGIN + key, and _UNREAD_TIME True on a time that time_format does not
    read); none without cfg.provenance."""
    origin = cfg.provenance.columns() if cfg.provenance else {}
    reads = read_provenance(
        origin,
        cfg.provenance and cfg.provenance.time_format,
        cfg.provenance and cfg.provenance.time_afternoon_before,
    )
    extra = [expr.alias(f"{_ORIGIN}{key}") for key, expr in reads.items()]
    if TIME in origin:
        unread = pl.col(origin[TIME]).is_not_null() & reads[TIME].is_null()
        extra.append(unread.alias(_UNREAD_TIME))
    return extra


def _scans(cfg: DatasetConfig) -> Iterator[tuple[Path, pl.LazyFrame]]:
    """Yield every CSV of cfg.path, in file order, and its rows, lazy.

    Each file is selected on the configured columns by name, so files
    of one dataset may differ in their other columns and in their
    order. The rows are as data.read_numbers returns them, then the
    provenance (_origin_exprs) and _HEADER, True on a header line the
    file repeats inside (data.header_rows).

    A provenance column missing from a file is read as null in it (its
    rows' provenance is unknown), and named on the console.

    Raises:
        ValueError: When a feature or the label is missing from a file.
    """
    columns = [*cfg.num_cols, *cfg.cat_cols, cfg.label_col]
    origin = cfg.provenance.columns() if cfg.provenance else {}
    extra = _origin_exprs(cfg)
    for file in csv_files(cfg.path):
        scan = pl.scan_csv(file, infer_schema=False)
        header = set(scan.collect_schema())
        missing = sorted(set(columns) - header)
        if missing:
            raise ValueError(f"columns missing from {file}: {missing}")
        absent = sorted(set(origin.values()) - header)
        if absent:  # a file without them: its rows' provenance is unknown
            print(f"[clean] {file.name}: provenance read as null: {absent}", flush=True)
            scan = scan.with_columns(pl.lit(None, pl.String).alias(c) for c in absent)
        flagged = scan.select(*columns, *extra, header_rows(columns).alias(_HEADER))
        yield file, read_numbers(flagged, cfg.num_cols, cfg.significant_digits)


def _compared(cfg: DatasetConfig) -> list[str]:
    """The columns two copies are equal on: the features, the label and UNPARSABLE."""
    return [*cfg.num_cols, *cfg.cat_cols, cfg.label_col, UNPARSABLE]


def _index(cfg: DatasetConfig) -> _Index:
    """Read every CSV once, in streaming, for the key and class of each row."""
    keys, labels, sources, kinds = [], [], [], Counter()
    label = pl.col(cfg.label_col).cast(pl.Categorical)
    for file, rows in _scans(cfg):
        part = rows.select(*row_key(_compared(cfg)), label, _HEADER)
        part = part.collect(engine="streaming")
        n_header = int(part[_HEADER].sum())
        part = part.filter(~pl.col(_HEADER))
        keys.append(part.select(KEY).to_numpy())
        labels.append(part[cfg.label_col])
        kinds.update(
            count_kinds(rows.filter(~pl.col(_HEADER)), cfg.num_cols, cfg.cat_cols)
        )
        sources.append({"file": file.name, "n_read": part.height, "n_header": n_header})
        print(
            f"[clean] {file.name}: {part.height} rows, {n_header} repeated headers",
            flush=True,
        )
    return _Index(
        keys=np.concatenate(keys),
        labels=pl.concat(labels),
        kinds={kind: kinds[kind] for kind in KINDS},
        sources=sources,
    )


def _read_kept(
    cfg: DatasetConfig, sizes: list[int], first: np.ndarray, row: np.ndarray
) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    """Read every CSV again, in streaming, for the rows kept only.

    Args:
        sizes: Per file, its rows read (without its header lines).
        first: The positions, among the rows read, of the rows of the
            data file (the first copy of each kept distinct row),
            ascending.
        row: Per row read, its row of the data file, -1 when it is not
            kept.

    Returns:
        The rows of the data file, in order (the columns _compared),
        and, with cfg.provenance, the provenance of every row read that
        is kept, in reading order, with its row of the data file in
        "row"; None without.
    """

    def among(positions: np.ndarray) -> pl.Expr:
        positions = pl.Series(positions, dtype=pl.Int64).implode()
        return pl.col(_POSITION).cast(pl.Int64).is_in(positions)

    data, origins, start = [], [], 0
    origin = [expr.meta.output_name() for expr in _origin_exprs(cfg)]
    for (_, rows), n in zip(_scans(cfg), sizes, strict=True):
        rows = rows.filter(~pl.col(_HEADER)).with_row_index(_POSITION)
        here = first[(first >= start) & (first < start + n)] - start
        part = rows.filter(among(here))
        data.append(part.select(_compared(cfg)).collect(engine="streaming"))
        if origin:
            of_file = row[start : start + n]
            kept = np.flatnonzero(of_file >= 0)
            part = rows.filter(among(kept))
            part = part.select(origin).collect(engine="streaming")
            origins.append(part.with_columns(pl.Series("row", of_file[kept])))
        start += n
    data = pl.concat(data, how="vertical", rechunk=False)
    return data, (pl.concat(origins, how="vertical") if origin else None)


def run(cfg: DatasetConfig, conn: sqlite3.Connection, run_id: int) -> None:
    """Read the CSV, drop duplicates, rare classes and constant columns.

    The rows are never held whole: the CSVs are read twice, in
    streaming, first for the key and class of every row, then for the
    rows kept. Order: read num_cols, cat_cols and label_col of each CSV
    of cfg.path (data.csv_files: one file or a glob pattern, in sorted
    order) as text, without the header lines a file repeats
    (data.header_rows), with the provenance columns of cfg.provenance
    when the dataset names them (provenance.read_provenance) ->
    data.read_numbers (num_cols as Float64; a text that is not a number
    becomes null, marked in UNPARSABLE) -> data.count_kinds, and
    data.row_key of every row: the key of the row equal on every
    column, the label and UNPARSABLE included but not the provenance ->
    data.group_copies (a distinct row, with DUPLICATES = its copies, 1
    when unique) -> data.rare_classes(min_class_count), on the distinct
    rows, dropped -> data.cap_classes(max_class_rows, seed) on the
    distinct rows of the other classes, the rest dropped -> read again:
    the first copy of each distinct row kept -> data.constant_columns
    of num_cols and cat_cols, dropped -> classes sorted, label_col
    replaced by its code in LABEL -> data.unparsable_cells of the kept
    rows and columns. No row is dropped for a missing or non-finite
    value, and nothing is scaled or encoded: stage prepare does it.

    Writes to output_dir: data.parquet (the kept num_cols as read, then
    DUPLICATES, the weight of the row and no feature, then the kept
    cat_cols as read, then LABEL as class code; row order = the CSV's
    order of first copies among the kept rows, so a row's position is
    its id everywhere downstream),
    classes.json (the names, indexed by code), unparsable.parquet (row of
    data.parquet, column: the cells whose text was not a number),
    provenance.parquet (with cfg.provenance, else removed: one row per
    row of the CSV that is a copy of a kept row, in reading order, so
    the copies of a row each keep their own: row, the row of
    data.parquet it is a copy of, then the configured keys of
    provenance.KEYS (Categorical, as read, null when empty) and
    provenance.TIME (Datetime, null when empty or not a time in
    time_format)).

    Logs: clean_summary (one row: n_raw, n_duplicate (copies dropped),
    n_rare (distinct rows of rare classes), n_capped (distinct rows
    past max_class_rows, 0 without), n_kept, n_classes, n_constant;
    n_kept = n_raw - n_duplicate - n_rare - n_capped; then n_empty,
    n_unparsable, n_nan, n_inf: the cells of the CSV rows read, by
    kind), dropped_classes (label, n_rows: distinct),
    dropped_columns (column), class_counts (label, n_read: the CSV
    rows read, n_distinct: the distinct rows, n_rows: those kept, under
    n_distinct only past max_class_rows), features (column, kind "num"
    | "cat": the kept ones), sources (file, n_read: its rows read,
    n_header: its repeated header lines dropped; one row per CSV, in
    reading order), provenance_columns (with cfg.provenance, one row per
    configured key: key, column, n_read (the rows of
    provenance.parquet), n_empty, n_unparsable (times that time_format
    does not read, 0 for the other keys), n_values (the distinct
    values, null for the time)).

    Raises:
        FileNotFoundError: When cfg.path matches no file.
        ValueError: When a configured column is missing from a CSV, a
            row has no label, or no class survives.
    """
    print(f"[clean] reading {cfg.path}", flush=True)
    index = _index(cfg)
    n_unlabelled = index.labels.null_count()
    if n_unlabelled:
        raise ValueError(f"{n_unlabelled} rows of {cfg.path} have no {cfg.label_col}")
    copies = group_copies(index.keys)
    labels = index.labels.gather(copies.first)
    rare = rare_classes(labels, cfg.min_class_count)
    common = ~labels.is_in(rare).to_numpy()
    kept = common.copy()
    kept[common] = cap_classes(
        labels.filter(common).to_physical().to_numpy(), cfg.max_class_rows, cfg.seed
    )
    if not kept.any():
        raise ValueError(f"no class has min_class_count={cfg.min_class_count} rows")
    row = np.where(kept, np.cumsum(kept) - 1, -1)[copies.group]
    sizes = [int(source["n_read"]) for source in index.sources]
    rows, origins = _read_kept(cfg, sizes, copies.first[kept], row)
    rows = rows.with_columns(pl.Series(DUPLICATES, copies.copies[kept]))

    n_raw, n_distinct = len(copies.group), len(copies.first)
    n_duplicate = n_raw - n_distinct
    n_rare = n_distinct - int(common.sum())
    n_capped = int(common.sum() - kept.sum())
    constant_cols = constant_columns(rows, [*cfg.num_cols, *cfg.cat_cols])
    features = feature_cols(
        [column for column in rows.columns if column not in constant_cols],
        cfg.num_cols,
        cfg.cat_cols,
    )
    print(
        f"[clean] {n_raw} rows, {n_duplicate} duplicates, {n_rare} in "
        f"{len(rare)} rare classes, {n_capped} over max_class_rows, "
        f"{len(constant_cols)} constant columns",
        flush=True,
    )
    classes = sorted(rows[cfg.label_col].unique().to_list())
    codes = pl.col(cfg.label_col).replace_strict(
        {name: code for code, name in enumerate(classes)}, return_dtype=pl.Int64
    )
    data = rows.select(*features.num, DUPLICATES, *features.cat, codes.alias(LABEL))

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    data.write_parquet(data_file(cfg))
    unparsable_cells(rows[UNPARSABLE], [*features.num, *features.cat]).write_parquet(
        unparsable_file(cfg)
    )
    classes_file(cfg).write_text(json.dumps(classes))
    provenance_rows = []
    if origins is None:
        provenance_file(cfg).unlink(missing_ok=True)
    else:
        provenance, provenance_rows = _provenance(origins, cfg.provenance.columns())
        provenance.write_parquet(provenance_file(cfg))
    print(f"[clean] saved {data.height} rows", flush=True)

    raw_counts = _counts(index.labels)
    distinct_counts = _counts(labels)
    kept_counts = _counts(rows[cfg.label_col])
    db.log_table(
        conn,
        run_id,
        "clean_summary",
        [
            {
                "n_raw": n_raw,
                "n_duplicate": n_duplicate,
                "n_rare": n_rare,
                "n_capped": n_capped,
                "n_kept": data.height,
                "n_classes": len(classes),
                "n_constant": len(constant_cols),
                **{f"n_{kind}": index.kinds[kind] for kind in KINDS},
            }
        ],
    )
    db.log_table(
        conn,
        run_id,
        "dropped_classes",
        [{"label": label, "n_rows": distinct_counts[label]} for label in rare],
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
                "n_distinct": distinct_counts[label],
                "n_rows": kept_counts[label],
            }
            for label in classes
        ],
    )
    db.log_table(conn, run_id, "sources", index.sources)
    db.log_table(
        conn,
        run_id,
        "features",
        [{"column": column, "kind": "num"} for column in features.num]
        + [{"column": column, "kind": "cat"} for column in features.cat],
    )
    db.log_table(conn, run_id, "provenance_columns", provenance_rows)


def _counts(labels: pl.Series) -> dict[str, int]:
    """Return the rows of each class, by name."""
    counts = labels.value_counts()
    return {str(label): int(n) for label, n in counts.iter_rows()}


def _provenance(
    origins: pl.DataFrame, columns: dict[str, str]
) -> tuple[pl.DataFrame, list[dict]]:
    """Return the rows of provenance.parquet and of the provenance_columns log.

    Args:
        origins: The provenance of every row read that is kept
            (_read_kept), with its row of the data file in "row".
        columns: The configured keys and their columns.
    """
    unread = int(origins[_UNREAD_TIME].sum()) if _UNREAD_TIME in origins.columns else 0
    rows = origins.select(
        "row", *(pl.col(f"{_ORIGIN}{key}").alias(key) for key in columns)
    )
    logged = []
    for key, column in columns.items():
        n_unparsable = unread if key == TIME else 0
        values = rows[key].drop_nulls()
        logged.append(
            {
                "key": key,
                "column": column,
                "n_read": rows.height,
                "n_empty": rows[key].null_count() - n_unparsable,
                "n_unparsable": n_unparsable,
                "n_values": None if key == TIME else values.n_unique(),
            }
        )
    print(
        f"[clean] provenance of {rows.height} rows: "
        + ", ".join(
            f"{r['key']} {r['n_empty']} empty"
            + (f", {r['n_unparsable']} not in time_format" if r["key"] == TIME else "")
            for r in logged
        ),
        flush=True,
    )
    return rows, logged

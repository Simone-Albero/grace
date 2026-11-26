"""Clean the raw rows: polars frames in and out.

A dataset may not fit in memory, so its rows are never held whole: a
row is first known by its key (row_key), two 64-bit hashes of its
columns, from which its copies are grouped (group_copies), the rare
classes found (rare_classes) and the large ones sampled (cap_classes);
only then are the rows kept read.
"""

from __future__ import annotations

__all__ = [
    "DUPLICATES",
    "KEY",
    "KINDS",
    "LABEL",
    "UNPARSABLE",
    "Copies",
    "Features",
    "cap_classes",
    "constant_columns",
    "count_kinds",
    "csv_files",
    "feature_cols",
    "group_copies",
    "header_rows",
    "rare_classes",
    "read_numbers",
    "round_significant",
    "row_key",
    "unparsable_cells",
]

import glob
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import numpy as np
import polars as pl

from src.graph import sample_rows

Frame = TypeVar("Frame", pl.DataFrame, pl.LazyFrame)

LABEL = "label"  # class code column of the data file
DUPLICATES = (
    "n_duplicates"  # copies of each kept row in the CSV, 1 when unique: its weight
)
UNPARSABLE = "__unparsable__"  # clean's mark of the numeric cells that were not numbers
KINDS = ("empty", "unparsable", "nan", "inf")  # the null and invalid values, by kind
KEY = ("__key_0__", "__key_1__")  # the two hashes of row_key


@dataclass(frozen=True)
class Features:
    """The input columns the data file holds.

    Attributes:
        num: The numeric columns.
        cat: The categorical columns, as text.
    """

    num: tuple[str, ...]
    cat: tuple[str, ...]


def feature_cols(
    columns: Sequence[str], num_cols: Sequence[str], cat_cols: Sequence[str]
) -> Features:
    """Return the configured columns that columns holds.

    Stage clean drops the constant columns, so the data file may lack
    some of the configured ones. DUPLICATES is never a feature: it is
    the weight of the row.

    Args:
        columns: The columns of the data file.
        num_cols: The configured numeric columns.
        cat_cols: The configured categorical columns.

    Returns:
        The present ones, in configured order.
    """
    present = set(columns)
    return Features(
        num=tuple(column for column in num_cols if column in present),
        cat=tuple(column for column in cat_cols if column in present),
    )


def csv_files(path: Path) -> list[Path]:
    """Return the CSVs a dataset is read from.

    A dataset split over several files (one per capture day, say) is
    configured as a glob pattern; its files are read in sorted order,
    so the rows keep the same order, and positions, on every run.

    Args:
        path: A file, or a glob pattern ("*", "?", "[...]") of files.

    Returns:
        [path] when path is a file, else the files the pattern matches,
        sorted.

    Raises:
        FileNotFoundError: When path is not a file and matches none.
    """
    if path.is_file():
        return [path]
    files = sorted(Path(match) for match in glob.glob(str(path)))
    files = [file for file in files if file.is_file()]
    if not files:
        raise FileNotFoundError(f"no CSV matches {path}")
    return files


def header_rows(columns: Sequence[str]) -> pl.Expr:
    """Flag the rows that repeat the header of the CSV.

    Some CSVs written in chunks (CIC-IDS2018 among them) repeat the
    header line inside the file; read as text, it is a row whose every
    cell is its column's name. A row that is so on some columns only is
    data.

    Args:
        columns: The columns read, at least one.

    Returns:
        A Boolean expression, True on a header row, never null.
    """
    return pl.all_horizontal(pl.col(c) == c for c in columns).fill_null(False)


def read_numbers(
    rows: Frame, num_cols: Sequence[str], significant_digits: int | None = None
) -> Frame:
    """Turn the numeric columns from text into numbers, marking the bad text.

    A cell that is null stays null ("empty"); a cell whose text is not
    a number becomes null too, and its column is named in UNPARSABLE,
    so the two stay apart (also for drop_duplicates, which compares
    UNPARSABLE as any column). "nan" and "inf" texts are numbers. With
    significant_digits, every number is rounded to them
    (round_significant), before any copy is found.

    Args:
        rows: The rows as read, every column text; lazy rows stay lazy,
            so a large CSV is never held as text.
        num_cols: The numeric columns.
        significant_digits: The significant digits every number keeps,
            None for all.

    Returns:
        rows with num_cols as Float64 and UNPARSABLE appended: the
        names of the row's unparsable columns in num_cols order, joined
        by ",", null when none.
    """
    bad = [
        pl.when(
            pl.col(c).is_not_null() & pl.col(c).cast(pl.Float64, strict=False).is_null()
        ).then(pl.lit(c))
        for c in num_cols
    ]
    marks = (
        pl.concat_str(bad, separator=",", ignore_nulls=True)
        if bad
        else pl.lit(None, pl.String)
    )
    numbers = pl.col(num_cols).cast(pl.Float64, strict=False)
    if significant_digits is not None:
        numbers = round_significant(numbers, significant_digits)
    return rows.with_columns(
        numbers,
        pl.when(marks != "").then(marks).alias(UNPARSABLE),
    )


def round_significant(expr: pl.Expr, digits: int) -> pl.Expr:
    """Round numbers to significant digits, the same way whatever digits they were written with.

    The same expression for every value, so that a number written with
    10 significant digits and the same number written with 15 give the
    same float. 0, NaN and infinities are kept, a null stays null.

    Args:
        expr: The Float64 numbers.
        digits: The significant digits kept, >= 1.

    Returns:
        The rounded numbers, Float64.
    """
    return expr.round_sig_figs(digits)


def unparsable_cells(marks: pl.Series, columns: Sequence[str]) -> pl.DataFrame:
    """List the cells read_numbers marked, row by row.

    Args:
        marks: UNPARSABLE of the rows, in their order.
        columns: The columns kept: a marked column not among them is
            left out.

    Returns:
        Columns row (Int64, the position in marks) and column (String),
        sorted by row then columns order.
    """
    kept = {column: i for i, column in enumerate(columns)}
    cells = [
        (row, column)
        for row, mark in enumerate(marks.to_list())
        if mark
        for column in mark.split(",")
        if column in kept
    ]
    cells.sort(key=lambda cell: (cell[0], kept[cell[1]]))
    return pl.DataFrame(
        {"row": [r for r, _ in cells], "column": [c for _, c in cells]},
        schema={"row": pl.Int64, "column": pl.String},
    )


def count_kinds(
    rows: Frame, num_cols: Sequence[str], cat_cols: Sequence[str]
) -> dict[str, int]:
    """Count the null and invalid cells of the rows, by kind.

    Args:
        rows: The output of read_numbers; lazy rows are counted in
            streaming, so they are never held whole.
        num_cols: The numeric columns.
        cat_cols: The categorical columns.

    Returns:
        By kind of KINDS, the cells: "empty" (null, in any column, but
        unparsable), "unparsable", "nan", "inf" (either sign); 0 for
        no row.
    """

    def total(exprs: list[pl.Expr]) -> pl.Expr:
        return pl.sum_horizontal(exprs) if exprs else pl.lit(0, pl.Int64)

    nums, every = list(num_cols), [*num_cols, *cat_cols]
    marks = pl.col(UNPARSABLE).str.count_matches(",").cast(pl.Int64) + 1
    counts = (
        rows.lazy()
        .select(
            marks.sum().alias("unparsable"),
            total([pl.col(c).null_count().cast(pl.Int64) for c in every]).alias(
                "nulls"
            ),
            total([pl.col(c).is_nan().sum().cast(pl.Int64) for c in nums]).alias("nan"),
            total([pl.col(c).is_infinite().sum().cast(pl.Int64) for c in nums]).alias(
                "inf"
            ),
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )
    unparsable = int(counts["unparsable"] or 0)
    return {
        "empty": int(counts["nulls"] or 0) - unparsable,
        "unparsable": unparsable,
        "nan": int(counts["nan"] or 0),
        "inf": int(counts["inf"] or 0),
    }


def row_key(columns: Sequence[str]) -> list[pl.Expr]:
    """Return the key of a row: two 64-bit hashes of its columns.

    Two rows are copies when every column is equal; the key stands for
    the row, so copies have one key, and two rows that differ share one
    with a chance of about n^2 / 2^129 over n rows (1e-23 at 6e7 rows).
    Equal as polars groups rows: null equals null, NaN equals NaN, -0.0
    equals 0.0, and a null differs from NaN and from the text "".

    Args:
        columns: The columns compared, at least one (of the output of
            read_numbers, so an unparsable cell differs from an empty one).

    Returns:
        Two UInt64 expressions, named KEY.
    """
    row = pl.struct(list(columns))
    return [row.hash(seed).alias(name) for seed, name in enumerate(KEY)]


@dataclass(frozen=True)
class Copies:
    """The copies of the rows read, grouped by key.

    A group is a distinct row; groups are numbered in the order of
    their first copy among the rows read.

    Attributes:
        group: Per row read, its group, int64, shape (n,).
        first: Per group, the position of its first copy among the
            rows read, int64, ascending.
        copies: Per group, its copies, int64, >= 1.
    """

    group: np.ndarray
    first: np.ndarray
    copies: np.ndarray


def group_copies(keys: np.ndarray) -> Copies:
    """Group the rows read that are copies of each other.

    Args:
        keys: The key of every row read, in reading order, uint64,
            shape (n, 2) (the two columns of row_key).

    Returns:
        The groups; none when n is 0.
    """
    keys = np.asarray(keys, dtype=np.uint64)
    n = len(keys)
    if n == 0:
        empty = np.empty(0, dtype=np.int64)
        return Copies(group=empty, first=empty, copies=empty)
    # Stable: the first row of each run of equal keys is its first copy.
    order = np.lexsort((keys[:, 1], keys[:, 0]))
    ordered = keys[order]
    starts = np.ones(n, dtype=bool)
    starts[1:] = (ordered[1:] != ordered[:-1]).any(axis=1)
    del ordered
    run = np.cumsum(starts, dtype=np.int64) - 1  # per sorted row, its run
    first = order[starts].astype(np.int64)
    copies = np.diff(np.r_[np.flatnonzero(starts), n]).astype(np.int64)
    by_first = np.argsort(first, kind="stable")
    number = np.empty(len(first), dtype=np.int64)
    number[by_first] = np.arange(len(first), dtype=np.int64)
    group = np.empty(n, dtype=np.int64)
    group[order] = number[run]
    return Copies(group=group, first=first[by_first], copies=copies[by_first])


def rare_classes(labels: pl.Series, min_count: int) -> list[str]:
    """Return the classes with fewer than min_count rows.

    Args:
        labels: The class of every distinct row.
        min_count: The fewest rows a class keeps (a class with exactly
            min_count rows stays).

    Returns:
        The rare classes, sorted.
    """
    counts = labels.value_counts()
    rare = counts.filter(pl.col("count") < min_count)[:, 0]
    return sorted(str(label) for label in rare.to_list())


def cap_classes(labels: np.ndarray, cap: int | None, seed: int) -> np.ndarray:
    """Sample every class down to cap rows.

    Each class over cap keeps a uniform sample of cap of its rows
    (graph.sample_rows with seed); the others keep every row.

    Args:
        labels: The class of every distinct row, shape (n,).
        cap: The most rows a class keeps; None keeps every row.
        seed: The seed of the draw, the same for every class.

    Returns:
        Per row, whether it is kept, bool, shape (n,).
    """
    labels = np.asarray(labels)
    kept = np.ones(len(labels), dtype=bool)
    if cap is None:
        return kept
    for label in np.unique(labels):
        rows = np.flatnonzero(labels == label)
        if len(rows) > cap:
            kept[rows] = False
            kept[rows[sample_rows(len(rows), cap, seed)]] = True
    return kept


def constant_columns(rows: pl.DataFrame, columns: Sequence[str]) -> list[str]:
    """Return the columns that hold a single distinct value.

    Args:
        rows: The rows.
        columns: The columns to check.

    Returns:
        The constant ones, in the order of columns; null counts as a
        value, so a column of nulls is constant and one mixing nulls
        and a value is not.
    """
    return [column for column in columns if rows[column].n_unique() == 1]

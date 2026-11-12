"""Clean the raw rows: polars frames in and out."""

from __future__ import annotations

__all__ = [
    "DUPLICATES",
    "KINDS",
    "LABEL",
    "UNPARSABLE",
    "Features",
    "constant_columns",
    "count_kinds",
    "drop_duplicates",
    "drop_rare_classes",
    "feature_cols",
    "read_numbers",
    "unparsable_cells",
]

from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

LABEL = "label"  # class code column of the data file
DUPLICATES = (
    "n_duplicates"  # copies of each kept row in the CSV, 1 when unique: its weight
)
UNPARSABLE = "__unparsable__"  # clean's mark of the numeric cells that were not numbers
KINDS = ("empty", "unparsable", "nan", "inf")  # the null and invalid values, by kind


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


def read_numbers(rows: pl.DataFrame, num_cols: Sequence[str]) -> pl.DataFrame:
    """Turn the numeric columns from text into numbers, marking the bad text.

    A cell that is null stays null ("empty"); a cell whose text is not
    a number becomes null too, and its column is named in UNPARSABLE,
    so the two stay apart (also for drop_duplicates, which compares
    UNPARSABLE as any column). "nan" and "inf" texts are numbers.

    Args:
        rows: The rows as read, every column text.
        num_cols: The numeric columns.

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
    return rows.with_columns(
        pl.col(num_cols).cast(pl.Float64, strict=False),
        pl.when(marks != "").then(marks).alias(UNPARSABLE),
    )


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
    rows: pl.DataFrame, num_cols: Sequence[str], cat_cols: Sequence[str]
) -> dict[str, int]:
    """Count the null and invalid cells of the rows, by kind.

    Args:
        rows: The output of read_numbers.
        num_cols: The numeric columns.
        cat_cols: The categorical columns.

    Returns:
        By kind of KINDS, the cells: "empty" (null, in any column, but
        unparsable), "unparsable", "nan", "inf" (either sign).
    """
    marks = rows[UNPARSABLE].drop_nulls().to_list()
    unparsable = sum(len(mark.split(",")) for mark in marks)
    nulls = sum(rows[c].null_count() for c in [*num_cols, *cat_cols])
    numbers = rows.select(num_cols)
    return {
        "empty": nulls - unparsable,
        "unparsable": unparsable,
        "nan": int(sum(numbers[c].is_nan().sum() for c in num_cols)),
        "inf": int(sum(numbers[c].is_infinite().sum() for c in num_cols)),
    }


def drop_duplicates(rows: pl.DataFrame) -> pl.DataFrame:
    """Keep one copy of each distinct row and count its copies.

    Two rows are copies when every column is equal, the label included,
    so equal features under different labels stay apart; null equals
    null and NaN equals NaN.

    Args:
        rows: The rows, without a DUPLICATES column.

    Returns:
        The first copy of each distinct row, in order of first
        appearance, with DUPLICATES appended: its number of copies
        (Int64), 1 for a row without duplicates.
    """
    count = pl.len().cast(pl.Int64).alias(DUPLICATES)
    return rows.group_by(rows.columns, maintain_order=True).agg(count)


def drop_rare_classes(
    rows: pl.DataFrame, label_col: str, min_count: int
) -> tuple[pl.DataFrame, list[str]]:
    """Drop every class with fewer than min_count rows.

    Args:
        rows: The rows.
        label_col: The class column.
        min_count: The fewest rows a class keeps (a class with exactly
            min_count rows stays).

    Returns:
        The rows of the kept classes, in their original order, and the
        dropped class names, sorted.
    """
    counts = rows[label_col].value_counts()
    rare = counts.filter(pl.col("count") < min_count)[label_col].to_list()
    return rows.filter(~pl.col(label_col).is_in(rare)), sorted(rare)


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

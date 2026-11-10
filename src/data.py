"""Clean the raw rows: polars frames in and out."""

from __future__ import annotations

__all__ = [
    "DUPLICATES",
    "LABEL",
    "Features",
    "constant_columns",
    "drop_duplicates",
    "drop_rare_classes",
    "feature_cols",
]

from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

LABEL = "label"  # class code column of the data file
DUPLICATES = "n_duplicates"  # copies of each kept row in the CSV, 1 when unique


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
    """Return the configured columns that columns holds, and DUPLICATES.

    Stage clean drops the constant columns, so the data file may lack
    some of the configured ones; it adds DUPLICATES, a numeric feature.

    Args:
        columns: The columns of the data file.
        num_cols: The configured numeric columns.
        cat_cols: The configured categorical columns.

    Returns:
        The present ones, in configured order; DUPLICATES last among
        the numeric ones when columns holds it.
    """
    present = set(columns)
    num = [column for column in num_cols if column in present]
    if DUPLICATES in present:
        num.append(DUPLICATES)
    return Features(
        num=tuple(num),
        cat=tuple(column for column in cat_cols if column in present),
    )


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

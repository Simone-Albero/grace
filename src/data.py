"""Clean, split and cap the raw rows: polars frames in and out."""

from __future__ import annotations

__all__ = [
    "LABEL",
    "SPLITS",
    "cap_per_class",
    "constant_columns",
    "drop_rare_classes",
    "stratified_split",
]

from collections.abc import Mapping, Sequence

import polars as pl

SPLITS = ("train", "val", "test")
LABEL = "label"  # class code column of every split file
_POSITION = "_position"


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


def _hash_rank(seed: int, label_col: str) -> pl.Expr:
    """Rank each row in its class by a seeded hash of its position."""
    return pl.col(_POSITION).hash(seed).rank("ordinal").over(label_col)


def stratified_split(
    rows: pl.DataFrame, label_col: str, shares: Mapping[str, float], seed: int
) -> dict[str, pl.DataFrame]:
    """Split the rows into train, val and test, stratified by class.

    Within each class the rows are ranked by a hash of their position
    seeded by seed; the first ceil(train · n) ranks go to train, the
    next up to ceil((train + val) · n) to val, the rest to test. So
    every split holds each class in proportion, and the result depends
    on the seed only.

    Args:
        rows: The rows.
        label_col: The class column.
        shares: The share of each of SPLITS; they sum to 1.
        seed: The hash seed.

    Returns:
        One frame per name of SPLITS, disjoint, together every row of
        rows, each in the original row order.
    """
    share = (_hash_rank(seed, label_col) - 1) / pl.len().over(label_col)
    train, val = shares["train"], shares["train"] + shares["val"]
    split = (
        pl.when(share < train)
        .then(pl.lit("train"))
        .when(share < val)
        .then(pl.lit("val"))
        .otherwise(pl.lit("test"))
    )
    assigned = rows.with_row_index(_POSITION).select(split.alias("split"))["split"]
    return {name: rows.filter(assigned == name) for name in SPLITS}


def cap_per_class(
    rows: pl.DataFrame, label_col: str, cap: int, seed: int
) -> pl.DataFrame:
    """Keep at most cap rows per class, drawn without replacement.

    The draw ranks the rows of each class by a hash of their position
    seeded by seed, so it depends on the seed only.

    Args:
        rows: The rows.
        label_col: The class column.
        cap: The most rows a class keeps, > 0; a smaller class is kept
            whole.
        seed: The hash seed.

    Returns:
        The kept rows, in their original order.
    """
    kept = rows.with_row_index(_POSITION).filter(_hash_rank(seed, label_col) <= cap)
    return kept.drop(_POSITION)

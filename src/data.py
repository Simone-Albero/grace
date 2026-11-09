"""Clean and split the raw rows: polars frames in, polars frames out."""

from __future__ import annotations

__all__ = ["SPLITS", "drop_incomplete", "drop_rare_classes", "stratified_split"]

from collections.abc import Mapping, Sequence

import polars as pl

SPLITS = ("train", "val", "test")


def drop_incomplete(rows: pl.DataFrame, num_cols: Sequence[str]) -> pl.DataFrame:
    """Drop the rows with a null or a non-finite number.

    Args:
        rows: The raw rows; num_cols are Float64, every other column
            text.
        num_cols: The numeric columns, where NaN and ±inf count as
            missing.

    Returns:
        The complete rows, in their original order.
    """
    finite = [pl.col(c).is_finite() for c in num_cols]
    return rows.drop_nulls().filter(*finite) if finite else rows.drop_nulls()


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
    rank = pl.col("_position").hash(seed).rank("ordinal").over(label_col)
    share = (rank - 1) / pl.len().over(label_col)
    train, val = shares["train"], shares["train"] + shares["val"]
    split = (
        pl.when(share < train)
        .then(pl.lit("train"))
        .when(share < val)
        .then(pl.lit("val"))
        .otherwise(pl.lit("test"))
    )
    assigned = rows.with_row_index("_position").select(split.alias("split"))["split"]
    return {name: rows.filter(assigned == name) for name in SPLITS}

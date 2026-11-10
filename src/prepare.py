"""The features every distance of the method is measured on.

Fitted on the cleaned rows by stage prepare. A number becomes
sign(x) · log1p(|x|), then is standardized (scikit-learn
StandardScaler: minus its mean, divided by its standard deviation); a
missing or non-finite number lands on the mean, 0, and MISSING counts
how many numbers of the row did, unscaled. A categorical column
becomes an integer code (TopNEncoder: its top_n most frequent values
their own code, every other one a shared "other" code), then a one-hot
block of one slot per code. The active slot of column c is worth
w_c = sqrt(cat_cost / g_c), with g_c = 1 - Σ p_s² over the shares p_s
of its slots in the fitted rows (the chance two random rows differ on
it): a mismatch costs 2 · cat_cost / g_c in squared distance, so on
average, g_c times that, the column adds 2 · cat_cost, as much as one
standardized number adds at cat_cost = 1. A rare slot is far from the
others, as a rare value of a standardized number is. Nothing here
depends on the metric: graph.space_points applies it to the numbers
only, and leaves MISSING out of every distance.
"""

from __future__ import annotations

__all__ = [
    "MISSING",
    "NULL",
    "OTHER",
    "Preparation",
    "TopNEncoder",
    "column_names",
    "fit_preparation",
    "signed_log1p",
    "transform",
]

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler

OTHER = "<other>"  # slot of the values outside a column's top_n
NULL = "<null>"  # name of the null value when it is among the top_n
MISSING = "n_missing"  # missing or non-finite numbers of a row, set to 0


def signed_log1p(x: np.ndarray) -> np.ndarray:
    """Return sign(x) · log1p(|x|), elementwise.

    Args:
        x: Any float array.

    Returns:
        An array of the same shape.
    """
    return np.sign(x) * np.log1p(np.abs(x))


class TopNEncoder:
    """Integer codes of one categorical column, fitted on the rows.

    Codes 1..top_n are the top_n most frequent values, the most frequent
    first (null counts as a value; ties: the smaller value first, null
    last); code 0 is every other value, unseen ones included ("other").

    Attributes:
        column: The column name.
        top_n: The most frequent values with their own code, >= 1.
        values_: The value of each code from 1, in code order; set by
            fit.
        has_other_: Whether the fitted column held a value outside
            values_; set by fit.
    """

    def __init__(self, column: str, top_n: int) -> None:
        self.column = column
        self.top_n = top_n

    def fit(self, values: pl.Series) -> TopNEncoder:
        """Learn the top_n most frequent values.

        Args:
            values: The column, as text.

        Returns:
            self.
        """
        counts = values.value_counts().sort(
            ["count", values.name], descending=[True, False], nulls_last=True
        )
        self.values_ = tuple(counts[values.name].head(self.top_n).to_list())
        self.has_other_ = counts.height > self.top_n
        return self

    def transform(self, values: pl.Series) -> np.ndarray:
        """Encode a column.

        Args:
            values: The column, as text.

        Returns:
            The int32 codes, one per value.
        """
        codes = {value: code for code, value in enumerate(self.values_, start=1)}
        null_code = codes.pop(None, 0)
        coded = values.replace_strict(codes, default=0, return_dtype=pl.Int32)
        # replace_strict sends null to the default; null has its own code.
        coded = coded.to_numpy().astype(np.int32)
        coded[values.is_null().to_numpy()] = null_code
        return coded

    def slot_names(self) -> list[str]:
        """Return the names of the one-hot slots, in slot order.

        Returns:
            f"{column}={value}" per code from 1 (NULL for null), then
            f"{column}={OTHER}" when has_other_.
        """
        names = [NULL if value is None else value for value in self.values_]
        if self.has_other_:
            names.append(OTHER)
        return [f"{self.column}={name}" for name in names]


@dataclass(frozen=True)
class Preparation:
    """Everything fitted on the rows that prepares a row.

    Attributes:
        num_cols: The numeric columns, in column order.
        scaler: The StandardScaler of the signed log of num_cols,
            fitted on the finite values.
        encoders: The TopNEncoder of each categorical column, in column
            order after num_cols.
        weights: The value of an active slot of each encoder,
            sqrt(cat_cost / g_c); 0 for a column whose fitted rows all
            share one slot (g_c = 0).
    """

    num_cols: tuple[str, ...]
    scaler: StandardScaler
    encoders: tuple[TopNEncoder, ...]
    weights: tuple[float, ...]


def fit_preparation(
    rows: pl.DataFrame,
    num_cols: Sequence[str],
    cat_cols: Sequence[str],
    *,
    top_n: int,
    cat_cost: float,
) -> Preparation:
    """Fit the preparation on the rows.

    Args:
        rows: The rows of the data file.
        num_cols: The numeric columns, at least one.
        cat_cols: The categorical columns, as text; may be empty.
        top_n: The values of a categorical column with their own code.
        cat_cost: The average squared distance a categorical column adds
            between two random rows, in units of a standardized
            number's (2), > 0.

    Returns:
        The fitted preparation.
    """
    scaler = StandardScaler().fit(_signed_log(rows, num_cols))
    encoders = tuple(TopNEncoder(name, top_n).fit(rows[name]) for name in cat_cols)
    weights = []
    for encoder in encoders:
        slots = _slots(encoder, encoder.transform(rows[encoder.column]))
        shares = np.bincount(slots, minlength=_n_slots(encoder)) / max(len(slots), 1)
        differ = 1.0 - float(np.sum(shares**2))
        weights.append(math.sqrt(cat_cost / differ) if differ > 1e-12 else 0.0)
    return Preparation(
        num_cols=tuple(num_cols),
        scaler=scaler,
        encoders=encoders,
        weights=tuple(weights),
    )


def column_names(preparation: Preparation) -> list[str]:
    """Return the names of the prepared columns.

    Args:
        preparation: The fitted preparation.

    Returns:
        num_cols, then MISSING, then the slot_names of each encoder.
    """
    names = [*preparation.num_cols, MISSING]
    for encoder in preparation.encoders:
        names += encoder.slot_names()
    return names


def transform(preparation: Preparation, rows: pl.DataFrame) -> pl.DataFrame:
    """Prepare rows.

    Args:
        preparation: The fitted preparation.
        rows: Rows of the data file.

    Returns:
        Float64 columns named by column_names, in row order: the
        standardized numbers (0 when missing or non-finite), then
        MISSING (how many of them were missing or non-finite, 0 when
        none; present even when no row has one), then per
        categorical column its slots, one of them set to the column's
        preparation.weights and the others 0 (all 0 for a value outside
        the top_n of a column fitted without an "other" slot, which
        only rows the preparation was not fitted on can hold).
    """
    logged = _signed_log(rows, preparation.num_cols)
    missing = np.isnan(logged).sum(axis=1, keepdims=True).astype(np.float64)
    numbers = preparation.scaler.transform(logged)
    blocks = [np.nan_to_num(numbers, nan=0.0), missing]
    for encoder, weight in zip(preparation.encoders, preparation.weights):
        n_slots = _n_slots(encoder)
        slot = _slots(encoder, encoder.transform(rows[encoder.column]))
        block = np.zeros((rows.height, n_slots))
        filled = slot < n_slots
        block[np.flatnonzero(filled), slot[filled]] = weight
        blocks.append(block)
    return pl.DataFrame(
        np.hstack(blocks), schema=column_names(preparation), orient="row"
    )


def _n_slots(encoder: TopNEncoder) -> int:
    """Return the one-hot slots of a fitted encoder."""
    return len(encoder.values_) + encoder.has_other_


def _slots(encoder: TopNEncoder, codes: np.ndarray) -> np.ndarray:
    """Return the slot of every code; _n_slots when it has none.

    Code c >= 1 is slot c - 1; code 0 is the "other" slot, last, which
    only an encoder with has_other_ holds.
    """
    other = len(encoder.values_) if encoder.has_other_ else _n_slots(encoder)
    return np.where(codes > 0, codes - 1, other).astype(np.int64)


def _signed_log(rows: pl.DataFrame, num_cols: Sequence[str]) -> np.ndarray:
    """Return signed_log1p of num_cols; NaN if missing or non-finite."""
    numbers = rows.select(num_cols).to_numpy().astype(np.float64)
    numbers = numbers.reshape(rows.height, len(num_cols))
    numbers[~np.isfinite(numbers)] = np.nan
    return signed_log1p(numbers)

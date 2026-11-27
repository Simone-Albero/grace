"""The features every distance of the method is measured on.

Fitted on the cleaned rows by stage prepare, block by block (the rows
are never held whole): the statistics of the fit are sums over the
rows. A number becomes
sign(x) · log1p(|x|), then is standardized (scikit-learn
StandardScaler: minus its mean, divided by its standard deviation); a
missing or non-finite number lands on the mean, 0, and MISSING counts
how many numbers of the row did, unscaled. A categorical column
becomes an integer code (TopNEncoder: its top_n most frequent values
their own code, every other one a shared "other" code), then a one-hot
block of one slot per code. The active slot of column c is worth
w_c = sqrt(1 / g_c), with g_c = 1 - Σ p_s² over the shares p_s of its
slots in the fitted rows (the chance two random rows differ on it): a
mismatch costs 2 / g_c in squared distance, so on average, g_c times
that, the column adds 2, as much as one standardized number adds. A
rare slot is far from the others, as a rare value of a standardized
number is. MISSING is a diagnostic, logged by stage prepare: no
distance sees it. The points are the standardized numbers then the
slots, in single precision (float32).
"""

from __future__ import annotations

__all__ = [
    "MISSING",
    "NULL",
    "OTHER",
    "Preparation",
    "TopNEncoder",
    "column_names",
    "column_sources",
    "fit_preparation",
    "signed_log1p",
    "transform",
]

import math
from collections.abc import Iterable, Sequence
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
        slot_counts_: The fitted rows of each one-hot slot, in slot
            order (slot_names); set by fit.
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
        return self.fit_counts(values.value_counts())

    def fit_counts(self, counts: pl.DataFrame) -> TopNEncoder:
        """Learn the top_n most frequent values from their counts.

        Args:
            counts: One row per distinct value of the column: the value
                (null included), in the column named self.column, and
                its rows, in "count" (value_counts, summed over blocks).

        Returns:
            self.
        """
        counts = counts.sort(
            ["count", self.column], descending=[True, False], nulls_last=True
        )
        self.values_ = tuple(counts[self.column].head(self.top_n).to_list())
        self.has_other_ = counts.height > self.top_n
        rest = int(counts["count"].slice(self.top_n).sum() or 0)
        top = [int(n) for n in counts["count"].head(self.top_n).to_list()]
        self.slot_counts_ = tuple(top + [rest] * self.has_other_)
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
            sqrt(1 / g_c); 0 for a column whose fitted rows all
            share one slot (g_c = 0).
    """

    num_cols: tuple[str, ...]
    scaler: StandardScaler
    encoders: tuple[TopNEncoder, ...]
    weights: tuple[float, ...]


def fit_preparation(
    blocks: Iterable[pl.DataFrame],
    num_cols: Sequence[str],
    cat_cols: Sequence[str],
    *,
    top_n: int,
) -> Preparation:
    """Fit the preparation on the rows, read block by block.

    The scaler is fitted with StandardScaler.partial_fit on each block
    (the same mean and variance as one fit, up to rounding); the counts
    of each categorical value are summed over the blocks.

    Args:
        blocks: The rows of the data file, in blocks, each with num_cols
            and cat_cols.
        num_cols: The numeric columns, at least one.
        cat_cols: The categorical columns, as text; may be empty.
        top_n: The values of a categorical column with their own code.

    Returns:
        The fitted preparation.
    """
    scaler = StandardScaler()
    counts: dict[str, list[pl.DataFrame]] = {name: [] for name in cat_cols}
    for block in blocks:
        scaler.partial_fit(_signed_log(block, num_cols))
        for name in cat_cols:
            counts[name].append(block[name].value_counts())
    encoders = []
    for name in cat_cols:
        summed = pl.concat(counts[name]).group_by(name).agg(pl.col("count").sum())
        encoders.append(TopNEncoder(name, top_n).fit_counts(summed))
    weights = []
    for encoder in encoders:
        shares = np.asarray(encoder.slot_counts_, dtype=np.float64)
        shares /= max(shares.sum(), 1.0)
        differ = 1.0 - float(np.sum(shares**2))
        weights.append(math.sqrt(1.0 / differ) if differ > 1e-12 else 0.0)
    return Preparation(
        num_cols=tuple(num_cols),
        scaler=scaler,
        encoders=tuple(encoders),
        weights=tuple(weights),
    )


def column_names(preparation: Preparation) -> list[str]:
    """Return the names of the columns of the points.

    Args:
        preparation: The fitted preparation.

    Returns:
        num_cols, then the slot_names of each encoder.
    """
    names = list(preparation.num_cols)
    for encoder in preparation.encoders:
        names += encoder.slot_names()
    return names


def column_sources(preparation: Preparation) -> list[str]:
    """Return the configured column behind every column of the points.

    Args:
        preparation: The fitted preparation.

    Returns:
        One name per column of column_names: a numeric column's own,
        the categorical column of a one-hot slot.
    """
    sources = list(preparation.num_cols)
    for encoder in preparation.encoders:
        sources += [encoder.column] * _n_slots(encoder)
    return sources


def transform(
    preparation: Preparation, rows: pl.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare rows.

    Args:
        preparation: The fitted preparation.
        rows: Rows of the data file.

    Returns:
        The points, float32, shape (n, columns of column_names), in row
        order: the standardized numbers (0 when missing or
        non-finite), then per categorical column its slots, one of them
        set to the column's preparation.weights and the others 0 (all 0
        for a value outside the top_n of a column fitted without an
        "other" slot, which only rows the preparation was not fitted on
        can hold); and MISSING of every row, shape (n,) int64: how many
        of its numbers were missing or non-finite.
    """
    logged = _signed_log(rows, preparation.num_cols)
    missing = np.isnan(logged).sum(axis=1).astype(np.int64)
    numbers = np.nan_to_num(preparation.scaler.transform(logged), nan=0.0)
    blocks = [numbers.astype(np.float32)]
    for encoder, weight in zip(preparation.encoders, preparation.weights):
        n_slots = _n_slots(encoder)
        slot = _slots(encoder, encoder.transform(rows[encoder.column]))
        block = np.zeros((rows.height, n_slots), dtype=np.float32)
        filled = slot < n_slots
        block[np.flatnonzero(filled), slot[filled]] = weight
        blocks.append(block)
    return np.hstack(blocks), missing


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

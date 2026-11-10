"""The euclidean space every distance of the method is measured in.

Fitted on the raw columns of the data by stage regions. A number
becomes sign(x) · log1p(|x|), minus its median, divided by its
interquartile range (scikit-learn RobustScaler); a missing or
non-finite number lands on the median, 0. A categorical column becomes
an integer code (TopNHashEncoder: frequent values their own code, the
rest hashed), then a one-hot block over its top_k most frequent codes
and one shared "other" slot. A slot is worth sqrt(cat_cost /
2), so two rows that differ on one categorical column are cat_cost
apart in squared distance. Under the cosine distance every row is then
scaled to unit length.
"""

from __future__ import annotations

__all__ = [
    "HASH_SALT",
    "Space",
    "TopNHashEncoder",
    "embed",
    "fit_space",
    "hash_bucket",
    "scale_for_metric",
    "signed_log1p",
]

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.preprocessing import RobustScaler

HASH_SALT = "cat-encoder-v1"


def signed_log1p(x: np.ndarray) -> np.ndarray:
    """Return sign(x) · log1p(|x|), elementwise.

    Args:
        x: Any float array.

    Returns:
        An array of the same shape.
    """
    return np.sign(x) * np.log1p(np.abs(x))


def hash_bucket(column: str, value: str, buckets: int) -> int:
    """Return the hash bucket of a categorical value.

    The bucket is the little-endian integer of the 8-byte blake2b digest
    of f"{HASH_SALT}|{column}|{value}", modulo buckets.

    Args:
        column: The column name, so equal values of two columns differ.
        value: The value, as text.
        buckets: The number of buckets, > 0.

    Returns:
        An integer in [0, buckets).
    """
    key = f"{HASH_SALT}|{column}|{value}".encode()
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, "little") % buckets


class TopNHashEncoder:
    """Integer codes of one categorical column, fitted on the data.

    Code 0 is null; codes 1..top_n are the top_n most frequent values of
    the data, the most frequent first (ties: the smaller value first);
    every other value, unseen ones included, is 1 + top_n +
    hash_bucket(...) when hash_buckets > 0, else 0. So the codes lie in
    [0, 1 + top_n + hash_buckets).

    Attributes:
        column: The column name, part of the hash key.
        top_n: The most frequent values with their own code.
        hash_buckets: The codes the other values share.
        codes_: The code of each frequent value, set by fit.
    """

    def __init__(self, column: str, top_n: int, hash_buckets: int) -> None:
        self.column = column
        self.top_n = top_n
        self.hash_buckets = hash_buckets

    def fit(self, values: pl.Series) -> TopNHashEncoder:
        """Learn the top_n most frequent non-null values.

        Args:
            values: The column, as text.

        Returns:
            self.
        """
        counts = values.drop_nulls().value_counts()
        frequent = counts.sort(["count", values.name], descending=[True, False])
        top = frequent[values.name].head(self.top_n).to_list()
        self.codes_ = {value: code for code, value in enumerate(top, start=1)}
        return self

    def transform(self, values: pl.Series) -> np.ndarray:
        """Encode a column.

        Args:
            values: The column, as text.

        Returns:
            The int32 codes, one per value.
        """
        first_hashed = 1 + self.top_n

        def code(value: str) -> int:
            if value in self.codes_:
                return self.codes_[value]
            if self.hash_buckets == 0:
                return 0
            return first_hashed + hash_bucket(self.column, value, self.hash_buckets)

        seen = values.drop_nulls().unique().to_list()
        codes = values.replace_strict(
            seen, [code(value) for value in seen], return_dtype=pl.Int32
        )
        # With nothing to replace, polars keeps the input's text dtype.
        return codes.cast(pl.Int32).fill_null(0).to_numpy()


@dataclass(frozen=True)
class Space:
    """Everything fitted on the data that places a row in the space.

    Attributes:
        num_cols: The numeric columns, in point order.
        scaler: The RobustScaler of the signed log of num_cols.
        encoders: The TopNHashEncoder of each categorical column, in
            point order after num_cols.
        kept: The kept codes of each categorical column, ascending.
        weight: The value of an active slot, sqrt(cat_cost / 2).
        metric: "euclidean" or "cosine".
    """

    num_cols: tuple[str, ...]
    scaler: RobustScaler
    encoders: tuple[TopNHashEncoder, ...]
    kept: tuple[np.ndarray, ...]
    weight: float
    metric: str


def fit_space(
    rows: pl.DataFrame,
    num_cols: Sequence[str],
    cat_cols: Sequence[str],
    *,
    top_n: int,
    hash_buckets: int,
    top_k: int,
    cat_cost: float,
    metric: str,
) -> Space:
    """Fit the space on the rows.

    Args:
        rows: The rows of the data file.
        num_cols: The numeric columns, at least one.
        cat_cols: The categorical columns, as text; may be empty.
        top_n: The values of a categorical column with their own code.
        hash_buckets: The codes the other values are hashed into.
        top_k: The codes per column with their own slot (ties: the
            smaller code first); 0 leaves only the "other" slot.
        cat_cost: The squared distance a mismatch adds, > 0.
        metric: "euclidean" or "cosine".

    Returns:
        The fitted space.
    """
    scaler = RobustScaler().fit(_signed_log(rows, num_cols))
    encoders = tuple(
        TopNHashEncoder(name, top_n, hash_buckets).fit(rows[name]) for name in cat_cols
    )
    kept = []
    for column in _codes(encoders, rows).T:
        codes, counts = np.unique(column, return_counts=True)
        order = np.lexsort((codes, -counts))
        kept.append(np.sort(codes[order[:top_k]]))
    return Space(
        num_cols=tuple(num_cols),
        scaler=scaler,
        encoders=encoders,
        kept=tuple(kept),
        weight=math.sqrt(cat_cost / 2),
        metric=metric,
    )


def embed(space: Space, rows: pl.DataFrame) -> np.ndarray:
    """Place rows in the space.

    Args:
        space: The fitted space.
        rows: Rows of the data file.

    Returns:
        Float64 points of shape (n, n_num + Σ (len(kept) + 1)): the
        scaled numbers, then per categorical column one slot per kept
        code and a last "other" slot, exactly one of them set to
        space.weight; then scale_for_metric(space.metric).
    """
    numbers = space.scaler.transform(_signed_log(rows, space.num_cols))
    blocks = [np.nan_to_num(numbers, nan=0.0)]
    for kept, column in zip(space.kept, _codes(space.encoders, rows).T):
        position = np.searchsorted(kept, column)
        found = position < len(kept)
        found[found] = kept[position[found]] == column[found]
        slot = np.where(found, position, len(kept))
        block = np.zeros((len(column), len(kept) + 1))
        block[np.arange(len(column)), slot] = space.weight
        blocks.append(block)
    return scale_for_metric(np.hstack(blocks), space.metric)


def scale_for_metric(points: np.ndarray, metric: str) -> np.ndarray:
    """Scale points so that euclidean distances serve the metric.

    Args:
        points: Shape (n, d).
        metric: "euclidean" (unchanged) or "cosine" (each row divided by
            max(its norm, 1e-8)).

    Returns:
        A new array of the same shape.
    """
    if metric == "cosine":
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        return points / np.maximum(norms, 1e-8)
    return np.array(points, dtype=np.float64)


def _signed_log(rows: pl.DataFrame, num_cols: Sequence[str]) -> np.ndarray:
    """Return signed_log1p of num_cols; NaN if missing or infinite."""
    numbers = rows.select(num_cols).to_numpy().astype(np.float64)
    numbers = numbers.reshape(rows.height, len(num_cols))
    numbers[~np.isfinite(numbers)] = np.nan
    return signed_log1p(numbers)


def _codes(encoders: Sequence[TopNHashEncoder], rows: pl.DataFrame) -> np.ndarray:
    columns = [encoder.transform(rows[encoder.column]) for encoder in encoders]
    return np.column_stack(columns) if columns else np.empty((rows.height, 0), int)

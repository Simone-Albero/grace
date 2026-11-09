"""Preprocessing fitted on train: scaling, categorical and class codes.

The prepared frame is what both the classifier and the region space
read: numeric columns as signed log1p then robust scaling, categorical
columns as integer codes (frequent values their own code, the rest
hashed), the class as an integer code in the column LABEL.
"""

from __future__ import annotations

__all__ = [
    "HASH_SALT",
    "LABEL",
    "Preprocessor",
    "TopNHashEncoder",
    "hash_bucket",
    "signed_log1p",
]

import hashlib
from collections.abc import Sequence

import numpy as np
import polars as pl
from sklearn.preprocessing import RobustScaler

LABEL = "label"  # class code column of every prepared split
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
    """Integer codes of one categorical column, fitted on train.

    Code 0 is null; codes 1..top_n are the top_n most frequent values of
    train, the most frequent first (ties: the smaller value first);
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
            values: The column on train, as text.

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
            seen, [code(v) for v in seen], return_dtype=pl.Int32
        )
        return codes.fill_null(0).to_numpy()


class Preprocessor:
    """Every column of a split, fitted on train.

    Attributes:
        num_cols: The numeric columns, in output order.
        cat_cols: The categorical columns, in output order after
            num_cols.
        label_col: The raw class column.
        top_n: Passed to every TopNHashEncoder.
        hash_buckets: Passed to every TopNHashEncoder.
        classes_: The class names, sorted; a class code is an index into
            it. Set by fit.
    """

    def __init__(
        self,
        num_cols: Sequence[str],
        cat_cols: Sequence[str],
        label_col: str,
        top_n: int,
        hash_buckets: int,
    ) -> None:
        self.num_cols = tuple(num_cols)
        self.cat_cols = tuple(cat_cols)
        self.label_col = label_col
        self.top_n = top_n
        self.hash_buckets = hash_buckets

    def fit(self, train: pl.DataFrame) -> Preprocessor:
        """Fit the numeric scaler, the encoders and the classes.

        The scaler subtracts the median and divides by the interquartile
        range (sklearn RobustScaler; a column with no spread is divided
        by 1).

        Args:
            train: The clean train rows.

        Returns:
            self.
        """
        self.scaler_ = RobustScaler().fit(self._numeric(train))
        self.encoders_ = [
            TopNHashEncoder(c, self.top_n, self.hash_buckets).fit(train[c])
            for c in self.cat_cols
        ]
        self.classes_ = tuple(sorted(train[self.label_col].unique().to_list()))
        return self

    def transform(self, rows: pl.DataFrame) -> pl.DataFrame:
        """Prepare a split.

        Args:
            rows: Clean rows with every input column.

        Returns:
            num_cols (Float64), cat_cols (Int32) and LABEL (Int64), in
            that order, one row per row of rows.

        Raises:
            ValueError: When rows holds a class not seen by fit.
        """
        unseen = set(rows[self.label_col].unique().to_list()) - set(self.classes_)
        if unseen:
            raise ValueError(f"classes not seen on train: {sorted(unseen)}")
        class_codes = {name: i for i, name in enumerate(self.classes_)}
        scaled = self.scaler_.transform(self._numeric(rows))
        columns = {c: scaled[:, i] for i, c in enumerate(self.num_cols)}
        for encoder in self.encoders_:
            columns[encoder.column] = encoder.transform(rows[encoder.column])
        columns[LABEL] = rows[self.label_col].replace_strict(
            class_codes, return_dtype=pl.Int64
        )
        return pl.DataFrame(columns).with_columns(
            pl.col(self.num_cols).cast(pl.Float64), pl.col(self.cat_cols).cast(pl.Int32)
        )

    def _numeric(self, rows: pl.DataFrame) -> np.ndarray:
        values = rows.select(self.num_cols).to_numpy().astype(np.float64)
        return signed_log1p(values.reshape(rows.height, len(self.num_cols)))

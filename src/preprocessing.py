"""The first block of every classifier pipeline, fitted on train.

Numbers are z-scored; each categorical column is one-hot over its top_n
most frequent values of train plus one "other" column collecting the
rest (rare, unseen, null). Non-finite numbers become NaN and stay NaN:
the pipeline imputes them for a classifier that cannot read them.
"""

from __future__ import annotations

__all__ = ["Features", "Preprocessor", "feature_cols"]

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass(frozen=True)
class Features:
    """The input columns a split file holds.

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

    Stage split drops the constant columns, so a split file may lack
    some of the configured ones.

    Args:
        columns: The columns of a split file.
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


class Preprocessor(BaseEstimator, TransformerMixin):
    """scikit-learn transformer from a polars frame to a float matrix.

    Attributes:
        features: The input columns.
        top_n: The most frequent values of a categorical column with a
            column of their own.
        mean_: The train mean of each number (NaN ignored), set by fit.
        std_: The train standard deviation of each number, 1 where it is
            0 or not finite, set by fit.
        levels_: The kept values of each categorical column, the most
            frequent first (ties: the smaller value first), set by fit.
    """

    def __init__(self, features: Features, top_n: int) -> None:
        self.features = features
        self.top_n = top_n

    def fit(self, X: pl.DataFrame, y: object = None) -> Preprocessor:
        """Learn the mean and spread of the numbers and the kept values.

        Args:
            X: The train rows.
            y: Ignored.

        Returns:
            self.
        """
        numbers = self._numbers(X)
        # An all-NaN column warns; it gets mean 0 and std 1 below.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean, std = np.nanmean(numbers, axis=0), np.nanstd(numbers, axis=0)
        self.mean_ = np.nan_to_num(mean)
        self.std_ = np.where(np.isfinite(std) & (std > 0), std, 1.0)
        self.levels_ = {
            column: X[column]
            .drop_nulls()
            .value_counts()
            .sort(["count", column], descending=[True, False])
            .head(self.top_n)[column]
            .to_list()
            for column in self.features.cat
        }
        return self

    def transform(self, X: pl.DataFrame) -> np.ndarray:
        """Return the z-scored numbers, then the one-hot blocks.

        Args:
            X: Rows with every column of features.

        Returns:
            Float64, shape (n, n_num + Σ (len(levels) + 1)); per
            categorical column one column per kept value, then "other".
        """
        parts = [(self._numbers(X) - self.mean_) / self.std_]
        flags = []
        for column, levels in self.levels_.items():
            flags += [pl.col(column).eq(value).fill_null(False) for value in levels]
            flags.append(pl.col(column).is_in(levels).fill_null(False).not_())
        if flags:
            onehot = X.select(
                flag.cast(pl.Float64).alias(f"_{i}") for i, flag in enumerate(flags)
            )
            parts.append(onehot.to_numpy())
        return np.hstack(parts)

    def _numbers(self, X: pl.DataFrame) -> np.ndarray:
        numbers = X.select(self.features.num).to_numpy().astype(np.float64)
        numbers = numbers.reshape(X.height, len(self.features.num))
        numbers[~np.isfinite(numbers)] = np.nan
        return numbers

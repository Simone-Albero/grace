"""Raw-table cleaning, stratified split, class sampling, encoders."""

__all__ = [
    "LogTransformer",
    "TopNHashEncoder",
    "build_preprocessor",
    "cap_per_class",
    "drop_nans",
    "drop_rare_classes",
    "ml_split",
    "stratified_holdout",
]

import hashlib
import math

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


def drop_nans(rows: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Drop the rows with a NaN or infinite value in `cols`.

    Args:
        rows: The table to clean.
        cols: The columns checked for NaN and infinite values.

    Returns:
        The remaining rows, with their index.
    """
    return rows.replace([np.inf, -np.inf], np.nan).dropna(subset=cols)


def drop_rare_classes(
    rows: pd.DataFrame, label_col: str, *, min_count: int
) -> pd.DataFrame:
    """Drop the rows of every class with fewer than `min_count` rows.

    Args:
        rows: The table to filter.
        label_col: The label column.
        min_count: The fewest rows a class keeps to stay.

    Returns:
        The rows of the remaining classes, with their index.
    """
    counts = rows[label_col].value_counts()
    return rows[~rows[label_col].isin(counts[counts < min_count].index)]


def _stratified_sample(
    rows: pd.DataFrame,
    label_col: str,
    per_group: int,
    *,
    random_state: int,
) -> pd.DataFrame:
    """Sample up to `per_group` rows per label, keeping the index."""
    return rows.groupby(rows[label_col].values, group_keys=False).apply(
        lambda group: group.sample(
            n=min(len(group), per_group), random_state=random_state
        )
    )


def cap_per_class(
    rows: pd.DataFrame,
    max_per_class: int | None,
    *,
    label_col: str,
    random_state: int,
) -> pd.DataFrame:
    """Sample every class down to at most `max_per_class` rows.

    A class under the cap stays whole, so the shares of the rare classes
    are kept and only the largest are flattened.

    Args:
        rows: The table to cap.
        max_per_class: The most rows of a class; None keeps every row.
        label_col: The label column.
        random_state: The seed of the sampling.

    Returns:
        The capped rows, with their index.
    """
    if max_per_class is None:
        return rows
    return _stratified_sample(rows, label_col, max_per_class, random_state=random_state)


def ml_split(
    rows: pd.DataFrame,
    *,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    random_state: int,
    label_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a table into stratified train, validation and test sets.

    Args:
        rows: The table to split.
        train_frac: The share of the rows for train.
        val_frac: The share for validation.
        test_frac: The share for test.
        random_state: The seed of the splits.
        label_col: The label column, which stratifies both splits.

    Returns:
        The train, validation and test rows.
    """
    train, rest = train_test_split(
        rows, train_size=train_frac, random_state=random_state, stratify=rows[label_col]
    )
    val, test = train_test_split(
        rest,
        train_size=val_frac / (val_frac + test_frac),
        random_state=random_state,
        stratify=rest[label_col],
    )
    return train, val, test


def stratified_holdout(
    rows: pd.DataFrame, *, holdout_frac: float, label_col: str, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a table into a fit part and a stratified holdout.

    Args:
        rows: The table to split.
        holdout_frac: The share of rows held out.
        label_col: The label column, which stratifies the split.
        random_state: The seed of the split.

    Returns:
        The fit rows and the holdout rows.

    Raises:
        ValueError: A class has fewer than two rows, or either side of
            the split would hold fewer rows than there are classes, so
            stratification could not keep every class on both sides.
    """
    counts = rows[label_col].value_counts()
    n_holdout = math.ceil(holdout_frac * len(rows))
    n_fit = len(rows) - n_holdout
    if counts.min() < 2 or n_fit < len(counts) or n_holdout < len(counts):
        raise ValueError(
            f"{len(rows)} rows of {len(counts)} classes (smallest {counts.min()}) "
            f"split by holdout_frac={holdout_frac} into {n_fit} fit and {n_holdout} "
            "holdout row(s) cannot keep every class on both sides: lower holdout_frac "
            "or give it more rows."
        )
    fit, holdout = train_test_split(
        rows,
        test_size=holdout_frac,
        random_state=random_state,
        stratify=rows[label_col],
    )
    return fit, holdout


class LogTransformer(BaseEstimator, TransformerMixin):
    """A signed log1p, which compresses a skewed column.

    It keeps the order of negatives. It is stateless, so it has no
    fitted attributes.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> "LogTransformer":
        """Return the transformer, which has nothing to learn.

        Args:
            X: The features, ignored.
            y: The labels, ignored.

        Returns:
            The transformer itself.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply sign(x) * log1p(|x|) to every value.

        Args:
            X: The numerical features.

        Returns:
            The transformed features, shaped like `X`.
        """
        return np.sign(X) * np.log1p(np.abs(X))


# Part of every bucket: changing it re-buckets every unseen value.
_HASH_KEY = "cat-encoder-v1"


class TopNHashEncoder(BaseEstimator, TransformerMixin):
    """An encoder of categoricals: 0 missing, 1..top_n frequent, hashes.

    Attributes:
        top_n: The most frequent values of a column that get an id.
        hash_buckets: The buckets the other values hash into; 0 sends
            them to the missing id.
        columns_: The fitted column names.
        category_maps_: By column, the id of each frequent value.
    """

    def __init__(self, *, top_n: int = 256, hash_buckets: int = 1024) -> None:
        self.top_n = top_n
        self.hash_buckets = hash_buckets

    def _hash_bucket(self, col: str, value: object, n_buckets: int) -> int:
        """Return the stable bucket index of a category value."""
        text = "NA" if pd.isna(value) else str(value)
        digest = hashlib.blake2b(
            f"{_HASH_KEY}|{col}|{text}".encode(), digest_size=8
        ).digest()
        return int.from_bytes(digest, byteorder="little") % n_buckets

    def fit(self, X: pd.DataFrame, y: object = None) -> "TopNHashEncoder":
        """Learn the top-N categories of every column.

        Args:
            X: The categorical features.
            y: The labels, ignored.

        Returns:
            The fitted encoder.
        """
        X = pd.DataFrame(X)
        self.columns_ = list(X.columns)
        self.category_maps_ = {
            col: {
                cat: i + 1
                for i, cat in enumerate(
                    X[col].value_counts(dropna=True).nlargest(self.top_n).index
                )
            }
            for col in self.columns_
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode every column into category ids, hash buckets or 0.

        Args:
            X: The categorical features; the fitted columns it lacks
                are skipped.

        Returns:
            The int32 ids, indexed like `X`.
        """
        X = pd.DataFrame(X)
        hashed_start = 1 + self.top_n
        out = {}
        for col in self.columns_:
            if col not in X.columns:
                continue
            category_map = self.category_maps_[col]
            column = X[col]
            codes = column.map(category_map)
            if self.hash_buckets > 0:
                unseen = codes.isna() & column.notna()
                hash_map = {
                    value: hashed_start
                    + self._hash_bucket(col, value, self.hash_buckets)
                    for value in column[unseen].unique()
                }
                codes = codes.where(~unseen, column.map(hash_map))
            codes = codes.fillna(0)
            out[col] = codes.to_numpy(dtype=np.int32)
        return pd.DataFrame(out, index=X.index)


def build_preprocessor(
    num_cols: list[str], cat_cols: list[str], *, top_n: int, hash_buckets: int
) -> ColumnTransformer:
    """Return the encoders prepare fits on the feature columns.

    Signed log with robust scaling, top-N categories with hash codes.

    Args:
        num_cols: The numerical feature names.
        cat_cols: The categorical feature names.
        top_n: The categories kept per column.
        hash_buckets: The buckets the other categories hash into.

    Returns:
        The unfitted transformer, with steps "num" and "cat" for the
        non-empty column lists and plain output column names.
    """
    transformers = []
    if num_cols:
        numerical = Pipeline(
            [("log_transformer", LogTransformer()), ("scaler", RobustScaler())]
        )
        transformers.append(("num", numerical, num_cols))
    if cat_cols:
        encoder = TopNHashEncoder(top_n=top_n, hash_buckets=hash_buckets)
        categorical = Pipeline([("top_n_encoder", encoder)])
        transformers.append(("cat", categorical, cat_cols))
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

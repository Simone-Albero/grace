"""Classifiers: registry, pipeline, balancing and probabilities.

A pipeline takes a pandas frame of prepared columns (numeric, then
categorical codes) and puts in front of the classifier the categorical
strategy of its ModelSpec:
- "onehot": one-hot of the codes, unknown codes all zero;
- "drop_cat": the categorical columns dropped;
- "native_sklearn": codes as pandas categories, the 255 most frequent
  kept (the rest NaN), read by the classifier as categorical;
- "native_xgb": codes as pandas categories, all kept,
  enable_categorical;
- None: the frame passed as it is, codes as numbers.
"""

from __future__ import annotations

__all__ = [
    "CappedCategoryEncoder",
    "MODELS",
    "build_pipeline",
    "predict_proba",
    "undersample",
]

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Each categorical strategy: the encoder of the categorical columns
# (None drops them) and the params that make a classifier read pandas
# categories.
_ENCODINGS: dict[str, tuple[Callable[[], Any] | None, dict[str, Any] | None]] = {
    "onehot": (
        lambda: OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        None,
    ),
    "drop_cat": (None, None),
    # HistGradientBoosting takes at most 255 categories per column.
    "native_sklearn": (
        lambda: CappedCategoryEncoder(255),
        {"categorical_features": "from_dtype"},
    ),
    "native_xgb": (
        lambda: CappedCategoryEncoder(None),
        {"enable_categorical": True, "tree_method": "hist"},
    ),
}


class CalibratedLinearSVC(BaseEstimator, ClassifierMixin):
    """A LinearSVC with sigmoid calibration (cv=3), for predict_proba.

    Attributes:
        C: The inverse regularization strength.
        max_iter: The most iterations of the LinearSVC.
        random_state: The seed of the LinearSVC.
        classes_: The class labels, set by fit.
    """

    def __init__(
        self, C: float = 1.0, max_iter: int = 2000, random_state: int | None = None
    ) -> None:
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> CalibratedLinearSVC:
        """Fit the LinearSVC and its calibration.

        Args:
            X: The features.
            y: The labels.

        Returns:
            self.
        """
        svc = LinearSVC(
            C=self.C, max_iter=self.max_iter, random_state=self.random_state
        )
        self.calibrated_ = CalibratedClassifierCV(svc, method="sigmoid", cv=3).fit(X, y)
        self.classes_ = self.calibrated_.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the predicted labels."""
        return self.calibrated_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return the calibrated probabilities, shape (n, n_classes)."""
        return self.calibrated_.predict_proba(X)


class CappedCategoryEncoder(BaseEstimator, TransformerMixin):
    """Cast columns to pandas categories of their most frequent values.

    Attributes:
        max_cardinality: The most categories per column (the rest NaN);
            None keeps every value seen by fit.
        categories_: The kept values of every column, set by fit.
    """

    def __init__(self, max_cardinality: int | None = None) -> None:
        self.max_cardinality = max_cardinality

    def fit(self, X: pd.DataFrame, y: object = None) -> CappedCategoryEncoder:
        """Learn the kept values of every column.

        Args:
            X: The categorical columns.
            y: Ignored.

        Returns:
            self.
        """
        self.categories_ = {}
        for column in X.columns:
            counts = X[column].value_counts()
            n_kept = (
                len(counts) if self.max_cardinality is None else self.max_cardinality
            )
            self.categories_[column] = counts.nlargest(n_kept).index
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return the columns as categoricals over the kept values."""
        encoded = X.copy()
        for column, kept in self.categories_.items():
            encoded[column] = pd.Categorical(
                encoded[column].where(encoded[column].isin(kept)), kept
            )
        return encoded

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Return the column names, unchanged."""
        return np.array(list(self.categories_))


@dataclass(frozen=True)
class ModelSpec:
    """How to build one classifier.

    Attributes:
        factory: Takes the params, returns an unfitted classifier with
            the scikit-learn API.
        categorical: A key of _ENCODINGS; None passes the frame through.
    """

    factory: Callable[..., Any]
    categorical: str | None


MODELS: dict[str, ModelSpec] = {
    "logistic_regression": ModelSpec(LogisticRegression, "onehot"),
    "lda": ModelSpec(LinearDiscriminantAnalysis, "onehot"),
    "linear_svc": ModelSpec(CalibratedLinearSVC, "onehot"),
    "knn": ModelSpec(KNeighborsClassifier, "onehot"),
    "mlp": ModelSpec(MLPClassifier, "onehot"),
    "decision_tree": ModelSpec(DecisionTreeClassifier, None),
    "random_forest": ModelSpec(RandomForestClassifier, None),
    "hist_gradient_boosting": ModelSpec(
        HistGradientBoostingClassifier, "native_sklearn"
    ),
    "xgboost": ModelSpec(XGBClassifier, "native_xgb"),
    "naive_bayes": ModelSpec(GaussianNB, "drop_cat"),
}


def undersample(labels: np.ndarray, seed: int) -> np.ndarray:
    """Return rows that cut every class to the size of the smallest.

    Args:
        labels: The class code of every row.
        seed: The seed of the draw within each class.

    Returns:
        The kept positions, ascending.
    """
    rng = np.random.default_rng(seed)
    smallest = np.bincount(labels)[np.unique(labels)].min()
    kept = [
        rng.choice(np.flatnonzero(labels == code), smallest, replace=False)
        for code in np.unique(labels)
    ]
    return np.sort(np.concatenate(kept))


def build_pipeline(
    name: str,
    params: Mapping[str, Any],
    num_cols: Sequence[str],
    cat_cols: Sequence[str],
    seed: int,
) -> Pipeline:
    """Build an unfitted pipeline: categorical encoding, classifier.

    The classifier gets params, plus random_state=seed when it takes one
    and params does not set it; list values become tuples (MLP layer
    sizes).

    Args:
        name: A key of MODELS.
        params: The classifier kwargs.
        num_cols: The numeric columns of the input frame.
        cat_cols: The categorical code columns of the input frame.
        seed: The default random_state.

    Returns:
        The pipeline, steps "encode" and "classifier".

    Raises:
        ValueError: When name is not a key of MODELS.
    """
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}, expected one of {sorted(MODELS)}")
    spec = MODELS[name]
    params = {k: tuple(v) if isinstance(v, list) else v for k, v in params.items()}
    if "random_state" in spec.factory().get_params() and "random_state" not in params:
        params["random_state"] = seed
    if spec.categorical is None:
        return Pipeline(
            [("encode", "passthrough"), ("classifier", spec.factory(**params))]
        )
    encoder, native = _ENCODINGS[spec.categorical]
    steps = [("num", "passthrough", list(num_cols))]
    if encoder is not None and cat_cols:
        steps.append(("cat", encoder(), list(cat_cols)))
    encode = ColumnTransformer(steps, remainder="drop", verbose_feature_names_out=False)
    if native is not None:
        # A categorical dtype survives only in a DataFrame.
        encode.set_output(transform="pandas")
        params = native | params
    return Pipeline([("encode", encode), ("classifier", spec.factory(**params))])


def predict_proba(model: Pipeline, X: pd.DataFrame, n_classes: int) -> np.ndarray:
    """Return the class probabilities of every row.

    Args:
        model: A pipeline fitted on class codes 0..n_classes-1, all
            present.
        X: The prepared rows.
        n_classes: The number of classes.

    Returns:
        Shape (n, n_classes); column j is class code j.

    Raises:
        ValueError: When the model returns another number of columns or
            a non-finite probability.
    """
    proba = model.predict_proba(X)
    if proba.shape[1] != n_classes or not np.isfinite(proba).all():
        raise ValueError(
            f"expected {n_classes} finite probability columns, "
            f"got {proba.shape[1]} (finite: {np.isfinite(proba).all()})"
        )
    return proba

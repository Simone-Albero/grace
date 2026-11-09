"""Classifiers: registry, pipeline and probabilities.

Every pipeline is preprocessing.Preprocessor (z-score and one-hot), a
median imputer for a classifier that cannot read NaN, then the
classifier. It takes the polars rows of a split file.
"""

from __future__ import annotations

__all__ = ["MODELS", "ModelSpec", "build_pipeline", "predict_proba"]

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.preprocessing import Features, Preprocessor

# Rows per predict_proba call: the one-hot matrix of a whole test split
# may not fit in memory.
_CHUNK_ROWS = 100_000


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


@dataclass(frozen=True)
class ModelSpec:
    """How to build one classifier.

    Attributes:
        factory: Takes the params, returns an unfitted classifier with
            the scikit-learn API.
        nan_ok: Whether the classifier reads NaN; otherwise the
            pipeline imputes the train median.
    """

    factory: Callable[..., Any]
    nan_ok: bool


MODELS: dict[str, ModelSpec] = {
    "logistic_regression": ModelSpec(LogisticRegression, nan_ok=False),
    "lda": ModelSpec(LinearDiscriminantAnalysis, nan_ok=False),
    "linear_svc": ModelSpec(CalibratedLinearSVC, nan_ok=False),
    "knn": ModelSpec(KNeighborsClassifier, nan_ok=False),
    "mlp": ModelSpec(MLPClassifier, nan_ok=False),
    "naive_bayes": ModelSpec(GaussianNB, nan_ok=False),
    "decision_tree": ModelSpec(DecisionTreeClassifier, nan_ok=True),
    "random_forest": ModelSpec(RandomForestClassifier, nan_ok=True),
    "hist_gradient_boosting": ModelSpec(HistGradientBoostingClassifier, nan_ok=True),
    "xgboost": ModelSpec(XGBClassifier, nan_ok=True),
}


def build_pipeline(
    name: str,
    params: Mapping[str, Any],
    features: Features,
    top_n: int,
    seed: int,
) -> Pipeline:
    """Build an unfitted pipeline: preprocessing, imputer, classifier.

    The imputer (median of train) is there only when the classifier
    cannot read NaN. The classifier gets params, plus random_state=seed
    when it takes one and params does not set it; list values become
    tuples (MLP layer sizes).

    Args:
        name: A key of MODELS.
        params: The classifier kwargs.
        features: The input columns.
        top_n: The kept values of each categorical column.
        seed: The default random_state.

    Returns:
        The pipeline, steps "preprocess", "impute" (if any) and
        "classifier".

    Raises:
        ValueError: When name is not a key of MODELS.
    """
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}, expected one of {sorted(MODELS)}")
    spec = MODELS[name]
    params = {k: tuple(v) if isinstance(v, list) else v for k, v in params.items()}
    if "random_state" in spec.factory().get_params() and "random_state" not in params:
        params["random_state"] = seed
    steps = [("preprocess", Preprocessor(features, top_n))]
    if not spec.nan_ok:
        steps.append(("impute", SimpleImputer(strategy="median")))
    steps.append(("classifier", spec.factory(**params)))
    return Pipeline(steps)


def predict_proba(model: Pipeline, rows: pl.DataFrame, n_classes: int) -> np.ndarray:
    """Return the class probabilities of every row.

    Args:
        model: A pipeline fitted on class codes 0..n_classes-1, all
            present.
        rows: The rows of a split file.
        n_classes: The number of classes.

    Returns:
        Shape (n, n_classes); column j is class code j.

    Raises:
        ValueError: When the model returns another number of columns or
            a non-finite probability.
    """
    proba = np.vstack(
        [model.predict_proba(chunk) for chunk in rows.iter_slices(_CHUNK_ROWS)]
    )
    if proba.shape[1] != n_classes or not np.isfinite(proba).all():
        raise ValueError(
            f"expected {n_classes} finite probability columns, "
            f"got {proba.shape[1]} (finite: {np.isfinite(proba).all()})"
        )
    return proba

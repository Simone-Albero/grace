"""Scores of the classifier and of the failure-rate predictors."""

from __future__ import annotations

__all__ = ["classification_metrics", "confusion", "rate_scores"]

from collections.abc import Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    r2_score,
)


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, classes: Sequence[str]
) -> list[dict[str, object]]:
    """Return accuracy, and precision, recall and F1 per class.

    Args:
        y_true: The class codes.
        y_pred: The predicted class codes.
        classes: The class names; code j is classes[j].

    Returns:
        Rows {label, metric, value}: label "all" with metric accuracy,
        macro_precision, macro_recall, macro_f1, weighted_precision,
        weighted_recall and weighted_f1; then one row per class name
        with metric precision, recall, f1 and support. An undefined
        precision or recall is 0.
    """
    codes = list(range(len(classes)))
    # Averaged over the classes that occur, as sklearn does by default.
    present = np.union1d(y_true, y_pred).tolist()
    rows: list[dict[str, object]] = [
        {
            "label": "all",
            "metric": "accuracy",
            "value": float(accuracy_score(y_true, y_pred)),
        }
    ]
    for average in ("macro", "weighted"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=present, average=average, zero_division=0
        )
        for metric, value in (("precision", precision), ("recall", recall), ("f1", f1)):
            rows.append(
                {"label": "all", "metric": f"{average}_{metric}", "value": float(value)}
            )
    per_class = precision_recall_fscore_support(
        y_true, y_pred, labels=codes, zero_division=0
    )
    for code, name in enumerate(classes):
        for metric, values in zip(("precision", "recall", "f1", "support"), per_class):
            cast = int if metric == "support" else float
            rows.append({"label": name, "metric": metric, "value": cast(values[code])})
    return rows


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Return the confusion counts.

    Args:
        y_true: The class codes.
        y_pred: The predicted class codes.
        n_classes: The number of classes.

    Returns:
        Shape (n_classes, n_classes): rows true, columns predicted.
    """
    return confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))


def rate_scores(predicted: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    """Score predicted against observed failure rates of regions.

    Args:
        predicted: The predicted rates.
        observed: The observed rates, same length.

    Returns:
        spearman (NaN when either side has std <= 1e-12), r2 (NaN with
        fewer than 2 values or a constant observed side), mae and mse.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    constant = predicted.std() <= 1e-12 or observed.std() <= 1e-12
    spearman = np.nan if constant else float(spearmanr(predicted, observed).statistic)
    r2 = (
        np.nan
        if len(observed) < 2 or observed.std() == 0
        else r2_score(observed, predicted)
    )
    error = predicted - observed
    return {
        "spearman": spearman,
        "r2": float(r2),
        "mae": float(np.abs(error).mean()),
        "mse": float((error**2).mean()),
    }

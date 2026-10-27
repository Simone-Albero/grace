"""Failure-rate scores and classification metrics."""

__all__ = [
    "check_probabilities",
    "compute_classification_metrics",
    "log_scores",
    "rate_scores",
    "spearman",
]

import logging
from collections.abc import Callable, Mapping

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.utils.multiclass import unique_labels

logger = logging.getLogger(__name__)


def check_probabilities(y_proba: np.ndarray, n_classes: int, split: str) -> None:
    """Check a split's probabilities: one finite column per class.

    Args:
        y_proba: The predicted probabilities, shape (n, columns).
        n_classes: The classes of the data.
        split: The split's name, for the error message.

    Raises:
        ValueError: A column count other than `n_classes`, or a value
            that is not finite.
    """
    if y_proba.shape[1] != n_classes:
        raise ValueError(
            f"The model gave {y_proba.shape[1]} probabilities per row on {split} for "
            f"{n_classes} classes: column k of predictions would not be class k."
        )
    if not np.isfinite(y_proba).all():
        raise ValueError(
            f"The model predicted non-finite probabilities on {split}: every "
            "confidence-based score downstream would be NaN."
        )


def spearman(values: np.ndarray, other: np.ndarray) -> float:
    """Return Spearman's rho, NaN when either side is constant.

    Args:
        values: One side of the correlation.
        other: The other side, the same length as `values`.

    Returns:
        The correlation, NaN when either side has no variance.
    """
    if np.std(values) > 1e-12 and np.std(other) > 1e-12:
        return float(spearmanr(values, other).statistic)
    return float("nan")


def rate_scores(predicted: ArrayLike, observed: ArrayLike) -> dict[str, float]:
    """Score predicted failure rates against the observed ones.

    Args:
        predicted: The predicted rate of every region.
        observed: The observed rate of every region.

    Returns:
        "spearman", "r2", "mae" and "mse". Spearman is NaN when either
        side is constant, R² when the observed rates are.
    """
    predicted = np.asarray(predicted, dtype=float)
    observed = np.asarray(observed, dtype=float)
    return {
        "spearman": spearman(predicted, observed),
        "r2": (
            float(r2_score(observed, predicted))
            if len(observed) > 1 and np.std(observed) > 0
            else float("nan")
        ),
        "mae": float(mean_absolute_error(observed, predicted)),
        "mse": float(np.mean((predicted - observed) ** 2)),
    }


def log_scores(name: str, scores: Mapping[str, float]) -> None:
    """Log a predictor's Spearman, R² and MAE on one line.

    Args:
        name: The predictor name.
        scores: Its scores, as `rate_scores` returns them.
    """
    logger.info(
        "%-9s rho=%.4f r2=%.4f mae=%.4f",
        name,
        scores["spearman"],
        scores["r2"],
        scores["mae"],
    )


_METRICS: list[tuple[str, Callable]] = [
    ("precision", precision_score),
    ("recall", recall_score),
    ("f1", f1_score),
]


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, object]:
    """Score a classification overall and per class.

    Args:
        y_true: The true class ids.
        y_pred: The predicted class ids.

    Returns:
        The "accuracy", the macro and weighted "precision", "recall"
        and "f1" (as `<name>_<average>`) and "classes", one dict with
        "class_id" and the three scores per class seen in either array.
    """
    metrics: dict[str, object] = {"accuracy": float(accuracy_score(y_true, y_pred))}

    for average in ("macro", "weighted"):
        for name, metric_fn in _METRICS:
            metrics[f"{name}_{average}"] = float(
                metric_fn(y_true, y_pred, average=average, zero_division=0)
            )

    # Every class observed or predicted, passed explicitly so the rows
    # below line up with the arrays sklearn returns.
    labels = unique_labels(y_true, y_pred)
    per_class = {
        name: metric_fn(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        ).tolist()
        for name, metric_fn in _METRICS
    }
    metrics["classes"] = [
        {
            "class_id": int(label),
            **{name: values[i] for name, values in per_class.items()},
        }
        for i, label in enumerate(labels)
    ]

    return metrics

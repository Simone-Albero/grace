"""Probability checks, predictor scores and classification metrics."""

__all__ = [
    "ap_lift",
    "auroc",
    "binned_mse_ratio",
    "check_probabilities",
    "compute_classification_metrics",
]

from collections.abc import Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.multiclass import unique_labels


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


def auroc(values: np.ndarray, errors: np.ndarray) -> float:
    """Return the chance that a failed row outranks a correct one.

    Args:
        values: Each row's failure probability.
        errors: Each row's failure, 0 or 1.

    Returns:
        The area under the ROC curve, 0.5 for a random order, NaN when
        every row failed or none did.
    """
    if errors.min() == errors.max():
        return float("nan")
    return float(roc_auc_score(errors, values))


def ap_lift(values: np.ndarray, errors: np.ndarray) -> float:
    """Return the average precision over the failure rate.

    Args:
        values: Each row's failure probability.
        errors: Each row's failure, 0 or 1.

    Returns:
        1 for a random order and 1 / failure rate at best, NaN when no
        row failed.
    """
    if not errors.any():
        return float("nan")
    return float(average_precision_score(errors, values) / errors.mean())


def binned_mse_ratio(
    values: np.ndarray, errors: np.ndarray, reference: np.ndarray, n_bins: int
) -> float:
    """Return the squared gap of the bins' rates over the reference's.

    The rows fall into up to `n_bins` bins of about equal size by
    `values`, equal values sharing a bin. The gap of a bin is its mean
    probability against its failure rate, so the failure's chance within
    a bin drops out, unlike in the per-row mean squared error. The
    sampling noise of each bin's rate stays: the score has a floor that
    falls as the rows grow.

    Args:
        values: Each row's failure probability.
        errors: Each row's failure, 0 or 1.
        reference: Each row's probability of the predictor to beat.
        n_bins: The bins to aim for.

    Returns:
        The gap of `values` over that of `reference`, summed over the
        rows; NaN when the reference has none.
    """
    distinct, inverse, sizes = np.unique(
        values, return_inverse=True, return_counts=True
    )
    # A value goes to the bin of its first rank, so equal values stay
    # together whatever share of the rows they hold.
    first_rank = np.cumsum(sizes) - sizes
    bins = (first_rank * n_bins // len(values))[inverse]
    counts = np.bincount(bins)
    rate = np.bincount(bins, weights=errors) / np.maximum(counts, 1)

    def gap(probabilities: np.ndarray) -> float:
        mean = np.bincount(bins, weights=probabilities) / np.maximum(counts, 1)
        return float((counts * (mean - rate) ** 2).sum())

    reference_gap = gap(reference)
    # Below the tolerance the gap is float noise around an exact fit.
    return gap(values) / reference_gap if reference_gap > 1e-12 else float("nan")


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

"""Failure-rate scores and classification metrics."""

__all__ = [
    "compute_classification_metrics",
    "group_scores",
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
    roc_auc_score,
)
from sklearn.utils.multiclass import unique_labels

logger = logging.getLogger(__name__)


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


def group_scores(
    predicted: ArrayLike, observed: ArrayLike, group: ArrayLike, *, reference: float
) -> dict[str, float]:
    """Score a per-row failure prediction, constant within its group.

    `brier` decomposes exactly into `uncertainty` (the variance of
    `observed`), `resolution` (how far each group's observed rate sits
    from the overall one) and `reliability` (how far each group's
    prediction sits from its observed rate):
    `brier = uncertainty - resolution + reliability`.

    Args:
        predicted: Each row's predicted failure rate, constant within
            its group.
        observed: Each row's observed failure, 0 or 1.
        group: Each row's group id.
        reference: The single rate a group-blind model would predict
            for every row, such as the global val rate. `skill` is the
            share of its Brier score `predicted` avoids.

    Returns:
        "brier", its decomposition "uncertainty", "resolution" and
        "reliability", "skill" and "auroc". "skill" is NaN when
        `reference` already scores zero Brier, "auroc" when `observed`
        has only one class.
    """
    predicted = np.asarray(predicted, dtype=float)
    observed = np.asarray(observed, dtype=float)
    group = np.asarray(group)
    n_rows = len(observed)

    brier = float(np.mean((predicted - observed) ** 2))
    base_rate = float(observed.mean())
    uncertainty = base_rate * (1 - base_rate)

    resolution = 0.0
    reliability = 0.0
    for group_id in np.unique(group):
        in_group = group == group_id
        share = int(in_group.sum()) / n_rows
        observed_rate = float(observed[in_group].mean())
        resolution += share * (observed_rate - base_rate) ** 2
        reliability += share * (float(predicted[in_group].mean()) - observed_rate) ** 2

    reference_brier = float(np.mean((reference - observed) ** 2))
    skill = 1 - brier / reference_brier if reference_brier > 1e-12 else float("nan")
    auroc = (
        float(roc_auc_score(observed, predicted)) if 0 < base_rate < 1 else float("nan")
    )

    return {
        "brier": brier,
        "skill": skill,
        "uncertainty": uncertainty,
        "resolution": resolution,
        "reliability": reliability,
        "auroc": auroc,
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

"""Failure rates per region: observed, and from the classifier's output.

val_rate is the region's failure rate on val, the rate of its class on
val when no val row reaches it. MCP and ATC read only the classifier's
maximum probability; mcp_cal and atc_cal read the probability of its
predicted class after bias-corrected temperature scaling (BCTS,
Alexandari et al. 2020) fitted on val: one temperature for its
overconfidence, one bias per class for a prior other than the one it
was fitted on (max_per_class balances train). Meta is a model of the
classifier's errors, fitted on labelled rows (val): it reads the row's
columns next to the classifier's probabilities, its confidence and its
predicted class, as the post-hoc error predictors of the literature do
(Risk Advisor, RED). The observed rate is the target of every
predictor; it needs the test labels.
"""

from __future__ import annotations

__all__ = [
    "BASELINES",
    "ERROR_MODEL",
    "atc_rate",
    "atc_threshold",
    "bcts_confidence",
    "error_inputs",
    "error_probability",
    "failure_rates",
    "fit_bcts",
    "mcp_rate",
    "meta_rate",
    "val_rate",
]

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.special import log_softmax, softmax

from src.models import build_pipeline, predict_proba
from src.preprocessing import Features

# The column of each baseline in failures_<model>.
BASELINES = ("val_rate", "mcp", "atc", "mcp_cal", "atc_cal", "meta")
ERROR_MODEL = "hist_gradient_boosting"  # the key of models.MODELS of meta


def failure_rates(
    region: np.ndarray, error: np.ndarray, regions: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Count the rows and errors of every region.

    Args:
        region: The region of every row.
        error: Whether the classifier got every row wrong.
        regions: The regions to report, in output order.

    Returns:
        n_eval (int64), n_error (int64) and n_error / n_eval (NaN when
        n_eval is 0), one value per region of regions.
    """
    n_eval = _by_region(np.ones(len(region)), region, regions, np.sum).astype(np.int64)
    n_error = _by_region(error.astype(np.float64), region, regions, np.sum)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(n_eval > 0, n_error / n_eval, np.nan)
    return n_eval, n_error.astype(np.int64), rate


def val_rate(
    region: np.ndarray,
    error: np.ndarray,
    regions: np.ndarray,
    region_class: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the failure rate of every region on labelled rows (val).

    A region no row reaches gets the rate of the rows of the regions of
    its class; a class no row reaches, the rate of all rows.

    Args:
        region: The region of every row.
        error: Whether the classifier got every row wrong.
        regions: The regions to report, in output order.
        region_class: The class of every region of regions.

    Returns:
        The rows of every region (int64) and its rate.
    """
    n_rows, n_error, rate = failure_rates(region, error, regions)
    classes = np.unique(region_class)
    rows_of = _by_region(n_rows.astype(np.float64), region_class, classes, np.sum)
    errors_of = _by_region(n_error.astype(np.float64), region_class, classes, np.sum)
    overall = float(np.mean(error)) if len(error) else np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        class_rate = np.where(rows_of > 0, errors_of / rows_of, overall)
    fallback = class_rate[np.searchsorted(classes, region_class)]
    return n_rows, np.where(n_rows > 0, rate, fallback)


def mcp_rate(
    confidence: np.ndarray, region: np.ndarray, regions: np.ndarray
) -> np.ndarray:
    """Return the mean of 1 - confidence over the rows of every region.

    Args:
        confidence: The maximum class probability of every row.
        region: The region of every row.
        regions: The regions to report, in output order.

    Returns:
        One value per region; NaN when it has no row.
    """
    return _by_region(1 - confidence, region, regions, np.mean)


def atc_threshold(confidence: np.ndarray, correct: np.ndarray) -> float:
    """Fit the ATC threshold on labelled rows (val).

    Among the distinct confidences and +inf, the cut whose count of rows
    with confidence < cut is closest to the count of wrong rows; ties:
    the smallest cut.

    Args:
        confidence: The maximum class probability of every row.
        correct: Whether the classifier got every row right.

    Returns:
        The threshold.
    """
    cuts = np.append(np.unique(confidence), np.inf)
    n_below = np.searchsorted(np.sort(confidence), cuts, side="left")
    n_wrong = int((~np.asarray(correct, dtype=bool)).sum())
    return float(cuts[np.argmin(np.abs(n_below - n_wrong))])


def atc_rate(
    confidence: np.ndarray, region: np.ndarray, regions: np.ndarray, threshold: float
) -> np.ndarray:
    """Return the share of rows under the threshold in every region.

    Args:
        confidence: The maximum class probability of every row.
        region: The region of every row.
        regions: The regions to report, in output order.
        threshold: From atc_threshold.

    Returns:
        One value per region; NaN when it has no row.
    """
    below = (confidence < threshold).astype(np.float64)
    return _by_region(below, region, regions, np.mean)


def fit_bcts(proba: np.ndarray, label: np.ndarray) -> tuple[float, np.ndarray]:
    """Fit bias-corrected temperature scaling on labelled rows (val).

    The calibrated probabilities are softmax(log(proba) / temperature +
    bias); temperature and bias minimise their mean negative
    log-likelihood of label (L-BFGS-B, analytic gradient). A
    probability is clipped to 1e-12 before its log.

    Args:
        proba: The class probabilities, shape (n, K).
        label: The class codes, 0..K-1.

    Returns:
        The temperature, > 0, and the bias, shape (K,); the bias is
        defined up to a constant.
    """
    logits = np.log(np.clip(proba, 1e-12, 1))
    rows = np.arange(len(label))

    def loss(theta: np.ndarray) -> tuple[float, np.ndarray]:
        temperature, bias = np.exp(theta[0]), theta[1:]
        log_p = log_softmax(logits / temperature + bias, axis=1)
        nll = -log_p[rows, label].mean()
        # d nll / d logit = p - onehot(label).
        residual = np.exp(log_p)
        residual[rows, label] -= 1
        grad_log_t = -(residual * logits).sum(axis=1).mean() / temperature
        return nll, np.concatenate([[grad_log_t], residual.mean(axis=0)])

    start = np.zeros(1 + proba.shape[1])
    theta = minimize(loss, start, jac=True, method="L-BFGS-B").x
    return float(np.exp(theta[0])), theta[1:]


def bcts_confidence(
    proba: np.ndarray, pred: np.ndarray, temperature: float, bias: np.ndarray
) -> np.ndarray:
    """Return the calibrated probability of the predicted class.

    The bias may move the argmax; the predicted class stays the
    classifier's, so the errors are still its own.

    Args:
        proba: The class probabilities, shape (n, K).
        pred: The class the classifier predicted for each row.
        temperature: From fit_bcts.
        bias: From fit_bcts.

    Returns:
        One probability per row.
    """
    logits = np.log(np.clip(proba, 1e-12, 1))
    calibrated = softmax(logits / temperature + bias, axis=1)
    return calibrated[np.arange(len(pred)), pred]


def error_inputs(
    rows: pl.DataFrame, predictions: pl.DataFrame, features: Features
) -> tuple[pl.DataFrame, Features]:
    """Put the classifier's output next to the columns of the rows.

    Args:
        rows: The rows of a split file.
        predictions: Their pred_<model>_<split>.parquet, row by row.
        features: The input columns of rows.

    Returns:
        The columns of features, then proba_0..proba_{K-1} and
        confidence (numbers) and pred (its class code, as text); and the
        features naming them.
    """
    outputs = [column for column in predictions.columns if column.startswith("proba_")]
    outputs.append("confidence")
    inputs = pl.concat(
        [
            rows.select(*features.num, *features.cat),
            predictions.select(*outputs, pl.col("pred").cast(pl.String)),
        ],
        how="horizontal",
    )
    return inputs, Features(num=(*features.num, *outputs), cat=(*features.cat, "pred"))


def error_probability(
    fit_rows: pl.DataFrame,
    fit_error: np.ndarray,
    rows: pl.DataFrame,
    features: Features,
    *,
    params: Mapping[str, Any],
    max_samples: int,
    top_n: int,
    seed: int,
) -> np.ndarray:
    """Fit meta on labelled rows and return its P(error) on other rows.

    Meta is the pipeline of models.build_pipeline(ERROR_MODEL), fitted
    unweighted on a uniform draw of at most max_samples fit_rows, so
    that its probabilities stay calibrated and their mean over a region
    estimates the region's failure rate.

    Args:
        fit_rows: The labelled rows, from error_inputs.
        fit_error: Whether the classifier got each of them wrong.
        rows: The rows to score, from error_inputs.
        features: The input columns, from error_inputs.
        params: The kwargs of the classifier.
        max_samples: The most fit_rows it is fitted on.
        top_n: The kept values of each categorical column.
        seed: The default random_state.

    Returns:
        P(error) of every row of rows; the share of wrong fit_rows for
        all of them when the classifier got all or none wrong.
    """
    fit_error = np.asarray(fit_error, dtype=bool)
    if fit_error.all() or not fit_error.any():
        return np.full(rows.height, fit_error.mean())
    if fit_rows.height > max_samples:
        rng = np.random.default_rng(seed)
        drawn = np.sort(rng.choice(fit_rows.height, max_samples, replace=False))
        fit_rows, fit_error = fit_rows[drawn], fit_error[drawn]
    model = build_pipeline(ERROR_MODEL, params, features, top_n, seed)
    model.fit(fit_rows, fit_error.astype(np.int64))
    return predict_proba(model, rows, 2)[:, 1]


def meta_rate(
    probability: np.ndarray, region: np.ndarray, regions: np.ndarray
) -> np.ndarray:
    """Return the mean P(error) of meta over the rows of every region.

    Args:
        probability: From error_probability, one per row.
        region: The region of every row.
        regions: The regions to report, in output order.

    Returns:
        One value per region; NaN when it has no row.
    """
    return _by_region(probability, region, regions, np.mean)


def _by_region(
    values: np.ndarray,
    region: np.ndarray,
    regions: np.ndarray,
    reduce: Callable[[np.ndarray], float],
) -> np.ndarray:
    """Reduce values per region; a mean over no row is NaN."""
    order = np.argsort(region, kind="stable")
    present, starts = np.unique(region[order], return_index=True)
    groups = dict(zip(present.tolist(), np.split(values[order], starts[1:])))
    empty = 0.0 if reduce is np.sum else np.nan
    return np.array(
        [reduce(groups[r]) if r in groups else empty for r in regions.tolist()],
        dtype=np.float64,
    )

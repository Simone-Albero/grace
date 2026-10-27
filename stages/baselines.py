"""Per-sample failure probabilities from the classifier's own outputs.

Four predictors, each a probability that the classifier fails on a row,
estimated on val and written for val and test: `val_rate`, the global
val failure rate; `confidence`, the isotonic calibration on val of one
minus the top class probability; `binned`, the val failure rate of the
row's bin of equal-mass confidence within its predicted class (one bin
per 150 val rows, at least one per class); and `gbm`, a gradient
boosting classifier trained on the out-of-fold errors of train from the
row's coordinates, confidence and predicted class, then calibrated
isotonically on val. No test label is read. Artifact: the probabilities
of every val and test row. DB rows written: metrics val_error_rate,
n_bins and n_gbm_rows.
"""

__all__ = ["CONFIG", "INPUTS", "FILES", "OUTPUTS", "run"]

import numpy as np
import pandas as pd

from src.baselines import (
    assign_class_bins,
    bin_count,
    calibrate_isotonic,
    class_confidence_edges,
    estimate_group_rates,
    fit_error_gbm,
)
from stages import Context

CONFIG = ("seed",)
INPUTS = ("prepare", "space", "classify", "crossfit")
FILES = ()
OUTPUTS = ("scores.parquet",)
_SPLITS = ("val", "test")


def _confidence(predictions: pd.DataFrame) -> np.ndarray:
    """Return each row's top class probability."""
    proba = predictions.filter(like="proba_")
    return proba.to_numpy(dtype=np.float64).max(axis=1)


def _features(ctx: Context, split: str, predictions: pd.DataFrame) -> pd.DataFrame:
    """Return a split's gbm features: points, confidence, prediction."""
    features = ctx.split_rows("space", "z.parquet", split).filter(like="z_")
    features = features.reset_index(drop=True)
    features["confidence"] = _confidence(predictions)
    features["predicted"] = predictions["y_pred"].to_numpy()
    return features


def run(ctx: Context) -> None:
    """Estimate each predictor on val and write it for val and test.

    Args:
        ctx: The stage context.
    """
    seed = ctx.cfg.seed
    out_of_fold = ctx.split_rows("crossfit", "oof.parquet", "train")
    predictions = {
        split: ctx.split_rows("classify", "predictions.parquet", split)
        for split in _SPLITS
    }
    confidence = {split: _confidence(predictions[split]) for split in _SPLITS}
    predicted = {split: predictions[split]["y_pred"].to_numpy() for split in _SPLITS}
    val_errors = (ctx.labels("val") != predicted["val"]).astype(np.float64)

    calibrated = {
        split: calibrate_isotonic(
            1.0 - confidence["val"], val_errors, 1.0 - split_confidence
        )
        for split, split_confidence in confidence.items()
    }

    # Bins of equal val mass inside each predicted class; val and test
    # are grouped in one call so that no two classes share a group id.
    n_bins = bin_count(len(val_errors), np.unique(predicted["val"]).size)
    edges = class_confidence_edges(confidence["val"], predicted["val"], n_bins)
    group = assign_class_bins(
        np.concatenate([confidence["val"], confidence["test"]]),
        np.concatenate([predicted["val"], predicted["test"]]),
        edges,
    )
    val_group, test_group = np.split(group, [len(val_errors)])
    group_rate = estimate_group_rates(val_group, val_errors, np.unique(group))
    binned = {
        "val": group_rate.loc[val_group].to_numpy(),
        "test": group_rate.loc[test_group].to_numpy(),
    }

    # Out-of-fold errors of train teach the gbm; val calibrates it.
    train_features = _features(ctx, "train", out_of_fold)
    train_errors = (ctx.labels("train") != out_of_fold["y_pred"].to_numpy()).astype(int)
    predict_failure = fit_error_gbm(train_features, train_errors, random_state=seed)
    raw_gbm = {
        split: predict_failure(_features(ctx, split, predictions[split]))
        for split in _SPLITS
    }
    gbm = {
        split: calibrate_isotonic(raw_gbm["val"], val_errors, split_scores)
        for split, split_scores in raw_gbm.items()
    }

    val_error_rate = float(val_errors.mean())
    scores = pd.concat(
        [
            pd.DataFrame(
                {
                    "split": split,
                    "row": np.arange(len(confidence[split]), dtype=np.int32),
                    "val_rate": np.full(len(confidence[split]), val_error_rate),
                    "confidence": calibrated[split],
                    "binned": binned[split],
                    "gbm": gbm[split],
                }
            ).astype(
                {
                    "val_rate": np.float32,
                    "confidence": np.float32,
                    "binned": np.float32,
                    "gbm": np.float32,
                }
            )
            for split in _SPLITS
        ],
        ignore_index=True,
    )
    scores["split"] = pd.Categorical(scores["split"], categories=list(_SPLITS))
    scores.to_parquet(ctx.out / "scores.parquet", index=False)
    ctx.save_metrics(
        {
            "val_error_rate": val_error_rate,
            "n_bins": len(np.unique(val_group)),
            "n_gbm_rows": len(train_features),
        }
    )

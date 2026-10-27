"""One classifier trained on train, with its val and test predictions.

Artifacts: predictions (val and test) and the model. DB rows written:
`class_metrics`, `classifier_grid` (ML grid search), `history` (DL
train loss) and metrics from the test predictions (accuracy,
macro/weighted precision, recall, f1), n_fit (the rows the model fit
on, the DL early-stopping holdout excluded) and, after a grid search,
best_score.
"""

__all__ = ["config_keys", "INPUTS", "FILES", "OUTPUTS", "run"]

import json
import logging
from collections.abc import Mapping

import joblib
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.pipeline import Pipeline
from torch import nn

from src import dl, ml
from src.data import cap_per_class
from src.scores import check_probabilities, compute_classification_metrics
from stages import LABEL, SPLITS, Context

logger = logging.getLogger(__name__)

_COMMON = ("classifier", "data.max_per_class", "seed")
_ML_ONLY = ("grid_search",)
# DL-only groups never re-train an ML model.
_DL_ONLY = (
    "loss",
    "optimizer",
    "scheduler",
    "fit.training",
    # The embedding cardinalities are top_n + hash_buckets.
    "data.top_n",
    "data.hash_buckets",
)
INPUTS = ("split", "prepare")
FILES = ()
# model/ holds model.joblib (ml) or model.pt (dl).
OUTPUTS = ("predictions.parquet", "model")


def config_keys(cfg: DictConfig) -> tuple[str, ...]:
    """Return the config keys a classifier of this kind depends on.

    Args:
        cfg: The config; its `classifier.kind` picks ML or DL keys.

    Returns:
        The dotted keys of the cache key.
    """
    return _COMMON + (_DL_ONLY if cfg.classifier.kind == "dl" else _ML_ONLY)


def _classifier_params(
    ctx: Context, *, n_classes: int, n_num: int, n_cat: int
) -> dict[str, object]:
    """Return the classifier `params`: DL shapes or ML seed and jobs."""
    cfg = ctx.cfg
    params = OmegaConf.to_container(cfg.classifier.params, resolve=True)
    if cfg.classifier.kind == "dl":
        return dl.model_params(
            params,
            n_classes=n_classes,
            n_num=n_num,
            n_cat=n_cat,
            cardinality=cfg.data.top_n + cfg.data.hash_buckets,
        )
    return ml.classifier_params(
        cfg.classifier.name, params, seed=cfg.seed, n_jobs=ctx.runtime.n_jobs
    )


def _fit_ml(
    ctx: Context,
    fit_rows: pd.DataFrame,
    *,
    num_cols: list[str],
    cat_cols: list[str],
    params: Mapping[str, object],
) -> tuple[Pipeline, dict[str, object] | None]:
    """Fit the ML classifier, by grid search when its config has a grid.

    Returns:
        The model and the search summary (None without a grid).
    """
    cfg = ctx.cfg
    name = cfg.classifier.name
    X, y = fit_rows[num_cols + cat_cols], fit_rows[LABEL].to_numpy()
    if "grid" in cfg.classifier and cfg.classifier.grid:
        logger.info(
            "Grid search for %s: scoring=%s, cv=%d",
            name,
            cfg.grid_search.scoring,
            cfg.grid_search.cv,
        )
        model, summary = ml.grid_search(
            name,
            params,
            OmegaConf.to_container(cfg.classifier.grid),
            X,
            y,
            num_cols=num_cols,
            cat_cols=cat_cols,
            scoring=cfg.grid_search.scoring,
            cv=cfg.grid_search.cv,
            max_samples=cfg.grid_search.max_samples,
            n_jobs=ctx.runtime.n_jobs,
            random_state=cfg.seed,
        )
        logger.info(
            "Best params: %s | best score (%s): %.4f",
            summary["best_params"],
            cfg.grid_search.scoring,
            summary["best_score"],
        )
        return model, summary
    logger.info("Training %s ...", name)
    return ml.build_pipeline(name, params, num_cols, cat_cols).fit(X, y), None


def _dl_training(ctx: Context, params: Mapping[str, object]) -> dl.DLTraining:
    """Return how the DL classifier trains, from the config."""
    cfg, runtime = ctx.cfg, ctx.runtime
    training = cfg.fit.training
    return dl.DLTraining(
        model=cfg.classifier.name,
        model_params=params,
        loss=cfg.loss.name,
        loss_params=OmegaConf.to_container(cfg.loss.params, resolve=True),
        optimizer=cfg.optimizer.name,
        optimizer_params=OmegaConf.to_container(cfg.optimizer.params, resolve=True),
        scheduler=cfg.scheduler.name,
        scheduler_params=OmegaConf.to_container(cfg.scheduler.params, resolve=True),
        epochs=training.epochs,
        max_grad_norm=training.max_grad_norm,
        patience=training.early_stopping.patience,
        min_delta=training.early_stopping.min_delta,
        holdout_frac=training.early_stopping.holdout,
        train_batching=OmegaConf.to_container(training.dataloader),
        holdout_batching=OmegaConf.to_container(training.early_stopping.dataloader),
        num_workers=runtime.num_workers,
        pin_memory=runtime.pin_memory,
        device=runtime.device,
    )


def _predict(
    ctx: Context,
    model: Pipeline | nn.Module,
    rows: pd.DataFrame,
    split: str,
    *,
    num_cols: list[str],
    cat_cols: list[str],
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict a split, with one finite probability per class."""
    if ctx.cfg.classifier.kind == "ml":
        y_pred, y_proba = ml.predict(model, rows[num_cols + cat_cols])
    else:
        y_pred, y_proba = dl.predict(
            model,
            rows,
            num_cols=num_cols,
            cat_cols=cat_cols,
            device=torch.device(ctx.runtime.device),
        )
    check_probabilities(y_proba, n_classes, split)
    return y_pred, y_proba


def _predictions_table(
    predicted: Mapping[str, tuple[np.ndarray, np.ndarray]], n_classes: int
) -> pd.DataFrame:
    """Return the predictions of every split in one table."""
    predictions = pd.concat(
        [
            pd.DataFrame(
                {
                    "split": name,
                    "row": np.arange(len(y_pred), dtype=np.int32),
                    "y_pred": y_pred.astype(np.int32),
                    **{
                        f"proba_{class_id}": y_proba[:, class_id].astype(np.float32)
                        for class_id in range(n_classes)
                    },
                }
            )
            for name, (y_pred, y_proba) in predicted.items()
        ],
        ignore_index=True,
    )
    predictions["split"] = pd.Categorical(
        predictions["split"], categories=list(predicted)
    )
    return predictions


def _grid_table(summary: Mapping[str, object]) -> pd.DataFrame:
    """Return one row per candidate of the grid search."""
    cv_results = summary["cv_results"]
    best = summary["best_params"]
    return pd.DataFrame(
        {
            "candidate": range(len(cv_results)),
            "params": [
                json.dumps(candidate["params"], sort_keys=True)
                for candidate in cv_results
            ],
            "mean_score": [candidate["mean_test_score"] for candidate in cv_results],
            "std_score": [candidate["std_test_score"] for candidate in cv_results],
            "best": [int(candidate["params"] == best) for candidate in cv_results],
        }
    )


def run(ctx: Context) -> None:
    """Train the classifier; write its predictions and metrics.

    Args:
        ctx: The stage context.

    Raises:
        ValueError: The loss weights do not match the classes, or the
            model predicts a wrong number of probabilities or a
            non-finite one.
    """
    cfg = ctx.cfg
    num_cols, cat_cols = ctx.features()
    n_classes = len(ctx.read("classes", "prepare"))
    prepared = ctx.inputs["prepare"].dir
    train, val, test = (
        pd.read_parquet(prepared / f"{split}.parquet") for split in SPLITS
    )
    logger.info("Classifier: %s (kind=%s)", cfg.classifier.name, cfg.classifier.kind)

    fit_rows = cap_per_class(
        train, cfg.data.max_per_class, label_col=LABEL, random_state=cfg.seed
    )
    params = _classifier_params(
        ctx, n_classes=n_classes, n_num=len(num_cols), n_cat=len(cat_cols)
    )
    model_dir = ctx.out / "model"
    model_dir.mkdir()
    if cfg.classifier.kind == "ml":
        model, summary = _fit_ml(
            ctx, fit_rows, num_cols=num_cols, cat_cols=cat_cols, params=params
        )
        losses = []
        n_fit = len(fit_rows)
        joblib.dump(model, model_dir / "model.joblib")
    else:
        model, losses, n_fit = dl.fit_classifier(
            fit_rows,
            num_cols=num_cols,
            cat_cols=cat_cols,
            label_col=LABEL,
            n_classes=n_classes,
            training=_dl_training(ctx, params),
            seed=cfg.seed,
        )
        summary = None
        torch.save(
            {
                "state_dict": model.state_dict(),
                "name": cfg.classifier.name,
                "params": params,
            },
            model_dir / "model.pt",
        )

    predicted = {
        name: _predict(
            ctx,
            model,
            rows,
            name,
            num_cols=num_cols,
            cat_cols=cat_cols,
            n_classes=n_classes,
        )
        for name, rows in (("val", val), ("test", test))
    }
    _predictions_table(predicted, n_classes).to_parquet(
        ctx.out / "predictions.parquet", index=False
    )

    y_true = test[LABEL].to_numpy()
    metrics = compute_classification_metrics(y_true, predicted["test"][0])
    per_class = pd.DataFrame(metrics.pop("classes"))
    ctx.write("class_metrics", per_class)
    metrics["n_fit"] = n_fit
    if summary is not None:
        metrics["best_score"] = summary["best_score"]
        ctx.write("classifier_grid", _grid_table(summary))
    if losses:
        ctx.write("history", pd.DataFrame({"step": range(len(losses)), "loss": losses}))
    ctx.save_metrics(metrics)

"""The classifier's out-of-fold predictions on train.

Train is split in `crossfit.n_folds` stratified folds; the classifier
trained as classify trains it (the same cap, the same hyperparameters,
no new search) on all but one fold predicts that fold, so every train
row is predicted by a model that never saw it. The grid search that
chose the hyperparameters did see every train row, so the error rate is
slightly optimistic for a classifier with a grid. Artifact: the
predictions of every train row, with the fold that held it out. DB rows
written: metrics n_folds, oof_error_rate and n_fit_mean (the rows a
fold's model fit on, the DL early-stopping holdout excluded).
"""

__all__ = ["config_keys", "INPUTS", "FILES", "OUTPUTS", "run"]

import json
import logging
from collections.abc import Mapping

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from torch import nn

from src import dl, ml
from src.data import cap_per_class
from src.scores import check_probabilities
from stages import LABEL, Context

logger = logging.getLogger(__name__)

# classify's keys, less the grid search a fold never repeats, plus
# the folds.
_COMMON = ("classifier", "data.max_per_class", "seed", "crossfit.n_folds")
_DL_ONLY = (
    "loss",
    "optimizer",
    "scheduler",
    "fit.training",
    # The embedding cardinalities are top_n + hash_buckets.
    "data.top_n",
    "data.hash_buckets",
)
INPUTS = ("split", "prepare", "classify")
FILES = ()
OUTPUTS = ("oof.parquet",)


def config_keys(cfg: DictConfig) -> tuple[str, ...]:
    """Return the config keys a classifier of this kind depends on.

    Args:
        cfg: The config; its `classifier.kind` picks the DL keys or not.

    Returns:
        The dotted keys of the cache key.
    """
    return _COMMON + (_DL_ONLY if cfg.classifier.kind == "dl" else ())


def _best_params(ctx: Context) -> dict[str, object]:
    """Return the parameters classify's grid search chose, if any."""
    grid = ctx.read("classifier_grid", "classify")
    if grid.empty:
        return {}
    return json.loads(grid.loc[grid["best"] == 1, "params"].iloc[0])


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


def _fit_fold(
    ctx: Context,
    fit_rows: pd.DataFrame,
    *,
    num_cols: list[str],
    cat_cols: list[str],
    n_classes: int,
    params: Mapping[str, object],
) -> tuple[Pipeline | nn.Module, int]:
    """Train the classifier on one fold; return it and its fit rows."""
    cfg = ctx.cfg
    if cfg.classifier.kind == "ml":
        model = ml.build_pipeline(cfg.classifier.name, params, num_cols, cat_cols)
        model.fit(fit_rows[num_cols + cat_cols], fit_rows[LABEL].to_numpy())
        return model, len(fit_rows)
    model, _, n_fit = dl.fit_classifier(
        fit_rows,
        num_cols=num_cols,
        cat_cols=cat_cols,
        label_col=LABEL,
        n_classes=n_classes,
        training=_dl_training(ctx, params),
        seed=cfg.seed,
    )
    return model, n_fit


def _predict_fold(
    ctx: Context,
    model: Pipeline | nn.Module,
    rows: pd.DataFrame,
    *,
    num_cols: list[str],
    cat_cols: list[str],
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict held-out rows, with one finite probability per class."""
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
    check_probabilities(y_proba, n_classes, "a held-out fold")
    return y_pred, y_proba


def run(ctx: Context) -> None:
    """Predict every train row with models that never saw it.

    Args:
        ctx: The stage context.

    Raises:
        ValueError: A class with fewer train rows than folds, or a model
            that predicts a wrong number of probabilities or a
            non-finite one.
    """
    cfg = ctx.cfg
    n_folds = cfg.crossfit.n_folds
    num_cols, cat_cols = ctx.features()
    n_classes = len(ctx.read("classes", "prepare"))
    train = pd.read_parquet(ctx.inputs["prepare"].dir / "train.parquet")
    labels = train[LABEL].to_numpy()
    n_thinnest = int(np.bincount(labels, minlength=n_classes).min())
    if n_thinnest < n_folds:
        raise ValueError(
            f"A class has {n_thinnest} train rows for crossfit.n_folds={n_folds}: "
            "every fold needs a row of every class; raise data.min_class_count "
            "or lower crossfit.n_folds."
        )

    if cfg.classifier.kind == "ml":
        params = {
            **ml.classifier_params(
                cfg.classifier.name,
                OmegaConf.to_container(cfg.classifier.params, resolve=True),
                seed=cfg.seed,
                n_jobs=ctx.runtime.n_jobs,
            ),
            **_best_params(ctx),
        }
    else:
        params = dl.model_params(
            OmegaConf.to_container(cfg.classifier.params, resolve=True),
            n_classes=n_classes,
            n_num=len(num_cols),
            n_cat=len(cat_cols),
            cardinality=cfg.data.top_n + cfg.data.hash_buckets,
        )

    held_out_fold = np.empty(len(train), dtype=np.int8)
    y_pred = np.empty(len(train), dtype=np.int32)
    y_proba = np.empty((len(train), n_classes), dtype=np.float32)
    n_fit_by_fold = []
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cfg.seed)
    for fold, (fit_positions, held_positions) in enumerate(
        folds.split(np.zeros(len(train)), labels)
    ):
        logger.info("Crossfit fold %d of %d", fold + 1, n_folds)
        fit_rows = cap_per_class(
            train.iloc[fit_positions],
            cfg.data.max_per_class,
            label_col=LABEL,
            random_state=cfg.seed,
        )
        model, n_fit = _fit_fold(
            ctx,
            fit_rows,
            num_cols=num_cols,
            cat_cols=cat_cols,
            n_classes=n_classes,
            params=params,
        )
        n_fit_by_fold.append(n_fit)
        held_pred, held_proba = _predict_fold(
            ctx,
            model,
            train.iloc[held_positions],
            num_cols=num_cols,
            cat_cols=cat_cols,
            n_classes=n_classes,
        )
        held_out_fold[held_positions] = fold
        y_pred[held_positions] = held_pred
        y_proba[held_positions] = held_proba

    predictions = pd.DataFrame(
        {
            "split": pd.Categorical(["train"] * len(train), categories=["train"]),
            "row": np.arange(len(train), dtype=np.int32),
            "fold": held_out_fold,
            "y_pred": y_pred,
            **{f"proba_{k}": y_proba[:, k] for k in range(n_classes)},
        }
    )
    predictions.to_parquet(ctx.out / "oof.parquet", index=False)
    ctx.save_metrics(
        {
            "n_folds": n_folds,
            "oof_error_rate": float(np.mean(y_pred != labels)),
            "n_fit_mean": float(np.mean(n_fit_by_fold)),
        }
    )

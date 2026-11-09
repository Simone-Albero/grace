"""Stage train: fit one classifier on train, predict val and test."""

from __future__ import annotations

__all__ = ["run"]

import sqlite3

import joblib
import numpy as np
import polars as pl

from src import db
from src.config import DatasetConfig
from src.data import SPLITS
from src.evaluation import classification_metrics, confusion
from src.models import build_pipeline, predict_proba, undersample
from src.prepare import LABEL
from stages.paths import model_file, predictions_file, read_classes, read_split


def run(cfg: DatasetConfig, conn: sqlite3.Connection, run_id: int) -> None:
    """Fit cfg.model on train and predict val and test.

    Train is undersampled to the smallest class when method.balance is
    "undersample" (models.undersample, seed). The pipeline of
    models.build_pipeline reads the prepared columns; its classes are
    the class codes. pred = argmax of the probabilities (ties: the lower
    code), confidence = their maximum.

    Writes to output_dir: models/<model>.joblib and
    pred_<model>_{val,test}.parquet (row, label, pred, confidence,
    proba_0..proba_{K-1}; row = position in the split).
    Logs: model_metrics (split, label, metric, value: evaluation
    .classification_metrics of val and of test), confusion (split, true,
    pred, count: the non-zero cells), train_stats (one row: n_train,
    n_fit, balance).

    Raises:
        ValueError: When cfg.model is None.
    """
    if cfg.model is None:
        raise ValueError(f"train of {cfg.name} needs a model, got none")
    name = cfg.model.name
    splits = {split: read_split(cfg, split) for split in SPLITS}
    classes = read_classes(cfg)
    features = [*cfg.num_cols, *cfg.cat_cols]
    labels = splits["train"][LABEL].to_numpy()
    fit_rows = np.arange(len(labels))
    if cfg.method.balance == "undersample":
        fit_rows = undersample(labels, cfg.seed)
    print(
        f"[train] fitting {name} on {len(fit_rows)} of {len(labels)} rows", flush=True
    )
    model = build_pipeline(name, cfg.model.params, cfg.num_cols, cfg.cat_cols, cfg.seed)
    X = splits["train"][fit_rows].select(features).to_pandas()
    model.fit(X, labels[fit_rows])
    path = model_file(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

    metric_rows, confusion_rows = [], []
    for split in ("val", "test"):
        frame = splits[split]
        proba = predict_proba(model, frame.select(features).to_pandas(), len(classes))
        predicted = proba.argmax(axis=1)
        truth = frame[LABEL].to_numpy()
        pl.DataFrame(
            {
                "row": np.arange(frame.height),
                "label": truth,
                "pred": predicted,
                "confidence": proba.max(axis=1),
                **{f"proba_{j}": proba[:, j] for j in range(len(classes))},
            }
        ).write_parquet(predictions_file(cfg, split))
        metrics = classification_metrics(truth, predicted, classes)
        metric_rows += [{"split": split, **row} for row in metrics]
        counts = confusion(truth, predicted, len(classes))
        confusion_rows += [
            {
                "split": split,
                "true": classes[true_code],
                "pred": classes[pred_code],
                "count": int(count),
            }
            for (true_code, pred_code), count in np.ndenumerate(counts)
            if count
        ]
        print(f"[train] {split}: accuracy {metrics[0]['value']:.4f}", flush=True)

    db.log_table(conn, run_id, "model_metrics", metric_rows)
    db.log_table(conn, run_id, "confusion", confusion_rows)
    db.log_table(
        conn,
        run_id,
        "train_stats",
        [
            {
                "n_train": len(labels),
                "n_fit": len(fit_rows),
                "balance": cfg.method.balance,
            }
        ],
    )

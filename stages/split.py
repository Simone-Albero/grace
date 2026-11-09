"""Stage split: clean, split and prepare the rows."""

from __future__ import annotations

__all__ = ["run"]

import json
import sqlite3

import joblib
import polars as pl

from src import db
from src.config import DatasetConfig
from src.data import SPLITS, drop_incomplete, drop_rare_classes, stratified_split
from src.prepare import LABEL, Preprocessor
from stages.paths import classes_file, preprocessor_file, split_file


def _read(cfg: DatasetConfig) -> pl.DataFrame:
    columns = [*cfg.num_cols, *cfg.cat_cols, cfg.label_col]
    scan = pl.scan_csv(cfg.path, infer_schema=False)
    missing = sorted(set(columns) - set(scan.collect_schema().names()))
    if missing:
        raise ValueError(f"columns missing from {cfg.path}: {missing}")
    numeric = pl.col(cfg.num_cols).cast(pl.Float64, strict=False)
    return scan.select(columns).with_columns(numeric).collect()


def run(cfg: DatasetConfig, conn: sqlite3.Connection, run_id: int) -> None:
    """Read the CSV, clean it, split it and prepare every split.

    Order: read num_cols (Float64), cat_cols and label_col (text) of
    cfg.path -> data.drop_incomplete ->
    data.drop_rare_classes(min_class_count) ->
    data.stratified_split(cfg.split, seed) -> prepare.Preprocessor
    fitted on train -> transform of every split.

    Writes to output_dir: {train,val,test}.parquet (num_cols, cat_cols,
    "label" as class code; row order = the split's order, so a row's
    position is its id everywhere downstream), classes.json,
    preprocessor.joblib.
    Logs: split_summary (one row: n_raw, n_incomplete, n_rare, n_kept,
    n_classes), dropped_classes (label, n_rows), split_class_counts
    (label, train, val, test), features (column, kind "num" | "cat").

    Raises:
        ValueError: When a configured column is missing from the CSV, or
            no class survives.
    """
    print(f"[split] reading {cfg.path}", flush=True)
    raw = _read(cfg)
    complete = drop_incomplete(raw, cfg.num_cols)
    kept, dropped = drop_rare_classes(complete, cfg.label_col, cfg.min_class_count)
    if kept.is_empty():
        raise ValueError(
            f"no class has min_class_count={cfg.min_class_count} complete rows"
        )
    print(
        f"[split] {raw.height} rows, {raw.height - complete.height} incomplete, "
        f"{complete.height - kept.height} in {len(dropped)} rare classes",
        flush=True,
    )
    shares = {"train": cfg.split.train, "val": cfg.split.val, "test": cfg.split.test}
    parts = stratified_split(kept, cfg.label_col, shares, cfg.seed)
    preprocessor = Preprocessor(
        cfg.num_cols, cfg.cat_cols, cfg.label_col, cfg.top_n, cfg.hash_buckets
    ).fit(parts["train"])

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name in SPLITS:
        prepared = preprocessor.transform(parts[name])
        prepared.write_parquet(split_file(cfg, name))
        counts[name] = prepared[LABEL].value_counts()
        print(f"[split] saved {name}: {prepared.height} rows", flush=True)
    joblib.dump(preprocessor, preprocessor_file(cfg))
    classes_file(cfg).write_text(json.dumps(list(preprocessor.classes_)))

    complete_counts = complete[cfg.label_col].value_counts()
    n_complete = dict(zip(*complete_counts.to_dict(as_series=False).values()))
    db.log_table(
        conn,
        run_id,
        "split_summary",
        [
            {
                "n_raw": raw.height,
                "n_incomplete": raw.height - complete.height,
                "n_rare": complete.height - kept.height,
                "n_kept": kept.height,
                "n_classes": len(preprocessor.classes_),
            }
        ],
    )
    db.log_table(
        conn,
        run_id,
        "dropped_classes",
        [{"label": label, "n_rows": n_complete[label]} for label in dropped],
    )
    per_split = {
        name: dict(zip(*split_counts.to_dict(as_series=False).values()))
        for name, split_counts in counts.items()
    }
    db.log_table(
        conn,
        run_id,
        "split_class_counts",
        [
            {
                "label": label,
                **{split: per_split[split].get(code, 0) for split in SPLITS},
            }
            for code, label in enumerate(preprocessor.classes_)
        ],
    )
    db.log_table(
        conn,
        run_id,
        "features",
        [{"column": column, "kind": "num"} for column in cfg.num_cols]
        + [{"column": column, "kind": "cat"} for column in cfg.cat_cols],
    )

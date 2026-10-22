"""The cleaned raw CSV, split into train, val and test (no fitting).

DB rows written: `features` (num columns then cat columns) and metrics
n_raw_rows, n_raw_columns, n_train, n_val, n_test.
"""

__all__ = ["CONFIG", "INPUTS", "FILES", "OUTPUTS", "run"]

import logging
from pathlib import Path

import pandas as pd

from src.data import drop_nans, drop_rare_classes, ml_split
from stages import LABEL, SPLITS, Context

logger = logging.getLogger(__name__)

CONFIG = (
    "data.path",
    "data.label_col",
    "data.num_cols",
    "data.cat_cols",
    "data.min_class_count",
    "data.train_frac",
    "data.val_frac",
    "data.test_frac",
    "seed",
)
INPUTS = ()
# A changed raw CSV invalidates the cache.
FILES = ("data.path",)
OUTPUTS = tuple(f"{name}.parquet" for name in SPLITS)


def run(ctx: Context) -> None:
    """Write the stratified splits of the cleaned raw data.

    Args:
        ctx: The stage context.

    Raises:
        FileNotFoundError: The raw CSV does not exist.
        ValueError: The CSV lacks a configured column or has no row.
    """
    data = ctx.cfg.data
    path = Path(data.path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data {str(path)!r} not found (synthetic_test: run `make generate`)."
        )
    num_cols, cat_cols = list(data.num_cols), list(data.cat_cols)
    columns = [*num_cols, *cat_cols, data.label_col]

    logger.info("Loading %s ...", path)
    header = pd.read_csv(path, nrows=0).columns
    missing = [name for name in columns if name not in header]
    if missing:
        raise ValueError(
            f"{str(path)!r} lacks the columns {missing}: fix the CSV, or "
            "data.num_cols, data.cat_cols and data.label_col in the config."
        )
    n_raw_columns = len(header)
    raw = pd.read_csv(path, usecols=columns)
    n_raw_rows = len(raw)
    if n_raw_rows == 0:
        raise ValueError(f"{str(path)!r} has a header but no row.")
    logger.info("Raw data loaded: %d rows, %d columns", n_raw_rows, n_raw_columns)

    cleaned = drop_nans(raw, columns)
    cleaned = drop_rare_classes(cleaned, data.label_col, min_count=data.min_class_count)
    parts = ml_split(
        cleaned,
        train_frac=data.train_frac,
        val_frac=data.val_frac,
        test_frac=data.test_frac,
        random_state=ctx.cfg.seed,
        label_col=data.label_col,
    )

    metrics = {"n_raw_rows": n_raw_rows, "n_raw_columns": n_raw_columns}
    for name, part in zip(SPLITS, parts):
        part = part[columns].rename(columns={data.label_col: LABEL})
        part = part.reset_index(drop=True)
        part.to_parquet(ctx.out / f"{name}.parquet", index=False)
        metrics[f"n_{name}"] = len(part)
        logger.info("Split %s: %d rows", name, len(part))

    names = [*num_cols, *cat_cols]
    ctx.write(
        "features",
        pd.DataFrame(
            {
                "position": range(len(names)),
                "name": names,
                "kind": ["num"] * len(num_cols) + ["cat"] * len(cat_cols),
            }
        ),
    )
    ctx.save_metrics(metrics)

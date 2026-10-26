"""Empirical failure rate of every region, measured on val and test.

DB rows written: `failures` (every region; `val_rate`/`test_rate` NULL
when the split has no row in it) and metrics val_error_rate,
test_error_rate, n_regions_with_val and n_regions_with_test.
"""

__all__ = ["CONFIG", "INPUTS", "FILES", "OUTPUTS", "run"]

import pandas as pd

from stages import Context

CONFIG = ()
INPUTS = ("prepare", "regions", "classify")
FILES = ()
# The stage writes DB rows only.
OUTPUTS = ()


def _counts(ctx: Context, split: str) -> pd.DataFrame:
    """Return each region's row and error count of one split.

    Raises:
        ValueError: The labels, regions and predictions of the split
            differ in length.
    """
    y_true = ctx.labels(split)
    assignments = ctx.split_rows("regions", "assignments.parquet", split, ["region_id"])
    region = assignments["region_id"].to_numpy()
    predicted = ctx.split_rows("classify", "predictions.parquet", split, ["y_pred"])
    y_pred = predicted["y_pred"].to_numpy()
    if not len(y_true) == len(region) == len(y_pred):
        raise ValueError(
            f"{split.capitalize()} rows disagree: {len(y_true)} labels, "
            f"{len(region)} routed by regions, {len(y_pred)} predicted by classify. "
            "Re-run `python -m stages regions classify --force`."
        )
    return (
        pd.DataFrame({"region_id": region, "error": y_true != y_pred})
        .groupby("region_id")["error"]
        .agg(n_rows="size", n_errors="sum")
    )


def run(ctx: Context) -> None:
    """Count the val and test errors of every region.

    Args:
        ctx: The stage context.

    Raises:
        ValueError: The labels, regions and predictions of a split
            differ in length.
    """
    table = ctx.read("regions", "regions")[["region_id"]].set_index("region_id")
    metrics: dict[str, float] = {}
    for split in ("val", "test"):
        counts = _counts(ctx, split).reindex(table.index).fillna(0).astype(int)
        n_rows, n_errors = counts["n_rows"], counts["n_errors"]
        table[f"n_{split}"] = n_rows
        table[f"n_{split}_error"] = n_errors
        table[f"{split}_rate"] = (n_errors / n_rows).where(n_rows > 0)
        metrics[f"{split}_error_rate"] = int(n_errors.sum()) / int(n_rows.sum())
        metrics[f"n_regions_with_{split}"] = int((n_rows > 0).sum())

    ctx.write("failures", table.reset_index())
    ctx.save_metrics(metrics)

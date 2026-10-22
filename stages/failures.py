"""Empirical failure rate of every region, measured on the test split.

DB rows written: `failures` (every region, NULL rate when no test row
fell in it) and metrics n_eval, n_error, error_rate and
n_regions_evaluated.
"""

__all__ = ["CONFIG", "INPUTS", "FILES", "OUTPUTS", "run"]

import pandas as pd

from stages import Context

CONFIG = ()
INPUTS = ("prepare", "regions", "classify")
FILES = ()
# The stage writes DB rows only.
OUTPUTS = ()


def run(ctx: Context) -> None:
    """Count the test errors of every region.

    Args:
        ctx: The stage context.

    Raises:
        ValueError: The labels, regions and predictions of the test
            split differ in length.
    """
    y_true = ctx.labels("test")
    assignments = ctx.split_rows(
        "regions", "assignments.parquet", "test", ["region_id"]
    )
    region = assignments["region_id"].to_numpy()
    predicted = ctx.split_rows("classify", "predictions.parquet", "test", ["y_pred"])
    y_pred = predicted["y_pred"].to_numpy()
    if not len(y_true) == len(region) == len(y_pred):
        raise ValueError(
            f"Test rows disagree: {len(y_true)} labels, {len(region)} routed by "
            f"regions, {len(y_pred)} predicted by classify. "
            "Re-run `python -m stages regions classify --force`."
        )

    counts = (
        pd.DataFrame({"region_id": region, "error": y_true != y_pred})
        .groupby("region_id")["error"]
        .agg(n_eval="size", n_error="sum")
    )
    table = (
        ctx.read("regions", "regions")[["region_id"]]
        .join(counts, on="region_id")
        .fillna({"n_eval": 0, "n_error": 0})
        .astype({"n_eval": int, "n_error": int})
    )
    table["failure_rate"] = (table["n_error"] / table["n_eval"]).where(
        table["n_eval"] > 0
    )
    ctx.write("failures", table)
    n_eval, n_error = int(table["n_eval"].sum()), int(table["n_error"].sum())
    ctx.save_metrics(
        {
            "n_eval": n_eval,
            "n_error": n_error,
            "error_rate": n_error / n_eval,
            "n_regions_evaluated": int((table["n_eval"] > 0).sum()),
        }
    )

"""MCP and ATC failure-rate baselines of every region, scored on test.

Both predict a region's failure rate from the classifier's
probabilities on the test rows routed to it, the ATC threshold fit on
val. A region is scored (`used`) when at least
`failure_regressor.min_eval_support` test rows fell in it: the regressor
is scored on the same regions. DB rows written: `baselines` (every
region) and the scores of MCP and ATC in metrics.
"""

__all__ = ["CONFIG", "INPUTS", "FILES", "OUTPUTS", "run"]

import logging

import numpy as np
import pandas as pd

from src.baselines import atc_rate, atc_threshold, mcp_rate
from src.scores import log_scores, rate_scores
from stages import Context

logger = logging.getLogger(__name__)

CONFIG = ("failure_regressor.min_eval_support",)
INPUTS = ("prepare", "regions", "classify", "failures")
FILES = ()
# The stage writes DB rows only.
OUTPUTS = ()


def _rates(ctx: Context) -> tuple[float, pd.Series, pd.Series]:
    """Return the ATC threshold and the MCP and ATC rates on test."""
    val = ctx.split_rows("classify", "predictions.parquet", "val")
    test = ctx.split_rows("classify", "predictions.parquet", "test")
    region = ctx.split_rows("regions", "assignments.parquet", "test", ["region_id"])
    region = region["region_id"]
    val_labels = ctx.labels("val")
    if len(val) != len(val_labels):
        raise ValueError(
            f"Rows disagree: {len(val_labels)} val labels for {len(val)} predictions. "
            "Re-run `python -m stages prepare classify --force`."
        )
    proba_columns = [col for col in test.columns if col.startswith("proba_")]
    val_confidence = val[proba_columns].to_numpy(np.float64).max(axis=1)
    test_confidence = test[proba_columns].to_numpy(np.float64).max(axis=1)
    threshold = atc_threshold(val_confidence, val_labels == val["y_pred"].to_numpy())
    return (
        threshold,
        mcp_rate(test_confidence, region.to_numpy()),
        atc_rate(test_confidence, region.to_numpy(), threshold=threshold),
    )


def _used_regions(
    failures: pd.DataFrame, min_eval_support: int
) -> tuple[pd.Index, pd.Series, pd.Series]:
    """Pick the regions with enough test rows to be scored.

    Args:
        failures: The failures table, indexed by region id.
        min_eval_support: The fewest test rows a scored region has.

    Returns:
        The ids of the used regions, and the masks of the regions with
        no test row and with fewer than `min_eval_support`.

    Raises:
        ValueError: No region has enough test rows.
    """
    # A region can end up with no test row, e.g. a small one.
    no_eval = failures["n_eval"] == 0
    low_support = ~no_eval & (failures["n_eval"] < min_eval_support)
    used = failures.index[~no_eval & ~low_support]
    n_excluded = int(no_eval.sum() + low_support.sum())
    # Losing over a fifth of the regions earns a warning: routine on a
    # small test split, whose rows starve per-region support.
    log = logger.warning if n_excluded > 0.2 * len(failures) else logger.info
    log(
        "%d/%d regions used (no test row: %d, fewer than %d: %d)",
        len(used),
        len(failures),
        no_eval.sum(),
        min_eval_support,
        low_support.sum(),
    )
    if used.empty:
        raise ValueError(
            f"No region has {min_eval_support} test rows or more "
            "(failure_regressor.min_eval_support). Lower it or evaluate on more "
            "test rows."
        )
    return used, no_eval, low_support


def run(ctx: Context) -> None:
    """Compute the MCP and ATC rate of each region and score them.

    Args:
        ctx: The stage context.

    Raises:
        ValueError: If the val or test rows of classify, regions and
            prepare disagree, or if no region has enough test rows.
    """
    failures = ctx.read("failures", "failures").set_index("region_id")
    used, no_eval, low_support = _used_regions(
        failures, ctx.cfg.failure_regressor.min_eval_support
    )

    threshold, mcp, atc = _rates(ctx)
    table = pd.DataFrame(
        {"used": failures.index.isin(used).astype(int)}, index=failures.index
    )
    table["mcp"] = mcp.reindex(failures.index)
    table["atc"] = atc.reindex(failures.index)
    ctx.write("baselines", table.reset_index())

    y = failures.loc[used, "failure_rate"].astype(float)
    scores = {
        name: rate_scores(rate.loc[used], y)
        for name, rate in (("mcp", mcp), ("atc", atc))
    }
    for name, score in scores.items():
        log_scores(name, score)

    metrics = {}
    for predictor, score in scores.items():
        metrics.update({f"{predictor}_{name}": value for name, value in score.items()})
    n_eval = failures.loc[used, "n_eval"].sum()
    ctx.save_metrics(
        {
            **metrics,
            "atc_threshold": threshold,
            "n_regions": len(failures),
            "n_regions_used": len(used),
            "n_excluded_no_eval": no_eval.sum(),
            "n_excluded_low_support": low_support.sum(),
            "error_rate_used": failures.loc[used, "n_error"].sum() / n_eval,
        }
    )

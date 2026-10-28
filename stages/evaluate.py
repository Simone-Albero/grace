"""Per-sample scores of every failure predictor against the test errors.

The only stage that reads the test labels to judge, besides classify's
own classification metrics. Predictors: `tree` (the row's corrected
probability), `tree_leaf` (its leaf's rate), and baselines' `val_rate`,
`confidence`, `binned` and `gbm`. Each is scored per test row, by the
mean squared error of its probability against the row's failure over
the one of the constant val rate (`mse_ratio`), the same over bins of
equal size by the probability (`binned_mse_ratio`), the AUROC and the
average precision over the failure rate (`ap_lift`). DB rows written:
`leaf_tests` (the test rows and failures of every leaf) and metrics
`<predictor>_<score>` of those four, plus test_error_rate and n_test.
"""

__all__ = ["CONFIG", "INPUTS", "FILES", "OUTPUTS", "run"]

import numpy as np
import pandas as pd

from src.scores import ap_lift, auroc, binned_mse_ratio
from stages import Context

CONFIG = ()
INPUTS = ("prepare", "classify", "tree", "baselines")
FILES = ()
# The stage writes DB rows only.
OUTPUTS = ()
_BASELINES = ("val_rate", "confidence", "binned", "gbm")
_N_BINS = 10


def _probabilities(ctx: Context) -> dict[str, np.ndarray]:
    """Return every predictor's probability for every test row."""
    tree = ctx.split_rows("tree", "rows.parquet", "test", ["q_leaf", "q"])
    scores = ctx.split_rows("baselines", "scores.parquet", "test", list(_BASELINES))
    return {
        "tree": tree["q"].to_numpy(dtype=np.float64),
        "tree_leaf": tree["q_leaf"].to_numpy(dtype=np.float64),
        **{name: scores[name].to_numpy(dtype=np.float64) for name in _BASELINES},
    }


def run(ctx: Context) -> None:
    """Score every predictor per test row and each leaf's test rate.

    Args:
        ctx: The stage context.

    Raises:
        ValueError: The test labels and predictions disagree in length.
    """
    predicted = ctx.split_rows("classify", "predictions.parquet", "test", ["y_pred"])
    labels = ctx.labels("test")
    if len(predicted) != len(labels):
        raise ValueError(
            f"Test rows disagree: {len(labels)} labels, {len(predicted)} "
            "predicted by classify. Re-run `python -m stages prepare classify "
            "--force`."
        )
    errors = (labels != predicted["y_pred"].to_numpy()).astype(np.float64)
    probabilities = _probabilities(ctx)
    for name, values in probabilities.items():
        if len(values) != len(errors):
            raise ValueError(
                f"{name} has {len(values)} test probabilities for "
                f"{len(errors)} test rows."
            )

    val_rate_mse = float(np.mean((probabilities["val_rate"] - errors) ** 2))
    metrics = {"test_error_rate": float(errors.mean()), "n_test": len(errors)}
    for name, values in probabilities.items():
        mse = float(np.mean((values - errors) ** 2))
        # A constant that is exact on test leaves no ratio to take.
        metrics[f"{name}_mse_ratio"] = (
            mse / val_rate_mse if val_rate_mse > 0 else float("nan")
        )
        metrics[f"{name}_binned_mse_ratio"] = binned_mse_ratio(
            values, errors, probabilities["val_rate"], _N_BINS
        )
        metrics[f"{name}_auroc"] = auroc(values, errors)
        metrics[f"{name}_ap_lift"] = ap_lift(values, errors)
    ctx.save_metrics(metrics)

    tree = ctx.split_rows("tree", "rows.parquet", "test", ["leaf", "q"])
    per_leaf = (
        pd.DataFrame({"leaf": tree["leaf"].to_numpy(), "error": errors, "q": tree["q"]})
        .groupby("leaf")
        .agg(
            n_test=("error", "size"),
            n_test_errors=("error", "sum"),
            mean_q=("q", "mean"),
        )
    )
    nodes = ctx.read("tree_nodes", "tree")
    leaves = per_leaf.reindex(nodes.loc[nodes["status"] == "leaf", "node_id"])
    leaves["n_test"] = leaves["n_test"].fillna(0).astype(int)
    leaves["n_test_errors"] = leaves["n_test_errors"].fillna(0).astype(int)
    leaves["test_rate"] = leaves["n_test_errors"] / leaves["n_test"].where(
        leaves["n_test"] > 0
    )
    ctx.write("leaf_tests", leaves.rename_axis("node_id").reset_index())

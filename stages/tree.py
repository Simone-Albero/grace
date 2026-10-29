"""Regions as the leaves of a tree, and each row's failure probability.

Val is halved at random, stratified by class. The tree is grown with
k-means splits of 2 to `tree.max_k` parts on the LDA space plus one
axis, the share of nearest train rows that disagree with the predicted
class (one root per predicted class), on the first half of val, where
the classifier's errors say where to stop and how deep to cut: every
leaf costs a price on top of its expected Brier score, so a split is
kept only when it saves more than the leaves it adds. The leaves take
the failure rate of the second half that reaches them, shrunk toward the
error rate of the first half. A per-row correction by the classifier's
confidence gives every row its probability. Test labels are never read.
Artifacts: the tree and its estimates, and every val and test row's
leaf, disagreement, confidence, leaf rate and probability. DB rows
written: `tree_nodes` and metrics n_nodes, n_leaves, n_empty_roots,
max_depth, n_grow, n_estimate, grow_error_rate, estimate_error_rate,
fh_a, fh_b, fh_tau2, beta, mu_d, sigma_d, d_scale, mu_c, sigma_c and
n_leaves_without_estimate.
"""

__all__ = ["CONFIG", "INPUTS", "FILES", "OUTPUTS", "run"]

import logging
from collections.abc import Mapping

import joblib
import numpy as np
import pandas as pd

from src import tree as trees
from src.data import halve
from stages import SPLITS, Context

logger = logging.getLogger(__name__)

CONFIG = ("tree", "seed")
INPUTS = ("prepare", "space", "classify")
FILES = ()
OUTPUTS = ("tree.joblib", "rows.parquet")
_PREDICTED_SPLITS = SPLITS[1:]


def _read_points(ctx: Context, split: str) -> np.ndarray:
    """Return a split's points in the LDA space, from space."""
    rows = ctx.split_rows("space", "z.parquet", split)
    columns = sorted(
        (name for name in rows.columns if name.startswith("z_")),
        key=lambda name: int(name[2:]),
    )
    return rows[columns].to_numpy(dtype=np.float64)


def _read_predictions(ctx: Context, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Return a split's predictions and top probabilities."""
    rows = ctx.split_rows("classify", "predictions.parquet", split)
    probabilities = rows.filter(like="proba_").to_numpy(dtype=np.float64)
    return rows["y_pred"].to_numpy(), probabilities.max(axis=1)


def _standardize(
    by_split: Mapping[str, np.ndarray], is_grow: np.ndarray
) -> tuple[dict[str, np.ndarray], float, float]:
    """Centre and scale every split by the grown half's mean and std.

    Args:
        by_split: One array per predicted split.
        is_grow: Whether each val row is in the half the tree grows on.

    Returns:
        The standardized splits, all zeros when the std of the grown
        half is at most 1e-12, then the mean and std of the grown half.
    """
    grown = by_split["val"][is_grow]
    mean, std = float(grown.mean()), float(grown.std())
    return (
        {
            split: (values - mean) / std if std > 1e-12 else np.zeros(len(values))
            for split, values in by_split.items()
        },
        mean,
        std,
    )


def _nan_to_none(values: np.ndarray) -> list[object]:
    """Return values as a list with None for NaN, for NULL."""
    return [None if np.isnan(value) else value for value in values]


def _node_table(
    tree: trees.Tree,
    *,
    is_leaf: np.ndarray,
    is_kept: np.ndarray,
    n_grow: np.ndarray,
    n_grow_errors: np.ndarray,
    cost: np.ndarray,
    best: np.ndarray,
    n_estimate: np.ndarray,
    n_estimate_errors: np.ndarray,
    node_rate: np.ndarray,
    estimate_weight: np.ndarray,
) -> pd.DataFrame:
    """Return the `tree_nodes` rows, one per node of the grown tree."""
    status = np.where(is_leaf, "leaf", np.where(is_kept, "internal", "pruned"))
    return pd.DataFrame(
        {
            "node_id": np.arange(len(tree.parent)),
            "parent_id": [
                None if parent < 0 else int(parent) for parent in tree.parent
            ],
            "class_id": tree.class_id,
            "depth": trees.node_depths(tree),
            "n_grow": n_grow.astype(np.int64),
            "n_grow_errors": n_grow_errors.astype(np.int64),
            "cost": cost,
            "best": best,
            "status": status,
            "n_estimate": [
                None if node_status == "pruned" else int(count)
                for node_status, count in zip(status, n_estimate, strict=True)
            ],
            "n_estimate_errors": [
                None if node_status == "pruned" else int(count)
                for node_status, count in zip(status, n_estimate_errors, strict=True)
            ],
            "q": _nan_to_none(np.where(is_leaf, node_rate, np.nan)),
            "gamma": _nan_to_none(np.where(is_leaf, estimate_weight, np.nan)),
        }
    )


def run(ctx: Context) -> None:
    """Grow, cut and estimate the tree; write every row's probability.

    Args:
        ctx: The stage context.

    Raises:
        ValueError: A class has fewer than two val rows, so val cannot
            be halved with the class on both halves.
    """
    cfg = ctx.cfg
    n_classes = len(ctx.read("classes", "prepare"))
    y_train, y_val = ctx.labels("train"), ctx.labels("val")
    points = {split: _read_points(ctx, split) for split in SPLITS}
    predicted, confidence = {}, {}
    for split in _PREDICTED_SPLITS:
        predicted[split], confidence[split] = _read_predictions(ctx, split)
    val_errors = (y_val != predicted["val"]).astype(np.float64)
    is_grow = halve(y_val, random_state=cfg.seed)

    neighbour_index = trees.build_neighbours(points["train"], n_jobs=ctx.runtime.n_jobs)
    n_neighbours = min(cfg.tree.k_neighbours, len(y_train))
    disagreement = {
        split: trees.neighbour_disagreement(
            neighbour_index,
            y_train,
            predicted[split],
            points[split],
            n_neighbours=n_neighbours,
        )
        for split in _PREDICTED_SPLITS
    }
    standardized_disagreement, disagreement_mean, disagreement_std = _standardize(
        disagreement, is_grow
    )

    # The disagreement is one more axis, weighted in units of the spread
    # of the LDA space. Without weight the axis is left out, so that the
    # geometric tree is grown on exactly its own points.
    disagreement_scale = cfg.tree.d_weight * float(
        np.sqrt(points["train"].var(axis=0).sum())
    )
    tree_points = points
    if disagreement_scale > 0:
        tree_points = {
            split: np.column_stack(
                [points[split], disagreement_scale * standardized_disagreement[split]]
            )
            for split in _PREDICTED_SPLITS
        }

    penalty = trees.leaf_penalty(val_errors[is_grow], cfg.tree.leaf_cost)
    tree, node_of_row = trees.grow_tree(
        tree_points["val"][is_grow],
        predicted["val"][is_grow],
        val_errors[is_grow],
        tree_points["val"][~is_grow],
        predicted["val"][~is_grow],
        n_classes=n_classes,
        n_min=cfg.tree.n_min,
        max_k=cfg.tree.max_k,
        penalty=penalty,
        random_state=cfg.seed,
    )
    n_grow = trees.subtree_counts(tree, node_of_row, np.ones(is_grow.sum()))
    n_grow_errors = trees.subtree_counts(tree, node_of_row, val_errors[is_grow])
    # The estimation rows are routed without their labels, to cost the
    # noise of the rate each node would get.
    estimate_node_of_row = trees.descend(
        tree, tree_points["val"][~is_grow], predicted["val"][~is_grow]
    )
    is_leaf, is_kept, cost, best = trees.prune_tree(
        tree,
        n_grow,
        n_grow_errors,
        trees.subtree_counts(
            tree, estimate_node_of_row, np.ones(len(estimate_node_of_row))
        ),
        penalty=penalty,
    )
    leaf_of_row = {
        split: trees.descend(tree, tree_points[split], predicted[split], stop=is_leaf)
        for split in _PREDICTED_SPLITS
    }

    # The labels of the estimation half are used only now, to estimate
    # the leaves.
    n_nodes = len(tree.parent)
    estimate_leaves = leaf_of_row["val"][~is_grow]
    estimate_errors = val_errors[~is_grow]
    n_estimate = np.bincount(estimate_leaves, minlength=n_nodes).astype(np.float64)
    n_estimate_errors = np.bincount(
        estimate_leaves, weights=estimate_errors, minlength=n_nodes
    )
    grow_error_rate = val_errors[is_grow].mean()
    covariate = np.where(
        n_grow > 0, (n_grow_errors + 1) / (n_grow + 2), grow_error_rate
    )
    has_estimate_rows = is_leaf & (n_estimate > 0)
    intercept, slope, tau2 = trees.fit_fay_herriot(
        n_estimate_errors[has_estimate_rows],
        n_estimate[has_estimate_rows],
        covariate[has_estimate_rows],
    )
    node_rate, estimate_weight = trees.shrink_rates(
        n_estimate_errors,
        n_estimate,
        covariate,
        intercept=intercept,
        slope=slope,
        tau2=tau2,
    )

    standardized_low_confidence, low_confidence_mean, low_confidence_std = _standardize(
        {split: trees.low_confidence(confidence[split]) for split in _PREDICTED_SPLITS},
        is_grow,
    )
    beta = trees.fit_beta(
        node_rate[estimate_leaves],
        standardized_low_confidence["val"][~is_grow],
        estimate_errors,
    )

    leaf_rate = {split: node_rate[leaf_of_row[split]] for split in _PREDICTED_SPLITS}
    rows = pd.concat(
        [
            pd.DataFrame(
                {
                    "split": split,
                    "row": np.arange(len(points[split]), dtype=np.int32),
                    "grow": is_grow if split == "val" else False,
                    "leaf": leaf_of_row[split].astype(np.int32),
                    "d": disagreement[split].astype(np.float32),
                    "confidence": confidence[split].astype(np.float32),
                    "q_leaf": leaf_rate[split].astype(np.float32),
                    "q": trees.correct_rates(
                        leaf_rate[split], standardized_low_confidence[split], beta
                    ).astype(np.float32),
                }
            )
            for split in _PREDICTED_SPLITS
        ],
        ignore_index=True,
    )
    rows["split"] = pd.Categorical(rows["split"], categories=_PREDICTED_SPLITS)
    rows.to_parquet(ctx.out / "rows.parquet", index=False)
    joblib.dump(
        {
            "tree": tree,
            "leaf": is_leaf,
            "rate": node_rate,
            "intercept": intercept,
            "slope": slope,
            "tau2": tau2,
            "beta": beta,
            "mean_d": disagreement_mean,
            "std_d": disagreement_std,
            "scale": disagreement_scale,
            "mean_c": low_confidence_mean,
            "std_c": low_confidence_std,
        },
        ctx.out / "tree.joblib",
    )

    ctx.write(
        "tree_nodes",
        _node_table(
            tree,
            is_leaf=is_leaf,
            is_kept=is_kept,
            n_grow=n_grow,
            n_grow_errors=n_grow_errors,
            cost=cost,
            best=best,
            n_estimate=trees.subtree_counts(
                tree, estimate_leaves, np.ones(len(estimate_leaves))
            ),
            n_estimate_errors=trees.subtree_counts(
                tree, estimate_leaves, estimate_errors
            ),
            node_rate=node_rate,
            estimate_weight=estimate_weight,
        ),
    )
    ctx.save_metrics(
        {
            "n_nodes": n_nodes,
            "n_leaves": int(is_leaf.sum()),
            "n_empty_roots": int((n_grow[tree.root] == 0).sum()),
            "max_depth": int(trees.node_depths(tree)[is_kept].max()),
            "n_grow": int(is_grow.sum()),
            "n_estimate": int((~is_grow).sum()),
            "grow_error_rate": float(grow_error_rate),
            "estimate_error_rate": float(estimate_errors.mean()),
            "fh_a": intercept,
            "fh_b": slope,
            "fh_tau2": tau2,
            "beta": beta,
            "mu_d": disagreement_mean,
            "sigma_d": disagreement_std,
            "d_scale": disagreement_scale,
            "mu_c": low_confidence_mean,
            "sigma_c": low_confidence_std,
            "n_leaves_without_estimate": int((is_leaf & (n_estimate == 0)).sum()),
        }
    )
    logger.info("Tree: %d leaves, beta=%.3f", int(is_leaf.sum()), beta)

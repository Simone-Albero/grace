"""Regions as the leaves of a tree, and each row's failure probability.

The tree is grown with 2-means on the LDA space plus one axis, the
share of nearest train rows that disagree with the predicted class
(one root per predicted class), cut where the out-of-fold errors of
train say a deeper cut costs more noise than it removes, and its leaves
take the failure rate of the validation rows that reach them, shrunk
toward the out-of-fold rate. A per-row correction by the classifier's
confidence gives every row its probability. Test labels are never
read. Artifacts: the tree and its estimates, and every row's leaf,
disagreement, confidence, leaf rate and probability. DB rows written:
`tree_nodes` and metrics n_nodes, n_leaves, n_empty_roots, max_depth,
oof_error_rate, val_error_rate, fh_a, fh_b, fh_tau2, beta, mu_d,
sigma_d, d_scale, mu_c, sigma_c and n_leaves_without_val.
"""

__all__ = ["CONFIG", "INPUTS", "FILES", "OUTPUTS", "run"]

import logging
from collections.abc import Mapping

import joblib
import numpy as np
import pandas as pd

from src import tree as trees
from stages import SPLITS, Context

logger = logging.getLogger(__name__)

CONFIG = ("tree", "seed")
INPUTS = ("prepare", "space", "classify", "crossfit")
FILES = ()
OUTPUTS = ("tree.joblib", "rows.parquet")


def _read_points(ctx: Context, split: str) -> np.ndarray:
    """Return a split's points in the LDA space, from space."""
    rows = ctx.split_rows("space", "z.parquet", split)
    columns = sorted(
        (name for name in rows.columns if name.startswith("z_")),
        key=lambda name: int(name[2:]),
    )
    return rows[columns].to_numpy(dtype=np.float64)


def _read_predictions(
    ctx: Context, stage: str, file: str, split: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return a split's predicted classes and top probabilities.

    Read from classify or crossfit.
    """
    rows = ctx.split_rows(stage, file, split)
    probabilities = rows.filter(like="proba_").to_numpy(dtype=np.float64)
    return rows["y_pred"].to_numpy(), probabilities.max(axis=1)


def _standardize(
    by_split: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], float, float]:
    """Centre and scale every split by the mean and std of train.

    Returns:
        The standardized splits, all zeros when the std of train is at
        most 1e-12, then the mean and std of train.
    """
    mean = float(by_split["train"].mean())
    std = float(by_split["train"].std())
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
    n_train: np.ndarray,
    n_train_errors: np.ndarray,
    cost: np.ndarray,
    best: np.ndarray,
    n_val: np.ndarray,
    n_val_errors: np.ndarray,
    node_rate: np.ndarray,
    val_weight: np.ndarray,
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
            "n_train": n_train.astype(np.int64),
            "n_train_errors": n_train_errors.astype(np.int64),
            "cost": cost,
            "best": best,
            "status": status,
            "n_val": [
                None if node_status == "pruned" else int(count)
                for node_status, count in zip(status, n_val, strict=True)
            ],
            "n_val_errors": [
                None if node_status == "pruned" else int(count)
                for node_status, count in zip(status, n_val_errors, strict=True)
            ],
            "q": _nan_to_none(np.where(is_leaf, node_rate, np.nan)),
            "gamma": _nan_to_none(np.where(is_leaf, val_weight, np.nan)),
        }
    )


def run(ctx: Context) -> None:
    """Grow, cut and estimate the tree; write every row's probability.

    Args:
        ctx: The stage context.
    """
    cfg = ctx.cfg
    n_classes = len(ctx.read("classes", "prepare"))
    y_train = ctx.labels("train")
    points = {split: _read_points(ctx, split) for split in SPLITS}
    sources = {
        "train": ("crossfit", "oof.parquet"),
        "val": ("classify", "predictions.parquet"),
        "test": ("classify", "predictions.parquet"),
    }
    predicted, confidence = {}, {}
    for split in SPLITS:
        predicted[split], confidence[split] = _read_predictions(
            ctx, *sources[split], split
        )
    train_errors = (predicted["train"] != y_train).astype(np.float64)

    neighbour_index = trees.build_neighbours(points["train"], n_jobs=ctx.runtime.n_jobs)
    n_neighbours = min(cfg.tree.k_neighbours, len(y_train) - 1)
    disagreement = {
        "train": trees.neighbour_disagreement(
            neighbour_index, y_train, predicted["train"], n_neighbours=n_neighbours
        )
    } | {
        split: trees.neighbour_disagreement(
            neighbour_index,
            y_train,
            predicted[split],
            n_neighbours=n_neighbours,
            points=points[split],
        )
        for split in SPLITS[1:]
    }
    standardized_disagreement, disagreement_mean, disagreement_std = _standardize(
        disagreement
    )

    # The disagreement is one more axis, weighted in units of the spread
    # of the LDA space. Without weight the axis is left out, so that the
    # geometric tree is grown on exactly its own points.
    disagreement_scale = cfg.tree.d_weight * float(
        np.sqrt(points["train"].var(axis=0).sum())
    )
    grown_points = points
    if disagreement_scale > 0:
        grown_points = {
            split: np.column_stack(
                [points[split], disagreement_scale * standardized_disagreement[split]]
            )
            for split in SPLITS
        }

    tree, node_of_row = trees.grow_tree(
        grown_points["train"],
        predicted["train"],
        n_classes=n_classes,
        n_min=cfg.tree.n_min,
        random_state=cfg.seed,
    )
    n_train = trees.subtree_counts(tree, node_of_row, np.ones(len(y_train)))
    n_train_errors = trees.subtree_counts(tree, node_of_row, train_errors)
    is_leaf, is_kept, cost, best = trees.prune_tree(
        tree,
        n_train,
        n_train_errors,
        n_train=len(y_train),
        n_val=len(points["val"]),
    )
    leaf_of_row = {
        split: trees.descend(tree, grown_points[split], predicted[split], stop=is_leaf)
        for split in SPLITS
    }

    # The validation labels are read only now, to estimate the leaves.
    val_errors = (ctx.labels("val") != predicted["val"]).astype(np.float64)
    n_nodes = len(tree.parent)
    n_val = np.bincount(leaf_of_row["val"], minlength=n_nodes).astype(np.float64)
    n_val_errors = np.bincount(
        leaf_of_row["val"], weights=val_errors, minlength=n_nodes
    )
    oof_error_rate = train_errors.mean()
    covariate = np.where(
        n_train > 0, (n_train_errors + 1) / (n_train + 2), oof_error_rate
    )
    has_val_rows = is_leaf & (n_val > 0)
    intercept, slope, tau2 = trees.fit_fay_herriot(
        n_val_errors[has_val_rows], n_val[has_val_rows], covariate[has_val_rows]
    )
    node_rate, val_weight = trees.shrink_rates(
        n_val_errors,
        n_val,
        covariate,
        intercept=intercept,
        slope=slope,
        tau2=tau2,
    )

    standardized_low_confidence, low_confidence_mean, low_confidence_std = _standardize(
        {split: trees.low_confidence(confidence[split]) for split in SPLITS}
    )
    beta = trees.fit_beta(
        node_rate[leaf_of_row["val"]], standardized_low_confidence["val"], val_errors
    )

    leaf_rate = {split: node_rate[leaf_of_row[split]] for split in SPLITS}
    rows = pd.concat(
        [
            pd.DataFrame(
                {
                    "split": split,
                    "row": np.arange(len(points[split]), dtype=np.int32),
                    "leaf": leaf_of_row[split].astype(np.int32),
                    "d": disagreement[split].astype(np.float32),
                    "confidence": confidence[split].astype(np.float32),
                    "q_leaf": leaf_rate[split].astype(np.float32),
                    "q": trees.correct_rates(
                        leaf_rate[split], standardized_low_confidence[split], beta
                    ).astype(np.float32),
                }
            )
            for split in SPLITS
        ],
        ignore_index=True,
    )
    rows["split"] = pd.Categorical(rows["split"], categories=SPLITS)
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
            n_train=n_train,
            n_train_errors=n_train_errors,
            cost=cost,
            best=best,
            n_val=trees.subtree_counts(
                tree, leaf_of_row["val"], np.ones(len(points["val"]))
            ),
            n_val_errors=trees.subtree_counts(tree, leaf_of_row["val"], val_errors),
            node_rate=node_rate,
            val_weight=val_weight,
        ),
    )
    ctx.save_metrics(
        {
            "n_nodes": n_nodes,
            "n_leaves": int(is_leaf.sum()),
            "n_empty_roots": int((n_train[tree.root] == 0).sum()),
            "max_depth": int(trees.node_depths(tree)[is_kept].max()),
            "oof_error_rate": float(oof_error_rate),
            "val_error_rate": float(val_errors.mean()),
            "fh_a": intercept,
            "fh_b": slope,
            "fh_tau2": tau2,
            "beta": beta,
            "mu_d": disagreement_mean,
            "sigma_d": disagreement_std,
            "d_scale": disagreement_scale,
            "mu_c": low_confidence_mean,
            "sigma_c": low_confidence_std,
            "n_leaves_without_val": int((is_leaf & (n_val == 0)).sum()),
        }
    )
    logger.info("Tree: %d leaves, beta=%.3f", int(is_leaf.sum()), beta)

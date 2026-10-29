"""Regions as the leaves of a tree grown on points, cut by errors.

A tree has one root per predicted class. Every node splits its rows into
2 to `max_k` parts with k-means, and a row goes to the nearest of the
centres: the rule that builds the tree is the rule that routes a new
row. The points are the rows' coordinates plus the share of nearest
train rows that disagree with the prediction, as one more axis. The
errors of the rows the tree grew on say how deep to cut, each leaf gets
the failure rate of the other rows that reach it, and a per-row
correction moves it with the classifier's own confidence. Every leaf
costs a price on top of its expected Brier score, so that a split must
pay for the leaves it adds; the same cost picks the number of parts of a
split, stops the growth and cuts the grown tree.
"""

__all__ = [
    "Tree",
    "build_neighbours",
    "correct_rates",
    "descend",
    "fit_beta",
    "fit_fay_herriot",
    "grow_tree",
    "leaf_brier",
    "leaf_penalty",
    "low_confidence",
    "neighbour_disagreement",
    "node_depths",
    "prune_tree",
    "shrink_rates",
    "subtree_counts",
]

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import NearestNeighbors

# Starts of each k-means, enough for a stable split of a small node.
_N_INIT = 4
# The bounds of the correction's slope; a larger one is a step function.
_BETA_BOUND = 10.0
# Rates are clipped away from 0 and 1 so that their logit is finite.
_RATE_EPS = 1e-6
# Rows per k-NN query, so that the neighbour lists stay small in memory.
_NEIGHBOUR_CHUNK = 100_000


@dataclass(frozen=True)
class Tree:
    """A forest of trees in arrays, children after their parent.

    The children of a node have consecutive ids.

    Attributes:
        parent: Each node's parent, -1 at a root.
        first_child: The id of each node's first child, -1 without
            children.
        n_children: The number of each node's children, 0 at a leaf.
        class_id: The predicted class whose root the node descends from.
        centroid: The k-means centre of the part of its parent a node
            holds, shape (n_nodes, m); NaN at a root.
        root: The root of each predicted class, by class id.
    """

    parent: np.ndarray
    first_child: np.ndarray
    n_children: np.ndarray
    class_id: np.ndarray
    centroid: np.ndarray
    root: np.ndarray


def leaf_penalty(errors: np.ndarray, leaf_cost: float) -> float:
    """Return the price of one leaf, in units of the Brier score.

    Args:
        errors: The classifier's errors, 0 or 1, on the rows the tree
            grows on.
        leaf_cost: The share of the Brier score of the constant
            predictor that a leaf must save to be worth keeping.

    Returns:
        `leaf_cost` times the Brier score of the constant predictor.
    """
    rate = float(np.mean(errors))
    return leaf_cost * rate * (1 - rate)


def leaf_brier(
    n_rows: ArrayLike,
    n_errors: ArrayLike,
    n_estimate: ArrayLike,
    n_total: float,
) -> np.ndarray:
    """Return the expected Brier score of nodes taken as leaves.

    A node costs its share of the spread of the errors it hides plus the
    noise of a rate estimated on the estimation rows that reach it.

    Args:
        n_rows: The rows the tree grew on, under each node.
        n_errors: The classifier's errors among them.
        n_estimate: The rows the leaf rates will be estimated on, under
            each node; their labels are not read.
        n_total: All the rows the tree grew on, against which a node's
            share is taken.

    Returns:
        The cost of each node, without the price of a leaf.
    """
    n_rows = np.asarray(n_rows, dtype=np.float64)
    n_errors = np.asarray(n_errors, dtype=np.float64)
    # A leaf no estimation row reaches takes the fitted line, so it is
    # costed as if one did.
    n_estimate_floored = np.maximum(np.asarray(n_estimate, dtype=np.float64), 1.0)
    observed = np.divide(
        n_errors, n_rows, out=np.zeros_like(n_errors), where=n_rows > 0
    )
    smoothed = (n_errors + 1) / (n_rows + 2)
    return (n_rows / n_total) * (
        observed * (1 - observed) * n_rows / np.maximum(n_rows - 1, 1)
        + smoothed * (1 - smoothed) / n_estimate_floored
    )


def _nearest_centre(points: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Return which centre each point is nearest, the first on a tie."""
    squared_distance = np.column_stack(
        [((points - centre) ** 2).sum(axis=1) for centre in centres]
    )
    return squared_distance.argmin(axis=1)


def _split(
    points: np.ndarray,
    errors: np.ndarray,
    estimate_points: np.ndarray,
    *,
    n_min: int,
    max_k: int,
    penalty: float,
    n_total: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Split a node into the 2 to `max_k` parts cheapest as leaves.

    Each part counts as a leaf: its `leaf_brier` plus `penalty`. A
    number of parts that would leave one with fewer than `n_min` rows
    is rejected, and the fewest parts win a tie.

    Args:
        points: The node's points, shape (n, m).
        errors: Whether the classifier failed on each of them.
        estimate_points: The node's estimation points.
        n_min: The fewest rows of a part.
        max_k: The most parts.
        penalty: The price of a leaf.
        n_total: All the rows the tree grows on, for `leaf_brier`.
        random_state: The seed of every k-means.

    Returns:
        The centres, shape (k, m), and the part of every point and of
        every estimation point; None when no number of parts is allowed
        or the points are all the same.
    """
    if np.ptp(points, axis=0).max() == 0:
        return None
    best_cost, best_split = np.inf, None
    for k in range(2, max_k + 1):
        if len(points) < k * n_min:
            break
        # Fewer distinct points than parts leaves an empty part, which
        # the `n_min` check below rejects.
        with warnings.catch_warnings(action="ignore", category=ConvergenceWarning):
            centres = (
                KMeans(n_clusters=k, n_init=_N_INIT, random_state=random_state)
                .fit(points)
                .cluster_centers_
            )
        part_of_row = _nearest_centre(points, centres)
        n_rows = np.bincount(part_of_row, minlength=k)
        if n_rows.min() < n_min:
            continue
        estimate_part_of_row = _nearest_centre(estimate_points, centres)
        brier = leaf_brier(
            n_rows,
            np.bincount(part_of_row, weights=errors, minlength=k),
            np.bincount(estimate_part_of_row, minlength=k),
            n_total,
        )
        cost = brier.sum() + k * penalty
        if cost < best_cost:
            best_cost = cost
            best_split = centres, part_of_row, estimate_part_of_row
    return best_split


def grow_tree(
    points: np.ndarray,
    predicted: np.ndarray,
    errors: np.ndarray,
    estimate_points: np.ndarray,
    estimate_predicted: np.ndarray,
    *,
    n_classes: int,
    n_min: int,
    max_k: int,
    penalty: float,
    random_state: int,
) -> tuple[Tree, np.ndarray]:
    """Grow a tree of k-means splits under a root per predicted class.

    A node splits while all its parts keep at least `n_min` rows and its
    `leaf_brier` is above `penalty`. The stop is exact: as a leaf a node
    costs its `leaf_brier` plus `penalty`, any cut below it at least
    twice `penalty`, so `prune_tree` keeps a node whose `leaf_brier` is
    at most `penalty` as a leaf. A class nobody predicted keeps an empty
    root.

    Args:
        points: The rows' points, shape (n, m).
        predicted: Each row's predicted class.
        errors: Whether the classifier failed on each row, 0 or 1.
        estimate_points: The points of the estimation rows, routed
            alongside to count the rows each node would estimate on.
        estimate_predicted: Their predicted class.
        n_classes: The classes; node `c` is the root of class `c`.
        n_min: The fewest rows of a part of a split.
        max_k: The most parts of a split, which takes the number of
            parts from 2 to `max_k` that costs least as leaves.
        penalty: The price of a leaf, from `leaf_penalty`.
        random_state: The seed of every k-means.

    Returns:
        The tree and the deepest node every row reaches.
    """
    parent = [-1] * n_classes
    first_child = [-1] * n_classes
    n_children = [0] * n_classes
    class_id = list(range(n_classes))
    centroid = [np.full(points.shape[1], np.nan)] * n_classes
    node_of_row = np.asarray(predicted, dtype=np.int64).copy()
    stack = [
        (
            root,
            np.flatnonzero(predicted == root),
            np.flatnonzero(estimate_predicted == root),
        )
        for root in range(n_classes)
    ]
    while stack:
        node, rows, estimate_rows = stack.pop()
        n_errors = errors[rows].sum()
        if leaf_brier(len(rows), n_errors, len(estimate_rows), len(points)) <= penalty:
            continue
        split = _split(
            points[rows],
            errors[rows],
            estimate_points[estimate_rows],
            n_min=n_min,
            max_k=max_k,
            penalty=penalty,
            n_total=len(points),
            random_state=random_state,
        )
        if split is None:
            continue
        centres, part_of_row, estimate_part_of_row = split
        first_child[node], n_children[node] = len(parent), len(centres)
        for part, centre in enumerate(centres):
            child = len(parent)
            parent.append(node)
            first_child.append(-1)
            n_children.append(0)
            class_id.append(class_id[node])
            centroid.append(centre)
            members = rows[part_of_row == part]
            node_of_row[members] = child
            stack.append((child, members, estimate_rows[estimate_part_of_row == part]))
    tree = Tree(
        parent=np.array(parent, dtype=np.int64),
        first_child=np.array(first_child, dtype=np.int64),
        n_children=np.array(n_children, dtype=np.int64),
        class_id=np.array(class_id, dtype=np.int64),
        centroid=np.vstack(centroid),
        root=np.arange(n_classes, dtype=np.int64),
    )
    return tree, node_of_row


def _nearest_child(tree: Tree, nodes: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Return each node's child whose centre is nearest its point.

    A tie goes to the first child, as in `_nearest_centre`, so that a
    routed row ends where the growth put it.
    """
    n_children = tree.n_children[nodes]
    first_child = tree.first_child[nodes]
    squared_distance = np.full((len(nodes), n_children.max()), np.inf)
    for offset in range(n_children.max()):
        has_child = offset < n_children
        child_centres = tree.centroid[first_child[has_child] + offset]
        squared_distance[has_child, offset] = (
            (points[has_child] - child_centres) ** 2
        ).sum(axis=1)
    return first_child + squared_distance.argmin(axis=1)


def descend(
    tree: Tree,
    points: np.ndarray,
    predicted: np.ndarray,
    *,
    stop: np.ndarray | None = None,
) -> np.ndarray:
    """Route rows down from their predicted class's root.

    Args:
        tree: The tree.
        points: The rows' points, shape (n, m).
        predicted: Each row's predicted class.
        stop: Per node, whether a row stops there, as the leaves of a
            pruned tree; None goes down to the deepest node.

    Returns:
        The node every row ends at.
    """
    node = tree.root[np.asarray(predicted, dtype=np.int64)].copy()
    while True:
        is_movable = tree.n_children[node] > 0
        if stop is not None:
            is_movable &= ~stop[node]
        active = np.flatnonzero(is_movable)
        if active.size == 0:
            return node
        node[active] = _nearest_child(tree, node[active], points[active])


def subtree_counts(
    tree: Tree, node_of_row: np.ndarray, values: np.ndarray
) -> np.ndarray:
    """Sum a per-row value over the rows under every node.

    Args:
        tree: The tree.
        node_of_row: The node each row ends at.
        values: One number per row, such as 1 or an error.

    Returns:
        Per node, the sum over the rows at or below it.
    """
    totals = np.bincount(node_of_row, weights=values, minlength=len(tree.parent))
    for node in range(len(totals) - 1, -1, -1):
        if tree.parent[node] >= 0:
            totals[tree.parent[node]] += totals[node]
    return totals


def node_depths(tree: Tree) -> np.ndarray:
    """Return each node's depth, 0 at a root."""
    depth = np.zeros(len(tree.parent), dtype=np.int64)
    for node in range(len(depth)):
        if tree.parent[node] >= 0:
            depth[node] = depth[tree.parent[node]] + 1
    return depth


def prune_tree(
    tree: Tree,
    n_rows: np.ndarray,
    n_errors: np.ndarray,
    n_estimate: np.ndarray,
    *,
    penalty: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cut every tree where its leaves' cost is least.

    A leaf costs its expected Brier score, from `leaf_brier`, plus
    `penalty`. A node stays a leaf unless its best subtrees, one per
    child, cost strictly less together, found bottom-up.

    Args:
        tree: The grown tree.
        n_rows: The rows the tree grew on, under every node.
        n_errors: The classifier's errors among them.
        n_estimate: The rows the leaf rates will be estimated on, under
            every node; their labels are not read.
        penalty: The price of a leaf, from `leaf_penalty`.

    Returns:
        Per node: whether it is a leaf of the cut, whether the cut keeps
        it (a leaf or above one), its own cost and the best cost of its
        subtree.
    """
    n_rows = np.asarray(n_rows, dtype=np.float64)
    cost = leaf_brier(n_rows, n_errors, n_estimate, n_rows[tree.root].sum()) + penalty
    best = cost.copy()
    is_leaf = np.ones(len(cost), dtype=bool)
    for node in range(len(cost) - 1, -1, -1):
        if tree.n_children[node] == 0:
            continue
        first_child = tree.first_child[node]
        below = best[first_child : first_child + tree.n_children[node]].sum()
        if below < cost[node]:
            best[node] = below
            is_leaf[node] = False
    is_kept = np.zeros(len(cost), dtype=bool)
    is_kept[tree.root] = True
    for node in range(len(cost)):
        if is_kept[node] and not is_leaf[node]:
            first_child = tree.first_child[node]
            is_kept[first_child : first_child + tree.n_children[node]] = True
    return is_leaf & is_kept, is_kept, cost, best


def _tau_squared(
    residual_squared: np.ndarray, noise: np.ndarray, degrees_of_freedom: int
) -> float:
    """Solve sum(r² / (tau² + v)) = dof for tau², 0 when it is below."""
    if (
        degrees_of_freedom <= 0
        or (residual_squared / noise).sum() <= degrees_of_freedom
    ):
        return 0.0
    low, high = 0.0, residual_squared.sum() / degrees_of_freedom
    for _ in range(60):
        middle = (low + high) / 2
        if (residual_squared / (middle + noise)).sum() > degrees_of_freedom:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def fit_fay_herriot(
    errors: np.ndarray, counts: np.ndarray, covariate: np.ndarray
) -> tuple[float, float, float]:
    """Fit the Fay-Herriot model of the leaves' observed rates.

    Each leaf's rate is the line `a + b * covariate` plus a spread of
    variance `tau²` that the line misses, observed with the binomial
    noise of its `counts`. A fit of fewer than three leaves, or of one
    covariate value, has no slope.

    Args:
        errors: Each leaf's errors among its estimation rows.
        counts: Each leaf's estimation rows, all above 0.
        covariate: Each leaf's error rate on the rows the tree grew on.

    Returns:
        The intercept, the slope and `tau²`, never negative.
    """
    errors = np.asarray(errors, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    if len(errors) == 0:
        return 0.0, 0.0, 0.0
    rate = errors / counts
    pooled = errors.sum() / counts.sum()
    with_slope = len(rate) >= 3 and np.ptp(covariate) > 0
    degrees_of_freedom = len(rate) - (2 if with_slope else 1)
    design = np.column_stack([np.ones(len(rate)), covariate])
    mean = np.full(len(rate), pooled)
    tau2 = float(rate.var()) if len(rate) > 1 else 0.0
    intercept, slope = pooled, 0.0
    for _ in range(50):
        noise = np.maximum(mean * (1 - mean), _RATE_EPS) / counts
        weight = 1 / (tau2 + noise)
        if with_slope:
            gram = design.T @ (design * weight[:, None])
            intercept, slope = np.linalg.solve(gram, design.T @ (weight * rate))
        else:
            intercept, slope = float((weight * rate).sum() / weight.sum()), 0.0
        residual = rate - intercept - slope * covariate
        new_tau2 = _tau_squared(residual**2, noise, degrees_of_freedom)
        new_mean = np.clip(intercept + slope * covariate, _RATE_EPS, 1 - _RATE_EPS)
        is_converged = (
            abs(new_tau2 - tau2) < 1e-12 and np.abs(new_mean - mean).max() < 1e-9
        )
        tau2, mean = new_tau2, new_mean
        if is_converged:
            break
    return float(intercept), float(slope), float(tau2)


def shrink_rates(
    errors: np.ndarray,
    counts: np.ndarray,
    covariate: np.ndarray,
    *,
    intercept: float,
    slope: float,
    tau2: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Move each leaf's observed rate toward the fitted line.

    Args:
        errors: Each leaf's errors among its estimation rows.
        counts: Each leaf's estimation rows; 0 is allowed.
        covariate: Each leaf's error rate on the rows the tree grew on.
        intercept: The line's intercept.
        slope: The line's slope.
        tau2: The spread the line misses.

    Returns:
        Each leaf's rate and the weight its own observed rate got, NaN
        for a leaf without estimation rows, which takes the line.
    """
    errors = np.asarray(errors, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    line = np.clip(intercept + slope * covariate, _RATE_EPS, 1 - _RATE_EPS)
    has_rows = counts > 0
    rate = np.divide(errors, counts, out=np.zeros_like(errors), where=has_rows)
    noise = line * (1 - line) / np.where(has_rows, counts, 1)
    weight = np.where(has_rows, tau2 / (tau2 + noise), np.nan)
    shrunk = np.where(has_rows, weight * rate + (1 - weight) * line, line)
    return np.clip(shrunk, 0.0, 1.0), weight


def build_neighbours(train_points: np.ndarray, *, n_jobs: int) -> NearestNeighbors:
    """Index the train points for `neighbour_disagreement`.

    Args:
        train_points: The train points, shape (n, m).
        n_jobs: The parallel jobs of a query.

    Returns:
        The fitted index.
    """
    return NearestNeighbors(n_jobs=n_jobs).fit(train_points)


def neighbour_disagreement(
    index: NearestNeighbors,
    y_train: np.ndarray,
    predicted: np.ndarray,
    points: np.ndarray,
    *,
    n_neighbours: int,
) -> np.ndarray:
    """Return the share of nearest train rows of another class.

    Args:
        index: The fitted index of the train points.
        y_train: The train labels.
        predicted: Each row's predicted class.
        points: The rows' points.
        n_neighbours: The neighbours.

    Returns:
        Share of the `n_neighbours` nearest train rows not labelled as
        predicted.
    """
    neighbours = np.vstack(
        [
            index.kneighbors(
                points[start : start + _NEIGHBOUR_CHUNK],
                n_neighbors=n_neighbours,
                return_distance=False,
            )
            for start in range(0, len(points), _NEIGHBOUR_CHUNK)
        ]
    )
    return (y_train[neighbours] != np.asarray(predicted)[:, None]).mean(axis=1)


def _logit(rate: np.ndarray) -> np.ndarray:
    """Return the logit of rates kept away from 0 and 1."""
    return logit(np.clip(rate, _RATE_EPS, 1 - _RATE_EPS))


def low_confidence(top_probability: np.ndarray) -> np.ndarray:
    """Return the log-odds that a row's predicted class is wrong.

    Args:
        top_probability: Each row's highest class probability.

    Returns:
        The logit of one minus it, finite even at a probability of 1.
    """
    return _logit(1.0 - np.asarray(top_probability, dtype=np.float64))


def correct_rates(
    leaf_rate: np.ndarray, standardized: np.ndarray, beta: float
) -> np.ndarray:
    """Move each row's leaf rate by its standardized signal.

    Args:
        leaf_rate: Each row's leaf rate.
        standardized: Each row's signal, centred and scaled.
        beta: The slope on the log-odds scale.

    Returns:
        Each row's failure probability.
    """
    return expit(_logit(leaf_rate) + beta * standardized)


def fit_beta(
    leaf_rate: np.ndarray, standardized: np.ndarray, errors: np.ndarray
) -> float:
    """Fit the slope of the correction on estimation rows.

    Args:
        leaf_rate: Each row's leaf rate.
        standardized: Each row's signal, centred and scaled.
        errors: Each row's failure, 0 or 1.

    Returns:
        The slope minimizing the squared error, 0 without any spread in
        `standardized`.
    """
    if not np.any(standardized):
        return 0.0
    base = _logit(leaf_rate)
    optimum = minimize_scalar(
        lambda beta: float(np.mean((expit(base + beta * standardized) - errors) ** 2)),
        bounds=(-_BETA_BOUND, _BETA_BOUND),
        method="bounded",
    )
    return float(optimum.x)

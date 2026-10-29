"""What the dashboard reads from the artifact files, not the DB."""

__all__ = ["ClassPlane", "DisagreementAxis", "class_plane", "confusion"]

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


def confusion(
    prepare_dir: Path, classify_dir: Path, n_classes: int
) -> np.ndarray | None:
    """Count the test rows by true and predicted class.

    Rows pair by position, as in the stages: prepare's test file holds
    the labels, classify's predictions say `row` within the split.

    Args:
        prepare_dir: The folder of a prepare execution.
        classify_dir: The folder of a classify execution.
        n_classes: The number of classes.

    Returns:
        The counts, true class by rows and predicted by columns; None if
        a file is missing or the two do not have the same test length.
    """
    labels_file = prepare_dir / "test.parquet"
    predictions_file = classify_dir / "predictions.parquet"
    if not (labels_file.is_file() and predictions_file.is_file()):
        return None
    true = pd.read_parquet(labels_file, columns=["label"])["label"].to_numpy()
    predictions = pd.read_parquet(predictions_file, columns=["split", "row", "y_pred"])
    test = predictions[predictions["split"] == "test"].sort_values("row")
    if len(test) != len(true):
        return None
    predicted = test["y_pred"].to_numpy()
    cells = n_classes * true.astype(np.int64) + predicted.astype(np.int64)
    return np.bincount(cells, minlength=n_classes**2).reshape(n_classes, n_classes)


# The grown rows kept per leaf, and the most points one map draws: a
# leaf keeps its rows up to the first, and the map thins every leaf to
# the same count to fit the second.
_ROWS_PER_LEAF = 1000
_MAX_POINTS = 8000


@dataclass(frozen=True)
class DisagreementAxis:
    """How the tree turned a row's disagreement `d` into its axis `s_d`.

    Attributes:
        mean: The mean of `d` on the rows the tree grew on.
        std: The standard deviation of `d` on them.
        scale: The weight of the standardized `d` among the LDA axes.
    """

    mean: float
    std: float
    scale: float


@dataclass(frozen=True)
class ClassPlane:
    """Train rows of the leaves of a class in a 2D plane.

    Attributes:
        leaves: The node id of the leaf each point belongs to.
        points: The `(x, y)` of each grown row drawn, one per row.
        shares: The share of the placed leaves' variance, in z or in z
            and `s_d`, that each of the two axes holds.
    """

    leaves: np.ndarray
    points: np.ndarray
    shares: tuple[float, float]


@lru_cache(maxsize=4)
def _leaf_stats(
    space_dir: Path, tree_dir: Path, stamp: tuple[int, ...]
) -> dict[int, tuple[int, np.ndarray, np.ndarray, np.ndarray]] | None:
    """Count each leaf's grown rows, with their moments and a sample.

    The grown rows are the half of val the tree grew on. The points are
    the axes of z followed by the row's disagreement `d`, so that a
    plane with or without `s_d` comes out of the same read.

    Cached, since reading the files takes seconds on a large dataset.

    Args:
        space_dir: The folder of a space execution.
        tree_dir: The folder of a tree execution.
        stamp: The size and mtime of the two files, so that a folder
            rewritten in place, as after deleting the DB and the
            artifacts, is read again.

    Returns:
        The `(n_rows, mean, covariance, sample)` in z and `d` of each
        leaf that has grown rows, by node id, the sample being at most
        `_ROWS_PER_LEAF` rows drawn at random; None if the two files do
        not have the same val length.
    """
    val_only = [("split", "==", "val")]
    z = pd.read_parquet(space_dir / "z.parquet", filters=val_only)
    z = z.sort_values("row")
    tree_rows = pd.read_parquet(
        tree_dir / "rows.parquet",
        columns=["row", "grow", "leaf", "d"],
        filters=val_only,
    ).sort_values("row")
    if len(z) != len(tree_rows):
        return None
    z_columns = sorted(
        (name for name in z.columns if name.startswith("z_")),
        key=lambda name: int(name[2:]),
    )
    is_grow = tree_rows["grow"].to_numpy()
    points = np.column_stack([z[z_columns].to_numpy(), tree_rows["d"].to_numpy()])
    points = points[is_grow]
    row_leaves = tree_rows["leaf"].to_numpy()[is_grow]
    order = np.argsort(row_leaves, kind="stable")
    node_ids, starts, sizes = np.unique(
        row_leaves[order], return_index=True, return_counts=True
    )
    # A fixed seed, so that a leaf shows the same rows on every request.
    rng = np.random.default_rng(0)
    stats = {}
    for node_id, start, n_rows in zip(node_ids, starts, sizes, strict=True):
        leaf_points = points[order[start : start + n_rows]].astype(np.float64)
        mean = leaf_points.mean(axis=0)
        sample = leaf_points[
            rng.choice(n_rows, min(n_rows, _ROWS_PER_LEAF), replace=False)
        ]
        stats[int(node_id)] = (
            int(n_rows),
            mean,
            leaf_points.T @ leaf_points / n_rows - np.outer(mean, mean),
            sample.astype(np.float32),
        )
    return stats


def _points_per_leaf(sample_sizes: Sequence[int], budget: int) -> int:
    """Find the most points to draw from each leaf within a budget.

    Args:
        sample_sizes: The rows each leaf has to draw from.
        budget: The most points in all.

    Returns:
        The cap on every leaf, the largest that keeps the total within
        the budget; at least 1.
    """
    sizes = np.asarray(sample_sizes)
    low, high = 1, int(sizes.max())
    while low < high:
        middle = (low + high + 1) // 2
        if np.minimum(sizes, middle).sum() <= budget:
            low = middle
        else:
            high = middle - 1
    return low


def _space_transform(
    n_axes: int, disagreement: DisagreementAxis | None
) -> tuple[np.ndarray, np.ndarray]:
    """Build the map from the cached points, z then `d`, to a space.

    The space is the one whose principal plane the map draws: z, or z
    and `s_d`.

    Args:
        n_axes: The number of axes of z.
        disagreement: How the tree built `s_d`; None to leave it out.

    Returns:
        The offset to subtract from a point and the matrix to multiply
        it by, which keeps z and turns `d` into `s_d` or drops it.
    """
    offset = np.zeros(n_axes + 1)
    if disagreement is None:
        return offset, np.eye(n_axes + 1)[:, :n_axes]
    weights = np.eye(n_axes + 1)
    offset[-1] = disagreement.mean
    weights[-1, -1] = disagreement.scale / disagreement.std
    return offset, weights


def class_plane(
    space_dir: Path,
    tree_dir: Path,
    leaf_ids: Sequence[int],
    *,
    disagreement: DisagreementAxis | None = None,
) -> ClassPlane | None:
    """Place a sample of the leaves' grown rows in their PCA plane.

    The plane is the first two principal axes of the leaves' grown rows
    in z, or in z with the axis `s_d` the tree also cut on, so it shows
    where the tree cut the space. Each leaf contributes a random sample
    of its rows, thinned to the same count per leaf so that the whole
    stays within `_MAX_POINTS`.

    Args:
        space_dir: The folder of a space execution.
        tree_dir: The folder of a tree execution.
        leaf_ids: The node ids of the leaves to place: a class's, or
            those below one of its nodes.
        disagreement: How the tree built `s_d`, to add it to the axes
            of z; None for z alone.

    Returns:
        The points; None if a file is missing or the two do not have
        the same val length. The points are empty when no leaf has a
        grown row. With one LDA axis and no `s_d`, the y of every point
        is 0.
    """
    files = (space_dir / "z.parquet", tree_dir / "rows.parquet")
    if not all(file.is_file() for file in files):
        return None
    file_stats = [file.stat() for file in files]
    stamp = (
        file_stats[0].st_size,
        file_stats[0].st_mtime_ns,
        file_stats[1].st_size,
        file_stats[1].st_mtime_ns,
    )
    leaf_stats = _leaf_stats(space_dir, tree_dir, stamp)
    if leaf_stats is None:
        return None
    selected = {
        node_id: leaf_stats[node_id] for node_id in leaf_ids if node_id in leaf_stats
    }
    if not selected:
        return ClassPlane(np.empty(0, dtype=np.int64), np.empty((0, 2)), (0.0, 0.0))
    n_axes = len(next(iter(selected.values()))[1]) - 1
    offset, weights = _space_transform(n_axes, disagreement)
    leaves = {
        node_id: (
            n_rows,
            (leaf_mean - offset) @ weights,
            weights.T @ leaf_cov @ weights,
            (sample - offset) @ weights,
        )
        for node_id, (n_rows, leaf_mean, leaf_cov, sample) in selected.items()
    }
    n_class_rows = sum(n_rows for n_rows, _, _, _ in leaves.values())
    mean = (
        sum(n_rows * leaf_mean for n_rows, leaf_mean, _, _ in leaves.values())
        / n_class_rows
    )
    # The covariance of the class is the within-leaf one plus the one
    # between the leaves' means.
    covariance = (
        sum(
            n_rows * (leaf_cov + np.outer(leaf_mean - mean, leaf_mean - mean))
            for n_rows, leaf_mean, leaf_cov, _ in leaves.values()
        )
        / n_class_rows
    )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    n_components = min(2, len(eigenvalues))
    components = eigenvectors[:, ::-1][:, :n_components]
    total_variance = max(float(eigenvalues.sum()), 0.0)
    shares = [
        float(eigenvalue) / total_variance if total_variance > 0 else 0.0
        for eigenvalue in eigenvalues[::-1][:n_components]
    ]
    shares += [0.0] * (2 - n_components)
    cap = _points_per_leaf([len(sample) for *_, sample in leaves.values()], _MAX_POINTS)
    drawn_leaves, drawn_points = [], []
    for node_id, (_, _, _, sample) in leaves.items():
        drawn = (sample[:cap] - mean) @ components
        if n_components == 1:
            drawn = np.column_stack([drawn, np.zeros(len(drawn))])
        drawn_leaves.append(np.full(len(drawn), node_id))
        drawn_points.append(drawn)
    return ClassPlane(
        np.concatenate(drawn_leaves),
        np.concatenate(drawn_points),
        (shares[0], shares[1]),
    )

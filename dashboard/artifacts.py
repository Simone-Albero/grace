"""What the dashboard reads from the artifact files, not the DB."""

__all__ = ["ClassPlane", "class_plane", "confusion"]

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


@dataclass(frozen=True)
class ClassPlane:
    """The leaves of one predicted class as circles in a 2D plane.

    Attributes:
        circles: The `(x, y, radius)` of each leaf by node id; a leaf
            that no train row reaches has none.
        shares: The share of the class's variance in z that each of the
            two axes holds.
    """

    circles: dict[int, tuple[float, float, float]]
    shares: tuple[float, float]


@lru_cache(maxsize=4)
def _leaf_moments(
    space_dir: Path, tree_dir: Path, stamp: tuple[int, ...]
) -> dict[int, tuple[int, np.ndarray, np.ndarray]] | None:
    """Count each leaf's train rows, with their mean and covariance.

    Cached, since reading the files takes seconds on a large dataset.

    Args:
        space_dir: The folder of a space execution.
        tree_dir: The folder of a tree execution.
        stamp: The size and mtime of the two files, so that a folder
            rewritten in place, as after deleting the DB and the
            artifacts, is read again.

    Returns:
        The `(n_rows, mean, covariance)` in z of each leaf that has
        train rows, by node id; None if the two files do not have the
        same train length.
    """
    train_only = [("split", "==", "train")]
    z = pd.read_parquet(space_dir / "z.parquet", filters=train_only)
    z = z.sort_values("row")
    tree_rows = pd.read_parquet(
        tree_dir / "rows.parquet", columns=["row", "leaf"], filters=train_only
    ).sort_values("row")
    if len(z) != len(tree_rows):
        return None
    z_columns = sorted(
        (name for name in z.columns if name.startswith("z_")),
        key=lambda name: int(name[2:]),
    )
    points = z[z_columns].to_numpy()
    row_leaves = tree_rows["leaf"].to_numpy()
    order = np.argsort(row_leaves, kind="stable")
    node_ids, starts, sizes = np.unique(
        row_leaves[order], return_index=True, return_counts=True
    )
    moments = {}
    for node_id, start, n_rows in zip(node_ids, starts, sizes, strict=True):
        leaf_points = points[order[start : start + n_rows]].astype(np.float64)
        mean = leaf_points.mean(axis=0)
        moments[int(node_id)] = (
            int(n_rows),
            mean,
            leaf_points.T @ leaf_points / n_rows - np.outer(mean, mean),
        )
    return moments


def class_plane(
    space_dir: Path, tree_dir: Path, leaf_ids: Sequence[int]
) -> ClassPlane | None:
    """Place the leaves of a class in its first two principal axes.

    The plane is the PCA of the class's train rows in z, so the leaves
    show where the tree cut the space the LDA made. A leaf's circle is
    centred on the mean of its train rows and its radius is their RMS
    distance from it in the plane.

    Args:
        space_dir: The folder of a space execution.
        tree_dir: The folder of a tree execution.
        leaf_ids: The node ids of the leaves of the class.

    Returns:
        The circles; None if a file is missing or the two do not have
        the same train length. The circles are empty when no leaf has a
        train row. With a single LDA axis the y of every circle is 0.
    """
    files = (space_dir / "z.parquet", tree_dir / "rows.parquet")
    if not all(file.is_file() for file in files):
        return None
    stats = [file.stat() for file in files]
    stamp = (
        stats[0].st_size,
        stats[0].st_mtime_ns,
        stats[1].st_size,
        stats[1].st_mtime_ns,
    )
    moments = _leaf_moments(space_dir, tree_dir, stamp)
    if moments is None:
        return None
    leaves = {node_id: moments[node_id] for node_id in leaf_ids if node_id in moments}
    if not leaves:
        return ClassPlane({}, (0.0, 0.0))
    n_class_rows = sum(n_rows for n_rows, _, _ in leaves.values())
    mean = (
        sum(n_rows * leaf_mean for n_rows, leaf_mean, _ in leaves.values())
        / n_class_rows
    )
    # The covariance of the class is the within-leaf one plus the one
    # between the leaves' means.
    covariance = (
        sum(
            n_rows * (leaf_cov + np.outer(leaf_mean - mean, leaf_mean - mean))
            for n_rows, leaf_mean, leaf_cov in leaves.values()
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
    circles = {}
    for node_id, (_, leaf_mean, leaf_cov) in leaves.items():
        centre = (leaf_mean - mean) @ components
        spread = np.trace(components.T @ leaf_cov @ components)
        radius = float(np.sqrt(max(spread, 0.0)))
        circles[node_id] = (
            float(centre[0]),
            float(centre[1]) if n_components > 1 else 0.0,
            radius,
        )
    return ClassPlane(circles, (shares[0], shares[1]))

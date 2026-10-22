"""The pure functions of src/ on toy data."""

import numpy as np
import pytest

from src import clustering
from src.graph import build_train_graph


def test_duplicate_points_join_inside_the_mst() -> None:
    """Check that identical rows join by their zero-length edges."""
    rng = np.random.default_rng(0)
    points = np.vstack([rng.normal(size=(50, 4)), np.tile(rng.normal(size=4), (10, 1))])
    duplicates = np.arange(50, 60)
    mst = build_train_graph(points, k=3, metric="euclidean", n_jobs=1).mst_edges
    assert len(mst) == len(points) - 1
    # Else each copy is bridged to node 0 alone, by a long edge.
    assert np.isin(mst, duplicates).all(axis=1).sum() == len(duplicates) - 1


def _split_on_refit(
    X: np.ndarray, *, max_fit_samples: int, random_state: int
) -> np.ndarray:
    """Label 2 clusters on the subsample and 3 on the full class."""
    return np.arange(len(X)) % (2 if len(X) <= max_fit_samples else 3)


@pytest.mark.parametrize("max_clusters", [2, 3])
def test_refit_keeps_at_most_max_clusters(
    monkeypatch: pytest.MonkeyPatch, max_clusters: int
) -> None:
    """Check that a subsampled class's refit stays within its cap."""
    monkeypatch.setitem(clustering.ALGORITHMS, "split_on_refit", _split_on_refit)
    rng = np.random.default_rng(0)
    nodes = clustering.KdnNodes(
        X=rng.normal(size=(60, 2)),
        y=np.repeat([0, 1], 30),
        row_ids=np.arange(1000, 1060),
    )
    labels, n_merged_clusters, _, _ = clustering.cluster_class(
        rng.normal(size=(210, 2)),
        row_ids=np.arange(210),
        label=0,
        algorithm="split_on_refit",
        params={},
        nodes=nodes,
        max_fit_samples=100,
        max_clusters=max_clusters,
        min_cluster_floor=10,
        hardness_k=3,
        eval_rows_per_train_row=1.0,
        metric="euclidean",
        random_state=0,
    )
    # Three refit clusters of 70 rows, all above the floor of 10: a tie
    # at the cut must still keep `max_clusters` of them.
    assert np.unique(labels).size == min(max_clusters, 3)
    assert n_merged_clusters == 3 - min(max_clusters, 3)

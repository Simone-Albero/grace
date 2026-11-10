"""src.clustering: hardness, merge, loss, grid, choice and assignment."""

import numpy as np
import pytest

from src.clustering import (
    ALGORITHMS,
    assign_nearest,
    candidate_grid,
    centroids,
    cluster_class,
    fit_hdbscan,
    fit_kmeans,
    granularity_loss,
    kdn,
    merge_small_clusters,
    n_clusters_grid,
    takes_n_clusters,
    tunable_params,
)
from src.graph import sample_rows


def test_tunable_params_and_n_clusters():
    assert tunable_params(fit_hdbscan) == (
        "min_cluster_size",
        "min_samples",
        "cluster_selection_method",
    )
    assert tunable_params(fit_kmeans) == ()
    assert takes_n_clusters(fit_kmeans)
    assert not takes_n_clusters(fit_hdbscan)


def test_kdn():
    labels = np.array([[0, 1, 1], [0, 0, 0]])
    np.testing.assert_allclose(kdn(labels, 0), [2 / 3, 0.0])


def test_centroids_skip_noise():
    X = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 10.0], [50.0, 50.0]])
    ids, centers = centroids(X, np.array([0, 0, 5, -1]))
    assert ids.tolist() == [0, 5]
    np.testing.assert_allclose(centers, [[1.0, 0.0], [10.0, 10.0]])


def test_merge_small_clusters():
    X = np.array([[0.0], [0.1], [0.2], [10.0], [10.1], [10.2], [9.5], [0.3]])
    labels = np.array([0, 0, 0, 1, 1, 1, 2, -1])
    merged = merge_small_clusters(X, labels, 2)
    assert merged.tolist() == [0, 0, 0, 1, 1, 1, 1, 0]


def test_merge_small_clusters_without_survivors():
    X = np.array([[0.0], [5.0], [9.0]])
    merged = merge_small_clusters(X, np.array([3, 4, -1]), 2)
    assert merged.tolist() == [0, 0, 0]


def test_granularity_loss_by_hand():
    labels = np.array([0, 0, 1, 1])
    hardness = np.array([0.0, 0.5, 1.0, 1.0])
    # within 0.5 · 0.125; noise 0.5 · 0.1875 / 2 + 0.5 · 1e-3 / 2
    loss = granularity_loss(labels, hardness, scale=1.0)
    assert loss == pytest.approx(0.0625 + 0.046875 + 0.00025)


def test_granularity_loss_pools_the_variance_of_a_singleton():
    labels = np.array([0, 0, 1])
    hardness = np.array([0.0, 1.0, 0.5])
    loss = granularity_loss(labels, hardness, scale=1.0)
    assert loss == pytest.approx(0.5 + 0.25 / 2 * 2 / 3 + 0.25 * 1 / 3)


def test_granularity_loss_scales_the_noise_by_the_class_rows():
    labels = np.array([0, 0, 1, 1])
    hardness = np.array([0.5, 0.5, 0.5, 0.5])
    small = granularity_loss(labels, hardness, scale=1.0)
    large = granularity_loss(labels, hardness, scale=10.0)
    assert large == pytest.approx(small / 10)


def test_n_clusters_grid():
    assert n_clusters_grid(1000, 20, 100) == [2, 3, 6, 12, 25, 50]
    assert n_clusters_grid(1000, 20, 10) == [2, 3, 6, 10]
    assert n_clusters_grid(10, 20, 100) == [2]


def test_candidate_grid():
    assert candidate_grid(fit_kmeans, {}, [2, 3]) == [
        {"n_clusters": 2},
        {"n_clusters": 3},
    ]
    params = {"min_cluster_size": [5, 10], "cluster_selection_method": "leaf"}
    assert candidate_grid(fit_hdbscan, params, [2, 3]) == [
        {"min_cluster_size": 5, "cluster_selection_method": "leaf"},
        {"min_cluster_size": 10, "cluster_selection_method": "leaf"},
    ]


def _blobs(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three far blobs of 100 rows; hardness 0, 1, then 0/1 in turn."""
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [50.0, 0.0], [0.0, 50.0]])
    X = np.vstack([rng.normal(c, 1.0, (100, 2)) for c in centers])
    blob = np.repeat([0, 1, 2], 100)
    hardness = np.concatenate([np.zeros(100), np.ones(100), np.tile([0.0, 1.0], 50)])
    return X, blob, hardness


@pytest.mark.parametrize("name", sorted(ALGORITHMS))
def test_every_algorithm_labels_every_row(name):
    X, _, _ = _blobs()
    algorithm = ALGORITHMS[name]
    extra = (
        {"n_clusters": 3} if takes_n_clusters(algorithm) else {"min_cluster_size": 10}
    )
    labels = algorithm(X, max_fit_samples=120, random_state=0, **extra)
    assert labels.shape == (300,)
    assert labels.min() >= -1


def test_cluster_class_never_mixes_blobs():
    X, blob, hardness = _blobs()
    result = cluster_class(
        X,
        np.arange(300),
        hardness,
        algorithm=fit_kmeans,
        params={},
        max_fit_samples=1000,
        min_cluster_floor=20,
        max_clusters=10,
        seed=0,
    )
    assert result.labels.shape == (300,)
    for label in np.unique(result.labels):
        assert len(np.unique(blob[result.labels == label])) == 1
    assert [c.params["n_clusters"] for c in result.candidates] == [2, 4, 8, 10]
    assert result.loss == min(c.loss for c in result.candidates)
    assert result.params in [c.params for c in result.candidates]


def test_cluster_class_respects_max_clusters_and_refits_past_the_sample():
    X, _, hardness = _blobs()
    scored = sample_rows(300, 100, 0)
    result = cluster_class(
        X,
        scored,
        hardness[scored],
        algorithm=fit_kmeans,
        params={},
        max_fit_samples=100,
        min_cluster_floor=20,
        max_clusters=2,
        seed=0,
    )
    assert result.labels.shape == (300,)
    assert len(np.unique(result.labels)) <= 2
    assert result.labels.min() >= 0


def test_assign_nearest():
    points = np.array([[0.0, 0.0], [9.0, 9.0], [5.0, 5.0]])
    centers = np.array([[10.0, 10.0], [1.0, 1.0], [9.0, 9.0]])
    assert assign_nearest(points, centers).tolist() == [1, 2, 1]
    ties = np.array([[1.0, 0.0], [-1.0, 0.0]])
    assert assign_nearest(np.zeros((1, 2)), ties).tolist() == [0]

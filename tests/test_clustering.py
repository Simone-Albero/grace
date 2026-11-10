"""src.clustering: weighted Ward tree, cut, vote, loss and choice."""

import itertools

import numpy as np
import pytest
from scipy.cluster.hierarchy import fcluster, is_valid_linkage, ward
from sklearn.metrics import adjusted_rand_score

from src.clustering import (
    best_cut,
    centroids,
    cluster_class,
    kdn,
    region_loss,
    vote,
    ward_tree,
)
from src.graph import sample_rows


def test_kdn():
    labels = np.array([[0, 1, 1], [0, 0, 0]])
    np.testing.assert_allclose(kdn(labels, 0), [2 / 3, 0.0])


def test_centroids_skip_noise():
    X = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 10.0], [50.0, 50.0]])
    ids, centers = centroids(X, np.array([0, 0, 5, -1]))
    assert ids.tolist() == [0, 5]
    np.testing.assert_allclose(centers, [[1.0, 0.0], [10.0, 10.0]])


def test_centroids_are_weighted():
    X = np.array([[0.0], [4.0], [10.0]])
    ids, centers = centroids(X, np.array([0, 0, 1]), np.array([3.0, 1.0, 2.0]))
    assert ids.tolist() == [0, 1]
    np.testing.assert_allclose(centers, [[1.0], [10.0]])


# --- ward_tree ---


def _linkage(tree, n: int) -> np.ndarray:
    """The tree as a scipy linkage matrix (sizes are row counts)."""
    sizes = np.ones(2 * n - 1)
    for j, (a, b) in enumerate(tree.children):
        sizes[n + j] = sizes[a] + sizes[b]
    return np.column_stack([tree.children, tree.heights, sizes[n:]]).astype(float)


def _cloud(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.vstack(
        [rng.normal(c, 1.0, size=(n // 3, 4)) for c in (0.0, 4.0, 8.0)]
        + [rng.normal(0.0, 1.0, size=(n - 3 * (n // 3), 4))]
    )


def test_ward_tree_with_unit_weights_is_scipy_ward():
    X = _cloud(90, 0)
    tree = ward_tree(X, np.ones(len(X)))
    Z = _linkage(tree, len(X))
    assert is_valid_linkage(Z)
    reference = ward(X)
    np.testing.assert_allclose(tree.heights, reference[:, 2])
    for k in (2, 3, 5, 9):
        ours = fcluster(Z, k, "maxclust")
        assert adjusted_rand_score(ours, fcluster(reference, k, "maxclust")) == 1.0


def test_ward_tree_weights_are_copies_of_the_rows():
    X = _cloud(60, 1)
    w = np.random.default_rng(1).integers(1, 4, size=len(X))
    tree = ward_tree(X, w.astype(float))
    reference = ward(np.repeat(X, w, axis=0))
    # The copies merge first, at height 0; the other merges are the tree's.
    np.testing.assert_allclose(tree.heights, reference[-(len(X) - 1) :, 2], atol=1e-9)
    Z = _linkage(tree, len(X))
    for k in (2, 3, 6):
        ours = np.repeat(fcluster(Z, k, "maxclust"), w)
        assert adjusted_rand_score(ours, fcluster(reference, k, "maxclust")) == 1.0


def test_ward_tree_orders_children_before_parents():
    X = _cloud(40, 2)
    tree = ward_tree(X, np.ones(len(X)))
    n = len(X)
    assert tree.children.shape == (n - 1, 2) and tree.children.dtype == np.int64
    assert (tree.children < n + np.arange(n - 1)[:, None]).all()
    assert sorted(tree.children.ravel().tolist()) == list(range(2 * n - 2))
    assert (np.diff(tree.heights) >= 0).all()


def test_ward_tree_of_one_row_has_no_merge():
    tree = ward_tree(np.zeros((1, 3)), np.ones(1))
    assert tree.children.shape == (0, 2) and len(tree.heights) == 0


# --- best_cut ---


def _cost(rows, hardness, weights, scale):
    """The cost of best_cut of one region, from its docstring."""
    total = weights.sum()

    def variance(index):
        w, h = weights[index], hardness[index]
        mean = np.average(h, weights=w)
        return np.average((h - mean) ** 2, weights=w) * len(index) / (len(index) - 1)

    rows = np.asarray(rows)
    w, h = weights[rows], hardness[rows]
    mean = np.average(h, weights=w)
    var = variance(rows) if len(rows) > 1 else variance(np.arange(len(hardness)))
    noise = max(mean * (1 - mean), 1e-3) / (len(rows) * scale)
    return w.sum() / total * (var + noise)


def _cuts(tree, n, floor):
    """Every cut of the tree under the pruning rule: lists of regions."""
    members = [[i] for i in range(n)]
    for a, b in tree.children:
        members.append(members[a] + members[b])

    def splits(v):
        if v < n:
            return []
        a, b = tree.children[v - n]
        small = [len(members[c]) < floor for c in (a, b)]
        if all(small):
            return []
        if small[0]:
            return splits(b)
        if small[1]:
            return splits(a)
        return [x + y for x, y in itertools.product(options(a), options(b))]

    def options(v):
        return [[members[v]], *splits(v)]

    return options(2 * n - 2)


def _objective(labels, hardness, weights, scale, penalty):
    regions = [np.flatnonzero(labels == r) for r in np.unique(labels[labels >= 0])]
    return sum(_cost(r, hardness, weights, scale) + penalty for r in regions)


@pytest.mark.parametrize("floor", [1, 2, 3])
@pytest.mark.parametrize("penalty", [0.0, 0.01])
def test_best_cut_is_the_least_of_every_cut(floor, penalty):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(9, 2))
    weights = rng.integers(1, 4, size=9).astype(float)
    hardness = rng.choice([0.0, 0.3, 1.0], size=9)
    tree = ward_tree(X, weights)
    labels, _ = best_cut(
        tree, weights, hardness, scale=2.0, floor=floor, penalty=penalty
    )
    best = min(
        sum(_cost(r, hardness, weights, 2.0) + penalty for r in cut)
        for cut in _cuts(tree, 9, floor)
    )
    got = _objective(labels, hardness, weights, 2.0, penalty)
    assert got == pytest.approx(best)


def _blobs(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three far blobs of 100 rows; hardness 0, 1, then 0/1 in turn."""
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [50.0, 0.0], [0.0, 50.0]])
    X = np.vstack([rng.normal(c, 1.0, (100, 2)) for c in centers])
    blob = np.repeat([0, 1, 2], 100)
    hardness = np.concatenate([np.zeros(100), np.ones(100), np.tile([0.0, 1.0], 50)])
    return X, blob, hardness


def _region_tree_is_valid(tree, labels):
    n = len(tree.parent)
    assert tree.parent[0] == -1 and (tree.parent[1:] >= 0).all()
    assert (tree.parent[1:] < np.arange(1, n)).all()
    children = np.bincount(tree.parent[1:], minlength=n)
    leaf = children == 0
    assert set(children[~leaf].tolist()) <= {2}
    assert (tree.region[~leaf] == -1).all()
    assert tree.region[leaf].tolist() == list(range(leaf.sum()))
    assert set(labels[labels >= 0].tolist()) == set(range(leaf.sum()))


def test_best_cut_splits_blobs_of_different_hardness():
    X, blob, hardness = _blobs()
    weights = np.ones(len(X))
    labels, tree = best_cut(
        ward_tree(X, weights), weights, hardness, scale=1.0, floor=20
    )
    _region_tree_is_valid(tree, labels)
    kept = labels >= 0
    for region in np.unique(labels[kept]):
        assert len(np.unique(blob[kept][labels[kept] == region])) == 1
    for b in (0, 1):
        assert len(np.unique(labels[(blob == b) & kept])) == 1


def test_best_cut_keeps_homogeneous_hardness_whole():
    X, _, _ = _blobs()
    weights = np.ones(len(X))
    labels, tree = best_cut(
        ward_tree(X, weights), weights, np.full(len(X), 0.2), scale=1.0, floor=5
    )
    assert labels.tolist() == [0] * len(X)
    assert tree.parent.tolist() == [-1] and tree.region.tolist() == [0]


def test_a_far_small_branch_is_pruned_without_blocking_its_sibling():
    rng = np.random.default_rng(4)
    X = np.vstack(
        [rng.normal(0, 0.3, (50, 2)), rng.normal([3, 0], 0.3, (50, 2)), [[500, 500]]]
    )
    hardness = np.concatenate([np.zeros(50), np.ones(50), [0.5]])
    weights = np.ones(len(X))
    labels, tree = best_cut(
        ward_tree(X, weights), weights, hardness, scale=1.0, floor=5
    )
    assert labels[-1] == -1
    assert len(np.unique(labels[:50])) == 1 and len(np.unique(labels[50:100])) == 1
    assert labels[0] != labels[50]
    assert len(tree.parent) == 3


def test_a_larger_penalty_never_gives_more_regions():
    X = _cloud(150, 5)
    rng = np.random.default_rng(5)
    hardness = rng.random(len(X)) * (X[:, 0] > 2)
    weights = rng.integers(1, 5, size=len(X)).astype(float)
    tree = ward_tree(X, weights)
    counts = []
    for penalty in [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
        labels, region_tree = best_cut(
            tree, weights, hardness, scale=1.0, floor=3, penalty=penalty
        )
        _region_tree_is_valid(region_tree, labels)
        counts.append(int((region_tree.region >= 0).sum()))
    assert counts == sorted(counts, reverse=True)
    assert counts[-1] == 1


# --- vote and loss ---


def test_vote_takes_the_majority_then_the_nearest():
    X = np.array([[0.0], [1.0], [1.1], [0.2]])
    labels = np.array([0, 1, 1, -1])
    assert vote(X, labels, 3).tolist() == [0, 1, 1, 1]
    assert vote(X, labels, 2).tolist() == [0, 1, 1, 0]
    assert vote(X, np.array([2, -1, -1, -1]), 5).tolist() == [2, 2, 2, 2]


def test_region_loss_by_hand():
    labels = np.array([0, 0, 1])
    hardness = np.array([0.0, 1.0, 0.5])
    weights = np.array([1.0, 3.0, 2.0])
    # Region 0: W 4/6, mean 0.75, variance 0.1875 · 2, noise 0.1875 / 2.
    first = 4 / 6 * (0.375 + 0.09375)
    # Region 1 has one row: the variance of all rows, 0.8333 / 6 · 3 / 2.
    second = 2 / 6 * (5 / 6 / 6 * 1.5 + 0.25)
    loss = region_loss(labels, hardness, weights, scale=1.0)
    assert loss == pytest.approx(first + second)


def test_region_loss_scales_the_noise_by_the_class_rows():
    labels = np.array([0, 0, 1, 1])
    hardness = np.full(4, 0.5)
    small = region_loss(labels, hardness, np.ones(4), scale=1.0)
    large = region_loss(labels, hardness, np.ones(4), scale=10.0)
    assert large == pytest.approx(small / 10)


# --- cluster_class ---


def test_cluster_class_on_every_row():
    X, blob, hardness = _blobs()
    result = cluster_class(
        X,
        np.ones(len(X)),
        np.arange(len(X)),
        hardness,
        min_cluster_floor=20,
        max_clusters=10,
        vote_k=5,
    )
    assert result.labels.shape == (300,) and result.labels.min() == 0
    _region_tree_is_valid(result.tree, result.labels)
    assert result.labels.max() + 1 <= 10
    assert result.penalty == 0.0
    for region in np.unique(result.labels):
        assert len(np.unique(blob[result.labels == region])) == 1
    penalties = [c.penalty for c in result.candidates]
    assert penalties == sorted(penalties, reverse=True) and penalties[-1] == 0.0
    assert result.candidates[0].n_clusters == 1


def test_cluster_class_votes_the_rows_outside_the_sample():
    X, _, hardness = _blobs()
    scored = sample_rows(300, 100, 0)
    result = cluster_class(
        X,
        np.ones(len(X)),
        scored,
        hardness[scored],
        min_cluster_floor=20,
        max_clusters=10,
        vote_k=5,
    )
    assert result.labels.shape == (300,) and result.labels.min() == 0
    assert result.n_voted >= 200
    _region_tree_is_valid(result.tree, result.labels)


def test_cluster_class_caps_the_regions_with_a_penalty():
    X, _, hardness = _blobs()
    result = cluster_class(
        X,
        np.ones(len(X)),
        np.arange(len(X)),
        hardness,
        min_cluster_floor=20,
        max_clusters=2,
        vote_k=5,
    )
    assert result.labels.max() + 1 <= 2
    assert result.penalty > 0.0

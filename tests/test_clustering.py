"""src.clustering: weighted Ward tree, halves, separation, cut, sewing, regions, overlap zones."""

import itertools

import numpy as np
import pytest
from scipy.cluster.hierarchy import fcluster, is_valid_linkage, ward
from sklearn.metrics import adjusted_rand_score

import src.clustering as clustering
from src.clustering import (
    centroids,
    cluster_class,
    line_spread,
    merge_pieces,
    node_members,
    node_separation,
    overlap_pairs,
    overlap_zones,
    pair_separation,
    region_stability,
    regions_tree,
    separated_cut,
    separation,
    split_halves,
    vote,
    ward_tree,
)


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


# --- vote ---


def test_vote_takes_the_majority_then_the_nearest():
    X = np.array([[0.0], [1.0], [1.1], [0.2]])
    labels = np.array([0, 1, 1, -1])
    assert vote(X, labels, 3).tolist() == [0, 1, 1, 1]
    assert vote(X, labels, 2).tolist() == [0, 1, 1, 0]
    assert vote(X, np.array([2, -1, -1, -1]), 5).tolist() == [2, 2, 2, 2]


# --- halves and members ---


def test_split_halves():
    A, B = split_halves(101, 1000, 0)
    assert (len(A), len(B)) == (51, 50)
    assert sorted(np.r_[A, B].tolist()) == list(range(101))
    assert A.tolist() == sorted(A.tolist()) and B.tolist() == sorted(B.tolist())
    assert A.dtype == np.int64 and B.dtype == np.int64
    again = split_halves(101, 1000, 0)
    assert A.tolist() == again[0].tolist() and B.tolist() == again[1].tolist()
    A, B = split_halves(10, 3, 1)
    assert (len(A), len(B)) == (3, 3) and not set(A) & set(B)
    A, B = split_halves(1, 5, 0)
    assert A.tolist() == [0] and B.tolist() == []


def _below(tree, n: int, v: int) -> set[int]:
    """The rows (leaves) below node v of a WardTree of n rows."""
    if v < n:
        return {v}
    a, b = tree.children[v - n]
    return _below(tree, n, a) | _below(tree, n, b)


def test_node_members_attach_each_row_of_B_to_its_nearest_row_of_A():
    rng = np.random.default_rng(0)
    XA, XB = rng.normal(size=(30, 2)), rng.normal(size=(50, 2))
    tree = ward_tree(XA, np.ones(30))
    members = node_members(XA, XB, tree)
    nearest = np.argmin(((XB[:, None, :] - XA[None, :, :]) ** 2).sum(2), axis=1)
    assert members.assign.tolist() == nearest.tolist()
    for v in range(2 * 30 - 1):
        rows = set(members.order[members.start[v] : members.stop[v]].tolist())
        below = _below(tree, 30, v)
        assert rows == {b for b in range(50) if nearest[b] in below}


# --- separation ---


def test_pair_separation_by_hand():
    # The line runs along x (from the centroids of A); y plays no part.
    tree_a, tree_b = np.array([[0.0, 0.0], [0.0, 2.0]]), np.array([[1.0, 1.0]])
    measure_a = np.array([[-1.0, 5.0], [1.0, -5.0]])  # mean 0, variance 1
    measure_b = np.array([[3.0, 100.0], [5.0, -100.0]])  # mean 4, variance 1
    S, low = pair_separation(tree_a, tree_b, measure_a, measure_b)
    assert S == pytest.approx(4 / np.sqrt(2))
    # Var(S) = (1/2 + 1/2) / 2 + 16 / (4 · 8) · 0 (fourth moment 1 = V²).
    assert low == pytest.approx(4 / np.sqrt(2) - 2 * np.sqrt(0.5))


def test_pair_separation_edge_cases():
    a, b = np.array([[0.0]]), np.array([[1.0]])
    spread = np.array([[-1.0], [1.0]])
    assert pair_separation(a, a, spread, spread + 3) == (0.0, -np.inf)
    assert pair_separation(
        a, b, np.array([[0.0], [0.0]]), np.array([[2.0], [2.0]])
    ) == (
        np.inf,
        np.inf,
    )
    S, low = pair_separation(a, b, spread + 3, spread)  # B lies the other way
    assert S == 0.0 and low < 0.0


def test_pair_separation_standard_error_matches_repeated_draws():
    rng = np.random.default_rng(1)
    a, b = np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]])
    draws = [
        pair_separation(
            a, b, rng.normal(size=(200, 2)), rng.normal(size=(200, 2)) + [3.0, 0.0]
        )
        for _ in range(400)
    ]
    S = np.array([d[0] for d in draws])
    se = (S - np.array([d[1] for d in draws])) / 2
    assert S.mean() == pytest.approx(3 / np.sqrt(2), rel=0.05)
    assert se.mean() == pytest.approx(S.std(), rel=0.15)


def test_node_separation_judges_only_merges_with_floor_rows_of_B():
    rng = np.random.default_rng(2)
    X = np.vstack([rng.normal(size=(300, 2)), rng.normal(size=(300, 2)) + [8.0, 0.0]])
    A, B = split_halves(len(X), 10_000, 0)
    tree = ward_tree(X[A], np.ones(len(A)))
    members = node_members(X[A], X[B], tree)
    S, low = node_separation(X[A], X[B], tree, members, 20)
    n = len(A)
    count = members.stop - members.start
    for j, (a, b) in enumerate(tree.children):
        judged = count[a] >= 20 and count[b] >= 20
        assert np.isnan(S[j]) != judged and np.isnan(low[j]) != judged
    a, b = tree.children[-1]
    rows = lambda v: members.order[members.start[v] : members.stop[v]]  # noqa: E731
    expected = pair_separation(
        X[A][sorted(_below(tree, n, a))],
        X[A][sorted(_below(tree, n, b))],
        X[B][rows(a)],
        X[B][rows(b)],
    )
    assert (S[-1], low[-1]) == pytest.approx(expected)
    assert low[-1] > 4.0  # the two blobs, 8 sd apart


# --- cut and sewing ---


def _line(
    k: int, n: int, d: int, gap: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """k blobs of n rows, sd 1 in d dimensions, gap apart along the first axis."""
    rng = np.random.default_rng(seed)
    X = np.vstack(
        [rng.normal(size=(n, d)) + np.r_[gap * i, np.zeros(d - 1)] for i in range(k)]
    )
    return X, np.repeat(np.arange(k), n)


def _cut_of(X, floor=20, threshold=4.0, seed=0):
    A, B = split_halves(len(X), 10_000, seed)
    tree = ward_tree(X[A], np.ones(len(A)))
    members = node_members(X[A], X[B], tree)
    S, low = node_separation(X[A], X[B], tree, members, floor)
    pieces = separated_cut(tree, members, floor, low, threshold)
    return A, B, tree, members, low, pieces


def test_separated_cut_goes_from_the_leaves_up():
    # Three blobs in a line: the root splits one blob from the other two,
    # whose spread along the line hides the gap; the split below does not.
    X, blob = _line(3, 600, 5, 8.0, 3)
    A, _, tree, _, low, pieces = _cut_of(X)
    assert low[-1] < 4.0
    kept = pieces >= 0
    for p in np.unique(pieces[kept]):
        # A row of a blob's tail may sit with the next blob.
        assert np.bincount(blob[A][pieces == p]).max() / np.sum(pieces == p) > 0.99
    assert set(blob[A][kept].tolist()) == {0, 1, 2}
    assert pieces[kept].min() == 0 and len(np.unique(pieces[kept])) == pieces.max() + 1


def test_separated_cut_prunes_a_small_far_branch_without_blocking_its_sibling():
    rng = np.random.default_rng(4)
    X = np.vstack(
        [
            rng.normal(size=(200, 2)),
            rng.normal(size=(200, 2)) + [10, 0],
            rng.normal(size=(6, 2)) * 0.1 + 500,
        ]
    )
    A, _, _, _, _, pieces = _cut_of(X)
    far = A >= 400
    assert (pieces[far] == -1).all()
    first, second = pieces[A < 200], pieces[(A >= 200) & (A < 400)]
    assert len(np.unique(first)) == 1 and len(np.unique(second)) == 1
    assert first[0] != second[0] and first[0] >= 0


def test_merge_pieces_sews_a_shattered_blob():
    # A satellite of 1% at 6 sd, which Ward joins deep inside the big
    # blob: reaching it cuts the big blob in pieces that are sewn back.
    rng = np.random.default_rng(1007)
    X = np.vstack([rng.normal(size=(7920, 2)), rng.normal(size=(80, 2)) + [6.0, 0.0]])
    A, B, _, members, _, pieces = _cut_of(X, threshold=3.0)
    regions, used = merge_pieces(X[A], X[B], pieces, members.assign, 3.0, 50)
    assert used == 3.0
    kept = regions >= 0
    assert regions[kept].max() == 1
    satellite = regions[kept & (A >= 7920)]
    big = regions[kept & (A < 7920)]
    own, home = np.bincount(satellite).argmax(), np.bincount(big).argmax()
    assert own != home
    assert np.mean(satellite == own) > 0.9 and np.mean(big == home) > 0.99


def test_merge_pieces_keeps_blobs_in_a_line_and_obeys_the_cap():
    X, blob = _line(3, 600, 5, 8.0, 3)
    A, B, _, members, _, pieces = _cut_of(X)
    regions, used = merge_pieces(X[A], X[B], pieces, members.assign, 4.0, 50)
    assert used == 4.0 and regions.max() == 2
    assert adjusted_rand_score(blob[A][regions >= 0], regions[regions >= 0]) > 0.99
    capped, used = merge_pieces(X[A], X[B], pieces, members.assign, 4.0, 2)
    assert capped.max() == 1 and used > 4.0


# --- cluster_class ---


def _class(X, numeric=None, weights=None, **changes):
    options = dict(
        max_fit_samples=20_000,
        min_cluster_floor=20,
        max_clusters=50,
        split_separation=4.0,
        vote_k=10,
        seed=0,
    )
    options.update(changes)
    numeric = np.ones(X.shape[1], dtype=bool) if numeric is None else numeric
    weights = np.ones(len(X)) if weights is None else weights
    return cluster_class(X, numeric, weights, **options)


def test_cluster_class_finds_the_blobs():
    X, blob = _line(3, 600, 5, 8.0, 3)
    result = _class(X)
    assert result.labels.shape == (1800,) and result.labels.min() == 0
    assert adjusted_rand_score(blob, result.labels) > 0.99
    assert result.threshold == 4.0 and result.n_pieces >= 3
    assert result.sample.tolist() == list(range(1800))  # every row in a half
    thresholds, counts = result.curve
    assert thresholds.tolist() == sorted(thresholds.tolist()) and 4.0 in thresholds
    assert counts.tolist() == sorted(counts.tolist(), reverse=True)


def test_cluster_class_one_gaussian_is_one_region():
    X = np.random.default_rng(5).normal(size=(4000, 5))
    assert _class(X).labels.max() == 0


def test_cluster_class_does_not_judge_a_split_on_the_rows_it_was_made_on():
    # 80 rows of one Gaussian in 60 dimensions: on the rows Ward built it
    # on, the root split looks far apart; on the other half it does not.
    X = np.random.default_rng(6).normal(size=(80, 60))
    A, B = split_halves(80, 10_000, 0)
    tree = ward_tree(X[A], np.ones(len(A)))
    a, b = tree.children[-1]
    left, right = sorted(_below(tree, len(A), a)), sorted(_below(tree, len(A), b))
    in_sample, _ = pair_separation(X[A][left], X[A][right], X[A][left], X[A][right])
    assert in_sample > 3.0
    assert _class(X, min_cluster_floor=5, split_separation=3.0).labels.max() == 0


def test_cluster_class_measures_the_numbers_only():
    # Two groups apart only on a one-hot column: the same numbers.
    rng = np.random.default_rng(7)
    numbers = rng.normal(size=(2000, 3))
    slots = np.zeros((2000, 2))
    slots[:1000, 0] = slots[1000:, 1] = np.sqrt(2.0)
    X = np.hstack([numbers, slots])
    numeric = np.array([True, True, True, False, False])
    assert _class(X, numeric).labels.max() == 0
    assert _class(X).labels.max() == 1  # judged on the slots too, they split


def test_cluster_class_copies_do_not_make_regions():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(3000, 2))
    weights = np.ones(3000)
    weights[rng.choice(3000, 5, replace=False)] = 1000.0
    assert _class(X, weights=weights).labels.max() == 0


def test_cluster_class_labels_only_the_rows_asked_and_votes_the_rest():
    X, _ = _line(2, 30_000, 2, 10.0, 9)  # 60000 rows: halves of 20000
    result = _class(X)
    assert len(result.sample) == 40_000
    assert result.n_voted >= 20_000
    only = result.sample[::7]
    again = _class(X, seed=1, only=only)
    assert again.labels.shape == only.shape


def test_cluster_class_caps_the_regions():
    X, _ = _line(3, 600, 5, 8.0, 3)
    result = _class(X, max_clusters=2)
    assert result.labels.max() == 1 and result.threshold > 4.0


# --- regions of a class ---


def test_region_stability_by_hand():
    labels = np.array([0, 0, 0, 0, 1, 1])
    np.testing.assert_allclose(region_stability(labels, labels), [1.0, 1.0])
    other = np.array([0, 0, 1, 1, 2, 2])
    np.testing.assert_allclose(region_stability(labels, other), [0.5, 1.0])
    np.testing.assert_allclose(
        region_stability(labels, np.zeros(6, int)), [4 / 6, 2 / 6]
    )


def _region_tree_is_valid(tree, labels):
    n = len(tree.parent)
    assert tree.parent[0] == -1 and (tree.parent[1:] >= 0).all()
    assert (tree.parent[1:] < np.arange(1, n)).all()
    children = np.bincount(tree.parent[1:], minlength=n)
    leaf = children == 0
    assert set(children[~leaf].tolist()) <= {2}
    assert (tree.region[~leaf] == -1).all()
    assert tree.region[leaf].tolist() == list(range(leaf.sum()))
    assert set(labels.tolist()) == set(range(leaf.sum()))


def test_regions_tree_leaves_are_the_regions_in_pre_order():
    X, blob = _line(4, 50, 2, 10.0, 10)
    labels = np.array([2, 0, 3, 1])[blob]
    renamed, tree = regions_tree(X, np.ones(len(X)), labels)
    _region_tree_is_valid(tree, renamed)
    assert adjusted_rand_score(labels, renamed) == 1.0
    one, single = regions_tree(X, np.ones(len(X)), np.zeros(len(X), int))
    assert one.tolist() == [0] * len(X)
    assert single.parent.tolist() == [-1] and single.region.tolist() == [0]


# --- overlap zones ---


def _spread_by_hand(X, labels, weights, centers, a, b) -> float:
    """The weighted sd of a's rows along the line from c_a to c_b."""
    line = centers[b] - centers[a]
    if not np.linalg.norm(line):
        return 0.0
    line = line / np.linalg.norm(line)
    rows = labels == a
    along = (X[rows] - centers[a]) @ line
    return float(np.sqrt(np.average(along**2, weights=weights[rows])))


@pytest.mark.parametrize("chunk", [65536, 3])
def test_line_spread_by_hand(monkeypatch, chunk):
    monkeypatch.setattr(clustering, "_CHUNK", chunk)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3)) * [3.0, 1.0, 0.2]
    labels = rng.integers(0, 4, size=40)
    labels[:4] = [0, 1, 2, 3]
    weights = rng.integers(1, 6, size=40).astype(float)
    _, centers = centroids(X, labels, weights)
    centers[3] = centers[2]  # two coinciding centroids
    spread = line_spread(X, labels, weights, centers)
    assert spread.shape == (4, 4)
    for a, b in itertools.product(range(4), repeat=2):
        expected = 0.0 if a == b else _spread_by_hand(X, labels, weights, centers, a, b)
        assert spread[a, b] == pytest.approx(expected, abs=1e-12)


def test_separation_by_hand():
    centers = np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 4.0], [0.0, 1.0]])
    spread = np.array(
        [
            [0.0, 1.0, 1.0, 0.0],
            [2.0, 0.0, 0.0, 5.0],
            [2.0, 0.0, 0.0, 5.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    apart = separation(centers, spread)
    np.testing.assert_array_equal(apart, apart.T)
    np.testing.assert_array_equal(np.diag(apart), 0.0)
    assert apart[0, 1] == pytest.approx(5.0 / np.sqrt(1.0 + 4.0))
    assert apart[1, 2] == 0.0  # the same centroid
    assert apart[0, 3] == np.inf  # apart, and neither spreads along the line
    assert apart[1, 3] == pytest.approx(np.sqrt(9.0 + 9.0) / 5.0)


def test_separation_sees_blobs_apart_in_many_dimensions():
    # Two blobs of sd 1 in 200 dimensions, their centroids 6 apart: far
    # apart along the line between them, though the root mean square
    # radius of each (about sqrt(200)) dwarfs the distance.
    rng = np.random.default_rng(1)
    d = 200
    X = rng.normal(size=(2000, d))
    X[1000:, 0] += 6.0
    labels = np.repeat([0, 1], 1000)
    weights = np.ones(2000)
    _, centers = centroids(X, labels, weights)
    apart = separation(centers, line_spread(X, labels, weights, centers))
    assert apart[0, 1] == pytest.approx(6.0 / np.sqrt(2.0), rel=0.1)
    X[1000:, 0] -= 5.5  # now 0.5 apart
    _, centers = centroids(X, labels, weights)
    apart = separation(centers, line_spread(X, labels, weights, centers))
    assert apart[0, 1] < 1.0


def _apart(m: int, pairs: dict[tuple[int, int], float], rest: float = 5.0):
    """A symmetric separation matrix, rest off the given pairs, 0 on the diagonal."""
    apart = np.full((m, m), rest)
    np.fill_diagonal(apart, 0.0)
    for (a, b), value in pairs.items():
        apart[a, b] = apart[b, a] = value
    return apart


def test_overlap_pairs_by_hand():
    apart = _apart(5, {(0, 1): 0.2, (1, 2): 0.5, (0, 2): 0.1, (3, 4): 0.5, (2, 4): 1.0})
    classes = np.array([0, 1, 0, 2, 1])
    pairs = overlap_pairs(apart, classes, 1.0)
    # (0, 2) is one class; (2, 4) sits on the threshold.
    assert pairs.tolist() == [[0, 1], [1, 2], [3, 4]]
    assert pairs.dtype == np.int64
    assert overlap_pairs(apart, classes, 0.1).shape == (0, 2)


def test_overlap_zones_by_hand():
    apart = _apart(6, {(0, 1): 0.2, (1, 2): 0.5, (0, 2): 0.1, (3, 4): 0.7, (2, 4): 1.0})
    classes = np.array([0, 1, 0, 2, 1, 2])
    zones = overlap_zones(apart, classes, 1.0)
    # Zone 0: 0 and 1 meet at 0.2, then 2 joins 1 at 0.5 (0-2 is one
    # class). Zone 1: 3 and 4 at 0.7. 2-4 sits on the threshold and 5
    # is close to nothing.
    assert zones.parent.tolist() == [-1, 0, 1, 1, 0, -1, 5, 5]
    assert zones.zone.tolist() == [0, 0, 0, 0, 0, 1, 1, 1]
    assert zones.region.tolist() == [-1, -1, 0, 1, 2, -1, 3, 4]
    np.testing.assert_array_equal(
        zones.separation, [0.5, 0.2, np.nan, np.nan, np.nan, 0.7, np.nan, np.nan]
    )
    assert zones.link.tolist() == [
        [1, 2],
        [0, 1],
        [-1, -1],
        [-1, -1],
        [-1, -1],
        [3, 4],
        [-1, -1],
        [-1, -1],
    ]


def test_overlap_zones_without_overlap_are_empty():
    zones = overlap_zones(_apart(3, {}), np.array([0, 1, 2]), 1.0)
    for array in (zones.parent, zones.zone, zones.region, zones.separation):
        assert array.shape == (0,)
    assert zones.link.shape == (0, 2)


def _zone_below(zones, node: int) -> set[int]:
    """The regions of the leaves below a node of the zones."""
    stack, regions = [node], set()
    while stack:
        v = stack.pop()
        if zones.region[v] >= 0:
            regions.add(int(zones.region[v]))
        stack += np.flatnonzero(zones.parent == v).tolist()
    return regions


def test_overlap_zones_are_the_components_of_the_pairs():
    rng = np.random.default_rng(4)
    m = 30
    upper = np.triu(rng.uniform(0.0, 4.0, size=(m, m)), 1)
    apart = upper + upper.T
    classes = rng.integers(0, 4, size=m)
    zones = overlap_zones(apart, classes, 1.0)
    pairs = overlap_pairs(apart, classes, 1.0)
    leaves = np.flatnonzero(zones.region >= 0)
    assert sorted(zones.region[leaves].tolist()) == sorted(set(pairs.ravel().tolist()))
    zone_of = dict(zip(zones.region[leaves].tolist(), zones.zone[leaves].tolist()))
    for a, b in pairs:
        assert zone_of[a] == zone_of[b]
    # A zone is connected by its pairs: each merge is one of them.
    internal = np.flatnonzero(zones.region < 0)
    for zone in np.unique(zones.zone):
        n_leaves = np.sum(zones.zone[leaves] == zone)
        assert np.sum(zones.zone[internal] == zone) == n_leaves - 1
    first = [
        np.nanmin(zones.separation[zones.zone == z]) for z in np.unique(zones.zone)
    ]
    assert first == sorted(first)
    for v in range(len(zones.parent)):
        up = zones.parent[v]
        if up >= 0:
            assert up < v and zones.zone[up] == zones.zone[v]
        else:
            assert v == 0 or zones.zone[v] == zones.zone[v - 1] + 1
    for v in internal:
        left, right = np.flatnonzero(zones.parent == v)
        a, b = zones.link[v]
        assert a in _zone_below(zones, left) and b in _zone_below(zones, right)
        assert classes[a] != classes[b]
        assert zones.separation[v] == apart[a, b] < 1.0
        for kid in (left, right):
            if zones.region[kid] < 0:
                assert zones.separation[kid] <= zones.separation[v]
        # Single linkage: no pair across the two sides is closer.
        across = [
            apart[x, y]
            for x in _zone_below(zones, left)
            for y in _zone_below(zones, right)
            if classes[x] != classes[y]
        ]
        assert min(across) == zones.separation[v]

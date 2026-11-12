"""src.cells: the weighted Ward tree, the cells of a class and their merge into regions."""

import numpy as np
from scipy.cluster.hierarchy import fcluster, is_valid_linkage, ward
from sklearn.metrics import adjusted_rand_score

from src.cells import CellTree, class_cells, merge_cells, ward_tree
from src.graph import sample_rows

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


# --- cells of a class ---


def _blobs(sizes, d=3, gap=12.0, seed=0) -> tuple[np.ndarray, np.ndarray]:
    """Blobs of the given sizes, sd 1 in d dimensions, gap apart along the first axis."""
    rng = np.random.default_rng(seed)
    X = np.vstack(
        [
            rng.normal(size=(n, d)) + np.r_[gap * i, np.zeros(d - 1)]
            for i, n in enumerate(sizes)
        ]
    )
    return X, np.repeat(np.arange(len(sizes)), sizes)


def _cells(X, weights=None, max_rows=500, min_rows=20, max_fit_samples=20_000, seed=0):
    weights = np.ones(len(X)) if weights is None else weights
    return class_cells(X, weights, max_rows, min_rows, max_fit_samples, seed)


def _tree_is_valid(tree: CellTree, labels: np.ndarray) -> None:
    """Pre-order, two children per internal node, the cells 0.. in order at the leaves."""
    k = len(tree.parent)
    assert tree.parent[0] == -1 and (tree.parent[1:] >= 0).all()
    assert (tree.parent[1:] < np.arange(1, k)).all()
    children = np.bincount(tree.parent[1:], minlength=k)
    leaf = children == 0
    assert set(children[~leaf].tolist()) <= {2}
    assert (tree.cell[~leaf] == -1).all()
    assert tree.cell[leaf].tolist() == list(range(leaf.sum()))
    assert sorted(set(labels.tolist())) == list(range(leaf.sum()))


def test_class_cells_keep_blobs_apart_and_whole():
    # Three blobs of 400: the root (1200) and the node of two blobs (800)
    # are split, each blob (400) is a cell.
    X, blob = _blobs([400, 400, 400])
    labels, tree = _cells(X, max_rows=500)
    _tree_is_valid(tree, labels)
    assert labels.max() == 2 and adjusted_rand_score(blob, labels) == 1.0
    assert len(tree.parent) == 5


def test_class_cells_split_a_large_blob_down_to_max_rows():
    X = np.random.default_rng(1).normal(size=(5000, 3))
    labels, tree = _cells(X, max_rows=400)
    _tree_is_valid(tree, labels)
    sizes = np.bincount(labels)
    assert sizes.min() >= 20 and sizes.max() <= 400 * 1.1
    assert np.median(sizes) >= 100 and len(sizes) >= 5000 // 400


def test_class_cells_of_a_small_class_are_one_cell():
    X = np.random.default_rng(2).normal(size=(30, 3))
    labels, tree = _cells(X, max_rows=500)
    assert labels.tolist() == [0] * 30
    assert tree.parent.tolist() == [-1] and tree.cell.tolist() == [0]
    labels, tree = _cells(X[:1])
    assert labels.tolist() == [0] and tree.cell.tolist() == [0]


def test_class_cells_leave_a_small_branch_to_the_nearest_cell():
    # 5 far rows: under min_rows, they are no cell of their own.
    X, blob = _blobs([600, 5], gap=80.0, seed=3)
    labels, tree = _cells(X, max_rows=400, min_rows=20)
    _tree_is_valid(tree, labels)
    assert np.bincount(labels).min() >= 20
    assert len(np.unique(labels[blob == 1])) == 1  # the far rows join one cell


def test_class_cells_give_every_row_the_cell_of_its_nearest_sampled_row():
    X, blob = _blobs([1000, 1000, 1000], seed=4)
    labels, tree = _cells(X, max_rows=1500, max_fit_samples=300, seed=7)
    _tree_is_valid(tree, labels)
    assert adjusted_rand_score(blob, labels) == 1.0
    A = sample_rows(len(X), 300, 7)
    nearest = np.argmin(((X[:, None, :] - X[A][None, :, :]) ** 2).sum(2), axis=1)
    assert (labels == labels[A][nearest]).all()


def test_class_cells_count_distinct_rows_not_copies():
    # A row of many copies weighs in the tree but counts once in a cell.
    X, blob = _blobs([300, 300], seed=5)
    weights = np.ones(600)
    weights[0] = 10_000.0
    labels, _ = _cells(X, weights, max_rows=400)
    assert adjusted_rand_score(blob, labels) == 1.0


# --- merge into regions ---

# root(0) -> 1, 4; 1 -> cells 0, 1; 4 -> cells 2, 3
_TREE = CellTree(
    parent=np.array([-1, 0, 1, 1, 0, 4, 4]),
    cell=np.array([-1, -1, 0, 1, -1, 2, 3]),
)


def test_merge_cells_by_hand():
    assert merge_cells(_TREE, ["a", "a", "a", "a"]).tolist() == [0, 0, 0, 0]
    assert merge_cells(_TREE, ["a", "a", "b", "b"]).tolist() == [0, 0, 1, 1]
    assert merge_cells(_TREE, ["a", "b", "b", "b"]).tolist() == [0, 1, 2, 2]
    assert merge_cells(_TREE, ["a", "a", "b", "a"]).tolist() == [0, 0, 1, 2]
    # Equal profiles in different branches stay apart.
    assert merge_cells(_TREE, ["a", "b", "a", "b"]).tolist() == [0, 1, 2, 3]


def test_merge_cells_compares_tuples_and_one_cell():
    profile = [((), (2,)), ((), (2,)), ((1,), ()), ((1,), ())]
    assert merge_cells(_TREE, profile).tolist() == [0, 0, 1, 1]
    one = CellTree(parent=np.array([-1]), cell=np.array([0]))
    assert merge_cells(one, [("x",)]).tolist() == [0]

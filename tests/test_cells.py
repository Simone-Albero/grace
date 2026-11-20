"""src.cells: the weighted Ward tree, the cells of a class and their merge into regions."""

import numpy as np
from scipy.cluster.hierarchy import fcluster, is_valid_linkage, ward
from sklearn.metrics import adjusted_rand_score

from src.cells import (
    CellTree,
    _squared_distances,
    class_cells,
    merge_cells,
    ward_tree,
)
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


def test_ward_tree_is_scipy_ward():
    X = _cloud(90, 0)
    tree = ward_tree(X)
    Z = _linkage(tree, len(X))
    assert is_valid_linkage(Z)
    reference = ward(X)
    np.testing.assert_allclose(tree.heights, reference[:, 2])
    for k in (2, 3, 5, 9):
        ours = fcluster(Z, k, "maxclust")
        assert adjusted_rand_score(ours, fcluster(reference, k, "maxclust")) == 1.0


def test_ward_tree_orders_children_before_parents():
    X = _cloud(40, 2)
    tree = ward_tree(X)
    n = len(X)
    assert tree.children.shape == (n - 1, 2) and tree.children.dtype == np.int64
    assert (tree.children < n + np.arange(n - 1)[:, None]).all()
    assert sorted(tree.children.ravel().tolist()) == list(range(2 * n - 2))
    assert (np.diff(tree.heights) >= 0).all()


def test_ward_tree_of_one_row_has_no_merge():
    tree = ward_tree(np.zeros((1, 3)))
    assert tree.children.shape == (0, 2) and len(tree.heights) == 0


def _hair_apart(n: int, seed: int) -> np.ndarray:
    """Rows a hair apart (1e-8) around one point, as distinct CSV rows prepared."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.5, size=96) + rng.normal(0.0, 1e-8, size=(n, 96))


def test_squared_distances_are_the_distances():
    X = _cloud(30, 3)
    for a in (0, 7, 29):
        np.testing.assert_allclose(
            _squared_distances(X, a), ((X - X[a]) ** 2).sum(axis=1), rtol=1e-12
        )
    assert _squared_distances(X, 7)[7] == 0.0


def test_squared_distances_are_exactly_symmetric_for_rows_a_hair_apart():
    # ||x||² + ||y||² - 2 x·y rounds a to b and b to a differently here, and the
    # nearest-neighbour chain of ward_tree then went round three rows forever.
    X = _hair_apart(600, 4)
    D = np.stack([_squared_distances(X, a) for a in range(40)])
    assert (D[:, :40] == D[:, :40].T).all()


def test_ward_tree_of_rows_a_hair_apart_ends():
    far = np.random.default_rng(5).normal(50.0, 3.0, size=(510, 96))
    X = np.vstack([_hair_apart(30, 5), far])
    tree = ward_tree(X)
    assert tree.children.shape == (len(X) - 1, 2)


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


def _cells(X, min_rows=20, max_fit_samples=20_000, seed=0):
    return class_cells(X, min_rows, max_fit_samples, seed)


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
    # Three blobs of 400, min_rows 250: the root (1200) and the node of two
    # blobs (800) are split, a blob (400 < 2 · 250) is a cell.
    X, blob = _blobs([400, 400, 400])
    labels, tree = _cells(X, min_rows=250)
    _tree_is_valid(tree, labels)
    assert labels.max() == 2 and adjusted_rand_score(blob, labels) == 1.0
    assert len(tree.parent) == 5


def test_class_cells_split_a_large_blob_down_to_two_min_rows():
    X = np.random.default_rng(1).normal(size=(5000, 3))
    labels, tree = _cells(X, min_rows=20)
    _tree_is_valid(tree, labels)
    sizes = np.bincount(labels)
    # under 2 · min_rows, but for the rows of the small branches left beside it
    assert sizes.min() >= 20 and np.median(sizes) < 40
    assert len(sizes) >= 5000 // 40


def test_class_cells_of_a_small_class_are_one_cell():
    X = np.random.default_rng(2).normal(size=(30, 3))
    labels, tree = _cells(X, min_rows=20)  # 30 < 2 · 20
    assert labels.tolist() == [0] * 30
    assert tree.parent.tolist() == [-1] and tree.cell.tolist() == [0]
    labels, tree = _cells(X[:1])
    assert labels.tolist() == [0] and tree.cell.tolist() == [0]


def test_class_cells_leave_a_small_branch_to_the_nearest_cell():
    # 5 far rows: under min_rows, they are no cell of their own.
    X, blob = _blobs([600, 5], gap=80.0, seed=3)
    labels, tree = _cells(X, min_rows=20)
    _tree_is_valid(tree, labels)
    assert np.bincount(labels).min() >= 20
    assert len(np.unique(labels[blob == 1])) == 1  # the far rows join one cell


def test_class_cells_give_every_row_the_cell_of_its_nearest_sampled_row():
    X, blob = _blobs([1000, 1000, 1000], seed=4)
    labels, tree = _cells(X, min_rows=600, max_fit_samples=300, seed=7)
    _tree_is_valid(tree, labels)
    assert adjusted_rand_score(blob, labels) == 1.0
    A = sample_rows(len(X), 300, 7)
    nearest = np.argmin(((X[:, None, :] - X[A][None, :, :]) ** 2).sum(2), axis=1)
    assert (labels == labels[A][nearest]).all()


# --- merge into regions ---

# root(0) -> 1, 4; 1 -> cells 0, 1; 4 -> cells 2, 3
_TREE = CellTree(
    parent=np.array([-1, 0, 1, 1, 0, 4, 4]),
    cell=np.array([-1, -1, 0, 1, -1, 2, 3]),
)


def _merge(*cells):
    """merge_cells of _TREE, each cell as (yes, no) strings of entries."""
    return merge_cells(_TREE, [set(y) for y, _ in cells], [set(n) for _, n in cells])


def test_merge_cells_by_hand():
    # a region is named by the node at the top of its branch
    assert _merge(("a", ""), ("a", ""), ("a", ""), ("a", "")).tolist() == [0, 0, 0, 0]
    assert _merge(("a", ""), ("a", ""), ("", "a"), ("", "a")).tolist() == [1, 1, 4, 4]
    assert _merge(("a", ""), ("", "a"), ("", "a"), ("", "a")).tolist() == [2, 3, 4, 4]
    # cells that agree in different branches stay apart
    assert _merge(("a", ""), ("", "a"), ("a", ""), ("", "a")).tolist() == [2, 3, 5, 6]


def test_merge_cells_never_split_on_an_undecided_cell():
    # cell 1 is undecided on a: the branch of cells 0 and 1 holds, and
    # with no contradiction anywhere the class is one region
    assert _merge(("a", ""), ("", ""), ("a", "b"), ("a", "b")).tolist() == [0] * 4
    # a yes against a no splits, on any entry
    assert _merge(("a", ""), ("", ""), ("a", "b"), ("ab", "")).tolist() == [1, 1, 5, 6]


def test_merge_cells_of_one_cell():
    one = CellTree(parent=np.array([-1]), cell=np.array([0]))
    assert merge_cells(one, [{"x"}], [set()]).tolist() == [0]

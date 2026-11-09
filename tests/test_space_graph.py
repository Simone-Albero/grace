"""src.space and src.graph on hand-made points."""

import math

import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from src.graph import build_graph, nearest_neighbors, sample_rows
from src.space import embed, fit_space, scale_for_metric


def test_space_slots_and_mismatch_cost():
    cat = np.array([[1], [1], [2], [3]])
    space = fit_space(cat, top_k=1, cat_cost=3.0)
    assert [k.tolist() for k in space.kept] == [[1]]
    assert space.weight == pytest.approx(math.sqrt(1.5))
    points = embed(space, np.zeros((4, 1)), cat)
    assert points.shape == (4, 1 + 2)
    assert points.dtype == np.float64
    np.testing.assert_allclose(points[0], [0.0, space.weight, 0.0])
    np.testing.assert_allclose(points[2], [0.0, 0.0, space.weight])
    assert np.sum((points[0] - points[2]) ** 2) == pytest.approx(3.0)
    assert np.sum((points[2] - points[3]) ** 2) == 0.0


def test_space_top_k_zero_and_no_categorical():
    cat = np.array([[1, 5], [2, 5]])
    assert embed(fit_space(cat, 0, 1.0), np.ones((2, 3)), cat).shape == (2, 5)
    none = np.empty((2, 0), dtype=np.int64)
    points = embed(fit_space(none, 16, 1.0), np.ones((2, 3)), none)
    np.testing.assert_array_equal(points, np.ones((2, 3)))


def test_scale_for_metric():
    points = np.array([[3.0, 4.0], [0.0, 0.0]])
    np.testing.assert_array_equal(scale_for_metric(points, "euclidean"), points)
    np.testing.assert_allclose(
        scale_for_metric(points, "cosine"), [[0.6, 0.8], [0.0, 0.0]]
    )


def test_sample_rows():
    np.testing.assert_array_equal(sample_rows(10, 20, 0), np.arange(10))
    rows = sample_rows(100, 10, 0)
    assert len(rows) == 10 == len(set(rows.tolist()))
    assert rows.tolist() == sorted(rows.tolist())
    assert rows.max() < 100
    np.testing.assert_array_equal(rows, sample_rows(100, 10, 0))


def test_nearest_neighbors_excludes_the_query_by_row_id():
    nodes = np.array([[0.0], [1.0], [2.0], [10.0]])
    rows = np.array([0, 1, 2, 3])
    idx, dist = nearest_neighbors(nodes, rows, nodes[:1], rows[:1], 2)
    assert idx.tolist() == [[1, 2]]
    np.testing.assert_allclose(dist, [[1.0, 2.0]])
    idx, _ = nearest_neighbors(nodes, rows, np.array([[0.0]]), np.array([99]), 2)
    assert idx.tolist() == [[0, 1]]
    idx, _ = nearest_neighbors(nodes, rows, nodes, rows, 10)
    assert idx.shape == (4, 3)


def test_nearest_neighbors_keeps_a_duplicate_under_another_row():
    nodes = np.array([[0.0], [0.0], [5.0]])
    idx, dist = nearest_neighbors(nodes, np.array([0, 1, 2]), nodes[:1], [0], 1)
    assert idx.tolist() == [[1]]
    assert dist.tolist() == [[0.0]]


def test_build_graph_spans_separate_clusters():
    rng = np.random.default_rng(0)
    points = np.vstack([rng.normal(0, 1, (20, 2)), rng.normal(100, 1, (20, 2))])
    rows = np.arange(40) * 3
    graph = build_graph(points, rows, k=3)
    np.testing.assert_array_equal(graph.rows, rows)
    assert graph.knn_idx.shape == graph.knn_dist.shape == (40, 3)
    assert not np.any(graph.knn_idx == np.arange(40)[:, None])
    assert graph.mst.shape == (39, 2)
    edges = coo_matrix((np.ones(39), (graph.mst[:, 0], graph.mst[:, 1])), (40, 40))
    assert connected_components(edges, directed=False)[0] == 1


def test_build_graph_needs_two_points():
    with pytest.raises(ValueError):
        build_graph(np.zeros((1, 2)), np.array([0]), k=3)


def test_duplicates_are_exactly_zero_apart():
    rng = np.random.default_rng(0)
    points = rng.normal(0, 100, (300, 54))
    points[150:] = points[:150]
    graph = build_graph(points, np.arange(300), k=5)
    assert (graph.knn_dist[:, 0] == 0.0).all()
    twins = {frozenset((i, i + 150)) for i in range(150)}
    assert not twins & {frozenset(edge) for edge in graph.mst.tolist()}

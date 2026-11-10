"""src.graph on hand-made points."""

import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from src.graph import build_graph, nearest_neighbors, sample_rows, space_points


def test_space_points_scale_only_the_numbers_for_the_metric():
    numbers = np.array([[3.0, 4.0], [0.0, 0.0]])
    slots = np.array([[0.0, 2.0], [2.0, 0.0]])
    euclidean = space_points(numbers, slots, "euclidean")
    np.testing.assert_array_equal(euclidean, np.hstack([numbers, slots]))
    cosine = space_points(numbers, slots, "cosine")
    np.testing.assert_allclose(cosine[:, :2], [[0.6, 0.8], [0.0, 0.0]])
    np.testing.assert_array_equal(cosine[:, 2:], slots)
    without_slots = space_points(numbers, np.empty((2, 0)), "cosine")
    assert without_slots.shape == (2, 2) and without_slots.dtype == np.float64


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

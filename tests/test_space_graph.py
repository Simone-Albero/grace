"""src.space and src.graph on hand-made points."""

import hashlib
import math

import numpy as np
import polars as pl
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from src.graph import build_graph, nearest_neighbors, sample_rows
from src.space import (
    HASH_SALT,
    TopNHashEncoder,
    embed,
    fit_space,
    hash_bucket,
    scale_for_metric,
    signed_log1p,
)


def test_signed_log1p():
    x = np.array([-3.0, 0.0, math.e - 1])
    np.testing.assert_allclose(signed_log1p(x), [-math.log(4), 0.0, 1.0])


def test_hash_bucket_is_the_salted_blake2b():
    digest = hashlib.blake2b(f"{HASH_SALT}|col|val".encode(), digest_size=8)
    expected = int.from_bytes(digest.digest(), "little") % 7
    assert hash_bucket("col", "val", 7) == expected
    assert 0 <= hash_bucket("other", "val", 7) < 7


def test_top_n_hash_encoder_codes():
    train = pl.Series("c", ["a", "a", "b", "b", "c", None, "a"])
    encoder = TopNHashEncoder("c", top_n=2, hash_buckets=5).fit(train)
    codes = encoder.transform(pl.Series("c", ["a", "b", "c", None, "zzz"]))
    assert codes.dtype == np.int32
    assert codes.tolist() == [
        1,
        2,
        3 + hash_bucket("c", "c", 5),
        0,
        3 + hash_bucket("c", "zzz", 5),
    ]


def test_top_n_hash_encoder_on_a_column_without_values():
    encoder = TopNHashEncoder("c", top_n=2, hash_buckets=5)
    encoder.fit(pl.Series("c", [None, None], dtype=pl.String))
    for values in ([None, None], []):
        codes = encoder.transform(pl.Series("c", values, dtype=pl.String))
        assert codes.dtype == np.int32
        assert codes.tolist() == [0] * len(values)


def test_top_n_hash_encoder_ties_and_no_buckets():
    encoder = TopNHashEncoder("c", top_n=1, hash_buckets=0)
    encoder.fit(pl.Series("c", ["b", "a"]))
    assert encoder.transform(pl.Series("c", ["a", "b"])).tolist() == [1, 0]


def _fit(train: pl.DataFrame, cat_cols=("c",), **overrides):
    options = dict(top_n=16, hash_buckets=0, top_k=1, cat_cost=3.0, metric="euclidean")
    options.update(overrides)
    return fit_space(train, ["n"], list(cat_cols), **options)


def test_space_slots_and_mismatch_cost():
    train = pl.DataFrame({"n": [0.0] * 4, "c": ["a", "a", "b", "c"]})
    space = _fit(train)
    assert [k.tolist() for k in space.kept] == [[1]]
    assert space.weight == pytest.approx(math.sqrt(1.5))
    points = embed(space, train)
    assert points.shape == (4, 1 + 2)
    assert points.dtype == np.float64
    np.testing.assert_allclose(points[0], [0.0, space.weight, 0.0])
    np.testing.assert_allclose(points[2], [0.0, 0.0, space.weight])
    assert np.sum((points[0] - points[2]) ** 2) == pytest.approx(3.0)
    assert np.sum((points[2] - points[3]) ** 2) == 0.0


def test_space_numbers_are_robust_scaled_and_missing_ones_sit_on_the_median():
    train = pl.DataFrame({"n": [0.0, 1.0, 2.0, 3.0, 4.0, None, float("inf")]})
    space = _fit(train, cat_cols=())
    logged = np.log1p(np.arange(5.0))
    q25, q50, q75 = np.percentile(logged, [25, 50, 75])
    points = embed(space, train)
    np.testing.assert_allclose(points[:5, 0], (logged - q50) / (q75 - q25))
    np.testing.assert_array_equal(points[5:, 0], [0.0, 0.0])


def test_space_top_k_zero_and_cosine():
    train = pl.DataFrame({"n": [1.0, 2.0], "c": ["x", "y"]})
    assert embed(_fit(train, top_k=0), train).shape == (2, 2)
    space = _fit(train, top_k=2, cat_cost=1.0, metric="cosine")
    np.testing.assert_allclose(np.linalg.norm(embed(space, train), axis=1), 1.0)


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

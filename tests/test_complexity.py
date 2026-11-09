"""src.complexity: hand-computed families and the whole table."""

import numpy as np
import pytest

from src.complexity.common import MeasuredSample, summarize
from src.complexity.dimensionality import compute_t_measures
from src.complexity.feature import compute_f_measures
from src.complexity.linearity import compute_l_measures
from src.complexity.measure import (
    MEASURES,
    compute_population_complexity,
    nearest_rivals,
)
from src.graph import build_graph


def test_summarize():
    assert summarize("f1", []) == {"f1_min": None, "f1_mean": None, "f1_max": None}
    assert summarize("f1", [1.0, 2.0, 3.0]) == {
        "f1_min": 1.0,
        "f1_mean": 2.0,
        "f1_max": 3.0,
    }


def test_measures_are_45_and_unique():
    assert len(MEASURES) == len(set(MEASURES)) == 45


def test_nearest_rivals_skip_the_same_class():
    distances = np.array(
        [[np.inf, 1.0, 5.0, 3.0], [1.0, np.inf, 2.0, 9.0], [5.0, 2.0, np.inf, 1.0]]
        + [[3.0, 9.0, 1.0, np.inf]]
    )
    rivals = nearest_rivals([10, 11, 20, 21], np.array([0, 0, 1, 1]), distances, 1)
    assert rivals == {10: [21], 11: [20], 20: [11], 21: [10]}


def _pair(own: list[float], rival: list[float]) -> tuple[np.ndarray, np.ndarray]:
    points = np.array(own + rival)[:, None]
    population = np.array([0] * len(own) + [1] * len(rival))
    return points, population


def test_f_measures_separated_by_hand():
    points, population = _pair([0.0, 2.0], [10.0, 12.0])
    f = compute_f_measures(points, population, {0: [1], 1: [0]}, metric="euclidean")
    assert f[0]["f1_mean"] == pytest.approx(1 / 51, rel=1e-6)
    assert f[0]["f2_mean"] == 0.0
    assert f[0]["f3_mean"] == 0.0
    assert f[0]["f4_mean"] == 0.0


def test_f_measures_overlapping_by_hand():
    points, population = _pair([0.0, 4.0], [2.0, 6.0])
    f = compute_f_measures(points, population, {0: [1], 1: [0]}, metric="euclidean")
    assert f[0]["f1_mean"] == pytest.approx(1 / 1.5, rel=1e-6)
    assert f[0]["f2_mean"] == pytest.approx(1 / 3, rel=1e-6)
    assert f[0]["f3_mean"] == pytest.approx(0.5)
    assert f[0]["f4_mean"] == pytest.approx(0.5)


def test_t_measures_of_a_line():
    x = np.arange(10, dtype=np.float64)
    X = np.column_stack([x, 2 * x])
    t = compute_t_measures(X, np.zeros(10, dtype=int), {0: 20}, metric="euclidean")
    assert t[0]["t2"] == pytest.approx(0.1)
    assert t[0]["t3"] == pytest.approx(0.05)
    assert t[0]["t4"] == pytest.approx(0.5)
    assert t[0]["anisotropy"] == pytest.approx(1.0)


def test_l_measures_of_separated_blobs():
    rng = np.random.default_rng(0)
    points = np.vstack([rng.normal(0, 1, (30, 2)), rng.normal(20, 1, (30, 2))])
    population = np.repeat([0, 1], 30)
    lin = compute_l_measures(
        points, population, {0: [1], 1: [0]}, metric="euclidean", random_state=0
    )
    assert lin[0] == {"l1": 0.0, "l2": 0.0, "l3": 0.0}


def _complexity(gap: float) -> dict[int, dict[str, float | None]]:
    """Two classes, two regions each, the classes `gap` apart."""
    rng = np.random.default_rng(0)
    centers = np.array([[0.0, 0.0], [0.0, 3.0], [gap, 0.0], [gap, 3.0]])
    X = np.vstack([rng.normal(c, 1.0, (40, 2)) for c in centers])
    population = np.repeat([0, 1, 2, 3], 40)
    graph = build_graph(X, np.arange(160), k=10)
    sample = MeasuredSample(X, population, graph.knn_idx, graph.knn_dist)
    return compute_population_complexity(
        graph,
        population,
        sample,
        population_class={0: 0, 1: 0, 2: 1, 3: 1},
        centroids={p: X[population == p].mean(axis=0) for p in range(4)},
        sizes={p: 40 for p in range(4)},
        top_k_clusters=2,
        metric="euclidean",
        silhouette_max_samples=1000,
        silhouette_min_per_cluster=10,
        random_state=0,
    )


def test_population_complexity_keys_and_direction():
    far, near = _complexity(gap=20.0), _complexity(gap=0.5)
    assert sorted(far) == [0, 1, 2, 3]
    assert list(far[0]) == list(MEASURES)
    assert far[0]["f1_mean"] < near[0]["f1_mean"]
    assert far[0]["n3_mean"] < near[0]["n3_mean"]
    assert far[0]["l2"] < near[0]["l2"]
    assert far[0]["dist_to_nearest_rival"] > near[0]["dist_to_nearest_rival"]
    assert far[0]["rival_classes"] <= near[0]["rival_classes"]

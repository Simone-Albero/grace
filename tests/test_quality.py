"""src.quality on hand-made rows."""

import numpy as np
import polars as pl
import pytest

from src.quality import (
    Pair,
    candidate_pairs,
    diversity,
    feature_separation,
    label_conflicts,
    linear_error,
    neighbour_overlap,
    pair_conflicts,
    region_samples,
)


def test_label_conflicts_group_equal_rows_of_other_labels():
    features = pl.DataFrame(
        {
            "n": [1.0, 1.0, 2.0, 2.0, 3.0, 1.0, 1.0],
            "c": ["a", "a", None, None, "b", "b", "a"],
        }
    )
    labels = np.array([0, 1, 0, 2, 0, 1, 2])
    groups = label_conflicts(features, labels)
    assert groups.tolist() == [0, 0, 1, 1, -1, -1, 0]
    assert groups.dtype == np.int64


def test_label_conflicts_without_any():
    features = pl.DataFrame({"n": [1.0, 2.0, 3.0]})
    assert label_conflicts(features, np.array([0, 1, 0])).tolist() == [-1, -1, -1]


def test_pair_conflicts_count_both_rows_of_every_pair():
    groups = np.array([0, 0, 1, 1, -1, -1, 0])
    labels = np.array([0, 1, 0, 2, 0, 1, 2])
    region = np.array([0, 3, 0, 5, 1, 3, 6])
    weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    assert pair_conflicts(groups, labels, region, weights) == {
        (0, 3): (2, 3.0),
        (0, 6): (2, 8.0),
        (3, 6): (2, 9.0),
        (0, 5): (2, 7.0),
    }


def test_pair_conflicts_sum_over_groups():
    groups = np.array([0, 0, 1, 1])
    labels = np.array([1, 0, 0, 1])
    region = np.array([2, 1, 1, 2])
    weights = np.array([1.0, 1.0, 2.0, 3.0])
    assert pair_conflicts(groups, labels, region, weights) == {(1, 2): (4, 7.0)}


def test_pair_conflicts_skip_rows_of_one_class():
    # One group, as without some columns: two rows of class 0 in regions
    # 0 and 1, one of class 1 in region 2, one more of class 0 in region 0.
    groups = np.array([0, 0, 0, 0])
    labels = np.array([0, 0, 1, 0])
    region = np.array([0, 1, 2, 0])
    weights = np.array([1.0, 2.0, 4.0, 8.0])
    assert pair_conflicts(groups, labels, region, weights) == {
        (0, 2): (3, 13.0),
        (1, 2): (2, 6.0),
    }


def _symmetric(m: int, values: dict) -> np.ndarray:
    matrix = np.zeros((m, m))
    for (a, b), value in values.items():
        matrix[a, b] = matrix[b, a] = value
    return matrix


def test_candidate_pairs_join_near_conflict_and_nearest():
    classes = np.array([0, 0, 1, 1])
    full = _symmetric(4, {(0, 1): 0.5, (0, 2): 1.0, (0, 3): 3.0, (1, 2): 5.0})
    full = full + _symmetric(4, {(1, 3): 4.0, (2, 3): 0.2})
    without = _symmetric(4, {(0, 1): 9, (0, 2): 9, (0, 3): 9, (1, 2): 9})
    without = without + _symmetric(4, {(1, 3): 1.5, (2, 3): 9})
    pairs = candidate_pairs([full, without], [(1, 2)], classes, 2.0)
    assert pairs == [
        Pair(0, 2, ("near", "nearest")),
        Pair(0, 3, ("nearest",)),
        Pair(1, 2, ("conflict",)),
        Pair(1, 3, ("near", "nearest")),
    ]


def test_candidate_pairs_of_one_class_are_none():
    full = _symmetric(2, {(0, 1): 0.1})
    assert candidate_pairs([full], [(0, 1)], np.array([0, 0]), 2.0) == []


def test_region_samples_cap_each_region():
    region = np.array([0, 0, 0, 1, 1, 2, 0])
    samples = region_samples(region, 2, seed=0)
    assert sorted(samples) == [0, 1, 2]
    assert len(samples[0]) == 2 and set(samples[0]) <= {0, 1, 2, 6}
    assert samples[1].tolist() == [3, 4] and samples[2].tolist() == [5]
    assert all((np.diff(rows) > 0).all() for rows in samples.values())
    again = region_samples(region, 2, seed=0)
    assert all(np.array_equal(samples[r], again[r]) for r in samples)


def test_feature_separation_finds_the_separating_column():
    Xa = np.array([[0.0, 0, 1], [1, 0, 1], [2, 0, 1]])
    Xb = np.array([[5.0, 0, 1], [6, 1, 0], [7, 0, 1]])
    result = feature_separation(Xa, Xb, np.array([0, 1, 1]))
    assert result.share == 1.0 and result.column == 0 and result.n_separating == 1


def test_feature_separation_takes_the_slots_of_a_column_together():
    Xa = np.array([[0.0, 0, 1], [1, 0, 1], [2, 0, 1]])
    Xb = np.array([[1.0, 1, 0], [1, 1, 0], [2, 0, 1]])
    result = feature_separation(Xa, Xb, np.array([0, 1, 1]))
    assert result.column == 1 and result.share == pytest.approx(2 / 6)
    assert result.n_separating == 0


def test_feature_separation_drops_with_an_outlier():
    Xa = np.arange(10.0)[:, None]
    Xb = np.concatenate([np.arange(20.0, 30.0), [-5.0]])[:, None]
    result = feature_separation(Xa, Xb, np.array([0]))
    assert result.share == pytest.approx(11 / 21) and result.n_separating == 0


def test_feature_separation_of_equal_regions_is_zero():
    X = np.array([[0.0, 1.0], [1.0, 0.0]])
    result = feature_separation(X, X, np.array([0, 1]))
    assert result.share == 0.0 and result.column == -1 and result.n_separating == 0


def test_neighbour_overlap_by_hand():
    Xa = np.array([[0.0], [1.0], [2.5]])
    Xb = np.array([[3.0], [10.0], [11.5]])
    result = neighbour_overlap(Xa, Xb, k=1)
    assert result.kdn == pytest.approx(1 / 3)
    assert result.enn == pytest.approx(1 / 3)
    assert result.n1 == pytest.approx(2 / 6)


def test_neighbour_overlap_of_blobs_apart():
    rng = np.random.default_rng(0)
    Xa, Xb = rng.normal(0, 1, (30, 2)), rng.normal(100, 1, (30, 2))
    result = neighbour_overlap(Xa, Xb, k=4)
    assert result.kdn == 0.0 and result.enn == 0.0
    assert result.n1 == pytest.approx(2 / 60)


def test_neighbour_overlap_of_one_blob():
    rng = np.random.default_rng(0)
    Xa, Xb = rng.normal(0, 1, (300, 2)), rng.normal(0, 1, (300, 2))
    result = neighbour_overlap(Xa, Xb, k=4)
    assert result.kdn == pytest.approx(0.5, abs=0.1)
    assert result.enn > 0.5 and result.n1 > 0.3


def test_neighbour_overlap_with_fewer_rows_than_k():
    result = neighbour_overlap(np.array([[0.0]]), np.array([[1.0]]), k=4)
    assert result.kdn == 1.0 and result.enn == 1.0 and result.n1 == 1.0


def _xor(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    def blob(x, y):
        return rng.normal((x, y), 0.5, (50, 2))

    return (
        np.vstack([blob(0, 0), blob(10, 10)]),
        np.vstack([blob(0, 10), blob(10, 0)]),
    )


def test_linear_error_of_a_curved_boundary():
    Xa, Xb = _xor(np.random.default_rng(0))
    assert linear_error(Xa, Xb) >= 0.2
    assert neighbour_overlap(Xa, Xb, k=4).kdn == 0.0


def test_linear_error_of_blobs_apart():
    rng = np.random.default_rng(0)
    assert linear_error(rng.normal(0, 1, (30, 2)), rng.normal(50, 1, (30, 2))) == 0.0


def test_diversity_scales_every_column_first():
    X = np.array([[0.0, 5.0], [10.0, 5.0], [5.0, 7.0], [10.0, 9.0]])
    pdd, tc = diversity(X, np.array([0, 0, 1, 1]), n_pairs=10, seed=0)
    # scaled: [0, 0], [1, 0] (one of norm 0); [0.5, 0.5], [1, 1] (cosine 1)
    np.testing.assert_allclose(pdd, [0.0, 1.0])
    np.testing.assert_allclose(tc, [0.0, 1.0])


def test_diversity_of_equal_orthogonal_and_single_rows():
    X = np.array([[1.0, 0], [1, 0], [1, 0], [1, 0], [0, 1], [0.3, 0.3], [0, 0], [0, 0]])
    region = np.array([0, 0, 0, 1, 1, 2, 3, 3])
    pdd, tc = diversity(X, region, n_pairs=50, seed=0)
    np.testing.assert_allclose(pdd, [1.0, 0.0, 1.0, 1.0])
    np.testing.assert_allclose(tc, [1.0, 0.0, 1.0, 1.0])


def test_diversity_counts_the_collapsed_pairs():
    angles = np.array([0.0, 0.01, np.pi / 2])
    X = np.column_stack([np.cos(angles), np.sin(angles)])
    pdd, tc = diversity(X, np.zeros(3, dtype=int), n_pairs=3000, seed=0)
    assert tc[0] == pytest.approx(1 / 3, abs=0.05)
    expected = (np.cos(0.01) + np.cos(np.pi / 2 - 0.01) + 0.0) / 3
    assert pdd[0] == pytest.approx(expected, abs=0.05)

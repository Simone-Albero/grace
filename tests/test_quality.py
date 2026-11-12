"""src.quality on hand-made rows."""

import numpy as np
import polars as pl
import pytest

from src.quality import (
    ISSUES,
    Pair,
    candidate_pairs,
    cover_hart_low,
    diversity,
    feature_dependence,
    invalid_values,
    label_conflicts,
    nearest_error,
    pair_conflicts,
    region_issues,
    region_samples,
    single_column,
    verdict,
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
    apart = _symmetric(4, {(0, 1): 0.5, (0, 2): 1.0, (0, 3): 3.0, (1, 2): 5.0})
    apart = apart + _symmetric(4, {(1, 3): 4.0, (2, 3): 0.2})
    pairs = candidate_pairs(apart, [(2, 1)], classes, 2.0)
    assert pairs == [
        Pair(0, 2, ("near", "nearest")),
        Pair(0, 3, ("nearest",)),
        Pair(1, 2, ("label_conflict",)),
        Pair(1, 3, ("nearest",)),
    ]


def test_candidate_pairs_of_one_class_are_none():
    apart = _symmetric(2, {(0, 1): 0.1})
    assert candidate_pairs(apart, [(0, 1)], np.array([0, 0]), 2.0) == []


def test_region_samples_cap_each_region():
    region = np.array([0, 0, 0, 1, 1, 2, 0])
    samples = region_samples(region, 2, seed=0)
    assert sorted(samples) == [0, 1, 2]
    assert len(samples[0]) == 2 and set(samples[0]) <= {0, 1, 2, 6}
    assert samples[1].tolist() == [3, 4] and samples[2].tolist() == [5]
    assert all((np.diff(rows) > 0).all() for rows in samples.values())
    again = region_samples(region, 2, seed=0)
    assert all(np.array_equal(samples[r], again[r]) for r in samples)


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


def test_invalid_values_by_kind():
    features = pl.DataFrame(
        {
            "n1": [1.0, None, float("nan"), float("inf")],
            "n2": [None, 2.0, -float("inf"), 3.0],
            "c": ["x", None, "y", "z"],
        }
    )
    unparsable = pl.DataFrame({"row": [1], "column": ["n1"]})
    cells = invalid_values(features, unparsable)
    assert cells.columns == ["row", "column", "kind"]
    assert cells.rows() == [
        (0, "n2", "empty"),
        (1, "n1", "unparsable"),
        (1, "c", "empty"),
        (2, "n1", "nan"),
        (2, "n2", "inf"),
        (3, "n1", "inf"),
    ]
    clean = pl.DataFrame({"n1": [1.0], "c": ["x"]})
    none = pl.DataFrame(
        {"row": [], "column": []}, schema={"row": pl.Int64, "column": pl.String}
    )
    assert invalid_values(clean, none).height == 0


def _sq(X: np.ndarray) -> np.ndarray:
    return ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)


def test_nearest_error_by_hand_with_a_tie():
    # a: 0, 1, 5; b: 5, 6, 10. a@5 and b@5 are each other's nearest; b@6 is
    # 1 from a@5 and from b@5, a tie that splits its vote.
    X = np.array([0.0, 1, 5, 5, 6, 10])[:, None]
    side = np.array([False, False, False, True, True, True])
    err_a, err_b = nearest_error(_sq(X), side)
    assert err_a == pytest.approx(1 / 3) and err_b == pytest.approx(0.5)


def test_nearest_error_does_not_depend_on_the_order_of_the_rows():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 3, size=(60, 2)).astype(float)  # many ties
    side = np.arange(60) >= 30
    err = nearest_error(_sq(X), side)
    swapped = nearest_error(_sq(X[::-1]), ~side[::-1])
    assert swapped == pytest.approx(err[::-1])
    order = rng.permutation(60)
    assert nearest_error(_sq(X[order]), side[order]) == pytest.approx(err)


def test_cover_hart_low():
    assert cover_hart_low(0.0) == 0.0
    assert cover_hart_low(0.18) == pytest.approx(0.1)
    assert cover_hart_low(0.5) == pytest.approx(0.5)
    assert cover_hart_low(0.9) == pytest.approx(0.5)


def test_single_column_loses_only_the_outlier():
    Xa = np.c_[np.arange(10.0), np.zeros(10)]
    Xb = np.c_[np.r_[np.arange(20.0, 30.0), -5.0], np.zeros(11)]
    result = single_column(Xa, Xb, np.array([0, 1]), np.array([True, True]), 0.95)
    assert result.column == 0
    assert result.side_a == 1.0 and result.side_b == pytest.approx(10 / 11)
    assert result.share == pytest.approx((1 + 10 / 11) / 2)  # 0.9545
    assert result.n_single_columns == 1
    stricter = single_column(Xa, Xb, np.array([0, 1]), np.array([True, True]), 0.96)
    assert stricter.n_single_columns == 0


def test_single_column_on_a_binary_column():
    a = np.zeros((100, 1))
    a[0] = 1.0
    b = np.ones((100, 1))
    b[0] = 0.0
    result = single_column(a, b, np.array([0]), np.array([True]), 0.95)
    assert (
        result.side_a == 0.99 and result.side_b == 0.99 and result.n_single_columns == 1
    )


def test_single_column_reads_the_slots_of_a_categorical_together():
    # column 0: a number equal on both sides; column 1: three slots
    va, vb = np.array([0, 1] * 10), np.array([2] * 19 + [0])
    Xa = np.c_[np.zeros(20), np.eye(3)[va]]
    Xb = np.c_[np.zeros(20), np.eye(3)[vb]]
    result = single_column(Xa, Xb, np.array([0, 1, 1, 1]), np.array([True, False]), 0.9)
    assert result.column == 1
    assert result.side_a == 1.0 and result.side_b == pytest.approx(19 / 20)
    assert result.n_single_columns == 1


def test_single_column_without_any_varying_column():
    X = np.ones((5, 2))
    result = single_column(X, X, np.array([0, 1]), np.array([True, True]), 0.95)
    assert result == type(result)(0.5, 0.5, -1, 0)


def _pair(case: str, n: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(n, 12)), rng.normal(size=(n, 12))
    column_of = np.arange(12)
    if case == "shortcut":
        b[:, 5] += 8
    elif case == "copies":
        b[:, 3] += 8
        a[:, 7] = a[:, 3] + rng.normal(0, 0.1, n)
        b[:, 7] = b[:, 3] + rng.normal(0, 0.1, n)
    elif case == "categorical":
        a = np.c_[a[:, :11], np.eye(6)[rng.integers(0, 3, n)]]
        b = np.c_[b[:, :11], np.eye(6)[rng.integers(3, 6, n)]]
        column_of = np.r_[np.arange(11), np.full(6, 11)]
    elif case == "spread":
        b[:, :10] += 1.5
    return a, b, column_of


def _judge(case: str, n_conflicts: int = 0) -> tuple[tuple[int, ...], str, int]:
    a, b, column_of = _pair(case)
    numeric = np.r_[np.ones(column_of.max(), bool), case != "categorical"]
    dependence = feature_dependence(a, b, column_of, 3, seed=0)
    share = single_column(a, b, column_of, numeric, 0.95).share
    return (dependence.group, *verdict(dependence.errors, n_conflicts, share, 0.1))


def test_feature_dependence_finds_the_shortcut_column():
    group, judged, k = _judge("shortcut")
    assert group[0] == 5 and (judged, k) == ("shortcut", 1)


def test_feature_dependence_finds_two_redundant_columns():
    group, judged, k = _judge("copies")
    assert set(group[:2]) == {3, 7} and (judged, k) == ("shortcut", 2)


def test_feature_dependence_takes_the_slots_of_a_column_together():
    group, judged, k = _judge("categorical")
    assert group[0] == 11 and (judged, k) == ("shortcut", 1)


def test_a_separation_spread_over_many_columns_is_robust():
    assert _judge("spread")[1:] == ("robust", 0)


def test_one_distribution_overlaps_or_conflicts():
    assert _judge("same")[1:] == ("class_overlap", 0)
    assert _judge("same", n_conflicts=3)[1:] == ("label_conflict", 0)


def test_feature_dependence_shapes_and_small_sides():
    a, b, column_of = _pair("shortcut", n=40)
    dependence = feature_dependence(a, b, column_of, 3, seed=(1, 2, 3))
    assert len(dependence.group) == 3 and len(set(dependence.group)) == 3
    assert dependence.errors.shape == (4, 2)
    assert ((dependence.errors >= 0) & (dependence.errors <= 1)).all()
    again = feature_dependence(a, b, column_of, 3, seed=(1, 2, 3))
    assert again.group == dependence.group
    np.testing.assert_array_equal(again.errors, dependence.errors)
    assert feature_dependence(a[:3], b, column_of, 3, seed=0) is None


def test_verdict_by_hand():
    def judge(errors, n_conflicts=0, share=0.5):
        return verdict(
            np.repeat(np.array(errors)[:, None], 2, 1), n_conflicts, share, 0.1
        )

    assert judge([0.2, 0.3], n_conflicts=3) == ("label_conflict", 0)
    assert judge([0.2, 0.3]) == ("class_overlap", 0)
    assert judge([0.2, 0.25, 0.35], share=0.95) == ("shortcut", 2)
    assert judge([0.05, 0.12, 0.2]) == ("shortcut", 2)
    assert judge([0.0, 0.08, 0.09]) == ("robust", 0)
    assert judge([0.09, 0.11, 0.12]) == ("robust", 0)
    sides = np.array([[0.0, 0.3], [0.5, 0.5]])
    assert verdict(sides, 0, 0.5, 0.1) == ("class_overlap", 0)


def test_region_issues_in_order():
    assert region_issues(0, 0.1, 0, ["robust", "label_conflict"], 0.9) == ()
    every = region_issues(2, 0.95, 1, ["class_overlap", "shortcut", "robust"], 0.9)
    assert every == ISSUES
    assert region_issues(0, 0.9, 0, [], 0.9) == ("duplicates",)

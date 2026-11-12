"""src.quality on hand-made rows."""

import numpy as np
import polars as pl
import pytest

from src.quality import (
    DEFECTS,
    RIVAL_DEFECTS,
    ROW_DEFECTS,
    cell_samples,
    cover_hart_low,
    defects_of,
    feature_dependence,
    invalid_values,
    label_conflicts,
    nearest_error,
    pair_conflicts,
    pair_confusion,
    rival_defects,
    rival_pairs,
    single_column,
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
    cell = np.array([0, 3, 0, 5, 1, 3, 6])
    weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    assert pair_conflicts(groups, labels, cell, weights) == {
        (0, 3): (2, 3.0),
        (0, 6): (2, 8.0),
        (3, 6): (2, 9.0),
        (0, 5): (2, 7.0),
    }


def test_pair_conflicts_sum_over_groups():
    groups = np.array([0, 0, 1, 1])
    labels = np.array([1, 0, 0, 1])
    cell = np.array([2, 1, 1, 2])
    weights = np.array([1.0, 1.0, 2.0, 3.0])
    assert pair_conflicts(groups, labels, cell, weights) == {(1, 2): (4, 7.0)}


def test_pair_conflicts_skip_rows_of_one_class():
    # One group, as without some columns: two rows of class 0 in cells
    # 0 and 1, one of class 1 in cell 2, one more of class 0 in cell 0.
    groups = np.array([0, 0, 0, 0])
    labels = np.array([0, 0, 1, 0])
    cell = np.array([0, 1, 2, 0])
    weights = np.array([1.0, 2.0, 4.0, 8.0])
    assert pair_conflicts(groups, labels, cell, weights) == {
        (0, 2): (3, 13.0),
        (1, 2): (2, 6.0),
    }


def test_rival_pairs_pair_each_cell_with_its_rows_nearest_rivals():
    # Regions 0 and 1 of class 0, 2 and 3 of class 1, 4 of class 2.
    points = np.array([[0.0], [1.0], [10.0], [11.0], [1.4], [2.0], [12.0], [100.0]])
    points = np.r_[points, [[200.0]]]
    cell = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4])
    cell_class = np.array([0, 0, 1, 1, 2])
    # 0 and 1 find 1.4 (2); 10 and 11 find 12 (3); 1.4 and 2 find 1 (0);
    # 12 and 100 find 11 (1); 200 finds 100 (3).
    assert rival_pairs(points, cell, cell_class) == [(0, 2), (1, 3), (3, 4)]
    order = np.random.default_rng(0).permutation(len(points))
    assert rival_pairs(points[order], cell[order], cell_class) == [
        (0, 2),
        (1, 3),
        (3, 4),
    ]


def test_rival_pairs_of_one_class_are_none():
    points = np.arange(4.0)[:, None]
    assert rival_pairs(points, np.array([0, 0, 1, 1]), np.array([0, 0])) == []


def test_cell_samples_cap_each_cell():
    cell = np.array([0, 0, 0, 1, 1, 2, 0])
    samples = cell_samples(cell, 2, seed=0)
    assert sorted(samples) == [0, 1, 2]
    assert len(samples[0]) == 2 and set(samples[0]) <= {0, 1, 2, 6}
    assert samples[1].tolist() == [3, 4] and samples[2].tolist() == [5]
    assert all((np.diff(rows) > 0).all() for rows in samples.values())
    again = cell_samples(cell, 2, seed=0)
    assert all(np.array_equal(samples[r], again[r]) for r in samples)


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
    result = single_column(Xa, Xb, np.array([0, 1]), np.array([True, True]))
    assert result.column == 0
    assert result.side_a == 1.0 and result.side_b == pytest.approx(10 / 11)
    assert result.share == pytest.approx((1 + 10 / 11) / 2)  # 0.9545


def test_single_column_on_a_binary_column():
    a = np.zeros((100, 1))
    a[0] = 1.0
    b = np.ones((100, 1))
    b[0] = 0.0
    result = single_column(a, b, np.array([0]), np.array([True]))
    assert result.side_a == 0.99 and result.side_b == 0.99


def test_single_column_reads_the_slots_of_a_categorical_together():
    # column 0: a number equal on both sides; column 1: three slots
    va, vb = np.array([0, 1] * 10), np.array([2] * 19 + [0])
    Xa = np.c_[np.zeros(20), np.eye(3)[va]]
    Xb = np.c_[np.zeros(20), np.eye(3)[vb]]
    result = single_column(Xa, Xb, np.array([0, 1, 1, 1]), np.array([True, False]))
    assert result.column == 1
    assert result.side_a == 1.0 and result.side_b == pytest.approx(19 / 20)


def test_single_column_without_any_varying_column():
    X = np.ones((5, 2))
    result = single_column(X, X, np.array([0, 1]), np.array([True, True]))
    assert result == type(result)(0.5, 0.5, -1)


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


def _judge(case: str) -> tuple[tuple[int, ...], tuple[str, ...], tuple[str, ...]]:
    """The columns removed, and the defects of each side against the other."""
    a, b, column_of = _pair(case)
    numeric = np.r_[np.ones(column_of.max(), bool), case != "categorical"]
    dependence = feature_dependence(a, b, column_of, 3, seed=0)
    confusion = pair_confusion(a, b, column_of, numeric, dependence)
    a_defects, b_defects = (
        defects_of(0, 10, 10.0, 0, *rival_defects([(1, confusion[:, side])], 0.1))
        for side in (0, 1)
    )
    return dependence.group, a_defects, b_defects


def test_feature_dependence_finds_the_shortcut_column():
    group, *defects = _judge("shortcut")
    assert group[0] == 5 and defects == [("shortcut",), ("shortcut",)]


def test_feature_dependence_finds_two_redundant_columns():
    group, *defects = _judge("copies")
    assert set(group[:2]) == {3, 7} and defects == [("shortcut",), ("shortcut",)]


def test_feature_dependence_takes_the_slots_of_a_column_together():
    group, *defects = _judge("categorical")
    assert group[0] == 11 and defects == [("shortcut",), ("shortcut",)]


def test_a_separation_spread_over_many_columns_is_no_defect():
    assert _judge("spread")[1:] == ((), ())


def test_one_distribution_overlaps():
    assert _judge("same")[1:] == (("class_overlap",), ("class_overlap",))


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


def test_pair_confusion_takes_the_better_classifier_at_every_step():
    # The categorical column splits the sides, 11 noisy numbers swamp the
    # 1-NN: with every column the single column wins; without it, neither.
    a, b, column_of = _pair("categorical")
    numeric = np.r_[np.ones(11, bool), False]
    dependence = feature_dependence(a, b, column_of, 3, seed=0)
    assert dependence.errors[0].mean() >= 0.1
    confusion = pair_confusion(a, b, column_of, numeric, dependence)
    assert confusion.shape == dependence.errors.shape
    np.testing.assert_array_equal(confusion[0], [0.0, 0.0])
    assert (confusion[1:] >= 0.1).all()
    assert (confusion <= dependence.errors).any(axis=1).all()


def test_single_column_skips_the_columns_asked():
    Xa = np.c_[np.arange(10.0), np.zeros(10)]
    Xb = np.c_[np.arange(20.0, 30.0), np.r_[np.ones(9), 0.0]]
    column_of, numeric = np.array([0, 1]), np.array([True, True])
    assert single_column(Xa, Xb, column_of, numeric).column == 0
    rest = single_column(Xa, Xb, column_of, numeric, skip=[0])
    assert rest.column == 1 and rest.side_b == pytest.approx(0.9)
    assert single_column(Xa, Xb, column_of, numeric, skip=[0, 1]).column == -1


def test_rival_defects_by_hand():
    def rivals(*pairs):
        return rival_defects([(c, np.array(e)) for c, e in pairs], 0.1)

    assert rivals() == ((), ())
    assert rivals((1, [0.0, 0.05, 0.08])) == ((), ())
    assert rivals((2, [0.1, 0.2])) == ((2,), ())  # at tau
    assert rivals((3, [0.05, 0.12, 0.2])) == ((), (3,))
    assert rivals((3, [0.05, 0.08, 0.1])) == ((), (3,))  # at tau, the last column
    # A rival it overlaps is no shortcut; another rival may be one.
    assert rivals((2, [0.3, 0.5])) == ((2,), ())
    assert rivals((4, [0.3]), (2, [0.0, 0.2]), (4, [0.5])) == ((4,), (2,))
    # Both against one class, through two of its cells.
    assert rivals((5, [0.2]), (5, [0.05, 0.3])) == ((5,), (5,))


def test_defects_of_by_hand():
    assert defects_of(0, 10, 10.0, 0, (), ()) == ()
    assert defects_of(1, 10, 10.0, 0, (), ()) == ("invalid_values",)
    assert defects_of(0, 10, 12.0, 0, (), ()) == ("duplicates",)
    assert defects_of(0, 10, 10.0, 2, (), ()) == ("label_conflict",)
    assert defects_of(0, 10, 10.0, 0, (2,), ()) == ("class_overlap",)
    assert defects_of(0, 10, 10.0, 0, (), (3,)) == ("shortcut",)
    assert defects_of(1, 10, 11.0, 1, [4], [2]) == DEFECTS
    assert DEFECTS == ROW_DEFECTS + RIVAL_DEFECTS

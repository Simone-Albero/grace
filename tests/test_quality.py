"""src.quality on hand-made rows."""

import math

import numpy as np
import polars as pl
import pytest

from src.quality import (
    CLASS_DEFECTS,
    DEFECTS,
    RIVAL_DEFECTS,
    ROW_DEFECTS,
    Dependence,
    SingleColumn,
    cell_samples,
    cell_variance,
    cover_hart_low,
    decisive_rows,
    defects_of,
    feature_dependence,
    invalid_values,
    label_conflicts,
    mean_bounds,
    nearest_error,
    pair_conflicts,
    pair_confusion,
    provenance_verdict,
    redundancy_verdict,
    rival_defects,
    rival_pairs,
    share_bounds,
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


def test_rows_of_a_class_spread_over_its_cells_are_confused_with_the_class():
    # A cell of 60 rows lying among the 1,000 rows of a class cut into 20
    # cells: against each of them few of its rows are confused, against
    # the class pooled nearly all, as quality_report measures it.
    rng = np.random.default_rng(0)
    a = np.c_[rng.uniform(0, 20, 60), rng.uniform(0, 1, 60)]
    cells = [np.c_[rng.uniform(i, i + 1, 50), rng.uniform(0, 1, 50)] for i in range(20)]

    def error_of_a(b: np.ndarray) -> float:
        side = np.r_[np.zeros(len(a), bool), np.ones(len(b), bool)]
        return nearest_error(_sq(np.r_[a, b]), side)[0]

    assert max(error_of_a(b) for b in cells) <= 0.15
    assert error_of_a(np.vstack(cells)) >= 0.9


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
    n = int(dependence.measure[: len(a)].sum())  # the rows of a side measured
    a_defects, b_defects = (
        defects_of(0, 10, 10.0, 0, _established(confusion[:, side], n))
        for side in (0, 1)
    )
    return dependence.group, a_defects, b_defects


def _established(steps, n: int) -> set[str]:
    """The rival defects established against one class, at tau 0.1 and 95%."""
    found = rival_defects([(1, steps, n)], 0.1, 0.95)
    return {defect for (defect, _), v in found.items() if v}


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
    # half of each side measured, the other half chooses
    assert dependence.measure.shape == (80,) and dependence.measure.dtype == bool
    assert dependence.measure[:40].sum() == dependence.measure[40:].sum() == 20
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


def test_single_column_is_measured_on_other_rows_than_it_is_chosen_on():
    # chosen: a below 9.5, b above; measured: one row of each side crosses
    Xa, Xb = np.arange(10.0)[:, None], np.arange(20.0, 30.0)[:, None]
    Ya, Yb = np.array([[0.0], [25.0]]), np.array([[30.0], [5.0]])
    column_of, numeric = np.array([0]), np.array([True])
    assert single_column(Xa, Xb, column_of, numeric) == SingleColumn(1.0, 1.0, 0)
    held = single_column(Xa, Xb, column_of, numeric, Ya=Ya, Yb=Yb)
    assert (held.side_a, held.side_b, held.column) == (0.5, 0.5, 0)


def test_single_column_sends_a_value_it_was_not_chosen_on_to_the_first_side():
    # one categorical column of three slots: value 0 on a, 1 on b, 2 unseen
    slots = np.eye(3)
    column_of, numeric = np.array([0, 0, 0]), np.array([False])
    Xa, Xb = slots[[0] * 5], slots[[1] * 5]
    held = single_column(Xa, Xb, column_of, numeric, Ya=slots[[2, 0]], Yb=slots[[2, 1]])
    assert (held.side_a, held.side_b, held.column) == (1.0, 0.5, 0)


def test_two_cells_of_one_distribution_overlap_even_when_small():
    # 20 rows a side, 10 numbers and a categorical of 30 values: chosen and
    # measured on the same rows, the single column finds a rule in the noise
    rng = np.random.default_rng(0)
    column_of = np.r_[np.arange(10), np.full(30, 10)]
    numeric = np.r_[np.ones(10, bool), False]

    def side() -> np.ndarray:
        return np.c_[rng.normal(size=(20, 10)), np.eye(30)[rng.integers(0, 30, 20)]]

    confusion = []
    for seed in range(20):
        a, b = side(), side()
        dependence = feature_dependence(a, b, column_of, 3, seed=seed)
        confusion.append(pair_confusion(a, b, column_of, numeric, dependence)[0].mean())
    assert np.mean(confusion) > 0.3 and min(confusion) >= 0.1


def test_pair_sizes_is_gone():
    # a cell is measured whole, so that min_rows holds what it promises;
    # only the rival class, pooled, is capped
    import src.quality as quality

    assert not hasattr(quality, "pair_sizes")


def test_pair_confusion_takes_the_classifier_whose_worse_side_errs_least():
    # One categorical column of two slots: a always 0; b chosen on 0 for 4
    # rows and 1 for 6 (0 to a, 1 to b), measured on 0 for 9 rows of 10.
    # The rule errs (0, 0.9), mean 0.45, the 1-NN (0.8, 0.2), mean 0.5: the
    # rule's worse side is the worse one, so the 1-NN is the better.
    slots = np.eye(2)
    a = slots[[0] * 20]
    b = slots[[0] * 9 + [1] + [0] * 4 + [1] * 6]
    half = np.r_[np.ones(10, bool), np.zeros(10, bool)]  # the first 10 measured
    measure = np.r_[half, half]
    dependence = Dependence((), np.array([[0.8, 0.2]]), measure)
    column_of, numeric = np.array([0, 0]), np.array([False])
    rule = single_column(a[10:], b[10:], column_of, numeric, Ya=a[:10], Yb=b[:10])
    assert (1 - rule.side_a, 1 - rule.side_b) == pytest.approx((0.0, 0.9))
    confusion = pair_confusion(a, b, column_of, numeric, dependence)
    np.testing.assert_array_equal(confusion, [[0.8, 0.2]])


def test_single_column_skips_the_columns_asked():
    Xa = np.c_[np.arange(10.0), np.zeros(10)]
    Xb = np.c_[np.arange(20.0, 30.0), np.r_[np.ones(9), 0.0]]
    column_of, numeric = np.array([0, 1]), np.array([True, True])
    assert single_column(Xa, Xb, column_of, numeric).column == 0
    rest = single_column(Xa, Xb, column_of, numeric, skip=[0])
    assert rest.column == 1 and rest.side_b == pytest.approx(0.9)
    assert single_column(Xa, Xb, column_of, numeric, skip=[0, 1]).column == -1


def test_share_bounds_are_wilson_one_sided():
    low, high = share_bounds(0.5, 100, 0.95)
    assert (low, high) == pytest.approx((0.41885, 0.58115), abs=1e-4)
    low, high = share_bounds(0.0, 100, 0.95)
    assert low == 0.0 and high == pytest.approx(2.7055 / 102.7055, abs=1e-4)
    assert share_bounds(1.0, 10, 0.95)[1] == pytest.approx(1.0)
    assert share_bounds(0.3, 0, 0.95) == (0.0, 1.0)
    # narrower with more rows, wider with more confidence
    assert share_bounds(0.3, 400, 0.95)[0] > share_bounds(0.3, 40, 0.95)[0]
    assert share_bounds(0.3, 40, 0.99)[0] < share_bounds(0.3, 40, 0.95)[0]


def test_decisive_rows_let_a_cell_lack_the_defect():
    # with no row confused, the upper bound must fall under tau: fewer
    # rows and no cell can ever be told to lack the defect
    for tau in (0.05, 0.1, 0.15):
        for confidence in (0.95, 0.99):
            n = decisive_rows(tau, confidence)
            assert share_bounds(0.0, n, confidence)[1] < tau
            assert share_bounds(0.0, n - 1, confidence)[1] >= tau
    assert decisive_rows(0.1, 0.95) == 25  # z^2 (1 - tau) / tau = 24.3
    assert decisive_rows(0.15, 0.95) == 16
    # a smaller tau or a higher confidence needs more rows
    assert decisive_rows(0.05, 0.95) > decisive_rows(0.1, 0.95)
    assert decisive_rows(0.1, 0.99) > decisive_rows(0.1, 0.95)


def test_mean_bounds_and_verdict():
    low, high = mean_bounds(2.0, 1.0, 25, 0.95)
    assert (low, high) == pytest.approx((2 - 1.64485 / 5, 2 + 1.64485 / 5), abs=1e-4)
    assert mean_bounds(2.0, 1.0, 0, 0.95) == (-math.inf, math.inf)
    assert verdict(0.2, 0.3, 0.1) is True and verdict(0.1, 0.3, 0.1) is True
    assert verdict(0.01, 0.09, 0.1) is False
    assert verdict(0.05, 0.15, 0.1) is None


def test_rival_defects_by_hand():
    found = rival_defects(
        [
            (1, np.array([0.8, 0.9]), 100),  # inside class 1
            (2, np.array([0.3, 0.3]), 100),  # overlaps class 2
            (3, np.array([0.0, 0.4, 0.5]), 100),  # apart but for one column
            (4, np.array([0.12, 0.3]), 10),  # too few rows to say
            (5, np.array([0.0, 0.02, 0.03]), 100),  # apart
        ],
        0.1,
        0.95,
    )
    expected = {
        1: (True, True, False),
        2: (False, True, False),
        3: (False, False, True),
        4: (False, None, None),
        5: (False, False, False),
    }
    for k, (suspect, overlap, shortcut) in expected.items():
        assert found[("suspect_label", k)] is suspect, k
        assert found[("class_overlap", k)] is overlap, k
        assert found[("shortcut", k)] is shortcut, k
    assert rival_defects([], 0.1, 0.95) == {}


def test_a_shortcut_is_how_far_the_confusion_rises_not_how_high_it_reaches():
    # Against class 1 the confusion reaches 0.14, over tau, but the key
    # columns add only 0.08 of the rows: they are not what holds the two
    # apart. Against class 2 they add 0.2 of them.
    found = rival_defects(
        [(1, np.array([0.06, 0.14]), 1000), (2, np.array([0.05, 0.25]), 100)],
        0.1,
        0.95,
    )
    assert found[("class_overlap", 1)] is False  # 0.06 of 1000 is short of tau
    assert found[("shortcut", 1)] is False  # by its value, 0.14, it would be True
    assert found[("shortcut", 2)] is True
    # apart (0 of 60 rows confused), but 60 rows cannot tell a rise of
    # 0.12 from tau: undecided, and no shortcut is claimed
    few = rival_defects([(3, np.array([0.0, 0.12]), 60)], 0.1, 0.95)
    assert few[("class_overlap", 3)] is False and few[("shortcut", 3)] is None


def test_cell_variance_by_hand():
    points = np.array([[0.0, 0.0], [2.0, 0.0], [5.0, 5.0], [0.0, 0.0]])
    cell = np.array([0, 0, 1, 2])
    points[3] = [9.0, 9.0]
    variance, sd = cell_variance(points, cell, 3)
    # cell 0: centre (1, 0), squared distances 1 and 1
    assert variance.tolist() == pytest.approx([1.0, 0.0, 0.0])
    assert sd.tolist() == pytest.approx([0.0, 0.0, 0.0])
    variance, sd = cell_variance(np.array([[0.0], [1.0], [5.0]]), np.zeros(3, int), 1)
    d = np.array([4.0, 1.0, 9.0])  # centre 2
    assert (variance[0], sd[0]) == pytest.approx((d.mean(), d.std()))


def test_redundancy_verdict_by_hand():
    assert redundancy_verdict(0.01, 0.005, 100, 1.0, 0.1, 0.95) is True
    assert redundancy_verdict(0.5, 0.2, 100, 1.0, 0.1, 0.95) is False
    assert redundancy_verdict(0.09, 0.1, 25, 1.0, 0.1, 0.95) is None
    assert redundancy_verdict(0.0, 0.0, 30, 0.0, 0.1, 0.95) is None


def test_provenance_verdict_by_hand():
    assert provenance_verdict(0.9, 0.8, 100, 0.1, 0.95) is True
    assert provenance_verdict(0.12, 0.02, 100, 0.1, 0.95) is False
    assert provenance_verdict(0.3, 0.2, 5, 0.1, 0.95) is None


def test_defects_of_by_hand():
    assert defects_of(0, 10, 10.0, 0, ()) == ()
    assert defects_of(1, 10, 10.0, 0, ()) == ("invalid_values",)
    assert defects_of(0, 10, 12.0, 0, ()) == ("duplicates",)
    assert defects_of(0, 10, 10.0, 2, ()) == ("label_conflict",)
    assert defects_of(0, 10, 10.0, 0, {"shortcut", "redundancy"}) == (
        "shortcut",
        "redundancy",
    )
    assert defects_of(1, 10, 11.0, 1, RIVAL_DEFECTS + CLASS_DEFECTS) == DEFECTS
    assert DEFECTS == ROW_DEFECTS + RIVAL_DEFECTS + CLASS_DEFECTS

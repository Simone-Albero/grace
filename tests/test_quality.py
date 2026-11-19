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
    RivalMeasure,
    Rule,
    SingleColumn,
    branch_spread,
    cell_samples,
    cell_variance,
    class_rule,
    column_values,
    dataset_variance,
    decisive_rows,
    defects_of,
    invalid_values,
    label_conflicts,
    mean_bounds,
    nearest_error,
    pair_conflicts,
    pair_confusion,
    repetition_verdicts,
    rival_defects,
    rival_pairs,
    rule_errors,
    share_bounds,
    single_column,
    variance_verdict,
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
    if case == "categorical":
        a = np.c_[a[:, :11], np.eye(6)[rng.integers(0, 3, n)]]
        b = np.c_[b[:, :11], np.eye(6)[rng.integers(3, 6, n)]]
        column_of = np.r_[np.arange(11), np.full(6, 11)]
    elif case == "spread":
        b[:, :10] += 1.5
    return a, b, column_of


def _confusion(case: str) -> np.ndarray:
    a, b, column_of = _pair(case)
    numeric = np.r_[np.ones(column_of.max(), bool), case != "categorical"]
    return pair_confusion(a, b, column_of, numeric, seed=0).confusion


def test_a_separation_spread_over_many_columns_is_apart():
    assert (_confusion("spread") < 0.1).all()


def test_one_distribution_overlaps():
    assert (_confusion("same") > 0.3).all()


def test_pair_confusion_halves_and_shapes():
    a, b, column_of = _pair("spread", n=40)
    numeric = np.ones(12, bool)
    got = pair_confusion(a, b, column_of, numeric, seed=(1, 2, 3))
    assert got.nn_error.shape == got.confusion.shape == (2,)
    assert ((got.confusion >= 0) & (got.confusion <= 1)).all()
    # half of each side measured, the other half chooses the single column
    assert got.measure.shape == (80,) and got.measure.dtype == bool
    assert got.measure[:40].sum() == got.measure[40:].sum() == 20
    again = pair_confusion(a, b, column_of, numeric, seed=(1, 2, 3))
    np.testing.assert_array_equal(again.measure, got.measure)
    np.testing.assert_array_equal(again.confusion, got.confusion)
    assert pair_confusion(a[:3], b, column_of, numeric, seed=0) is None


def test_pair_confusion_takes_the_better_classifier():
    # The categorical column splits the sides, 11 noisy numbers swamp the
    # 1-NN: the single column wins.
    a, b, column_of = _pair("categorical")
    numeric = np.r_[np.ones(11, bool), False]
    got = pair_confusion(a, b, column_of, numeric, seed=0)
    assert got.nn_error.mean() >= 0.1
    np.testing.assert_array_equal(got.confusion, [0.0, 0.0])
    assert got.single.column == 11


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
    # measured on the same rows, the single column would find a rule in the
    # noise; measured on the other half, it does not
    rng = np.random.default_rng(0)
    column_of = np.r_[np.arange(10), np.full(30, 10)]
    numeric = np.r_[np.ones(10, bool), False]

    def side() -> np.ndarray:
        return np.c_[rng.normal(size=(20, 10)), np.eye(30)[rng.integers(0, 30, 20)]]

    confusion = []
    for seed in range(20):
        a, b = side(), side()
        confusion.append(
            pair_confusion(a, b, column_of, numeric, seed).confusion.mean()
        )
    assert np.mean(confusion) > 0.3 and min(confusion) >= 0.1


def test_pair_confusion_takes_the_classifier_whose_worse_side_errs_least():
    # One categorical column of two slots: a always 0; on b, value 1 on 7
    # rows of 20. The single column sends 1 to b and 0 to a: it errs on
    # none of a and on the 0s of b, its worse side; the 1-NN can do no
    # better on b, ties sharing the vote. The better is the one whose worse
    # side errs less, never the mean of the two sides.
    slots = np.eye(2)
    a, b = slots[[0] * 20], slots[[0] * 13 + [1] * 7]
    column_of, numeric = np.array([0, 0]), np.array([False])
    got = pair_confusion(a, b, column_of, numeric, seed=0)
    rule = 1 - np.array([got.single.side_a, got.single.side_b])
    better = got.nn_error if got.nn_error.max() <= rule.max() else rule
    np.testing.assert_array_equal(got.confusion, better)


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


def _measure(k, confusion, n_scored=100, rules=((0.5, 0.5),), n_cell=100, n_rival=1000):
    return RivalMeasure(k, confusion, n_scored, np.array(rules, float), n_cell, n_rival)


def test_rival_defects_by_hand():
    found = rival_defects(
        [
            _measure(1, 0.8),  # inside class 1
            _measure(2, 0.3),  # overlaps class 2
            _measure(3, 0.0, rules=[(0.0, 0.01), (0.3, 0.2)]),  # one column apart
            _measure(4, 0.12, n_scored=10),  # too few rows to say
            _measure(5, 0.0, rules=[(0.3, 0.02), (0.02, 0.4)]),  # no column both
            _measure(6, 0.3, rules=[(0.0, 0.0)]),  # a column, but overlapping
            _measure(7, None),  # a side too small
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
        6: (False, True, False),
        7: (None, None, None),
    }
    for k, (suspect, overlap, shortcut) in expected.items():
        assert found[("suspect_label", k)] is suspect, k
        assert found[("class_overlap", k)] is overlap, k
        assert found[("shortcut", k)] is shortcut, k
    assert rival_defects([], 0.1, 0.95) == {}


def test_a_shortcut_column_is_read_with_the_intervals_of_both_errors():
    # 0.05 of 60 rows of the cell: its upper bound is over 0.1; the same
    # error on 1000 rows is under it. The rival side alike.
    few = rival_defects([_measure(1, 0.0, rules=[(0.05, 0.0)], n_cell=60)], 0.1, 0.95)
    assert few[("shortcut", 1)] is None
    many = rival_defects(
        [_measure(1, 0.0, rules=[(0.05, 0.0)], n_cell=1000)], 0.1, 0.95
    )
    assert many[("shortcut", 1)] is True
    rival = rival_defects(
        [_measure(1, 0.0, rules=[(0.0, 0.08)], n_rival=60)], 0.1, 0.95
    )
    assert rival[("shortcut", 1)] is None
    # no column at all (none under one half): no shortcut
    bare = rival_defects([_measure(1, 0.0, rules=np.empty((0, 2)))], 0.1, 0.95)
    assert bare[("shortcut", 1)] is False


def test_rule_first_sends_each_value_to_its_side():
    v = np.array([1.0, 2.0, 3.0])
    assert Rule("le", 2.0).first(v).tolist() == [True, True, False]
    assert Rule("lt", 2.0).first(v).tolist() == [True, False, False]
    assert Rule("ge", 2.0).first(v).tolist() == [False, True, True]
    assert Rule("gt", 2.0).first(v).tolist() == [False, False, True]
    assert Rule("values", np.array([1.0, 3.0])).first(v).tolist() == [
        False,
        True,
        False,
    ]


def _balanced(rule: Rule, va: np.ndarray, vb: np.ndarray) -> float:
    return (rule.first(va).mean() + 1 - rule.first(vb).mean()) / 2


def test_class_rule_is_the_best_threshold_either_way():
    va, vb = np.arange(10.0), np.arange(5.0, 25.0)
    rule = class_rule(va, vb, numeric=True)
    # a below 5 is all right on b, and 5 of 10 of a: 0.75; a below 9 gets
    # all of a and 15 of 20 of b: 0.875
    assert _balanced(rule, va, vb) == pytest.approx(0.875)
    flipped = class_rule(vb, va, numeric=True)
    assert _balanced(flipped, vb, va) == pytest.approx(0.875)


def test_class_rule_is_exact_against_every_threshold():
    rng = np.random.default_rng(3)
    for _ in range(20):
        va = rng.integers(0, 30, rng.integers(1, 40)).astype(float)
        vb = rng.integers(10, 40, rng.integers(1, 400)).astype(float)
        best = max(
            max(_balanced(Rule(kind, t), va, vb) for kind in ("le", "lt", "ge", "gt"))
            for t in np.unique(np.r_[va, vb])
        )
        assert _balanced(class_rule(va, vb, True), va, vb) == pytest.approx(best)


def test_class_rule_on_a_categorical_sends_each_value_to_its_larger_share():
    va, vb = np.array([0, 0, 1, 2, 2]), np.array([1, 1, 2, 2, 2, 3])
    rule = class_rule(va.astype(float), vb.astype(float), numeric=False)
    # 0: a only; 1: 1/5 on a, 2/6 on b; 2: 2/5 on a, 3/6 on b; 3: b only
    assert rule.kind == "values" and rule.at.tolist() == [1.0, 2.0, 3.0]
    assert rule.first(np.array([4.0])).tolist() == [True]  # unseen: to a


def test_class_rule_without_a_rule_better_than_chance():
    rule = class_rule(np.ones(5), np.ones(8), numeric=True)
    assert _balanced(rule, np.ones(5), np.ones(8)) == pytest.approx(0.5)


def test_rule_errors_read_the_rule_of_the_classes_on_every_cell():
    # class 0 below 10, class 1 above, but cell 1 of class 0 lies among
    # class 1: the classes' rule leaves it on the wrong side
    values = np.r_[np.arange(10.0), np.arange(10.0), np.arange(20.0, 40.0), [30.0] * 4]
    labels = np.r_[np.zeros(20, int), np.ones(24, int)]
    cell = np.r_[np.zeros(10, int), np.ones(10, int), np.full(24, 2)]
    values[10:14] = 30.0  # 4 of the 10 rows of cell 1 lie in class 1
    cell_class = np.array([0, 0, 1])
    got = rule_errors(values, labels, cell, [(0, 1), (1, 1), (2, 0)], cell_class, True)
    assert got.shape == (3, 2)
    rule = class_rule(values[labels == 0], values[labels == 1], True)
    wrong_b = rule.first(values[labels == 1]).mean()
    assert got[0].tolist() == pytest.approx([0.0, wrong_b])
    assert got[1].tolist() == pytest.approx([0.4, wrong_b])
    # the other way round: the error on class 2's cell, then on class 0
    back = class_rule(values[labels == 1], values[labels == 0], True)
    assert got[2, 1] == pytest.approx(back.first(values[labels == 0]).mean())


def test_column_values_reads_a_number_or_the_active_slot():
    assert column_values(np.array([[1.5], [2.0]]), True).tolist() == [1.5, 2.0]
    slots = np.array([[0.0, 0.7, 0.0], [0.0, 0.0, 0.0], [0.7, 0.0, 0.0]])
    assert column_values(slots, False).tolist() == [1.0, -1.0, 0.0]


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


def test_variance_verdict_by_hand():
    assert variance_verdict(0.01, 0.005, 100, 1.0, 0.1, 0.95) is True
    assert variance_verdict(0.5, 0.2, 100, 1.0, 0.1, 0.95) is False
    assert variance_verdict(0.09, 0.1, 25, 1.0, 0.1, 0.95) is None
    assert variance_verdict(0.0, 0.0, 30, 0.0, 0.1, 0.95) is None


def test_dataset_variance_is_the_mean_squared_distance_to_the_centroid():
    points = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 3.0]])
    centre = points.mean(axis=0)
    expected = ((points - centre) ** 2).sum(axis=1).mean()
    assert dataset_variance(points) == pytest.approx(expected)
    assert dataset_variance(np.ones((4, 3))) == pytest.approx(0.0)


def _tree():
    """Two classes: 0 -> (1 -> (cell 0, cell 1), cell 2); 5 -> (cell 3, cell 4)."""
    parent = np.array([-1, 0, 1, 1, 0, -1, 5, 5])
    node_cell = np.array([-1, -1, 0, 1, 2, -1, 3, 4])
    return parent, node_cell


def test_branch_spread_is_exact_on_every_branch():
    rng = np.random.default_rng(1)
    parent, node_cell = _tree()
    cell = rng.integers(0, 5, 300)
    points = rng.normal(size=(300, 4)) * np.array([1.0, 3.0, 0.5, 10.0]) + 50.0
    variance, sd, n = branch_spread(points, cell, parent, node_cell)
    below = {0: [0, 1, 2], 1: [0, 1], 5: [3, 4]}
    for node in range(len(parent)):
        cells = below.get(node, [node_cell[node]])
        rows = points[np.isin(cell, cells)]
        d = ((rows - rows.mean(axis=0)) ** 2).sum(axis=1)
        assert n[node] == len(rows)
        assert variance[node] == pytest.approx(d.mean(), rel=1e-9)
        assert sd[node] == pytest.approx(d.std(), rel=1e-6)
    # a leaf, as cell_variance
    alone, alone_sd = cell_variance(points, cell, 5)
    assert variance[node_cell >= 0] == pytest.approx(alone[node_cell[node_cell >= 0]])
    assert sd[node_cell >= 0] == pytest.approx(alone_sd[node_cell[node_cell >= 0]])


def test_repetition_verdicts_by_hand():
    parent, node_cell = _tree()
    n_read = np.array([100.0, 95.0, 50.0, 45.0, 5.0, 100.0, 50.0, 50.0])
    rows = np.full(8, 1000.0)
    # node 1 (95% of class 0) tight; node 4 tight but 5% of its class; class 1
    # loose at the root, tight in node 6, undecided in node 7
    variance = np.array([5.0, 0.1, 0.1, 0.1, 0.1, 5.0, 0.1, 0.99])
    sd = np.array([1.0, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.5])
    found, top = repetition_verdicts(
        parent, node_cell, n_read, (variance, sd, rows), 10.0, 0.1, 0.95
    )
    assert found == [True, True, False, True, None]
    assert top.tolist() == [1, 1, -1, 6, -1]
    # against a wide dataset every root is tight: the largest branch, the root
    found, top = repetition_verdicts(
        parent, node_cell, n_read, (variance, sd, rows), 1000.0, 0.01, 0.95
    )
    assert found == [True] * 5 and top.tolist() == [0, 0, 0, 5, 5]


def test_defects_of_by_hand():
    assert defects_of(0, 10, 10.0, 0, ()) == ()
    assert defects_of(1, 10, 10.0, 0, ()) == ("invalid_values",)
    assert defects_of(0, 10, 12.0, 0, ()) == ("duplicates",)
    assert defects_of(0, 10, 10.0, 2, ()) == ("label_conflict",)
    assert defects_of(0, 10, 10.0, 0, {"off_scenario", "shortcut", "repetition"}) == (
        "shortcut",
        "repetition",
        "off_scenario",
    )
    assert defects_of(1, 10, 11.0, 1, RIVAL_DEFECTS + CLASS_DEFECTS) == DEFECTS
    assert DEFECTS == ROW_DEFECTS + RIVAL_DEFECTS + CLASS_DEFECTS

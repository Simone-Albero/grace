"""src.quality on hand-made rows."""

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
    branch_spread,
    class_rule,
    column_values,
    dataset_variance,
    defects_of,
    invalid_values,
    label_conflicts,
    nearest_rivals,
    pair_conflicts,
    repetition_verdicts,
    resolution_rows,
    rival_defects,
    rival_shares,
    rule_errors,
    share_bounds,
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


# --- the rows confused with a rival class ---


def _brute(points: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The distance to the nearest row of another class and of its own (itself left out)."""
    d = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    other = labels[:, None] != labels[None, :]
    return np.where(other, d, np.inf).min(1), np.where(~other, d, np.inf).min(1)


def test_nearest_rivals_by_hand():
    # class 0 at 0, 1, 4; class 1 at 4.5, 10; class 2 at 20, alone
    points = np.array([0.0, 1.0, 4.0, 4.5, 10.0, 20.0])[:, None]
    labels = np.array([0, 0, 0, 1, 1, 2])
    rival_row, confused = nearest_rivals(points, labels)
    assert rival_row.dtype == np.int64
    assert rival_row.tolist() == [3, 3, 3, 2, 2, 4]
    assert confused.tolist() == [0.0, 0.0, 1.0, 1.0, 0.0, 1.0]


def test_nearest_rivals_count_a_tie_one_half():
    # 0 is 1 from its own row and 1 from the rival row
    points = np.array([0.0, 1.0, -1.0])[:, None]
    rival_row, confused = nearest_rivals(points, np.array([0, 0, 1]))
    assert rival_row.tolist() == [2, 2, 0]
    assert confused.tolist() == [0.5, 0.0, 1.0]


def test_nearest_rivals_of_one_class_are_none():
    rival_row, confused = nearest_rivals(np.arange(4.0)[:, None], np.zeros(4, int))
    assert rival_row.tolist() == [-1] * 4 and confused.tolist() == [0.0] * 4


def test_nearest_rivals_are_exact_and_do_not_depend_on_the_order():
    rng = np.random.default_rng(0)
    points = rng.normal(size=(300, 5))
    labels = rng.integers(0, 4, 300)
    rival_row, confused = nearest_rivals(points, labels)
    rival, own = _brute(points, labels)
    assert (labels[rival_row] != labels).all()
    np.testing.assert_allclose(
        np.linalg.norm(points[rival_row] - points, axis=1), rival
    )
    np.testing.assert_array_equal(confused, (rival < own).astype(float))
    order = rng.permutation(300)
    again_row, again = nearest_rivals(points[order], labels[order])
    np.testing.assert_array_equal(again, confused[order])
    np.testing.assert_array_equal(order[again_row], rival_row[order])


def test_rival_shares_by_hand():
    # cells 0 and 1 of class 0, 2 of class 1, 3 of class 2
    cell = np.array([0, 0, 1, 2, 2, 3])
    cell_class = np.array([0, 0, 1, 2])
    rival_row = np.array([3, 4, 3, 2, 5, 4])
    confused = np.array([0.0, 0.5, 1.0, 1.0, 0.0, 1.0])
    assert rival_shares(cell, rival_row, confused, cell_class) == [
        (0, 1, [2], 0.25),
        (1, 1, [2], 1.0),
        (2, 0, [1], 0.5),
        (2, 2, [3], 0.0),
        (3, 1, [2], 1.0),
    ]
    none = np.full(6, -1)
    assert rival_shares(cell, none, np.zeros(6), cell_class) == []


def test_rows_of_a_class_spread_over_its_cells_are_confused_with_the_class():
    # A cell of 60 rows lying among the 1,000 rows of a class cut into 20
    # cells: its rows are confused with the class whole, whatever its cells.
    rng = np.random.default_rng(0)
    a = np.c_[rng.uniform(0, 20, 60), rng.uniform(0, 1, 60)]
    b = np.vstack(
        [np.c_[rng.uniform(i, i + 1, 50), rng.uniform(0, 1, 50)] for i in range(20)]
    )
    cell = np.r_[np.zeros(60, int), 1 + np.repeat(np.arange(20), 50)]
    cell_class = np.r_[0, np.ones(20, int)]
    rival_row, confused = nearest_rivals(np.r_[a, b], cell_class[cell])
    shares = rival_shares(cell, rival_row, confused, cell_class)
    (own,) = [s for s in shares if s[0] == 0]
    assert own[1] == 1 and own[3] >= 0.85 and len(own[2]) >= 15


def test_two_classes_of_one_distribution_overlap():
    rng = np.random.default_rng(0)
    points = rng.normal(size=(400, 12))
    _, confused = nearest_rivals(points, np.repeat([0, 1], 200))
    assert confused.mean() > 0.3


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


def test_resolution_rows_decide_a_share_tau_away_from_a_threshold():
    # on that many rows a share tau past tau or one half has the defect,
    # one tau short of it has none; on one row fewer some is undecided
    for tau in (0.05, 0.1, 0.15, 0.2):
        for confidence in (0.95, 0.99):
            n = resolution_rows(tau, confidence)
            for t in (tau, 0.5):
                assert verdict(*share_bounds(t + tau, n, confidence), t) is True
                assert verdict(*share_bounds(t - tau, n, confidence), t) is False
            fewer = [
                verdict(*share_bounds(t + sign * tau, n - 1, confidence), t)
                for t in (tau, 0.5)
                for sign in (1, -1)
            ]
            assert None in fewer
    # z^2 / (4 tau^2) = 270.5, 67.6, 30.1, 16.9: one half asks for the most
    assert [resolution_rows(t, 0.95) for t in (0.05, 0.1, 0.15, 0.2)] == [
        271,
        68,
        31,
        17,
    ]
    assert resolution_rows(0.1, 0.99) > resolution_rows(0.1, 0.95)
    assert resolution_rows(0.5, 0.95) >= 1


def test_verdict_by_hand():
    assert verdict(0.2, 0.3, 0.1) is True and verdict(0.1, 0.3, 0.1) is True
    assert verdict(0.01, 0.09, 0.1) is False
    assert verdict(0.05, 0.15, 0.1) is None


def _measure(k, share, n_rows=100, rules=((0.5, 0.5),), n_rival=1000):
    return RivalMeasure(k, share, n_rows, np.array(rules, float), n_rival)


def test_rival_defects_by_hand():
    found = rival_defects(
        [
            _measure(1, 0.8),  # inside class 1
            _measure(2, 0.3),  # overlaps class 2
            _measure(3, 0.0, rules=[(0.0, 0.01), (0.3, 0.2)]),  # one column apart
            _measure(4, 0.12, n_rows=10),  # too few rows to say
            _measure(5, 0.0, rules=[(0.3, 0.02), (0.02, 0.4)]),  # no column both
            _measure(6, 0.3, rules=[(0.0, 0.0)]),  # a column, and overlapping
        ],
        0.1,
        0.95,
    )
    expected = {
        1: (True, True, False),
        2: (False, True, False),
        3: (False, False, True),
        4: (False, None, False),
        5: (False, False, False),
        6: (False, True, True),
    }
    for k, (suspect, overlap, shortcut) in expected.items():
        assert found[("suspect_label", k)] is suspect, k
        assert found[("class_overlap", k)] is overlap, k
        assert found[("shortcut", k)] is shortcut, k
    assert rival_defects([], 0.1, 0.95) == {}


def test_a_shortcut_column_is_read_with_the_intervals_of_both_errors():
    # 0.05 of 60 rows of the cell: its upper bound is over 0.1; the same
    # error on 1000 rows is under it. The rival side alike.
    few = rival_defects([_measure(1, 0.0, rules=[(0.05, 0.0)], n_rows=60)], 0.1, 0.95)
    assert few[("shortcut", 1)] is None
    many = rival_defects(
        [_measure(1, 0.0, rules=[(0.05, 0.0)], n_rows=1000)], 0.1, 0.95
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
    variance, n = branch_spread(points, cell, parent, node_cell)
    below = {0: [0, 1, 2], 1: [0, 1], 5: [3, 4]}
    for node in range(len(parent)):
        cells = below.get(node, [node_cell[node]])
        rows = points[np.isin(cell, cells)]
        d = ((rows - rows.mean(axis=0)) ** 2).sum(axis=1)
        assert n[node] == len(rows)
        assert variance[node] == pytest.approx(d.mean(), rel=1e-9)
    # the root of all the rows, as dataset_variance
    whole = np.array([-1, 0, 0, 0, 0, 0])
    variance, n = branch_spread(points, cell, whole, np.array([-1, 0, 1, 2, 3, 4]))
    assert n[0] == 300
    assert variance[0] == pytest.approx(dataset_variance(points), rel=1e-9)


def test_repetition_verdicts_by_hand():
    parent, node_cell = _tree()
    n_read = np.array([100.0, 95.0, 50.0, 45.0, 5.0, 100.0, 50.0, 50.0])
    # node 1 (95% of class 0) tight; node 4 tight but 5% of its class; class 1
    # loose at the root, tight in node 6, loose in node 7; at tau times the
    # dataset's variance, 1.0, a branch is still tight
    variance = np.array([5.0, 1.0, 0.1, 0.1, 0.1, 5.0, 0.1, 1.01])
    found, top = repetition_verdicts(parent, node_cell, n_read, variance, 10.0, 0.1)
    assert found == [True, True, False, True, False]
    assert top.tolist() == [1, 1, -1, 6, -1]
    # against a wide dataset every root is tight: the largest branch, the root
    found, top = repetition_verdicts(parent, node_cell, n_read, variance, 1000.0, 0.01)
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

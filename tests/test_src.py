"""The pure functions of src/ on toy data."""

import math

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.model_selection import GridSearchCV

from src import baselines, tree
from src.complexity.common import weighted_group_measures
from src.data import cap_per_class, stratified_holdout
from src.graph import build_train_graph
from src.ml import grid_search
from src.scores import ap_lift, auroc, binned_mse_ratio


def test_stratified_holdout_is_stratified_and_deterministic() -> None:
    """Check the holdout's size, its per-class split and its seed."""
    rows = pd.DataFrame({"label": np.repeat([0, 1], [80, 20]), "x": np.arange(100)})
    fit, holdout = stratified_holdout(
        rows, holdout_frac=0.1, label_col="label", random_state=0
    )
    assert len(holdout) == math.ceil(0.1 * len(rows))
    assert len(fit) + len(holdout) == len(rows)
    assert holdout["label"].value_counts().to_dict() == {0: 8, 1: 2}
    fit_again, holdout_again = stratified_holdout(
        rows, holdout_frac=0.1, label_col="label", random_state=0
    )
    assert_frame_equal(holdout, holdout_again)
    assert_frame_equal(fit, fit_again)
    # A different seed must draw different rows, not just give the
    # same result independently of random_state.
    _, holdout_other_seed = stratified_holdout(
        rows, holdout_frac=0.1, label_col="label", random_state=1
    )
    assert not holdout_other_seed["x"].equals(holdout["x"])


def test_stratified_holdout_rejects_a_split_too_small_for_its_classes() -> None:
    """Check it names the condition, not sklearn's own opaque error."""
    rows = pd.DataFrame({"label": np.repeat([0, 1, 2], 10), "x": np.arange(30)})
    with pytest.raises(ValueError, match="holdout_frac"):
        stratified_holdout(rows, holdout_frac=0.05, label_col="label", random_state=0)


def test_cap_per_class_keeps_small_classes_whole() -> None:
    """Check a class under the cap is kept, a larger one cut to it."""
    rows = pd.DataFrame(
        {"label": np.repeat([0, 1, 2], [50, 8, 20]), "x": np.arange(78)}
    )
    capped = cap_per_class(rows, 20, label_col="label", random_state=0)
    assert capped["label"].value_counts().to_dict() == {0: 20, 2: 20, 1: 8}
    assert cap_per_class(rows, None, label_col="label", random_state=0) is rows


def test_grid_search_subsample_keeps_every_class() -> None:
    """Check a search subsample never leaves a class under `cv` rows."""
    rng = np.random.default_rng(0)
    labels = np.repeat([0, 1, 2], [400, 400, 6])
    X = pd.DataFrame({"num_1": rng.normal(size=len(labels)) + labels})
    seen = []

    class _Spy(GridSearchCV):
        def fit(self, X, y, **kwargs):
            seen.append(np.bincount(np.asarray(y), minlength=3))
            return super().fit(X, y, **kwargs)

    import src.ml as ml

    original = ml.GridSearchCV
    ml.GridSearchCV = _Spy
    try:
        grid_search(
            "logistic_regression",
            {"max_iter": 200},
            {"C": [1.0]},
            X,
            labels,
            num_cols=["num_1"],
            cat_cols=[],
            scoring="f1_macro",
            cv=3,
            max_samples=100,
            n_jobs=1,
            random_state=0,
        )
    finally:
        ml.GridSearchCV = original
    assert seen[0].min() >= 3


def test_duplicate_points_join_inside_the_mst() -> None:
    """Check that identical rows join by their zero-length edges."""
    rng = np.random.default_rng(0)
    points = np.vstack([rng.normal(size=(50, 4)), np.tile(rng.normal(size=4), (10, 1))])
    duplicates = np.arange(50, 60)
    mst = build_train_graph(points, k=3, n_jobs=1).mst_edges
    assert len(mst) == len(points) - 1
    # Else each copy is bridged to node 0 alone, by a long edge.
    assert np.isin(mst, duplicates).all(axis=1).sum() == len(duplicates) - 1


def test_tied_confidences_share_a_bin() -> None:
    """Check that a tie on every confidence leaves one bin, index 0."""
    confidence = np.full(10, 0.5)
    edges = baselines.confidence_edges(confidence, n_bins=4)
    assert edges.size == 1
    assert (baselines.assign_bins(confidence, edges) == 0).all()


def test_class_bins_cover_every_class() -> None:
    """Check each predicted class gets bins, a tiny minority one too."""
    rng = np.random.default_rng(1)
    confidence = rng.uniform(size=101)
    predicted_class = np.array([0] * 95 + [1] * 5 + [2])
    edges = baselines.class_confidence_edges(confidence, predicted_class, n_bins=4)
    assert set(edges) == {0, 1, 2}
    assert len(edges[0]) >= 2 and len(edges[1]) >= 2
    bins = baselines.assign_class_bins(confidence, predicted_class, edges)
    # Every group id belongs to exactly one predicted class: none is
    # shared across classes, unlike a plain within-class bin index.
    for class_id in (0, 1, 2):
        own_ids = set(bins[predicted_class == class_id])
        other_ids = set(bins[predicted_class != class_id])
        assert not own_ids & other_ids


def test_class_missing_from_edges_gets_its_own_group() -> None:
    """Check a class edges never saw gets one id, not a KeyError."""
    confidence = np.array([0.1, 0.9, 0.5])
    predicted_class = np.array([0, 0, 2])
    edges = baselines.class_confidence_edges(
        confidence[:2], predicted_class[:2], n_bins=2
    )
    assert set(edges) == {0}
    bins = baselines.assign_class_bins(confidence, predicted_class, edges)
    assert bins[2] not in bins[:2]


def test_class_bins_stay_consistent_across_a_split_call() -> None:
    """Check one call on two row sets' union, then split, numbers alike.

    `assign_class_bins` numbers a call's groups from the classes that
    call is given, so calling it separately on two row sets that
    predict different classes is not guaranteed to agree on an id; a
    caller with two sets to route (such as val and test) must instead
    call it once on their union and split the result.
    """
    first_confidence = np.array([0.6, 0.65, 0.3, 0.35])
    first_predicted = np.array([0, 0, 2, 2])
    second_confidence = np.array([0.62, 0.5, 0.32, 0.4])
    second_predicted = np.array([0, 1, 2, 2])  # class 1 only here.
    edges = baselines.class_confidence_edges(
        first_confidence, first_predicted, n_bins=2
    )

    confidence = np.concatenate([first_confidence, second_confidence])
    predicted = np.concatenate([first_predicted, second_predicted])
    group = baselines.assign_class_bins(confidence, predicted, edges)
    first_group, second_group = np.split(group, [len(first_confidence)])

    for class_id in (0, 1, 2):
        ids = np.concatenate(
            [
                first_group[first_predicted == class_id],
                second_group[second_predicted == class_id],
            ]
        )
        # Every row of a class gets the same id, wherever it came from.
        assert (ids == ids[0]).all()
    own_ids = {class_id: group[predicted == class_id][0] for class_id in (0, 1, 2)}
    assert len(set(own_ids.values())) == len(own_ids)


def test_apportion_sums_to_the_total() -> None:
    """Check the largest-remainder shares always sum to the total."""
    cases = [
        (15, [90, 5, 5]),  # np.round over-allocated this to 16.
        (15, [1] * 10),  # np.round over-allocated this to 20.
        (10, [1, 1, 1]),  # np.round under-allocated this to 9.
        (3, [998, 1, 1]),  # flooring every share to 1 overshoots 3.
    ]
    for total, weights in cases:
        allocated = baselines._apportion(total, np.array(weights))
        assert int(allocated.sum()) == total
        assert (allocated >= 1).all()


def test_empty_group_takes_global_rate() -> None:
    """Check a group id with no rows gets the global rate, not NaN."""
    group = np.array([0, 0, 1, 1, 1])
    observed = np.array([1, 0, 0, 1, 1])
    rates = baselines.estimate_group_rates(group, observed, [0, 1, 2])
    assert rates.loc[0] == pytest.approx(0.5)
    assert rates.loc[1] == pytest.approx(2 / 3)
    assert rates.loc[2] == pytest.approx(observed.mean())


def _blobs(n_per_blob: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Return three 2-D blobs and each row's predicted class, 0 or 1."""
    rng = np.random.default_rng(0)
    centres = np.array([[0.0, 0.0], [6.0, 0.0], [0.0, 6.0]])
    points = np.vstack(
        [centre + rng.normal(size=(n_per_blob, 2)) for centre in centres]
    )
    predicted = np.repeat([0, 0, 1], n_per_blob)
    return points, predicted


def test_route_reproduces_growth_of_train_rows() -> None:
    """Check descend ends each row where the growth put it."""
    points, predicted = _blobs()
    grown, node_of_row = tree.grow_tree(
        points, predicted, n_classes=2, n_min=10, random_state=0
    )
    assert len(grown.parent) > 2
    assert (tree.descend(grown, points, predicted) == node_of_row).all()


def test_no_grown_child_under_n_min() -> None:
    """Check every split leaves n_min rows or more on each side."""
    points, predicted = _blobs()
    grown, node_of_row = tree.grow_tree(
        points, predicted, n_classes=2, n_min=15, random_state=0
    )
    n_rows = tree.subtree_counts(grown, node_of_row, np.ones(len(points)))
    children = np.flatnonzero(grown.parent >= 0)
    assert (n_rows[children] >= 15).all()
    assert n_rows[grown.root].sum() == len(points)


def test_identical_points_make_a_leaf() -> None:
    """Check a node whose points are all the same is never split."""
    points = np.ones((100, 2))
    grown, _ = tree.grow_tree(
        points, np.zeros(100, dtype=int), n_classes=1, n_min=5, random_state=0
    )
    assert len(grown.parent) == 1 and grown.left[0] == -1


def _cut_costs(grown: tree.Tree, cost: np.ndarray, node: int) -> list[float]:
    """List the cost of every cut of the subtree under `node`."""
    cuts = [cost[node]]
    if grown.left[node] >= 0:
        for left in _cut_costs(grown, cost, grown.left[node]):
            for right in _cut_costs(grown, cost, grown.right[node]):
                cuts.append(left + right)
    return cuts


def test_prune_matches_brute_force() -> None:
    """Check the bottom-up cut is the cheapest of all cuts."""
    points, predicted = _blobs(80)
    rng = np.random.default_rng(1)
    errors = (rng.random(len(points)) < np.where(points[:, 0] > 3, 0.6, 0.05)).astype(
        float
    )
    grown, node_of_row = tree.grow_tree(
        points, predicted, n_classes=2, n_min=10, random_state=0
    )
    n_rows = tree.subtree_counts(grown, node_of_row, np.ones(len(points)))
    n_errors = tree.subtree_counts(grown, node_of_row, errors)
    is_leaf, is_kept, cost, best = tree.prune_tree(
        grown, n_rows, n_errors, n_train=len(points), n_val=50
    )
    for root in grown.root:
        assert best[root] == pytest.approx(min(_cut_costs(grown, cost, root)))
    assert cost[is_leaf].sum() == pytest.approx(best[grown.root].sum())
    assert (is_leaf <= is_kept).all()


def test_tau2_never_negative_and_no_val_leaf_takes_regression() -> None:
    """Check tau2 is not negative and an unseen leaf takes the line."""
    counts = np.array([40.0, 60.0, 50.0, 80.0])
    errors = np.array([4.0, 6.0, 5.0, 8.0])
    covariate = np.array([0.1, 0.1, 0.1, 0.1])
    intercept, slope, tau2 = tree.fit_fay_herriot(errors, counts, covariate)
    assert tau2 >= 0 and slope == 0
    rate, weight = tree.shrink_rates(
        np.array([0.0, 5.0]),
        np.array([0.0, 50.0]),
        np.array([0.2, 0.1]),
        intercept=0.05,
        slope=0.5,
        tau2=0.01,
    )
    assert rate[0] == pytest.approx(0.15) and np.isnan(weight[0])
    assert 0.0 <= rate[1] <= 1.0 and 0.0 < weight[1] < 1.0


def test_disagreement_excludes_self() -> None:
    """Check a train row is never its own neighbour."""
    points = np.arange(10.0)[:, None]
    labels = np.arange(10)
    index = tree.build_neighbours(points, n_jobs=1)
    own = tree.neighbour_disagreement(index, labels, labels, n_neighbours=3)
    assert (own == 1.0).all()
    other = tree.neighbour_disagreement(
        index, labels, labels, n_neighbours=3, points=points
    )
    assert (other < 1.0).all()


def test_class_never_predicted_oof_gets_an_empty_root() -> None:
    """Check a class nobody predicted keeps a root with no rows."""
    points, predicted = _blobs()
    grown, node_of_row = tree.grow_tree(
        points, predicted, n_classes=3, n_min=10, random_state=0
    )
    n_rows = tree.subtree_counts(grown, node_of_row, np.ones(len(points)))
    assert n_rows[grown.root[2]] == 0 and grown.left[grown.root[2]] == -1
    is_leaf, _, _, _ = tree.prune_tree(
        grown, n_rows, n_rows * 0, n_train=len(points), n_val=50
    )
    assert is_leaf[grown.root[2]]
    reached = tree.descend(grown, points[:3], np.array([2, 2, 2]), stop=is_leaf)
    assert (reached == grown.root[2]).all()


def test_leaf_measures_are_row_weighted_and_skip_undefined_ones() -> None:
    """Check a leaf averages its populations by rows, ignoring None."""
    measures = {
        10: {"f1": 1.0, "n1": None},
        11: {"f1": 4.0, "n1": 2.0},
        20: {"f1": 5.0, "n1": None},
    }
    leaves = weighted_group_measures(
        measures, group={10: 0, 11: 0, 20: 1}, weight={10: 30, 11: 10, 20: 7}
    )
    assert leaves[0]["f1"] == pytest.approx((30 * 1.0 + 10 * 4.0) / 40)
    assert leaves[0]["n1"] == pytest.approx(2.0)
    assert leaves[1]["f1"] == 5.0 and leaves[1]["n1"] is None


def test_bin_count_is_one_per_150_rows_and_at_least_one_per_class() -> None:
    """Check the bin rule and its floor of one bin per class."""
    assert baselines.bin_count(10_275, 9) == 68
    assert baselines.bin_count(300, 9) == 9


def test_calibrate_isotonic_never_decreases_in_the_score() -> None:
    """Check the map is monotone and stays a probability."""
    rng = np.random.default_rng(0)
    score = rng.random(500)
    errors = (rng.random(500) < score).astype(float)
    mapped = baselines.calibrate_isotonic(score, errors, np.linspace(-1, 2, 50))
    assert (np.diff(mapped) >= 0).all() and mapped.min() >= 0 and mapped.max() <= 1


def test_error_gbm_is_constant_when_every_row_has_the_same_outcome() -> None:
    """Check a one-valued target gives its rate, not a failed fit."""
    features = pd.DataFrame({"z_0": np.arange(10.0), "predicted": np.zeros(10, int)})
    predict = baselines.fit_error_gbm(features, np.zeros(10), random_state=0)
    assert (predict(features) == 0.0).all()


def test_low_confidence_is_finite_and_falls_with_confidence() -> None:
    """Check the log-odds of a wrong class: falling, finite at 1."""
    top_probability = np.array([0.5, 0.9, 0.99, 1.0])
    log_odds = tree.low_confidence(top_probability)
    assert np.isfinite(log_odds).all()
    assert (np.diff(log_odds) < 0).all()
    assert log_odds[0] == pytest.approx(0.0)


def test_a_constant_ranks_like_a_random_order() -> None:
    """Check a constant gets 0.5, 1 and 1 on the three scores."""
    errors = np.array([1.0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    # Off the failure rate, since a constant that is exact has no ratio.
    constant = np.full(10, 0.3)
    assert auroc(constant, errors) == pytest.approx(0.5)
    assert ap_lift(constant, errors) == pytest.approx(1.0)
    assert binned_mse_ratio(constant, errors, constant, 5) == pytest.approx(1.0)


def test_a_perfect_order_reaches_the_ceiling_of_the_ranking_scores() -> None:
    """Check failures ranked first reach AUROC 1 and AP / p 1/p."""
    errors = np.array([1.0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    values = np.linspace(0.9, 0.1, 10)
    assert auroc(values, errors) == 1.0
    assert ap_lift(values, errors) == pytest.approx(1 / errors.mean())


def test_binned_mse_ratio_is_zero_when_every_bin_rate_is_right() -> None:
    """Check chance inside a bin is ignored, only its rate counts."""
    errors = np.array([0.0, 0, 0, 1] * 5 + [0.0, 1, 1, 1] * 5)
    values = np.array([0.25] * 20 + [0.75] * 20)
    reference = np.full(40, errors.mean())
    assert binned_mse_ratio(values, errors, reference, 2) == pytest.approx(0.0)
    assert np.mean((values - errors) ** 2) > 0.1


def test_binned_mse_ratio_keeps_a_dominant_lowest_value_in_its_own_bin() -> None:
    """Check a mass point at the minimum leaves the rest a bin."""
    values = np.repeat([0.01, 0.9], [950, 50])
    errors = np.repeat([0.0, 1.0], [950, 50])
    reference = np.full(1000, 0.06)
    # Merged into one bin, the two gaps would cancel to a fifth.
    expected = (950 * 0.01**2 + 50 * 0.1**2) / (950 * 0.06**2 + 50 * 0.94**2)
    assert binned_mse_ratio(values, errors, reference, 10) == pytest.approx(expected)


def test_binned_mse_ratio_is_nan_when_the_reference_fits_by_float_noise() -> None:
    """Check a reference equal to the failure rate leaves no ratio."""
    errors = np.array([1.0] * 100 + [0.0] * 900)
    values = np.full(1000, 0.3)
    assert math.isnan(binned_mse_ratio(values, errors, np.full(1000, 0.1), 10))


def test_scores_are_nan_where_they_have_nothing_to_compare() -> None:
    """Check no failure, or only failures, leaves a score undefined."""
    values = np.linspace(0.1, 0.9, 10)
    for errors in (np.zeros(10), np.ones(10)):
        assert math.isnan(auroc(values, errors))
    assert math.isnan(ap_lift(values, np.zeros(10)))
    errors = np.array([0.0, 1] * 5)
    assert math.isnan(binned_mse_ratio(values, errors, np.full(10, 0.5), 5))

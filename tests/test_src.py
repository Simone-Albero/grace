"""The pure functions of src/ on toy data."""

import math

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.ensemble import RandomForestRegressor

from src import clustering, partitions, scores
from src.data import stratified_holdout
from src.graph import build_train_graph
from src.regressor import fit_failure_regressor


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


def test_duplicate_points_join_inside_the_mst() -> None:
    """Check that identical rows join by their zero-length edges."""
    rng = np.random.default_rng(0)
    points = np.vstack([rng.normal(size=(50, 4)), np.tile(rng.normal(size=4), (10, 1))])
    duplicates = np.arange(50, 60)
    mst = build_train_graph(points, k=3, metric="euclidean", n_jobs=1).mst_edges
    assert len(mst) == len(points) - 1
    # Else each copy is bridged to node 0 alone, by a long edge.
    assert np.isin(mst, duplicates).all(axis=1).sum() == len(duplicates) - 1


def _split_on_refit(
    X: np.ndarray, *, max_fit_samples: int, random_state: int
) -> np.ndarray:
    """Label 2 clusters on the subsample and 3 on the full class."""
    return np.arange(len(X)) % (2 if len(X) <= max_fit_samples else 3)


@pytest.mark.parametrize("max_clusters", [2, 3])
def test_refit_keeps_at_most_max_clusters(
    monkeypatch: pytest.MonkeyPatch, max_clusters: int
) -> None:
    """Check that a subsampled class's refit stays within its cap."""
    monkeypatch.setitem(clustering.ALGORITHMS, "split_on_refit", _split_on_refit)
    rng = np.random.default_rng(0)
    nodes = clustering.KdnNodes(
        X=rng.normal(size=(60, 2)),
        y=np.repeat([0, 1], 30),
        row_ids=np.arange(1000, 1060),
    )
    labels, n_merged_clusters, _, _ = clustering.cluster_class(
        rng.normal(size=(210, 2)),
        row_ids=np.arange(210),
        label=0,
        algorithm="split_on_refit",
        params={},
        nodes=nodes,
        max_fit_samples=100,
        max_clusters=max_clusters,
        min_cluster_floor=10,
        hardness_k=3,
        eval_rows_per_train_row=1.0,
        metric="euclidean",
        random_state=0,
    )
    # Three refit clusters of 70 rows, all above the floor of 10: a tie
    # at the cut must still keep `max_clusters` of them.
    assert np.unique(labels).size == min(max_clusters, 3)
    assert n_merged_clusters == 3 - min(max_clusters, 3)


def test_fit_failure_regressor_passes_sample_weight_to_the_forest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check every forest fit, in the inner search too, is weighted."""
    calls: list[np.ndarray | None] = []
    original_fit = RandomForestRegressor.fit

    def spy_fit(
        self: RandomForestRegressor,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: pd.Series | None = None,
    ) -> RandomForestRegressor:
        calls.append(sample_weight)
        return original_fit(self, X, y, sample_weight=sample_weight)

    monkeypatch.setattr(RandomForestRegressor, "fit", spy_fit)

    # 8 regions and 2 inner splits: enough for the inner search to run,
    # not just the no-search branch of `_fit_model`.
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
    y = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    weight = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 50.0])

    fit_failure_regressor(
        X,
        y,
        sample_weight=weight,
        params={},
        param_grid={"n_estimators": [5, 10]},
        n_outer_splits=2,
        n_inner_splits=2,
        n_iter=1,
        n_jobs=1,
        random_state=0,
    )

    # Every candidate's fit, in every inner and outer fold, plus the
    # refit: none silently dropped the weight.
    assert len(calls) > 3
    assert all(call is not None for call in calls)
    assert any(call.max() == pytest.approx(50.0) for call in calls)


def test_tied_confidences_share_a_bin() -> None:
    """Check that a tie on every confidence leaves one bin, index 0."""
    confidence = np.full(10, 0.5)
    edges = partitions.confidence_edges(confidence, n_bins=4)
    assert edges.size == 1
    assert (partitions.assign_bins(confidence, edges) == 0).all()


def test_class_bins_cover_every_class() -> None:
    """Check each predicted class gets bins, a tiny minority one too."""
    rng = np.random.default_rng(1)
    confidence = rng.uniform(size=101)
    predicted_class = np.array([0] * 95 + [1] * 5 + [2])
    edges = partitions.class_confidence_edges(confidence, predicted_class, n_bins=4)
    assert set(edges) == {0, 1, 2}
    assert len(edges[0]) >= 2 and len(edges[1]) >= 2
    bins = partitions.assign_class_bins(confidence, predicted_class, edges)
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
    edges = partitions.class_confidence_edges(
        confidence[:2], predicted_class[:2], n_bins=2
    )
    assert set(edges) == {0}
    bins = partitions.assign_class_bins(confidence, predicted_class, edges)
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
    edges = partitions.class_confidence_edges(
        first_confidence, first_predicted, n_bins=2
    )

    confidence = np.concatenate([first_confidence, second_confidence])
    predicted = np.concatenate([first_predicted, second_predicted])
    group = partitions.assign_class_bins(confidence, predicted, edges)
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
        allocated = partitions._apportion(total, np.array(weights))
        assert int(allocated.sum()) == total
        assert (allocated >= 1).all()


def test_empty_group_takes_global_rate() -> None:
    """Check a group id with no rows gets the global rate, not NaN."""
    group = np.array([0, 0, 1, 1, 1])
    observed = np.array([1, 0, 0, 1, 1])
    rates = partitions.estimate_group_rates(group, observed, [0, 1, 2])
    assert rates.loc[0] == pytest.approx(0.5)
    assert rates.loc[1] == pytest.approx(2 / 3)
    assert rates.loc[2] == pytest.approx(observed.mean())


def test_kdn_per_row_labels_match_one_class() -> None:
    """Check per-row labels reproduce separate single-class calls."""
    rng = np.random.default_rng(3)
    nodes = clustering.KdnNodes(
        X=rng.normal(size=(60, 2)),
        y=np.repeat([0, 1], 30),
        row_ids=np.arange(1000, 1060),
    )
    X = rng.normal(size=(20, 2))
    row_ids = np.arange(2000, 2020)
    labels = np.repeat([0, 1], 10)

    mixed = clustering.compute_kdn(X, row_ids=row_ids, labels=labels, nodes=nodes, k=5)
    for label in (0, 1):
        in_class = labels == label
        per_class = clustering.compute_kdn(
            X[in_class],
            row_ids=row_ids[in_class],
            labels=np.full(in_class.sum(), label),
            nodes=nodes,
            k=5,
        )
        assert np.allclose(mixed[in_class], per_class)


def test_level_pick_matches_cluster_class() -> None:
    """Check the level kmeans_levels scores best is cluster_class's."""
    rng = np.random.default_rng(6)
    X = np.vstack(
        [
            rng.normal(loc=loc, scale=0.3, size=(60, 2))
            for loc in ([0, 0], [5, 5], [0, 5], [5, 0], [2.5, 2.5])
        ]
    )
    row_ids = np.arange(len(X))
    nodes = clustering.KdnNodes(X=X, y=np.zeros(len(X), dtype=int), row_ids=row_ids)
    min_cluster_floor = 20
    hardness_k = 5
    eval_rows_per_train_row = 0.4

    best_labels, _, _, best_loss = clustering.cluster_class(
        X,
        row_ids=row_ids,
        label=0,
        algorithm="kmeans",
        params={},
        nodes=nodes,
        max_fit_samples=10000,
        max_clusters=1000,
        min_cluster_floor=min_cluster_floor,
        hardness_k=hardness_k,
        eval_rows_per_train_row=eval_rows_per_train_row,
        metric="euclidean",
        random_state=7,
    )

    levels = clustering.kmeans_levels(
        X,
        min_cluster_floor=min_cluster_floor,
        max_fit_samples=10000,
        metric="euclidean",
        random_state=7,
    )
    hardness = clustering.compute_kdn(
        X,
        row_ids=row_ids,
        labels=np.zeros(len(X), dtype=int),
        nodes=nodes,
        k=hardness_k,
    )
    losses = [
        clustering.granularity_loss(
            level_labels,
            hardness,
            n_class=len(X),
            eval_rows_per_train_row=eval_rows_per_train_row,
        )
        for level_labels in levels
    ]
    best_level = int(np.argmin(losses))
    assert losses[best_level] == pytest.approx(best_loss)
    assert np.array_equal(levels[best_level], best_labels)


def test_null_draw_is_seeded() -> None:
    """Check random_voronoi is reproducible from its own rng seed."""
    X = np.random.default_rng(8).normal(size=(100, 2))
    draw_args = dict(min_cluster_floor=2, metric="euclidean")
    first = clustering.random_voronoi(X, 10, rng=np.random.default_rng(0), **draw_args)
    again = clustering.random_voronoi(X, 10, rng=np.random.default_rng(0), **draw_args)
    different = clustering.random_voronoi(
        X, 10, rng=np.random.default_rng(1), **draw_args
    )
    assert np.array_equal(first, again)
    assert not np.array_equal(first, different)


def test_brier_decomposition_is_exact() -> None:
    """Check brier decomposes exactly into its three terms."""
    rng = np.random.default_rng(2)
    group = rng.integers(0, 6, size=500)
    observed = rng.binomial(1, 0.3, size=500).astype(float)
    group_rate = pd.Series(observed).groupby(group).transform("mean")
    predicted = np.clip(group_rate.to_numpy() + rng.normal(scale=0.05, size=1), 0, 1)
    score = scores.group_scores(predicted, observed, group, reference=0.3)
    assert score["brier"] == pytest.approx(
        score["uncertainty"] - score["resolution"] + score["reliability"], abs=1e-12
    )


def test_kmeans_levels_leave_no_undersized_group() -> None:
    """Check every level clears the floor, or merges to one group."""
    rng = np.random.default_rng(4)
    X = np.vstack([rng.normal(size=(300, 2)), rng.normal(loc=8, size=(7, 2))])
    levels = clustering.kmeans_levels(
        X,
        min_cluster_floor=20,
        max_fit_samples=10000,
        metric="euclidean",
        random_state=0,
    )
    assert len(levels) == len(clustering.level_counts(len(X), 20))
    for labels in levels:
        _, counts = np.unique(labels, return_counts=True)
        assert counts.min() >= 20 or counts.size == 1
    # The finest level must actually survive as more than one group,
    # not merge everything away: a mutant that always collapses to one
    # group would otherwise still pass.
    _, finest_counts = np.unique(levels[0], return_counts=True)
    assert finest_counts.size > 1


def test_random_voronoi_leaves_no_undersized_cell() -> None:
    """Check cells under the floor merge, more than one group left."""
    X = np.random.default_rng(5).normal(size=(200, 2))
    labels = clustering.random_voronoi(
        X,
        20,
        min_cluster_floor=8,
        metric="euclidean",
        rng=np.random.default_rng(0),
    )
    assert labels.shape == (200,)
    _, counts = np.unique(labels, return_counts=True)
    assert counts.min() >= 8
    assert counts.size > 1


def test_random_voronoi_caps_n_cells_at_the_row_count() -> None:
    """Check asking for more cells than rows draws at most one each."""
    X = np.random.default_rng(5).normal(size=(40, 2))
    labels = clustering.random_voronoi(
        X,
        1000,
        min_cluster_floor=4,
        metric="euclidean",
        rng=np.random.default_rng(0),
    )
    assert labels.shape == (40,)
    _, counts = np.unique(labels, return_counts=True)
    assert counts.min() >= 4 or counts.size == 1

"""The pure functions of src/ on toy data."""

import math

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.ensemble import RandomForestRegressor

from src import clustering
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

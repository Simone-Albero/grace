"""src.models, src.evaluation and src.baselines on hand-made data."""

import numpy as np
import pandas as pd
import pytest

from src.baselines import atc_rate, atc_threshold, failure_rates, mcp_rate
from src.evaluation import classification_metrics, confusion, rate_scores
from src.models import (
    MODELS,
    CappedCategoryEncoder,
    build_pipeline,
    predict_proba,
    undersample,
)


def test_undersample():
    labels = np.array([0] * 5 + [1] * 2 + [2] * 3)
    kept = undersample(labels, seed=0)
    assert kept.tolist() == sorted(kept.tolist())
    assert np.bincount(labels[kept]).tolist() == [2, 2, 2]
    np.testing.assert_array_equal(kept, undersample(labels, seed=0))


def _toy(n: int = 90) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    y = np.arange(n) % 3
    frame = pd.DataFrame(
        {
            "a": y + rng.normal(0, 0.3, n),
            "b": rng.normal(0, 1, n),
            "c": (y * 2 + rng.integers(0, 2, n)).astype(np.int32),
        }
    )
    return frame, y


@pytest.mark.parametrize("name", sorted(MODELS))
def test_every_model_gives_probabilities(name):
    X, y = _toy()
    model = build_pipeline(name, {}, ["a", "b"], ["c"], seed=0).fit(X, y)
    proba = predict_proba(model, X, 3)
    assert proba.shape == (90, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert (proba.argmax(axis=1) == y).mean() > 0.6


def test_build_pipeline_seed_and_params():
    model = build_pipeline("random_forest", {"n_estimators": 5}, ["a"], [], seed=7)
    assert model.named_steps["classifier"].random_state == 7
    assert model.named_steps["classifier"].n_estimators == 5
    model = build_pipeline("random_forest", {"random_state": 1}, ["a"], [], seed=7)
    assert model.named_steps["classifier"].random_state == 1
    with pytest.raises(ValueError, match="nope"):
        build_pipeline("nope", {}, ["a"], [], seed=0)


def test_predict_proba_checks_the_classes():
    X, y = _toy()
    model = build_pipeline("naive_bayes", {}, ["a", "b"], ["c"], seed=0).fit(X, y)
    with pytest.raises(ValueError):
        predict_proba(model, X, 4)


def test_capped_category_encoder():
    encoder = CappedCategoryEncoder(max_cardinality=1)
    out = encoder.fit(pd.DataFrame({"c": [1, 1, 2]})).transform(
        pd.DataFrame({"c": [1, 2, 3]})
    )
    assert isinstance(out["c"].dtype, pd.CategoricalDtype)
    assert out["c"].isna().tolist() == [False, True, True]


def test_classification_metrics_and_confusion():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    rows = classification_metrics(y_true, y_pred, ["a", "b"])
    value = {(r["label"], r["metric"]): r["value"] for r in rows}
    assert value[("all", "accuracy")] == pytest.approx(0.75)
    assert value[("a", "precision")] == pytest.approx(1.0)
    assert value[("a", "recall")] == pytest.approx(0.5)
    assert value[("b", "precision")] == pytest.approx(2 / 3)
    assert value[("b", "f1")] == pytest.approx(0.8)
    assert value[("a", "support")] == 2
    assert value[("all", "macro_f1")] == pytest.approx((2 / 3 + 0.8) / 2)
    assert confusion(y_true, y_pred, 2).tolist() == [[1, 1], [0, 2]]


def test_rate_scores():
    observed = np.array([0.1, 0.2, 0.4, 0.8])
    scores = rate_scores(observed / 2, observed)
    assert scores["spearman"] == pytest.approx(1.0)
    assert scores["mae"] == pytest.approx(np.mean(observed / 2))
    assert scores["mse"] == pytest.approx(np.mean((observed / 2) ** 2))
    assert np.isnan(rate_scores(np.full(4, 0.3), observed)["spearman"])
    assert np.isnan(rate_scores(observed, np.full(4, 0.3))["r2"])


def test_failure_rates():
    n_eval, n_error, rate = failure_rates(
        np.array([1, 1, 2, 2, 2]),
        np.array([True, False, False, False, True]),
        np.array([1, 2, 3]),
    )
    assert n_eval.tolist() == [2, 3, 0]
    assert n_error.tolist() == [1, 1, 0]
    np.testing.assert_allclose(rate, [0.5, 1 / 3, np.nan])


def test_mcp_and_atc_rates():
    confidence = np.array([0.9, 0.7, 0.6])
    region = np.array([1, 1, 2])
    regions = np.array([1, 2, 3])
    np.testing.assert_allclose(
        mcp_rate(confidence, region, regions), [0.2, 0.4, np.nan]
    )
    np.testing.assert_allclose(
        atc_rate(confidence, region, regions, 0.8), [0.5, 1.0, np.nan]
    )


def test_atc_threshold():
    correct = np.array([True, True, False, False])
    assert atc_threshold(np.array([0.9, 0.8, 0.7, 0.6]), correct) == 0.8
    assert atc_threshold(np.array([0.5, 0.9]), np.array([True, False])) == 0.9
    ties = atc_threshold(np.array([0.5, 0.5, 0.9]), np.array([False, True, True]))
    assert ties == 0.5
    assert atc_threshold(np.array([0.5, 0.6]), np.array([False, False])) == np.inf


def test_averages_skip_classes_absent_from_truth_and_predictions():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 0, 1])
    rows = classification_metrics(y_true, y_pred, ["a", "b", "never"])
    value = {(r["label"], r["metric"]): r["value"] for r in rows}
    assert value[("all", "macro_f1")] == pytest.approx(2 / 3)

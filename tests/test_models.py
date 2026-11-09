"""src.models, src.evaluation and src.baselines on hand-made data."""

import numpy as np
import polars as pl
import pytest

from src.baselines import (
    atc_rate,
    atc_threshold,
    bcts_confidence,
    error_inputs,
    error_probability,
    failure_rates,
    fit_bcts,
    mcp_rate,
    meta_rate,
)
from src.evaluation import classification_metrics, confusion, rate_scores
from src.models import MODELS, build_pipeline, predict_proba
from src.preprocessing import Features

FEATURES = Features(num=("a", "b"), cat=("c",))


def _toy(n: int = 90) -> tuple[pl.DataFrame, np.ndarray]:
    """Three classes; a separates them, c half does, a few a missing."""
    rng = np.random.default_rng(0)
    y = np.arange(n) % 3
    a_values = y + rng.normal(0, 0.3, n)
    a_values[:3] = [np.nan, np.inf, np.nan]
    frame = pl.DataFrame(
        {
            "a": a_values,
            "b": rng.normal(0, 1, n),
            "c": [f"v{code}" for code in y * 2 + rng.integers(0, 2, n)],
        }
    )
    return frame, y


@pytest.mark.parametrize("name", sorted(MODELS))
def test_every_model_gives_probabilities_despite_missing_numbers(name):
    X, y = _toy()
    model = build_pipeline(name, {}, FEATURES, top_n=4, seed=0).fit(X, y)
    assert ("impute" in model.named_steps) is not MODELS[name].nan_ok
    proba = predict_proba(model, X, 3)
    assert proba.shape == (90, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert (proba.argmax(axis=1) == y).mean() > 0.6


def test_build_pipeline_seed_and_params():
    model = build_pipeline("random_forest", {"n_estimators": 5}, FEATURES, 4, seed=7)
    assert model.named_steps["classifier"].random_state == 7
    assert model.named_steps["classifier"].n_estimators == 5
    model = build_pipeline("random_forest", {"random_state": 1}, FEATURES, 4, seed=7)
    assert model.named_steps["classifier"].random_state == 1
    with pytest.raises(ValueError, match="nope"):
        build_pipeline("nope", {}, FEATURES, 4, seed=0)


def test_predict_proba_checks_the_classes_and_spans_chunks(monkeypatch):
    X, y = _toy()
    model = build_pipeline("naive_bayes", {}, FEATURES, 4, seed=0).fit(X, y)
    whole = predict_proba(model, X, 3)
    monkeypatch.setattr("src.models._CHUNK_ROWS", 7)
    np.testing.assert_array_equal(predict_proba(model, X, 3), whole)
    with pytest.raises(ValueError):
        predict_proba(model, X, 4)


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


def test_bcts_corrects_the_prior():
    # A classifier fitted on balanced classes says 0.5 everywhere, but
    # 80% of the rows are class 0: the bias must move it to 0.8.
    proba = np.full((1000, 2), 0.5)
    label = (np.arange(1000) >= 800).astype(int)
    temperature, bias = fit_bcts(proba, label)
    pred = np.array([0, 1])
    confidence = bcts_confidence(proba[:2], pred, temperature, bias)
    np.testing.assert_allclose(confidence, [0.8, 0.2], atol=1e-4)


def test_bcts_softens_an_overconfident_classifier():
    # Always 0.99 sure of the right class, which it gets right 70% of
    # the time: the calibrated confidence must fall to about 0.7.
    n = 2000
    label = np.arange(n) % 2
    pred = np.where(np.arange(n) // 2 % 10 < 7, label, 1 - label)
    proba = np.full((n, 2), 0.01)
    proba[np.arange(n), pred] = 0.99
    temperature, bias = fit_bcts(proba, label)
    assert temperature > 1
    confidence = bcts_confidence(proba, pred, temperature, bias)
    np.testing.assert_allclose(confidence, 0.7, atol=1e-3)


def _classifier_output(n: int, wrong: np.ndarray) -> pl.DataFrame:
    """Two classes, confidence 0.8 on every row, whatever wrong is."""
    pred = np.zeros(n, dtype=np.int64)
    return pl.DataFrame(
        {
            "row": np.arange(n),
            "label": np.where(wrong, 1, 0),
            "pred": pred,
            "confidence": np.full(n, 0.8),
            "proba_0": np.full(n, 0.8),
            "proba_1": np.full(n, 0.2),
        }
    )


def test_error_inputs_append_the_classifier_output():
    rows, _ = _toy(6)
    inputs, features = error_inputs(rows, _classifier_output(6, np.zeros(6)), FEATURES)
    assert features == Features(
        num=("a", "b", "proba_0", "proba_1", "confidence"), cat=("c", "pred")
    )
    assert set(inputs.columns) == {*features.num, *features.cat}
    assert inputs["pred"].dtype == pl.String


def test_meta_learns_where_the_classifier_fails():
    # The classifier is wrong exactly where a > 0, whatever its
    # confidence: only the row's columns tell meta where it fails.
    rng = np.random.default_rng(0)
    n = 2000
    a = rng.normal(size=n)
    rows = pl.DataFrame({"a": a, "b": rng.normal(size=n), "c": ["x"] * n})
    wrong = a > 0
    inputs, features = error_inputs(rows, _classifier_output(n, wrong), FEATURES)
    half = n // 2
    probability = error_probability(
        inputs[:half],
        wrong[:half],
        inputs[half:],
        features,
        params={"max_iter": 50},
        max_samples=800,
        top_n=5,
        seed=0,
    )
    region = wrong[half:].astype(int)
    rate = meta_rate(probability, region, np.array([0, 1, 2]))
    assert rate[0] < 0.1 and rate[1] > 0.9 and np.isnan(rate[2])


def test_meta_without_errors_to_learn_is_constant():
    rows, _ = _toy(6)
    inputs, features = error_inputs(rows, _classifier_output(6, np.zeros(6)), FEATURES)
    probability = error_probability(
        inputs,
        np.zeros(6, bool),
        inputs,
        features,
        params={},
        max_samples=6,
        top_n=5,
        seed=0,
    )
    np.testing.assert_array_equal(probability, np.zeros(6))


def test_averages_skip_classes_absent_from_truth_and_predictions():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 0, 1])
    rows = classification_metrics(y_true, y_pred, ["a", "b", "never"])
    value = {(r["label"], r["metric"]): r["value"] for r in rows}
    assert value[("all", "macro_f1")] == pytest.approx(2 / 3)

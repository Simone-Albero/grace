"""src.ablation: feature sets, within-class ranking, paired bootstrap."""

import numpy as np
import pytest

from src.ablation import (
    feature_sets,
    paired_bootstrap,
    residual_corr,
    within_class_spearman,
)


def test_feature_sets():
    sets = feature_sets(3, ["mcp", "meta"])
    assert list(sets) == [
        "geometry",
        "signals",
        "geometry+signals",
        "mcp",
        "geometry+mcp",
        "meta",
        "geometry+meta",
    ]
    assert sets["geometry"].tolist() == [0, 1, 2]
    assert sets["signals"].tolist() == [3, 4]
    assert sets["meta"].tolist() == [4]
    assert sets["geometry+meta"].tolist() == [0, 1, 2, 4]


def test_within_class_spearman_ignores_small_and_constant_classes():
    observed = np.r_[np.arange(10.0), np.arange(10.0), np.zeros(10), [1.0, 2.0]]
    predicted = np.r_[np.arange(10.0), -np.arange(10.0), np.arange(10.0), [2, 1]]
    group = np.repeat([0, 1, 2, 3], [10, 10, 10, 2])
    # Classes 0 and 1 count (+1 and -1); 2 is constant, 3 too small.
    assert within_class_spearman(predicted, observed, group) == pytest.approx(0.0)
    assert np.isnan(within_class_spearman(predicted, observed, group, 11))


def test_residual_corr():
    observed = np.array([0.1, 0.5, 0.3, 0.9])
    shift = np.array([0.1, -0.1, 0.2, 0.0])
    assert residual_corr(observed + shift, observed + 2 * shift, observed) == (
        pytest.approx(1.0)
    )
    assert np.isnan(residual_corr(observed, observed + shift, observed))


def test_paired_bootstrap_brackets_a_known_gain():
    rng = np.random.default_rng(0)
    observed = rng.uniform(0, 1, 200)
    better = observed + rng.normal(0, 0.01, 200)
    worse = observed + rng.normal(0, 0.1, 200)
    weight = rng.uniform(1, 10, 200)
    result = paired_bootstrap(better, worse, observed, weight, n_bootstrap=300, seed=0)
    full = np.abs(better - observed).mean() - np.abs(worse - observed).mean()
    assert result["d_mae"] == pytest.approx(full)
    for key in ("d_mae", "d_wmae"):
        assert result[f"{key}_lo"] <= result[key] <= result[f"{key}_hi"] < 0
    assert result["d_spearman_lo"] > 0
    same = paired_bootstrap(worse, worse, observed, weight, n_bootstrap=50, seed=0)
    assert same["d_mae_lo"] == same["d_mae_hi"] == 0

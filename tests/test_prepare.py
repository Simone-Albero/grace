"""src.prepare on hand-made rows."""

import math

import numpy as np
import polars as pl
import pytest

from src.prepare import (
    MISSING,
    NULL,
    OTHER,
    TopNEncoder,
    column_names,
    fit_preparation,
    signed_log1p,
    transform,
)


def test_signed_log1p():
    x = np.array([-3.0, 0.0, math.e - 1])
    np.testing.assert_allclose(signed_log1p(x), [-math.log(4), 0.0, 1.0])


def test_top_n_encoder_codes_the_frequent_values_and_shares_the_rest():
    fitted = pl.Series("c", ["a", "a", "b", "b", "c", None, "a"])
    encoder = TopNEncoder("c", top_n=2).fit(fitted)
    assert encoder.values_ == ("a", "b")
    assert encoder.has_other_
    codes = encoder.transform(pl.Series("c", ["a", "b", "c", None, "zzz"]))
    assert codes.dtype == np.int32
    assert codes.tolist() == [1, 2, 0, 0, 0]
    assert encoder.slot_names() == ["c=a", "c=b", f"c={OTHER}"]


def test_top_n_encoder_counts_null_as_a_value():
    encoder = TopNEncoder("c", top_n=1).fit(pl.Series("c", [None, None, "x"]))
    assert encoder.values_ == (None,)
    assert encoder.transform(pl.Series("c", [None, "x"])).tolist() == [1, 0]
    assert encoder.slot_names() == [f"c={NULL}", f"c={OTHER}"]


def test_top_n_encoder_ties_put_the_smaller_value_first_and_null_last():
    encoder = TopNEncoder("c", top_n=1).fit(pl.Series("c", ["b", "a"]))
    assert encoder.values_ == ("a",)
    encoder = TopNEncoder("c", top_n=1).fit(pl.Series("c", [None, "z"]))
    assert encoder.values_ == ("z",)


def test_top_n_encoder_has_no_other_slot_when_every_value_fits():
    encoder = TopNEncoder("c", top_n=5).fit(pl.Series("c", ["a", "b", "a"]))
    assert encoder.values_ == ("a", "b")
    assert not encoder.has_other_
    assert encoder.slot_names() == ["c=a", "c=b"]
    empty = encoder.transform(pl.Series("c", [], dtype=pl.String))
    assert empty.dtype == np.int32 and len(empty) == 0


def test_numbers_are_standardized_and_missing_ones_sit_on_the_mean():
    rows = pl.DataFrame({"n": [0.0, 1.0, 2.0, 3.0, 4.0, None, float("inf")]})
    preparation = fit_preparation(rows, ["n"], [], top_n=4, cat_cost=1.0)
    logged = np.log1p(np.arange(5.0))
    prepared = transform(preparation, rows)
    assert prepared.columns == ["n", MISSING]
    expected = (logged - logged.mean()) / logged.std()
    np.testing.assert_allclose(prepared["n"].to_numpy()[:5], expected)
    np.testing.assert_array_equal(prepared["n"].to_numpy()[5:], [0.0, 0.0])
    assert prepared[MISSING].to_list() == [0.0] * 5 + [1.0, 1.0]


def test_missing_counts_the_numbers_of_a_row_set_to_0():
    nan, inf = float("nan"), float("inf")
    rows = pl.DataFrame(
        {"m": [1.0, None, nan, 2.0], "n": [-inf, None, 3.0, 4.0], "c": ["x"] * 4}
    )
    preparation = fit_preparation(rows, ["m", "n"], ["c"], top_n=1, cat_cost=1.0)
    prepared = transform(preparation, rows)
    assert prepared[MISSING].to_list() == [1.0, 2.0, 1.0, 0.0]
    finite = pl.DataFrame({"m": [1.0, 2.0], "n": [3.0, 5.0]})
    preparation = fit_preparation(finite, ["m", "n"], [], top_n=1, cat_cost=1.0)
    assert transform(preparation, finite)[MISSING].to_list() == [0.0, 0.0]


def test_a_categorical_mismatch_costs_cat_cost_over_the_chance_of_one():
    rows = pl.DataFrame({"n": [0.0] * 4, "c": ["a", "a", "b", "c"]})
    preparation = fit_preparation(rows, ["n"], ["c"], top_n=1, cat_cost=3.0)
    # Slots a and <other>, half the rows each: two rows differ with 1/2.
    (weight,) = preparation.weights
    assert weight == pytest.approx(math.sqrt(3.0 / 0.5))
    prepared = transform(preparation, rows)
    names = ["n", MISSING, "c=a", f"c={OTHER}"]
    assert prepared.columns == column_names(preparation) == names
    assert set(prepared.dtypes) == {pl.Float64}
    points = prepared.to_numpy()
    np.testing.assert_allclose(points[0], [0.0, 0.0, weight, 0.0])
    np.testing.assert_allclose(points[2], [0.0, 0.0, 0.0, weight])
    assert np.sum((points[0] - points[2]) ** 2) == pytest.approx(2 * 3.0 / 0.5)
    assert np.sum((points[2] - points[3]) ** 2) == 0.0


def test_a_categorical_column_adds_on_average_what_a_number_adds():
    rng = np.random.default_rng(0)
    values = rng.choice(["a", "b", "c", "d"], size=4000, p=[0.7, 0.2, 0.06, 0.04])
    rows = pl.DataFrame({"n": rng.normal(size=4000), "c": values})
    preparation = fit_preparation(rows, ["n"], ["c"], top_n=2, cat_cost=1.0)
    prepared = transform(preparation, rows)
    pairs = rng.integers(0, 4000, size=(20000, 2))
    slots = prepared.select(pl.selectors.starts_with("c=")).to_numpy()
    cat = np.sum((slots[pairs[:, 0]] - slots[pairs[:, 1]]) ** 2, axis=1).mean()
    num = (
        (prepared["n"].to_numpy()[pairs[:, 0]] - prepared["n"].to_numpy()[pairs[:, 1]])
        ** 2
    ).mean()
    assert cat == pytest.approx(2.0, rel=0.05)
    assert num == pytest.approx(2.0, rel=0.05)


def test_a_column_with_a_single_slot_weighs_nothing():
    rows = pl.DataFrame({"n": [1.0, 2.0], "c": ["a", "a"]})
    preparation = fit_preparation(rows, ["n"], ["c"], top_n=1, cat_cost=1.0)
    assert preparation.weights == (0.0,)
    assert transform(preparation, rows)["c=a"].to_list() == [0.0, 0.0]


def test_columns_follow_the_column_order():
    rows = pl.DataFrame(
        {"m": [1.0, 2.0], "n": [3.0, 5.0], "c": ["x", "y"], "d": ["u", "u"]}
    )
    preparation = fit_preparation(rows, ["n", "m"], ["d", "c"], top_n=2, cat_cost=1.0)
    assert column_names(preparation) == ["n", "m", MISSING, "d=u", "c=x", "c=y"]
    assert transform(preparation, rows).height == 2

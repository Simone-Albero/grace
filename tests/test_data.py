"""src.data and src.preprocessing on hand-made frames."""

import numpy as np
import polars as pl

from src.data import (
    SPLITS,
    cap_per_class,
    constant_columns,
    drop_rare_classes,
    stratified_split,
)
from src.preprocessing import Features, Preprocessor, feature_cols


def test_drop_rare_classes_keeps_order_and_the_class_at_the_floor():
    df = pl.DataFrame({"y": ["a", "c", "b", "a", "b", "a"], "v": list(range(6))})
    kept, dropped = drop_rare_classes(df, "y", 2)
    assert kept["v"].to_list() == [0, 2, 3, 4, 5]
    assert dropped == ["c"]


def _frame(counts: dict[str, int]) -> pl.DataFrame:
    labels = []
    for label, n_rows in counts.items():
        labels += [label] * n_rows
    return pl.DataFrame({"y": labels, "v": list(range(len(labels)))})


def test_stratified_split_is_a_stratified_partition():
    df = _frame({"a": 600, "b": 400})
    shares = {"train": 0.4, "val": 0.15, "test": 0.45}
    parts = stratified_split(df, "y", shares, seed=1)
    assert tuple(parts) == SPLITS
    values = pl.concat([parts[name]["v"] for name in SPLITS]).to_list()
    assert sorted(values) == list(range(1000))
    for name, part in parts.items():
        assert part["v"].to_list() == sorted(part["v"].to_list())
        for label, n in (("a", 600), ("b", 400)):
            got = (part["y"] == label).sum()
            assert abs(got - shares[name] * n) <= 1, (name, label, got)


def test_stratified_split_depends_on_the_seed_only():
    df = _frame({"a": 300, "b": 200})
    shares = {"train": 0.5, "val": 0.2, "test": 0.3}
    first = stratified_split(df, "y", shares, seed=3)
    again = stratified_split(df, "y", shares, seed=3)
    other = stratified_split(df, "y", shares, seed=4)
    assert first["train"].equals(again["train"])
    assert not first["train"].equals(other["train"])


def test_constant_columns_count_null_as_a_value():
    rows = pl.DataFrame(
        {
            "same": [1.0, 1.0, 1.0],
            "nulls": [None, None, None],
            "mixed": [None, 2.0, 2.0],
            "text": ["a", "a", "a"],
            "varied": [1.0, 2.0, 3.0],
        }
    )
    assert constant_columns(rows, rows.columns) == ["same", "nulls", "text"]
    assert constant_columns(rows, ["varied", "same"]) == ["same"]


def test_cap_per_class_keeps_order_and_small_classes():
    rows = _frame({"a": 50, "b": 3})
    capped = cap_per_class(rows, "y", 10, seed=0)
    assert capped["y"].value_counts().sort("y")["count"].to_list() == [10, 3]
    values = capped["v"].to_list()
    assert values == sorted(values)
    assert set(values) >= {50, 51, 52}
    assert capped.equals(cap_per_class(rows, "y", 10, seed=0))
    assert not capped.equals(cap_per_class(rows, "y", 10, seed=1))


def test_feature_cols_keep_the_present_ones_in_order():
    features = feature_cols(["c", "b", "label"], ["a", "b"], ["c", "d"])
    assert features == Features(num=("b",), cat=("c",))


def test_preprocessor_z_scores_and_one_hot_with_other():
    train = pl.DataFrame(
        {
            "n": [1.0, 3.0, float("inf"), None],
            "c": ["x", "x", "y", None],
        }
    )
    preprocessor = Preprocessor(Features(num=("n",), cat=("c",)), top_n=1)
    preprocessor.fit(train)
    assert preprocessor.levels_ == {"c": ["x"]}
    out = preprocessor.transform(
        pl.DataFrame({"n": [2.0, float("nan")], "c": ["z", "x"]})
    )
    assert out.shape == (2, 3)
    assert out[0, 0] == 0.0
    assert np.isnan(out[1, 0])
    np.testing.assert_array_equal(out[:, 1:], [[0.0, 1.0], [1.0, 0.0]])


def test_preprocessor_without_levels_puts_everything_in_other():
    train = pl.DataFrame({"n": [0.0, 0.0], "c": ["x", "y"]})
    preprocessor = Preprocessor(Features(num=("n",), cat=("c",)), top_n=0)
    out = preprocessor.fit(train).transform(train)
    np.testing.assert_array_equal(out, [[0.0, 1.0], [0.0, 1.0]])

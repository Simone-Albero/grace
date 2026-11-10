"""src.data on hand-made frames."""

import polars as pl

from src.data import (
    DUPLICATES,
    Features,
    constant_columns,
    drop_duplicates,
    drop_rare_classes,
    feature_cols,
)


def test_drop_rare_classes_keeps_order_and_the_class_at_the_floor():
    df = pl.DataFrame({"y": ["a", "c", "b", "a", "b", "a"], "v": list(range(6))})
    kept, dropped = drop_rare_classes(df, "y", 2)
    assert kept["v"].to_list() == [0, 2, 3, 4, 5]
    assert dropped == ["c"]


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


def test_feature_cols_keep_the_present_ones_in_order():
    features = feature_cols(["c", "b", "label"], ["a", "b"], ["c", "d"])
    assert features == Features(num=("b",), cat=("c",))
    with_count = feature_cols(["c", "b", DUPLICATES, "label"], ["a", "b"], ["c"])
    assert with_count == Features(num=("b", DUPLICATES), cat=("c",))


def test_drop_duplicates_counts_copies_within_a_label():
    nan = float("nan")
    rows = pl.DataFrame(
        {
            "v": [1.0, nan, 1.0, nan, None, None, 1.0],
            "c": ["x", "y", "x", "y", None, None, "x"],
            "y": ["a", "a", "a", "a", "b", "b", "b"],
        }
    )
    kept = drop_duplicates(rows)
    assert kept.columns == ["v", "c", "y", DUPLICATES]
    assert kept["c"].to_list() == ["x", "y", None, "x"]
    assert kept["y"].to_list() == ["a", "a", "b", "b"]
    assert kept[DUPLICATES].to_list() == [2, 2, 2, 1]

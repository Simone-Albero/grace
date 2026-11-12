"""src.data on hand-made frames."""

import math

import polars as pl

from src.data import (
    DUPLICATES,
    KINDS,
    UNPARSABLE,
    Features,
    constant_columns,
    count_kinds,
    drop_duplicates,
    drop_rare_classes,
    feature_cols,
    read_numbers,
    unparsable_cells,
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
    assert with_count == Features(num=("b",), cat=("c",))


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


def _text() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "a": ["1", None, "x", "nan", "-inf", "z"],
            "b": ["2", "y", "3", "4", "5", "w"],
            "c": ["p", None, "q", "r", "s", "t"],
        }
    )


def test_read_numbers_marks_the_text_that_is_not_a_number():
    rows = read_numbers(_text(), ["a", "b"])
    assert rows.columns == ["a", "b", "c", UNPARSABLE]
    assert rows["a"].dtype == pl.Float64 and rows["c"].dtype == pl.String
    a = rows["a"].to_list()
    assert a[0] == 1.0 and a[1] is None and a[2] is None and a[5] is None
    assert math.isnan(a[3]) and a[4] == -math.inf
    assert rows[UNPARSABLE].to_list() == [None, "b", "a", None, None, "a,b"]


def test_count_kinds_tells_empty_from_unparsable():
    rows = read_numbers(_text(), ["a", "b"])
    counts = count_kinds(rows, ["a", "b"], ["c"])
    assert set(counts) == set(KINDS)
    assert counts == {"empty": 2, "unparsable": 4, "nan": 1, "inf": 1}


def test_unparsable_cells_list_the_kept_columns_row_by_row():
    marks = pl.Series(UNPARSABLE, ["b", None, "a,b"])
    cells = unparsable_cells(marks, ["a", "b"])
    assert cells.columns == ["row", "column"]
    assert cells.rows() == [(0, "b"), (2, "a"), (2, "b")]
    assert unparsable_cells(marks, ["a"]).rows() == [(2, "a")]
    assert unparsable_cells(pl.Series(UNPARSABLE, [None], pl.String), ["a"]).height == 0


def test_drop_duplicates_keeps_an_unparsable_cell_apart_from_an_empty_one():
    text = pl.DataFrame({"a": ["n/a", None, None], "y": ["k", "k", "k"]})
    kept = drop_duplicates(read_numbers(text, ["a"]))
    assert kept[UNPARSABLE].to_list() == ["a", None]
    assert kept[DUPLICATES].to_list() == [1, 2]

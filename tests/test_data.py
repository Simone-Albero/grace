"""src.data on hand-made frames."""

import math

import numpy as np
import polars as pl
import pytest

from src.data import (
    DUPLICATES,
    KEY,
    KINDS,
    UNPARSABLE,
    Copies,
    Features,
    cap_classes,
    constant_columns,
    count_kinds,
    csv_files,
    feature_cols,
    group_copies,
    header_rows,
    rare_classes,
    read_numbers,
    row_key,
    unparsable_cells,
)


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


def _copies(rows: pl.DataFrame) -> Copies:
    keys = rows.select(row_key(rows.columns)).to_numpy()
    return group_copies(keys)


def test_row_keys_group_the_copies_within_a_label():
    nan = float("nan")
    rows = pl.DataFrame(
        {
            "v": [1.0, nan, 1.0, nan, None, None, 1.0, -0.0, 0.0, nan],
            "c": ["x", "y", "x", "y", None, None, "x", "z", "z", None],
            "y": ["a", "a", "a", "a", "b", "b", "b", "b", "b", "b"],
        }
    )
    copies = _copies(rows)
    # Groups in order of first copy; null = null, NaN = NaN, -0.0 = 0.0,
    # and the label keeps equal features apart.
    assert copies.group.dtype == np.int64
    assert copies.group.tolist() == [0, 1, 0, 1, 2, 2, 3, 4, 4, 5]
    assert copies.first.tolist() == [0, 1, 4, 6, 7, 9]
    assert copies.copies.tolist() == [2, 2, 2, 1, 2, 1]


def test_row_key_names_two_unsigned_hashes():
    keys = pl.DataFrame({"v": [1.0, 2.0]}).select(row_key(["v"]))
    assert tuple(keys.columns) == KEY
    assert keys.dtypes == [pl.UInt64, pl.UInt64]


def test_group_copies_of_no_row_are_none():
    copies = group_copies(np.empty((0, 2), dtype=np.uint64))
    assert len(copies.group) == len(copies.first) == len(copies.copies) == 0


def test_rare_classes_keep_the_class_at_the_floor():
    labels = pl.Series(["a", "c", "b", "a", "b", "a", "d"])
    assert rare_classes(labels, 2) == ["c", "d"]
    assert rare_classes(labels, 1) == []


def test_cap_classes_sample_only_the_classes_over_the_cap():
    labels = np.array(["a"] * 10 + ["b"] * 3 + ["a"] * 5 + ["c"] * 4)
    kept = cap_classes(labels, 4, seed=0)
    assert kept.dtype == bool and kept.shape == labels.shape
    counts = {k: int(kept[labels == k].sum()) for k in "abc"}
    assert counts == {"a": 4, "b": 3, "c": 4}
    assert np.array_equal(kept, cap_classes(labels, 4, seed=0))
    assert not np.array_equal(kept, cap_classes(labels, 4, seed=1))
    assert cap_classes(labels, None, seed=0).all()


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


def test_row_keys_keep_an_unparsable_cell_apart_from_an_empty_one():
    text = pl.DataFrame({"a": ["n/a", None, None], "y": ["k", "k", "k"]})
    copies = _copies(read_numbers(text, ["a"]))
    assert copies.group.tolist() == [0, 1, 1]
    assert copies.copies.tolist() == [1, 2]


def test_count_kinds_of_lazy_rows_as_of_rows_in_memory():
    rows = read_numbers(_text(), ["a", "b"])
    lazy = count_kinds(rows.lazy(), ["a", "b"], ["c"])
    assert lazy == count_kinds(rows, ["a", "b"], ["c"])
    assert count_kinds(rows.head(0), ["a", "b"], ["c"]) == dict.fromkeys(KINDS, 0)


def test_csv_files_take_a_file_or_the_sorted_matches_of_a_glob(tmp_path):
    for name in ("b.csv", "a.csv", "c.txt"):
        (tmp_path / name).write_text("v\n1\n")
    (tmp_path / "d.csv").mkdir()
    assert csv_files(tmp_path / "b.csv") == [tmp_path / "b.csv"]
    assert csv_files(tmp_path / "*.csv") == [tmp_path / "a.csv", tmp_path / "b.csv"]
    with pytest.raises(FileNotFoundError, match="no CSV matches"):
        csv_files(tmp_path / "*.parquet")


def test_header_rows_flag_only_rows_that_are_the_header_on_every_column():
    rows = pl.DataFrame(
        {
            "a": ["a", "1", "a", None, "a"],
            "b": ["b", "b", "2", "b", None],
        }
    )
    flags = rows.select(header_rows(["a", "b"]))[:, 0].to_list()
    assert flags == [True, False, False, False, False]

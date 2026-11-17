"""src.provenance on hand-made rows."""

import math
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from src.provenance import TIME, concentration, excess, read_provenance, time_window

FORMAT = "%Y-%m-%d %H:%M:%S%.f"


def test_read_provenance_reads_values_as_categories_and_times_in_their_format():
    text = pl.DataFrame(
        {
            "Src": ["1.2.3.4", None, "1.2.3.4", "5.6.7.8"],
            "When": ["2018-02-20 08:00:01.5", "n/a", None, "2018-02-20 08:00:02"],
        }
    )
    exprs = read_provenance({"src_ip": "Src", TIME: "When"}, FORMAT)
    assert list(exprs) == ["src_ip", TIME]
    read = text.select(**exprs)
    assert read["src_ip"].dtype == pl.Categorical
    assert read["src_ip"].cast(pl.String).to_list() == text["Src"].to_list()
    assert read[TIME].dtype == pl.Datetime("us")
    # A text that is no time in the format is unknown, as an empty one;
    # a time without a fraction of a second is read all the same.
    assert read[TIME].to_list() == [
        datetime(2018, 2, 20, 8, 0, 1, 500000),
        None,
        None,
        datetime(2018, 2, 20, 8, 0, 2),
    ]


def test_read_provenance_without_columns_reads_nothing():
    assert read_provenance({}, None) == {}


def _entropy_number(counts) -> float:
    p = np.asarray(counts) / sum(counts)
    return math.exp(-(p * np.log(p)).sum())


def test_concentration_of_each_group_by_hand():
    group = np.array([0, 0, 0, 0, 1, 1, 1, 2, 0, 4])
    values = pl.Series(["a", "b", "a", None, "z", "y", "x", None, "a", "q"])
    out = concentration(group, values, 5)
    assert out.columns == ["top", "top_share", "n_values", "eff_n"]
    assert out.schema["n_values"] == pl.Int64 and out.height == 5
    # Group 0: a three times, b once, the null left out.
    assert out.row(0)[:3] == ("a", 0.75, 2)
    assert out["eff_n"][0] == pytest.approx(_entropy_number([3, 1]))
    # Group 1: three values once each, a tie to the first in text order;
    # as many values as equally frequent ones.
    assert out.row(1)[:3] == ("x", pytest.approx(1 / 3), 3)
    assert out["eff_n"][1] == pytest.approx(3)
    # Group 4: a single value.
    assert out.row(4) == ("q", 1.0, 1, pytest.approx(1.0))
    # Group 2 holds a null only, group 3 no row.
    for g in (2, 3):
        assert out.row(g) == (None, None, 0, None)
    as_categories = concentration(group, values.cast(pl.Categorical), 5)
    assert as_categories.equals(out)


def test_excess_against_the_typical_group_of_its_class_by_hand():
    # Class 0: groups 0-3; class 1: groups 4-7; class 2: groups 8-9.
    rows = [
        (0, "a"), (0, "a"), (0, "a"), (0, "b"),
        (1, "a"), (1, "b"),
        (2, "a"), (2, "a"), (2, "a"), (2, "a"),
        (3, "a"), (3, "c"), (3, "c"), (3, "c"),
        (4, "a"), (4, "a"), (4, "d"), (4, "d"),
        (5, "d"), (5, "e"),
        (6, "d"), (6, "e"), (6, None),
        (7, None),
        (8, "e"), (8, "f"), (8, "f"),
    ]  # fmt: skip
    order = np.random.default_rng(0).permutation(len(rows))  # in no order
    group = np.array([rows[i][0] for i in order])
    values = pl.Series([rows[i][1] for i in order])
    group_class = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
    out = excess(group, values, group_class, 10)
    assert out.columns == ["excess_value", "excess"]
    assert out.schema == {"excess_value": pl.String, "excess": pl.Float64}
    # Class 0, four groups with a value, the typical share the 2nd
    # smallest: a is in all four (0.75, 0.5, 1, 0.25), typically 0.5; b
    # in half of them and c in one, typically 0.
    # Group 0: a 0.75 - 0.5 and b 0.25 - 0 tie, to the first in text order.
    assert out.row(0) == ("a", 0.25)
    assert out.row(1) == ("b", 0.5)
    assert out.row(2) == ("a", 0.5)
    # Its share of a is under the typical one, of c far over.
    assert out.row(3) == ("c", 0.75)
    # Class 1, three groups with a value (group 7 holds a null only), the
    # 2nd smallest: d is in all three at 0.5, e in two of three (more
    # than half: 0, 0.5, 0.5 gives 0.5), a in one (0); class 0's a,
    # typically 0.5 there, is not class 1's.
    assert out.row(4) == ("a", 0.5)
    assert out.row(5) == out.row(6) == ("d", 0.0)
    assert out.row(7) == (None, None)
    # Class 2, one group with a value: it is its own typical group.
    assert out.row(8) == ("e", 0.0)
    assert out.row(9) == (None, None)
    as_categories = excess(group, values.cast(pl.Categorical), group_class, 10)
    assert as_categories.equals(out)


def test_time_window_holds_the_central_share_of_the_rows_by_hand():
    start = datetime(2018, 2, 20, 8)

    def at(seconds):
        return start + timedelta(seconds=seconds)

    # Group 0: 21 times a second apart, the last a day later; 1: one
    # time; 2: two; 3: a null only; 4: no row.
    times = [at(s) for s in [*range(20), 86400]] + [at(0), at(0), at(30), None]
    group = np.array([0] * 21 + [1, 2, 2, 3])
    order = np.random.default_rng(0).permutation(len(group))  # in no order
    series = pl.Series([times[i] for i in order], dtype=pl.Datetime("us"))
    out = time_window(group[order], series, 5, share=0.9)
    assert out.columns == ["time_from", "time_to", "time_window"]
    # k = floor(20 * 0.1 / 2) = 1: the first and the last time are left
    # out, the day-late one among them.
    assert out.row(0) == (at(1), at(19), 18.0)
    assert out.row(1) == (at(0), at(0), 0.0)
    # k = floor(1 * 0.05) = 0: both times.
    assert out.row(2) == (at(0), at(30), 30.0)
    assert out.row(3) == out.row(4) == (None, None, None)
    # k = floor(20 * 0.5 / 2) = 5.
    wide = time_window(group[order], series, 5, share=0.5)
    assert wide.row(0) == (at(5), at(15), 10.0)

"""src.provenance on hand-made rows."""

import math
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from src.provenance import TIME, concentration, read_provenance, time_window

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


def test_time_window_leaves_out_tau_of_the_rows_by_hand():
    start = datetime(2018, 2, 20, 8)

    def at(seconds):
        return start + timedelta(seconds=seconds)

    # Group 0: 21 times a second apart, the last a day later; 1: one
    # time; 2: two; 3: a null only; 4: no row.
    times = [at(s) for s in [*range(20), 86400]] + [at(0), at(0), at(30), None]
    group = np.array([0] * 21 + [1, 2, 2, 3])
    order = np.random.default_rng(0).permutation(len(group))  # in no order
    series = pl.Series([times[i] for i in order], dtype=pl.Datetime("us"))
    out = time_window(group[order], series, 5, tau=0.1)
    assert out.columns == ["time_from", "time_to", "time_window"]
    # k = floor(20 * 0.1 / 2) = 1: the first and the last time are left
    # out, the day-late one among them.
    assert out.row(0) == (at(1), at(19), 18.0)
    assert out.row(1) == (at(0), at(0), 0.0)
    # k = floor(1 * 0.05) = 0: both times.
    assert out.row(2) == (at(0), at(30), 30.0)
    assert out.row(3) == out.row(4) == (None, None, None)
    # k = floor(20 * 0.5 / 2) = 5.
    wide = time_window(group[order], series, 5, tau=0.5)
    assert wide.row(0) == (at(5), at(15), 10.0)

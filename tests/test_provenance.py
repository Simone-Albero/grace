"""src.provenance on hand-made rows."""

import math
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from src.config import AttackScenario, ScenarioConfig
from src.provenance import (
    SCENARIO_KEYS,
    TIME,
    breach_shares,
    concentration,
    read_provenance,
    scenario_breaches,
)

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


def _origin() -> tuple[pl.DataFrame, np.ndarray]:
    """Six CSV rows: two benign, three of attack A, one of attack B."""
    t = datetime(2018, 2, 20, 10, 0, 0)
    origin = pl.DataFrame(
        {
            "row": [0, 1, 2, 3, 4, 5],
            "src_ip": ["9.9.9.9", "1.1.1.1", "1.1.1.1", "2.2.2.2", None, "3.3.3.3"],
            "dst_port": ["53", "443", "80", "53", "80", "123"],
            TIME: [t, t, t, t + timedelta(hours=2), t, None],
        }
    ).with_columns(pl.col("src_ip", "dst_port").cast(pl.Categorical))
    return origin, np.array([0, 0, 1, 1, 1, 2])


def test_scenario_breaches_rule_on_the_attacks_only():
    origin, row_class = _origin()
    t = datetime(2018, 2, 20, 10, 0, 0)
    scenario = ScenarioConfig(
        benign="Benign",
        background_ports=(53, 123),
        attacks={
            "A": AttackScenario(
                src_ip=("1.1.1.1",), time=((t - timedelta(minutes=5), t),)
            ),
        },
    )
    got = scenario_breaches(origin, row_class, ["Benign", "A", "B"], scenario)
    assert list(got) == ["dst_port", "src_ip", TIME]
    nan = math.nan
    # the benign rows break nothing; B has the ports only
    np.testing.assert_array_equal(got["dst_port"], [nan, nan, 0, 1, 0, 1])
    np.testing.assert_array_equal(got["src_ip"], [nan, nan, 0, 1, nan, nan])
    # the window is closed at its end; B has no window
    np.testing.assert_array_equal(got[TIME], [nan, nan, 0, 1, 0, nan])


def test_scenario_breaches_hold_the_keys_declared_only():
    origin, row_class = _origin()
    ports = ScenarioConfig(benign="Benign", background_ports=(53,))
    got = scenario_breaches(origin, row_class, ["Benign", "A", "B"], ports)
    assert list(got) == ["dst_port"]
    # a key the provenance lacks is not judged
    bare = scenario_breaches(
        origin.drop("dst_port"), row_class, ["Benign", "A", "B"], ports
    )
    assert bare == {}
    assert set(SCENARIO_KEYS) == {"dst_port", "src_ip", TIME}


def test_scenario_breaches_refuse_a_class_they_do_not_know():
    origin, row_class = _origin()
    for scenario in (
        ScenarioConfig(benign="Normal"),
        ScenarioConfig(benign="Benign", attacks={"C": AttackScenario(("1.1.1.1",))}),
    ):
        with pytest.raises(ValueError, match="scenario"):
            scenario_breaches(origin, row_class, ["Benign", "A", "B"], scenario)


def test_breach_shares_per_group_over_its_judged_rows():
    group = np.array([0, 0, 0, 1, 1, 2])
    breach = np.array([1.0, 0.0, np.nan, np.nan, np.nan, 1.0])
    share, n = breach_shares(group, breach, 4)
    np.testing.assert_array_equal(share, [0.5, np.nan, 1.0, np.nan])
    np.testing.assert_array_equal(n, [2, 0, 1, 0])

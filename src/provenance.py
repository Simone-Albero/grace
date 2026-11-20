"""Provenance: where each row of the CSV comes from, and the scenario it should fit.

A dataset may name the columns of its rows' source and destination
(address and port) and time (ProvenanceConfig). Stage clean reads them
for every row of the CSV, copies included (the copies of a row may come
from different hosts), and stage quality_report describes them per
cell (concentration). A dataset may also declare its scenario, what its
documentation says of where and when each class was generated
(ScenarioConfig): the ports of the background traffic, and per attack
its hosts and time windows. A row of an attack class that falls outside
them breaks the scenario (scenario_breaches); a row breaks it when it
breaks any of its keys (any_breach), and a cell with tau of its CSV rows
doing so has the defect off_scenario (src.quality).
"""

from __future__ import annotations

__all__ = [
    "KEYS",
    "any_breach",
    "SCENARIO_KEYS",
    "TIME",
    "breach_shares",
    "concentration",
    "read_provenance",
    "scenario_breaches",
]

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from src.config import ScenarioConfig

KEYS = ("src_ip", "src_port", "dst_ip", "dst_port")  # measured as values, in this order
TIME = "time"  # the time of the row
# The keys a scenario rules on, in this order: a background port as the destination of an
# attack, a source that is none of the attack's hosts, a time out of its windows.
SCENARIO_KEYS = ("dst_port", "src_ip", TIME)


def read_provenance(
    columns: Mapping[str, str], time_format: str | None
) -> dict[str, pl.Expr]:
    """Return the expressions that read the provenance of rows read as text.

    Args:
        columns: The configured keys of KEYS and TIME, each with the
            column of the CSV it is read from.
        time_format: The format of the TIME column's text (chrono's
            strftime, as polars reads it, e.g. "%Y-%m-%d %H:%M:%S%.f");
            None without TIME.

    Returns:
        By key, in the order of columns, the expression of its value: a
        key of KEYS as Categorical text, null when empty; TIME as
        Datetime("us"), null when empty or not a time in time_format.
    """
    exprs = {}
    for key, column in columns.items():
        if key == TIME:
            exprs[key] = pl.col(column).str.to_datetime(
                time_format, time_unit="us", strict=False
            )
        else:
            exprs[key] = pl.col(column).cast(pl.Categorical)
    return exprs


def concentration(group: np.ndarray, values: pl.Series, m: int) -> pl.DataFrame:
    """Return, per group of rows, how concentrated their values are.

    Args:
        group: The group of every row (its cell), in [0, m).
        values: The value of every row, text or Categorical; a null is
            unknown and left out.
        m: The number of groups.

    Returns:
        One row per group, 0 to m - 1 in order: top (String: the most
        frequent value, ties to the first in text order), top_share
        (Float64: its share of the group's rows with a value), n_values
        (Int64: the distinct values) and eff_n (Float64: exp of the
        Shannon entropy of the values' shares, Hill's number of order 1:
        as many equally frequent values have the same entropy; 1 for a
        single value, n_values when all are equally frequent). A group
        without a value has n_values 0 and the others null.
    """
    rows = pl.DataFrame({"group": group, "value": values}).drop_nulls("value")
    counts = (
        rows.group_by("group", "value")
        .len()
        .with_columns(pl.col("value").cast(pl.String))
        .with_columns(share=pl.col("len") / pl.col("len").sum().over("group"))
    )
    by_group = counts.group_by("group").agg(
        top=pl.col("value").sort_by(["len", "value"], descending=[True, False]).first(),
        top_share=pl.col("share").max(),
        n_values=pl.len().cast(pl.Int64),
        eff_n=(-(pl.col("share") * pl.col("share").log()).sum()).exp(),
    )
    every = pl.DataFrame({"group": np.arange(m, dtype=group.dtype)})
    return (
        every.join(by_group, on="group", how="left")
        .sort("group")
        .with_columns(pl.col("n_values").fill_null(0))
        .drop("group")
    )


def scenario_breaches(
    origin: pl.DataFrame,
    row_class: np.ndarray,
    classes: Sequence[str],
    scenario: "ScenarioConfig",
) -> dict[str, np.ndarray]:
    """Return, per key of the scenario, whether each CSV row breaks it.

    A row of the benign class breaks no rule. A row of another class
    breaks dst_port when its destination port is one of the background
    ports, src_ip when its source is none of its class's hosts, time
    when its time lies in none of its class's windows (each window
    closed). A key the scenario does not rule on for the row's class,
    or a value unknown (null), leaves the row unjudged on it.

    Args:
        origin: The provenance of the CSV rows (paths.read_provenance),
            one row per CSV row, with the columns of the keys it names.
        row_class: The class code of every CSV row, shape (n,).
        classes: The class names, by code.
        scenario: The dataset's scenario.

    Returns:
        By key of SCENARIO_KEYS the scenario rules on and origin holds,
        in that order, per CSV row: 1.0 when it breaks the rule, 0.0
        when it fits it, NaN when unjudged; shape (n,) float64.

    Raises:
        ValueError: When the scenario names a class that is not one of
            classes.
    """
    named = [scenario.benign, *scenario.attacks]
    unknown = [name for name in named if name not in classes]
    if unknown:
        raise ValueError(f"the scenario names classes the data lacks: {unknown}")
    code = {name: k for k, name in enumerate(classes)}
    row_class = np.asarray(row_class)
    found = {}
    if scenario.background_ports and "dst_port" in origin.columns:
        port = origin["dst_port"].cast(pl.String)
        ports = [str(p) for p in scenario.background_ports]
        breaks = port.is_in(ports).cast(pl.Float64).fill_null(np.nan).to_numpy()
        found["dst_port"] = np.where(row_class != code[scenario.benign], breaks, np.nan)
    for key in ("src_ip", TIME):
        declared = any(getattr(a, key) for a in scenario.attacks.values())
        if not declared or key not in origin.columns:
            continue
        breaks = np.full(origin.height, np.nan)
        for name, attack in scenario.attacks.items():
            rule = getattr(attack, key)
            if not rule:
                continue
            own = row_class == code[name]
            if key == TIME:
                t = origin[TIME].to_numpy()[own]
                inside = np.zeros(len(t), dtype=bool)
                for start, end in rule:
                    start, end = np.datetime64(start, "us"), np.datetime64(end, "us")
                    inside |= (t >= start) & (t <= end)
                breaks[own] = np.where(np.isnat(t), np.nan, ~inside)
            else:
                hosts = origin[key].cast(pl.String).to_numpy()[own]
                unknown = np.array([h is None for h in hosts], dtype=bool)
                breaks[own] = np.where(unknown, np.nan, ~np.isin(hosts, list(rule)))
        found[key] = breaks
    return found


def any_breach(breaches: Mapping[str, np.ndarray]) -> np.ndarray:
    """Return, per CSV row, whether it breaks any key of the scenario.

    Args:
        breaches: By key, per CSV row, 1.0, 0.0 or NaN
            (scenario_breaches); at least one key.

    Returns:
        Per CSV row, shape (n,): 1.0 when it breaks a key, 0.0 when it
        is judged on a key and breaks none, NaN when no key judges it.
    """
    return np.fmax.reduce(np.vstack(list(breaches.values())), axis=0)


def breach_shares(
    group: np.ndarray, breach: np.ndarray, m: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return, per group of CSV rows, the share of its judged rows that break a rule.

    Args:
        group: The group of every CSV row (its cell), in [0, m).
        breach: Per CSV row, 1.0, 0.0 or NaN (scenario_breaches).
        m: The number of groups.

    Returns:
        Per group, shape (m,): the share of its judged rows that break
        the rule (NaN for a group with none judged) and the judged rows.
    """
    known = ~np.isnan(breach)
    n = np.bincount(group[known], minlength=m)
    broken = np.bincount(group[known], weights=breach[known], minlength=m)
    share = np.full(m, np.nan)
    np.divide(broken, n, out=share, where=n > 0)
    return share, n

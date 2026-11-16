"""Provenance: where each row of the CSV comes from, and how concentrated it is per cell.

A dataset may name the columns of its rows' source and destination
(address and port) and time (ProvenanceConfig): never features, they
say whether a group of rows is many hosts at many times or one host in
a few minutes. Stage clean reads them for every row of the CSV, copies
included (the copies of a row may come from different hosts), and stage
quality_report measures them per cell.
"""

from __future__ import annotations

__all__ = ["KEYS", "TIME", "concentration", "read_provenance", "time_window"]

from collections.abc import Mapping

import numpy as np
import polars as pl

KEYS = ("src_ip", "src_port", "dst_ip", "dst_port")  # measured as values, in this order
TIME = "time"  # measured as a window


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


def time_window(
    group: np.ndarray, times: pl.Series, m: int, tau: float
) -> pl.DataFrame:
    """Return, per group of rows, the window of time that holds the central 1 - tau of them.

    With the n times of a group sorted, the window leaves out the k
    first and the k last, k = floor((n - 1) * tau / 2): from the tau/2
    quantile (the lower of the two times around it) to the 1 - tau/2
    quantile (the higher). At most tau of the group's rows lie outside
    it, so a few stray rows (copies read on another day, say) do not
    widen it.

    Args:
        group: The group of every row (its cell), in [0, m).
        times: The time of every row, Datetime; a null is unknown and
            left out.
        m: The number of groups.
        tau: The share of a group's rows that may lie outside its
            window, quality.tau.

    Returns:
        One row per group, 0 to m - 1 in order: time_from and time_to
        (Datetime, as times) and time_window (Float64: the seconds from
        time_from to time_to); null for a group without a time.
    """
    known = times.is_not_null().to_numpy()
    at = times.dt.epoch("us").to_numpy()[known]
    own = np.asarray(group)[known]
    order = np.lexsort((at, own))
    at, own = at[order], own[order]
    n = np.bincount(own, minlength=m)
    start = np.concatenate([[0], np.cumsum(n)[:-1]])
    # floor((n - 1) * tau / 2), with a margin for rounding: 20 * 0.05 may land under 1
    k = np.floor((np.maximum(n, 1) - 1) * tau / 2 + 1e-9).astype(np.int64)
    has = n > 0
    first = np.where(has, start + k, 0)
    last = np.where(has, start + n - 1 - k, 0)
    lo = pl.Series(np.where(has, at[first] if len(at) else 0, 0))
    hi = pl.Series(np.where(has, at[last] if len(at) else 0, 0))
    window = pl.DataFrame({"lo": lo, "hi": hi, "has": has})
    time = times.dtype
    return window.select(
        time_from=pl.when("has").then(pl.col("lo").cast(pl.Datetime("us"))).cast(time),
        time_to=pl.when("has").then(pl.col("hi").cast(pl.Datetime("us"))).cast(time),
        time_window=pl.when("has").then((pl.col("hi") - pl.col("lo")) / 1e6),
    )

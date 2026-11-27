"""Provenance: where each row of the CSV comes from, as support of the verdicts.

A dataset may name the columns of its rows' source and destination
(address and port) and time (ProvenanceConfig). Stage clean reads them
for every row of the CSV, copies included (the copies of a row may come
from different hosts). Stage quality_report measures them between each
distinct row read and its nearest rival row (key_coincidence,
time_nearness), with its nearest row of its own class as the control,
the chance read on the rows read of each class, and describes the hosts
of each region over all its rows (concentration). They never decide a
verdict: they support one.
"""

from __future__ import annotations

__all__ = [
    "KEYS",
    "TIME",
    "concentration",
    "key_coincidence",
    "read_provenance",
    "time_nearness",
]

from collections.abc import Mapping

import numpy as np
import polars as pl

KEYS = ("src_ip", "src_port", "dst_ip", "dst_port")  # measured as values, in this order
TIME = "time"  # the time of the row


def read_provenance(
    columns: Mapping[str, str],
    time_format: str | None,
    afternoon_before: int | None = None,
) -> dict[str, pl.Expr]:
    """Return the expressions that read the provenance of rows read as text.

    Args:
        columns: The configured keys of KEYS and TIME, each with the
            column of the CSV it is read from.
        time_format: The format of the TIME column's text (chrono's
            strftime, as polars reads it, e.g. "%Y-%m-%d %H:%M:%S%.f");
            None without TIME.
        afternoon_before: For a clock of 12 hours written without AM or
            PM, the first hour of the morning the capture holds: a time
            whose hour is under it is of the afternoon, and gets 12
            hours. None reads the hours as written.

    Returns:
        By key, in the order of columns, the expression of its value: a
        key of KEYS as Categorical text, null when empty; TIME as
        Datetime("us"), null when empty or not a time in time_format.
    """
    exprs = {}
    for key, column in columns.items():
        if key == TIME:
            time = pl.col(column).str.to_datetime(
                time_format, time_unit="us", strict=False
            )
            if afternoon_before is not None:
                late = time.dt.hour() < afternoon_before
                time = pl.when(late).then(time + pl.duration(hours=12)).otherwise(time)
            exprs[key] = time
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
        (Float64: its share of the group's rows with a value) and
        n_values (Int64: the distinct values). A group without a value
        has n_values 0 and the others null.
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
    )
    every = pl.DataFrame({"group": np.arange(m, dtype=group.dtype)})
    return (
        every.join(by_group, on="group", how="left")
        .sort("group")
        .with_columns(pl.col("n_values").fill_null(0))
        .drop("group")
    )


def key_coincidence(
    row: np.ndarray,
    values: pl.Series,
    labels: np.ndarray,
    rival_row: np.ndarray,
    own_row: np.ndarray,
    measured: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return, per distinct row, how often it shares a key's value with its nearest rival and own rows.

    For a distinct row i and a neighbour u, the chance that a copy of
    each, drawn at random, have the same value: the sum over the values
    of p_i · p_u, p the share of a row's CSV rows with the value (nulls
    left out), 1 or 0 for rows of one copy. Chance: the same between a
    row drawn at random from i's class and one from its rival's class,
    the sum of q_a · q_k with q the mean of p over the measured distinct
    rows of a class (with a known value). The own row is the control:
    neighbours in the features share values for reasons that are no
    labels' (the traffic of a day, a capture's artifacts, a key that is
    also a feature).

    Args:
        row: The row of the data file every CSV row is a copy of, shape
            (n_csv,).
        values: The key's value of every CSV row, text or Categorical,
            null when unknown.
        labels: The class code of every row of the data file, shape (n,).
        rival_row: The nearest row of another class of every row
            (quality.nearest_rivals), -1 for none.
        own_row: Its nearest other row of its class, -1 for none.
        measured: Per row of the data file, whether it is read (the
            rows the chance is the mean over), shape (n,) bool.

    Returns:
        Per row of the data file, shape (n,) float: the coincidence with
        its rival row, with its own row, and the chance between its
        class and its rival's; NaN where the row or either neighbour has
        no known value.
    """
    labels = np.asarray(labels)
    n = len(labels)
    known = (
        pl.DataFrame({"row": np.asarray(row, dtype=np.int64), "value": values})
        .with_columns(pl.col("value").cast(pl.String))
        .drop_nulls("value")
    )
    p = (
        known.group_by("row", "value")
        .len()
        .with_columns(p=pl.col("len") / pl.col("len").sum().over("row"))
        .select("row", "value", "p")
    )
    has = np.zeros(n, dtype=bool)
    has[p["row"].unique().to_numpy()] = True

    def coincidence(neighbour: np.ndarray) -> np.ndarray:
        pairs = pl.DataFrame({"row": np.arange(n), "other": neighbour}).filter(
            pl.col("other") >= 0
        )
        other = p.rename({"row": "other", "p": "q"})
        shared = (
            pairs.join(p, on="row")
            .join(other, on=["other", "value"])
            .group_by("row")
            .agg((pl.col("p") * pl.col("q")).sum().alias("sum"))
        )
        out = np.zeros(n)
        out[shared["row"].to_numpy()] = shared["sum"].to_numpy()
        return out

    rival_row, own_row = np.asarray(rival_row), np.asarray(own_row)
    every = (
        has
        & (rival_row >= 0)
        & (own_row >= 0)
        & has[np.maximum(rival_row, 0)]
        & has[np.maximum(own_row, 0)]
    )
    measured = np.asarray(measured, dtype=bool)
    read = p.filter(pl.Series(measured)[p["row"].to_numpy()])
    of_class = (
        read.with_columns(cls=pl.Series(labels)[read["row"].to_numpy()].alias("cls"))
        .group_by("cls", "value")
        .agg(pl.col("p").sum())
    )
    rows_of = np.bincount(labels[has & measured], minlength=int(labels.max()) + 1)
    q = of_class.with_columns(
        q=pl.col("p") / pl.Series(rows_of)[of_class["cls"].to_numpy()]
    ).select("cls", "value", "q")
    chance_of = (
        q.join(q.rename({"cls": "rival", "q": "q_rival"}), on="value")
        .group_by("cls", "rival")
        .agg((pl.col("q") * pl.col("q_rival")).sum().alias("chance"))
    )
    table = np.zeros((len(rows_of), len(rows_of)))
    table[chance_of["cls"].to_numpy(), chance_of["rival"].to_numpy()] = chance_of[
        "chance"
    ].to_numpy()
    same, own = coincidence(rival_row), coincidence(own_row)
    chance = table[labels, labels[np.maximum(rival_row, 0)]]
    return tuple(np.where(every, x, np.nan) for x in (same, own, chance))


def time_nearness(
    row: np.ndarray,
    times: np.ndarray,
    labels: np.ndarray,
    neighbour: np.ndarray,
    own: bool,
    measured: np.ndarray,
) -> np.ndarray:
    """Return, per distinct row, how near in time it lies to a neighbour, against chance.

    The copies of a row count as different times, and one is enough.
    For a row i and its neighbour u: g, the least distance between a
    copy of i and a copy of u; s, the share of the CSV rows of the
    measured rows of u's class (with a known time) inside the union of
    [t - g, t + g] around
    the copies of i, the bounds closed (a tie counts as near); m, the
    copies of u with a known time. The measure is 1 - (1 - s)^m: the
    chance that m copies drawn at random from u's class put one as
    near. With no tie to the time it is uniform, of mean one half; near
    0, the row and its neighbour are nearer in time than chance. For
    the own row (own True) the copies of i leave the count and the
    total.

    Args:
        row: The row of the data file every CSV row is a copy of, shape
            (n_csv,).
        times: The time of every CSV row, datetime64[us], NaT when
            unknown.
        labels: The class code of every row of the data file, shape (n,).
        neighbour: The neighbour of every row of the data file (its
            nearest rival or own row), -1 for none.
        own: Whether the neighbour is of the row's own class.
        measured: Per row of the data file, whether it is read (the
            rows whose CSV rows s is a share of), shape (n,) bool; a row
            with a neighbour must be read.

    Returns:
        Per row of the data file, shape (n,) float in [0, 1]; NaN where
        the row or its neighbour has no known time, or it has no
        neighbour.
    """
    row, labels = np.asarray(row, dtype=np.int64), np.asarray(labels)
    neighbour = np.asarray(neighbour, dtype=np.int64)
    n = len(labels)
    stamp = np.asarray(times, dtype="datetime64[us]")
    known = ~np.isnat(stamp)
    at, t = row[known], stamp[known].astype(np.int64)
    order = np.lexsort((t, at))
    at, t = at[order], t[order]
    start = np.searchsorted(at, np.arange(n), side="left")
    stop = np.searchsorted(at, np.arange(n), side="right")
    copies = stop - start
    read = np.asarray(measured, dtype=bool)[at]
    of_class = {
        k: np.sort(t[read & (labels[at] == k)]) for k in np.unique(labels[at])
    }  # the known times of the CSV rows of each class's rows read
    out = np.full(n, np.nan)
    judged = np.flatnonzero((neighbour >= 0) & (copies > 0))
    judged = judged[copies[neighbour[judged]] > 0]
    single = judged[(copies[judged] == 1) & (copies[neighbour[judged]] == 1)]
    for k in np.unique(labels[neighbour[single]]):  # one copy on each side: vectorized
        rows = single[labels[neighbour[single]] == k]
        mine, theirs = t[start[rows]], t[start[neighbour[rows]]]
        gap = np.abs(mine - theirs)
        times_k = of_class[k]
        inside = np.searchsorted(times_k, mine + gap, side="right") - np.searchsorted(
            times_k, mine - gap, side="left"
        )
        total = np.full(len(rows), len(times_k), dtype=np.float64)
        if own:  # the row itself leaves the count and the total
            inside, total = inside - 1, total - 1
        with np.errstate(divide="ignore", invalid="ignore"):
            out[rows] = np.where(total > 0, inside / total, np.nan)
    for i in np.setdiff1d(judged, single):  # copies on some side: one at a time
        u = neighbour[i]
        mine, theirs = t[start[i] : stop[i]], t[start[u] : stop[u]]
        near = np.clip(np.searchsorted(theirs, mine), 1, len(theirs)) - 1
        gap = min(
            np.abs(mine - theirs[near]).min(),
            np.abs(mine - theirs[np.minimum(near + 1, len(theirs) - 1)]).min(),
        )
        times_k = of_class[labels[u]]
        inside, end = 0, None
        for low, high in zip(
            mine - gap, mine + gap
        ):  # mine is sorted: merge as it goes
            if end is not None and low <= end:
                low = end + 1
            if low <= high:
                inside += np.searchsorted(
                    times_k, high, side="right"
                ) - np.searchsorted(times_k, low, side="left")
            end = high if end is None else max(end, high)
        total = len(times_k)
        if own:
            inside, total = inside - len(mine), total - len(mine)
        if total > 0:
            s = inside / total
            out[i] = 1 - (1 - s) ** len(theirs)
    return out

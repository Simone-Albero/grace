"""Stage quality_report: the data-quality report of the regions and their pairs."""

from __future__ import annotations

__all__ = ["run"]

import json
import math
import sqlite3
from dataclasses import dataclass

import numpy as np
import polars as pl

from src import db
from src.clustering import centroids, line_spread, separation
from src.config import DatasetConfig
from src.data import DUPLICATES, LABEL
from src.quality import (
    Pair,
    candidate_pairs,
    diversity,
    feature_separation,
    label_conflicts,
    linear_error,
    neighbour_overlap,
    pair_conflicts,
    region_samples,
)
from stages.paths import (
    assignments_file,
    need,
    prepared_columns,
    read_classes,
    read_data,
    read_points,
    read_weights,
    regions_file,
)

MEASURES = ("f3", "f3_feature", "n_separating", "kdn", "enn", "n1", "l2")
SPACES = ("", "_ns")  # the suffix of the full space and of the one without shortcuts
SCOPES = {"": "all", "_ns": "ns"}


@dataclass(frozen=True)
class _Space:
    """What the pairs need of one space: no point but the sampled ones.

    Attributes:
        separation: clustering.separation of every pair of regions.
        rows: By region, its sampled rows' points.
        column_of: The column (into names) of each column of the points.
        names: The configured columns behind the points.
    """

    separation: np.ndarray
    rows: dict[int, np.ndarray]
    column_of: np.ndarray
    names: list[str]


def run(cfg: DatasetConfig, conn: sqlite3.Connection, run_id: int) -> None:
    """Report the quality of the regions and of their pairs.

    Reads the data file (the features as read, DUPLICATES), the points
    (paths.read_points, as stage regions; again without shortcut_cols
    when any, the space "ns"), regions.parquet and assignments.parquet,
    which must be of the same rows ("run regions again" otherwise).
    Order: quality.region_samples (max_rows_per_region, seed) ->
    quality.diversity of every region on the points (diversity_pairs,
    seed) -> per space: the weighted centroids of the regions ->
    clustering.line_spread -> clustering.separation. Then
    quality.label_conflicts on every feature (and on every feature but
    shortcut_cols) -> quality.pair_conflicts -> quality.candidate_pairs
    under quality.candidate_separation -> per pair and space, on the
    sampled rows: quality.feature_separation, quality.neighbour_overlap
    (k) and quality.linear_error (null when a side has fewer than 2
    sampled rows).

    A pair is a shortcut when its kdn without shortcut_cols reaches
    highlight.kdn while its kdn does not, or when it shares more
    conflicting rows without them. The ns columns and shortcut are null
    without shortcut_cols.

    Logs: quality_pairs (region_a, class_a, region_b, class_b, reasons
    joined by ",", sep, sep_ns, n_conflicts, n_conflicts_read,
    n_conflicts_ns, n_conflicts_ns_read, shortcut, then f3, f3_feature,
    n_separating, kdn, enn, n1, l2 and the same with _ns),
    quality_regions (region, class, n_rows, n_read, dup_share = 1 -
    n_rows / n_read, conflict_rows, conflict_read, pdd, tc, rival: the
    pair's other region of highest enn, and that pair's enn, kdn, f3,
    l2, shortcut: whether any of its pairs is one), quality_classes
    (class, n_rows, n_read, dup_share, conflict_share: conflicting CSV
    rows over n_read, enn: of its regions' rival pairs, weighted by
    n_rows, f3 and f3_feature: the most over its pairs that at most one
    feature separates for every row (null without), shortcut_share:
    rows of regions with a shortcut pair over n_rows, pdd weighted by
    n_rows, tc: the most over its regions, n_pairs),
    label_conflict_groups (scope "all" | "ns", group, row, class,
    region, n_read), quality_stats (one row: shortcut_cols and highlight
    as JSON, candidate_separation, k, max_rows_per_region,
    diversity_pairs, n_pairs, n_near, n_conflict, n_nearest, n_groups,
    n_groups_ns). n_rows counts distinct rows, n_read rows of the CSV.
    """
    quality = cfg.method.quality
    classes = read_classes(cfg)
    regions = pl.read_parquet(need(regions_file(cfg), "regions")).sort("region")
    assignments = pl.read_parquet(need(assignments_file(cfg), "regions"))
    region = assignments.sort("row")["region"].to_numpy()
    region_class = regions["class"].to_numpy()
    weights = read_weights(cfg)
    points, labels = read_points(cfg)
    if len(region) != len(labels) or not np.array_equal(region_class[region], labels):
        raise ValueError("the regions were built on other data: run regions again")
    samples = region_samples(region, quality.max_rows_per_region, cfg.seed)
    print(f"[quality_report] diversity of {len(region_class)} regions", flush=True)
    pdd, tc = diversity(points, region, quality.diversity_pairs, cfg.seed)

    data = read_data(cfg)
    features = data.drop(DUPLICATES, LABEL)
    shortcuts = [c for c in cfg.shortcut_cols if c in features.columns]
    spaces = {"": _space(points, region, weights, samples, prepared_columns(cfg))}
    del points
    groups = {"": label_conflicts(features, labels)}
    if cfg.shortcut_cols:
        drop = cfg.shortcut_cols
        points, _ = read_points(cfg, drop=drop)
        names = prepared_columns(cfg, drop=drop)
        spaces["_ns"] = _space(points, region, weights, samples, names)
        del points
        groups["_ns"] = label_conflicts(features.drop(shortcuts), labels)
    conflicts = {
        s: pair_conflicts(g, labels, region, weights) for s, g in groups.items()
    }
    pairs = candidate_pairs(
        [space.separation for space in spaces.values()],
        set().union(*conflicts.values()),
        region_class,
        quality.candidate_separation,
    )
    print(f"[quality_report] measuring {len(pairs)} pairs of regions", flush=True)
    pair_rows = []
    for i, pair in enumerate(pairs):
        pair_rows.append(_pair_row(pair, spaces, conflicts, region_class, classes, cfg))
        if (i + 1) % 100 == 0:
            print(f"[quality_report] {i + 1}/{len(pairs)} pairs", flush=True)

    sizes = regions["n_rows"].to_numpy()
    reads = regions["n_read"].to_numpy()
    region_rows = _region_rows(
        pair_rows, groups[""], region, region_class, classes, sizes, reads, weights
    )
    for row, own_pdd, own_tc in zip(region_rows, pdd, tc):
        row["pdd"], row["tc"] = float(own_pdd), float(own_tc)
    db.log_table(conn, run_id, "quality_pairs", pair_rows)
    db.log_table(conn, run_id, "quality_regions", region_rows)
    db.log_table(
        conn, run_id, "quality_classes", _class_rows(region_rows, pair_rows, classes)
    )
    db.log_table(
        conn,
        run_id,
        "label_conflict_groups",
        [
            {
                "scope": SCOPES[suffix],
                "group": int(g[row]),
                "row": int(row),
                "class": classes[labels[row]],
                "region": int(region[row]),
                "n_read": float(weights[row]),
            }
            for suffix, g in groups.items()
            for row in np.flatnonzero(g >= 0)
        ],
    )
    db.log_table(
        conn,
        run_id,
        "quality_stats",
        [
            {
                "shortcut_cols": json.dumps(list(cfg.shortcut_cols)),
                "highlight": json.dumps(quality.highlight),
                "candidate_separation": quality.candidate_separation,
                "k": quality.k,
                "max_rows_per_region": quality.max_rows_per_region,
                "diversity_pairs": quality.diversity_pairs,
                "n_pairs": len(pairs),
                **{
                    f"n_{reason}": sum(reason in p.reasons for p in pairs)
                    for reason in ("near", "conflict", "nearest")
                },
                **{
                    f"n_groups{suffix}": (
                        int(groups[suffix].max()) + 1 if suffix in groups else None
                    )
                    for suffix in SPACES
                },
            }
        ],
    )
    print(
        f"[quality_report] {len(pairs)} pairs, "
        f"{int(groups[''].max()) + 1} label conflicts",
        flush=True,
    )


def _space(
    points: np.ndarray,
    region: np.ndarray,
    weights: np.ndarray,
    samples: dict[int, np.ndarray],
    names: list[str],
) -> _Space:
    """Return the separation of the regions and their sampled rows."""
    _, centers = centroids(points, region, weights)
    apart = separation(centers, line_spread(points, region, weights, centers))
    unique = list(dict.fromkeys(names))
    return _Space(
        separation=apart,
        rows={own: points[rows] for own, rows in samples.items()},
        column_of=np.array([unique.index(name) for name in names], dtype=np.int64),
        names=unique,
    )


def _pair_row(
    pair: Pair,
    spaces: dict[str, _Space],
    conflicts: dict[str, dict[tuple[int, int], tuple[int, float]]],
    region_class: np.ndarray,
    classes: list[str],
    cfg: DatasetConfig,
) -> dict[str, object]:
    """Return the quality_pairs row of one pair."""
    a, b = pair.a, pair.b
    row: dict[str, object] = {
        "region_a": a,
        "class_a": classes[region_class[a]],
        "region_b": b,
        "class_b": classes[region_class[b]],
        "reasons": ",".join(pair.reasons),
    }
    for suffix in SPACES:
        space = spaces.get(suffix)
        row[f"sep{suffix}"] = None if space is None else float(space.separation[a, b])
    for suffix in SPACES:
        n, read = (None, None)
        if suffix in conflicts:
            n, read = conflicts[suffix].get((a, b), (0, 0.0))
        row[f"n_conflicts{suffix}"], row[f"n_conflicts{suffix}_read"] = n, read
    row["shortcut"] = None
    for suffix in SPACES:
        space = spaces.get(suffix)
        values = (
            dict.fromkeys(MEASURES)
            if space is None
            else _pair_measures(
                space.rows[a],
                space.rows[b],
                space.column_of,
                space.names,
                cfg.method.quality.k,
            )
        )
        row.update({f"{key}{suffix}": value for key, value in values.items()})
    if "_ns" in spaces:
        highlight = cfg.method.quality.highlight["kdn"]
        row["shortcut"] = bool(
            (row["kdn_ns"] >= highlight > row["kdn"])
            or row["n_conflicts_ns"] > row["n_conflicts"]
        )
    return row


def _pair_measures(
    Xa: np.ndarray, Xb: np.ndarray, column_of: np.ndarray, names: list[str], k: int
) -> dict[str, float | int | str | None]:
    """Return MEASURES of one pair in one space."""
    separated = feature_separation(Xa, Xb, column_of)
    overlap = neighbour_overlap(Xa, Xb, k)
    return {
        "f3": separated.share,
        "f3_feature": names[separated.column] if separated.column >= 0 else None,
        "n_separating": separated.n_separating,
        "kdn": overlap.kdn,
        "enn": overlap.enn,
        "n1": overlap.n1,
        "l2": linear_error(Xa, Xb) if min(len(Xa), len(Xb)) >= 2 else None,
    }


def _region_rows(
    pair_rows: list[dict[str, object]],
    groups: np.ndarray,
    region: np.ndarray,
    region_class: np.ndarray,
    classes: list[str],
    sizes: np.ndarray,
    reads: np.ndarray,
    weights: np.ndarray,
) -> list[dict[str, object]]:
    """Return the quality_regions rows but pdd and tc, by region id."""
    m = len(region_class)
    conflicting = groups >= 0
    conflict_rows = np.bincount(region[conflicting], minlength=m)
    conflict_read = np.bincount(
        region[conflicting], weights=weights[conflicting], minlength=m
    )
    worst: dict[int, dict[str, object]] = {}
    shortcut = np.zeros(m, dtype=bool)
    for p in pair_rows:
        for own, other in (("region_a", "region_b"), ("region_b", "region_a")):
            r = p[own]
            if r not in worst or p["enn"] > worst[r]["enn"]:
                worst[r] = {**p, "rival": p[other]}
            shortcut[r] |= bool(p["shortcut"])
    rows = []
    for r in range(m):
        rival = worst.get(r, {})
        rows.append(
            {
                "region": r,
                "class": classes[region_class[r]],
                "n_rows": int(sizes[r]),
                "n_read": float(reads[r]),
                "dup_share": float(1 - sizes[r] / reads[r]),
                "conflict_rows": int(conflict_rows[r]),
                "conflict_read": float(conflict_read[r]),
                "pdd": None,
                "tc": None,
                **{key: rival.get(key) for key in ("rival", "enn", "kdn", "f3", "l2")},
                "shortcut": bool(shortcut[r]),
            }
        )
    return rows


def _class_rows(
    region_rows: list[dict[str, object]],
    pair_rows: list[dict[str, object]],
    classes: list[str],
) -> list[dict[str, object]]:
    """Return the quality_classes rows, the roll-up of the regions."""
    rows = []
    for name in classes:
        own = [r for r in region_rows if r["class"] == name]
        if not own:
            continue
        n_rows = sum(r["n_rows"] for r in own)
        n_read = sum(r["n_read"] for r in own)
        pairs = [p for p in pair_rows if name in (p["class_a"], p["class_b"])]
        single = [p for p in pairs if p["n_separating"] <= 1]
        best = max(single, key=lambda p: p["f3"], default=None)
        rows.append(
            {
                "class": name,
                "n_rows": n_rows,
                "n_read": n_read,
                "dup_share": 1 - n_rows / n_read,
                "conflict_share": sum(r["conflict_read"] for r in own) / n_read,
                "enn": _weighted(own, "enn"),
                "f3": None if best is None else best["f3"],
                "f3_feature": None if best is None else best["f3_feature"],
                "shortcut_share": sum(r["n_rows"] for r in own if r["shortcut"])
                / n_rows,
                "pdd": _weighted(own, "pdd"),
                "tc": max(r["tc"] for r in own),
                "n_pairs": len(pairs),
            }
        )
    return rows


def _weighted(rows: list[dict[str, object]], key: str) -> float | None:
    """Return the mean of key over rows, weighted by n_rows; None if none."""
    known = [r for r in rows if r[key] is not None]
    if not known:
        return None
    total = sum(r["n_rows"] for r in known)
    return _number(sum(r[key] * r["n_rows"] for r in known) / total)


def _number(value: float | None) -> float | None:
    """Return value as a float, None when undefined or not finite."""
    return None if value is None or not math.isfinite(value) else float(value)

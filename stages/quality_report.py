"""Stage quality_report: the data-quality report of the regions, their pairs and the classes."""

from __future__ import annotations

__all__ = ["run"]

import json
import math
import sqlite3
from collections import Counter

import numpy as np
import polars as pl

from src import db
from src.clustering import centroids, line_spread, separation
from src.config import DatasetConfig, QualityConfig
from src.data import DUPLICATES, KINDS, LABEL
from src.quality import (
    ISSUES,
    REASONS,
    VERDICTS,
    Pair,
    candidate_pairs,
    cover_hart_low,
    diversity,
    feature_dependence,
    invalid_values,
    label_conflicts,
    pair_conflicts,
    region_issues,
    region_samples,
    single_column,
    verdict,
)
from stages.paths import (
    assignments_file,
    need,
    prepared_columns,
    read_classes,
    read_data,
    read_points,
    read_unparsable,
    read_weights,
    regions_file,
)

SOLID = "solid"  # the status of a region without problems; "flagged" otherwise
CLASS_ISSUES = ("low_diversity", "collapse")


def run(cfg: DatasetConfig, conn: sqlite3.Connection, run_id: int) -> None:
    """Report the quality of the regions, of their pairs and of the classes.

    Reads the data file (the features as read, DUPLICATES), the cells
    stage clean could not read as numbers (paths.read_unparsable), the
    points (paths.read_points, as stage regions), regions.parquet and
    assignments.parquet, which must be of the same rows ("run regions
    again" otherwise). With q = method.quality, tau =
    q.thresholds["nn_error"]. Order: quality.diversity of every class on
    the points (q.diversity_pairs, seed) -> quality.region_samples
    (q.max_rows_per_region, seed) -> the weighted centroids of the
    regions -> clustering.line_spread -> clustering.separation ->
    quality.invalid_values -> quality.label_conflicts on every feature
    -> quality.pair_conflicts -> quality.candidate_pairs under
    q.candidate_separation -> per pair, on the sampled rows:
    quality.single_column (q.thresholds["single_column_acc"]),
    quality.feature_dependence (q.max_group, seed (seed, a, b)) and
    quality.verdict (tau; null when a side has fewer than 4 sampled
    rows) -> per region: its rival, its pairs' verdicts,
    quality.region_issues (q.thresholds["dup_share"]) -> per class:
    the roll-up, and low_diversity (pdd >= q.thresholds["pdd"]) and
    collapse (tc >= q.thresholds["tc"]).

    Logs: quality_pairs (region_a, class_a, region_b, class_b, reasons
    joined by ",", sep, n_conflicts, n_conflicts_read; nn_error_a,
    nn_error_b, nn_error (their mean), bayes_error_low
    (quality.cover_hart_low of nn_error); shortcut_columns (the columns
    removed, joined by ",": the group's first columns the shortcut
    depends on, the whole group otherwise), nn_error_without_a,
    nn_error_without_b, nn_error_without (the errors without them),
    nn_error_rise = nn_error_without - nn_error, verdict;
    single_column_acc_a, single_column_acc_b, single_column_acc (their
    mean), single_column, n_single_columns), quality_regions (region, class,
    n_rows, n_read, n_duplicates = n_read - n_rows, dup_share = 1 -
    n_rows / n_read, conflict_rows, conflict_read, missing_rows and
    missing_read (rows with a null or invalid value), then per kind of
    data.KINDS <kind>_rows; rival: the region of the pair where this
    region's side errs the most (ties to the lower id), with nn_error,
    bayes_error_low, single_column_acc, single_column of its own side
    and the pair's verdict, shortcut_columns (when the verdict is
    "shortcut"); pairs_<verdict>: its pairs by verdict; issues joined
    by ",", status "solid" or "flagged"), quality_classes (class,
    n_rows, n_read, n_duplicates, dup_share, conflict_share and
    missing_share over n_read, nn_error: its regions' weighted by
    n_rows, single_column_acc and single_column: the most over its
    pairs that at most one column separates (null without), pdd, tc, issues of
    low_diversity and collapse, solid_share and share_<issue> of its
    CSV rows, pairs_<verdict>, n_pairs), region_missing (region,
    column, kind, n_rows, n_read: the null and invalid cells; no row
    without one), label_conflict_groups (group, row, class, region,
    n_read), quality_stats (one row: thresholds as JSON,
    candidate_separation, max_rows_per_region, diversity_pairs,
    max_group, n_pairs, n_near, n_label_conflict, n_nearest, n_groups,
    n_regions, n_solid, solid_share and share_<issue> of the CSV rows,
    pairs_<verdict>). n_rows counts distinct rows, n_read rows of the
    CSV.
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
    names = prepared_columns(cfg)
    columns = list(dict.fromkeys(names))
    column_of = np.array([columns.index(name) for name in names], dtype=np.int64)
    numeric = np.isin(columns, cfg.num_cols)
    print(f"[quality_report] diversity of {len(classes)} classes", flush=True)
    pdd, tc = diversity(points, labels, quality.diversity_pairs, cfg.seed)
    samples = region_samples(region, quality.max_rows_per_region, cfg.seed)
    rows_of = {own: points[rows] for own, rows in samples.items()}
    _, centers = centroids(points, region, weights)
    apart = separation(centers, line_spread(points, region, weights, centers))
    del points

    data = read_data(cfg)
    features = data.drop(DUPLICATES, LABEL)
    invalid = invalid_values(features, read_unparsable(cfg))
    groups = label_conflicts(features, labels)
    conflicts = pair_conflicts(groups, labels, region, weights)
    pairs = candidate_pairs(
        apart, set(conflicts), region_class, quality.candidate_separation
    )
    print(f"[quality_report] measuring {len(pairs)} pairs of regions", flush=True)
    pair_rows = []
    for i, pair in enumerate(pairs):
        n, read = conflicts.get((pair.a, pair.b), (0, 0.0))
        pair_rows.append(
            {
                "region_a": pair.a,
                "class_a": classes[region_class[pair.a]],
                "region_b": pair.b,
                "class_b": classes[region_class[pair.b]],
                "reasons": ",".join(pair.reasons),
                "sep": float(apart[pair.a, pair.b]),
                "n_conflicts": n,
                "n_conflicts_read": read,
                **_pair_measures(
                    pair, rows_of, column_of, numeric, columns, n, quality, cfg.seed
                ),
            }
        )
        if (i + 1) % 100 == 0:
            print(f"[quality_report] {i + 1}/{len(pairs)} pairs", flush=True)

    missing_cells = _missing_rows(invalid, region, weights)
    region_rows = _region_rows(
        pair_rows,
        groups,
        invalid,
        region,
        region_class,
        classes,
        regions["n_rows"].to_numpy(),
        regions["n_read"].to_numpy(),
        weights,
        quality.thresholds["dup_share"],
    )
    class_rows = _class_rows(region_rows, pair_rows, classes, pdd, tc, quality)
    n_read = sum(r["n_read"] for r in region_rows)
    n_solid = sum(r["status"] == SOLID for r in region_rows)
    print(
        f"[quality_report] {len(pairs)} pairs, {int(groups.max()) + 1} label "
        f"conflicts, {n_solid}/{len(region_rows)} regions solid",
        flush=True,
    )
    db.log_table(conn, run_id, "quality_pairs", pair_rows)
    db.log_table(conn, run_id, "quality_regions", region_rows)
    db.log_table(conn, run_id, "quality_classes", class_rows)
    db.log_table(conn, run_id, "region_missing", missing_cells)
    db.log_table(
        conn,
        run_id,
        "label_conflict_groups",
        [
            {
                "group": int(groups[row]),
                "row": int(row),
                "class": classes[labels[row]],
                "region": int(region[row]),
                "n_read": float(weights[row]),
            }
            for row in np.flatnonzero(groups >= 0)
        ],
    )
    db.log_table(
        conn,
        run_id,
        "quality_stats",
        [
            {
                "thresholds": json.dumps(quality.thresholds),
                "candidate_separation": quality.candidate_separation,
                "max_rows_per_region": quality.max_rows_per_region,
                "diversity_pairs": quality.diversity_pairs,
                "max_group": quality.max_group,
                "n_pairs": len(pairs),
                **{
                    f"n_{reason}": sum(reason in p.reasons for p in pairs)
                    for reason in REASONS
                },
                "n_groups": int(groups.max()) + 1,
                "n_regions": len(region_rows),
                "n_solid": n_solid,
                **_coverage(region_rows, n_read),
                **_verdict_counts(pair_rows),
            }
        ],
    )


def _pair_measures(
    pair: Pair,
    rows_of: dict[int, np.ndarray],
    column_of: np.ndarray,
    numeric: np.ndarray,
    columns: list[str],
    n_conflicts: int,
    quality: QualityConfig,
    seed: int,
) -> dict[str, object]:
    """Return the measures of one pair on its regions' sampled rows."""
    Xa, Xb = rows_of[pair.a], rows_of[pair.b]
    tau = quality.thresholds["nn_error"]
    one = single_column(
        Xa, Xb, column_of, numeric, quality.thresholds["single_column_acc"]
    )
    dependence = feature_dependence(
        Xa, Xb, column_of, quality.max_group, (seed, pair.a, pair.b)
    )
    row: dict[str, object] = dict.fromkeys(
        ("nn_error_a", "nn_error_b", "nn_error", "bayes_error_low")
        + ("shortcut_columns", "nn_error_without_a", "nn_error_without_b")
        + ("nn_error_without", "nn_error_rise", "verdict")
    )
    if dependence is not None:
        errors = dependence.errors
        judged, k = verdict(errors, n_conflicts, one.share, tau)
        last = k if judged == "shortcut" else len(dependence.group)
        err, without = float(errors[0].mean()), float(errors[last].mean())
        row.update(
            nn_error_a=float(errors[0, 0]),
            nn_error_b=float(errors[0, 1]),
            nn_error=err,
            bayes_error_low=cover_hart_low(err),
            shortcut_columns=",".join(columns[c] for c in dependence.group[:last]),
            nn_error_without_a=float(errors[last, 0]),
            nn_error_without_b=float(errors[last, 1]),
            nn_error_without=without,
            nn_error_rise=without - err,
            verdict=judged,
        )
    return {
        **row,
        "single_column_acc_a": one.side_a,
        "single_column_acc_b": one.side_b,
        "single_column_acc": one.share,
        "single_column": columns[one.column] if one.column >= 0 else None,
        "n_single_columns": one.n_single_columns,
    }


def _missing_rows(
    invalid: pl.DataFrame, region: np.ndarray, weights: np.ndarray
) -> list[dict[str, object]]:
    """Return the region_missing rows: the null and invalid cells per region, column and kind."""
    cells = invalid.with_columns(
        region=pl.Series(region)[invalid["row"]],
        weight=pl.Series(weights)[invalid["row"]],
    )
    return (
        cells.group_by("region", "column", "kind")
        .agg(n_rows=pl.len(), n_read=pl.col("weight").sum())
        .sort("region", "column", "kind")
        .to_dicts()
    )


def _region_rows(
    pair_rows: list[dict[str, object]],
    groups: np.ndarray,
    invalid: pl.DataFrame,
    region: np.ndarray,
    region_class: np.ndarray,
    classes: list[str],
    sizes: np.ndarray,
    reads: np.ndarray,
    weights: np.ndarray,
    dup_threshold: float,
) -> list[dict[str, object]]:
    """Return the quality_regions rows, by region id."""
    m = len(region_class)
    conflicting = groups >= 0
    conflict_rows = np.bincount(region[conflicting], minlength=m)
    conflict_read = np.bincount(
        region[conflicting], weights=weights[conflicting], minlength=m
    )
    hit = np.unique(invalid["row"].to_numpy())
    missing_rows = np.bincount(region[hit], minlength=m)
    missing_read = np.bincount(region[hit], weights=weights[hit], minlength=m)
    by_kind = {}
    for kind in KINDS:
        rows = np.unique(invalid.filter(pl.col("kind") == kind)["row"].to_numpy())
        by_kind[kind] = np.bincount(region[rows], minlength=m)
    sides: dict[int, list[dict[str, object]]] = {r: [] for r in range(m)}
    for p in pair_rows:
        for own, other, s in (
            ("region_a", "region_b", "a"),
            ("region_b", "region_a", "b"),
        ):
            sides[p[own]].append(
                {
                    **p,
                    "rival": p[other],
                    "own_err": p[f"nn_error_{s}"],
                    "own_one": p[f"single_column_acc_{s}"],
                }
            )
    rows = []
    for r in range(m):
        mine = sides[r]
        judged = [s for s in mine if s["own_err"] is not None]
        rival = max(judged, key=lambda s: (s["own_err"], -s["rival"]), default=None)
        verdicts = [s["verdict"] for s in mine if s["verdict"] is not None]
        dup_share = float(1 - sizes[r] / reads[r])
        issues = region_issues(
            int(missing_rows[r]),
            dup_share,
            int(conflict_rows[r]),
            verdicts,
            dup_threshold,
        )
        rows.append(
            {
                "region": r,
                "class": classes[region_class[r]],
                "n_rows": int(sizes[r]),
                "n_read": float(reads[r]),
                "n_duplicates": float(reads[r] - sizes[r]),
                "dup_share": dup_share,
                "conflict_rows": int(conflict_rows[r]),
                "conflict_read": float(conflict_read[r]),
                "missing_rows": int(missing_rows[r]),
                "missing_read": float(missing_read[r]),
                **{f"{kind}_rows": int(by_kind[kind][r]) for kind in KINDS},
                "rival": None if rival is None else rival["rival"],
                "nn_error": None if rival is None else rival["own_err"],
                "bayes_error_low": (
                    None if rival is None else cover_hart_low(rival["own_err"])
                ),
                "single_column_acc": None if rival is None else rival["own_one"],
                "single_column": None if rival is None else rival["single_column"],
                "verdict": None if rival is None else rival["verdict"],
                "shortcut_columns": (
                    rival["shortcut_columns"]
                    if rival is not None and rival["verdict"] == "shortcut"
                    else None
                ),
                **{f"pairs_{v}": Counter(verdicts)[v] for v in VERDICTS},
                "issues": ",".join(issues) or None,
                "status": "flagged" if issues else SOLID,
            }
        )
    return rows


def _class_rows(
    region_rows: list[dict[str, object]],
    pair_rows: list[dict[str, object]],
    classes: list[str],
    pdd: np.ndarray,
    tc: np.ndarray,
    quality: QualityConfig,
) -> list[dict[str, object]]:
    """Return the quality_classes rows, the roll-up of the regions."""
    rows = []
    for code, name in enumerate(classes):
        own = [r for r in region_rows if r["class"] == name]
        if not own:
            continue
        n_rows = sum(r["n_rows"] for r in own)
        n_read = sum(r["n_read"] for r in own)
        pairs = [p for p in pair_rows if name in (p["class_a"], p["class_b"])]
        single = [p for p in pairs if p["n_single_columns"] <= 1]
        best = max(single, key=lambda p: p["single_column_acc"], default=None)
        known = [r for r in own if r["nn_error"] is not None]
        issues = [
            issue
            for issue, value, key in zip(
                CLASS_ISSUES, (pdd[code], tc[code]), ("pdd", "tc")
            )
            if value >= quality.thresholds[key]
        ]
        rows.append(
            {
                "class": name,
                "n_rows": n_rows,
                "n_read": n_read,
                "n_duplicates": n_read - n_rows,
                "dup_share": 1 - n_rows / n_read,
                "conflict_share": sum(r["conflict_read"] for r in own) / n_read,
                "missing_share": sum(r["missing_read"] for r in own) / n_read,
                "nn_error": _number(
                    sum(r["nn_error"] * r["n_rows"] for r in known)
                    / sum(r["n_rows"] for r in known)
                    if known
                    else None
                ),
                "single_column_acc": (
                    None if best is None else best["single_column_acc"]
                ),
                "single_column": None if best is None else best["single_column"],
                "pdd": float(pdd[code]),
                "tc": float(tc[code]),
                "issues": ",".join(issues) or None,
                **_coverage(own, n_read),
                **_verdict_counts(pairs),
                "n_pairs": len(pairs),
            }
        )
    return rows


def _coverage(region_rows: list[dict[str, object]], n_read: float) -> dict[str, float]:
    """Return solid_share and share_<issue>: the CSV rows of those regions over n_read."""
    out = {
        "solid_share": sum(r["n_read"] for r in region_rows if r["status"] == SOLID)
        / n_read
    }
    for issue in ISSUES:
        out[f"share_{issue}"] = (
            sum(
                r["n_read"]
                for r in region_rows
                if issue in (r["issues"] or "").split(",")
            )
            / n_read
        )
    return out


def _verdict_counts(pair_rows: list[dict[str, object]]) -> dict[str, int]:
    """Return pairs_<verdict>: the pairs of each verdict."""
    counts = Counter(p["verdict"] for p in pair_rows)
    return {f"pairs_{v}": counts[v] for v in VERDICTS}


def _number(value: float | None) -> float | None:
    """Return value as a float, None when undefined or not finite."""
    return None if value is None or not math.isfinite(value) else float(value)

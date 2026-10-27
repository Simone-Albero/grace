"""The context of each tab of a run page: tiles, tables and figures."""

__all__ = ["TABS", "build"]

import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path

from dashboard import artifacts, charts, queries
from dashboard.queries import PREDICTORS, SCORES, predictor_stage

# Each tab with the done stages that open it: any of them does.
TABS = {
    "overview": (),
    "data": ("split",),
    "classifier": ("classify",),
    "regions": ("regions",),
    "failures": ("failures",),
    "predictors": ("baselines", "regress"),
    "partitions": ("partitions",),
}
# The stages whose config a tab shows at its foot.
_SETTINGS = {
    "overview": (),
    "data": ("split", "prepare"),
    "classifier": ("classify",),
    "regions": ("regions",),
    "failures": ("failures",),
    "predictors": ("baselines", "regress"),
    "partitions": ("partitions",),
}
# The tiles of each overview area: (stage, metric, label, kind).
_OVERVIEW = {
    "data": [
        ("split", "n_raw_rows", "raw rows", "count"),
        ("split", "n_train", "train", "count"),
        ("split", "n_val", "val", "count"),
        ("split", "n_test", "test", "count"),
    ],
    "classifier": [
        ("classify", "accuracy", "accuracy", "score"),
        ("classify", "f1_macro", "F1 macro", "score"),
    ],
    "regions": [("regions", "n_regions", "regions", "count")],
    "failures": [
        ("failures", "val_error_rate", "val error rate", "score"),
        ("failures", "test_error_rate", "test error rate", "score"),
    ],
    "predictors": [
        ("regress", "regressor_spearman", "RF Spearman", "score"),
        ("regress", "regressor_mse", "RF MSE", "score"),
        ("baselines", "val_spearman", "val Spearman", "score"),
        ("baselines", "val_mse", "val MSE", "score"),
    ],
    "partitions": [("partitions", "reference_brier", "reference Brier", "score")],
}
# The metrics each tab shows as tiles: (stage, metric, label, kind).
_TILES = {
    "data": [
        ("split", "n_raw_rows", "raw rows", "count"),
        ("split", "n_raw_columns", "raw columns", "count"),
    ],
    "classifier": [
        ("classify", "accuracy", "accuracy", "score"),
        ("classify", "precision_macro", "precision macro", "score"),
        ("classify", "recall_macro", "recall macro", "score"),
        ("classify", "f1_macro", "F1 macro", "score"),
        ("classify", "f1_weighted", "F1 weighted", "score"),
        ("classify", "best_score", "best CV score", "score"),
        ("classify", "n_fit", "rows fitted", "count"),
    ],
    "regions": [
        ("regions", "n_regions", "regions", "count"),
        ("regions", "n_graph_rows", "graph rows", "count"),
        ("regions", "eval_rows_per_train_row", "eval rows / train row", "score"),
    ],
    "failures": [
        ("failures", "val_error_rate", "val error rate", "score"),
        ("failures", "test_error_rate", "test error rate", "score"),
        ("failures", "n_regions_with_val", "regions with val rows", "count"),
        ("failures", "n_regions_with_test", "regions with test rows", "count"),
    ],
    "partitions": [
        ("partitions", "reference_brier", "reference Brier", "score"),
        ("partitions", "val_error_rate", "val error rate", "score"),
        ("partitions", "test_error_rate", "test error rate", "score"),
        ("partitions", "inter_fit_rows", "inter fit rows", "count"),
    ],
}
# The preprocessing of prepare, in the order it applies.
_PREPROCESSING = {
    "num": "signed log1p, then a RobustScaler (median and IQR)",
    "cat": (
        "integer codes: 0 missing, 1..top_n the most frequent values, "
        "then hash buckets"
    ),
}
_WORST_REGIONS = 15


_Metrics = Mapping[str, Mapping[str, float | None]]


def _tiles(metrics: _Metrics, specs: Sequence[tuple[str, str, str, str]]) -> list[dict]:
    """Build the tiles of the metrics that exist."""
    return [
        {"label": label, "value": metrics[stage][name], "kind": kind}
        for stage, name, label, kind in specs
        if name in metrics.get(stage, {})
    ]


def _dropped(split: Mapping[str, float | None]) -> float | None:
    """Return the raw rows that no split holds, if the metrics say."""
    names = ("n_raw_rows", "n_train", "n_val", "n_test")
    if any(split.get(name) is None for name in names):
        return None
    return split["n_raw_rows"] - sum(split[name] for name in names[1:])


def _data(
    db: sqlite3.Connection,
    ids: Mapping[str, int],
    metrics: _Metrics,
    configs: Mapping[str, Mapping],
) -> dict:
    """Split sizes, preprocessing and features."""
    split = metrics["split"]
    splits = ("train", "val", "test")
    n_split_rows = sum(split.get(f"n_{name}") or 0 for name in splits) or 1
    shares = [
        {
            "label": name,
            "rows": split.get(f"n_{name}"),
            "share": (split.get(f"n_{name}") or 0) / n_split_rows,
        }
        for name in splits
    ]
    feature_rows = queries.features(db, ids["split"])
    n_by_kind = {
        kind: sum(row["kind"] == kind for row in feature_rows)
        for kind in ("num", "cat")
    }
    class_rows = queries.classes(db, ids["prepare"]) if "prepare" in ids else []
    return {
        "tiles": _tiles(metrics, _TILES["data"]),
        "dropped": _dropped(split),
        "shares": shares,
        "kinds": n_by_kind,
        "n_classes": len(class_rows),
        "classes": class_rows,
        "features": feature_rows,
        "preprocessing": _PREPROCESSING,
        "prepare_config": configs.get("prepare", {}),
        "split_config": configs["split"],
        "figures": {"split": charts.split_bars(class_rows)} if class_rows else {},
    }


def _classifier(
    db: sqlite3.Connection,
    ids: Mapping[str, int],
    metrics: _Metrics,
    root: Path,
    dirs: Mapping[str, str],
) -> dict:
    """Scores, confusion matrix, grid search and training loss."""
    classify_id = ids["classify"]
    class_rows = (
        queries.class_metrics(db, ids["prepare"], classify_id)
        if "prepare" in ids
        else []
    )
    figures = {}
    confusion_note = None
    if class_rows:
        figures["scores"] = charts.class_score_bars(class_rows)
        counts = artifacts.confusion(
            root / dirs["prepare"], root / dirs["classify"], len(class_rows)
        )
        if counts is None:
            confusion_note = (
                "The confusion matrix needs "
                f"{root / dirs['prepare'] / 'test.parquet'} and "
                f"{root / dirs['classify'] / 'predictions.parquet'}, and they "
                "are missing or disagree in length."
            )
        else:
            figures["confusion"] = charts.confusion_heatmap(
                counts, [row["name"] for row in class_rows]
            )
    grid = queries.classifier_grid(db, classify_id)
    if grid:
        figures["grid"] = charts.grid_search(grid)
    history = queries.history(db, classify_id)
    if history:
        figures["loss"] = charts.loss_curve(history)
    return {
        "tiles": _tiles(metrics, _TILES["classifier"]),
        "classes": class_rows,
        "grid": sorted(grid, key=lambda row: -(row["mean_score"] or 0)),
        "figures": figures,
        "confusion_note": confusion_note,
    }


def _regions(db: sqlite3.Connection, ids: Mapping[str, int], metrics: _Metrics) -> dict:
    """Clustering of each class and the region explorer's figures."""
    clusterings = queries.class_clusterings(db, ids["regions"], ids["prepare"])
    _, rows = queries.regions(db, ids, measures=False)
    figures = {}
    if clusterings:
        figures["per_class"] = charts.regions_per_class(clusterings)
    if rows:
        figures["sizes"] = charts.region_size_box(rows)
    return {
        "tiles": _tiles(metrics, _TILES["regions"]),
        "clusterings": clusterings,
        "figures": figures,
    }


def _failures(
    db: sqlite3.Connection, ids: Mapping[str, int], metrics: _Metrics, run_id: int
) -> dict:
    """The empirical failure rates of the regions."""
    _, rows = queries.regions(db, ids, measures=False)
    worst = sorted(
        (row for row in rows if row["test_rate"] is not None),
        key=lambda row: -row["test_rate"],
    )[:_WORST_REGIONS]
    figures = {
        "scatter": charts.val_test_scatter(rows, run_id=run_id),
        "classes": charts.class_rate_bars(rows),
        "ecdf": charts.rate_ecdf(rows),
        "bin_counts": charts.region_rate_bins(rows, measure="count"),
        "bin_sizes": charts.region_rate_bins(rows, measure="size"),
        "gap": charts.class_rate_gap(rows, run_id=run_id),
    }
    return {
        "tiles": _tiles(metrics, _TILES["failures"]),
        "worst": worst,
        "figures": figures,
    }


def _scores(metrics: _Metrics) -> dict[str, dict]:
    """Collect each predictor's scores from the stage metrics."""
    scores = {}
    for key in PREDICTORS:
        stage_metrics = metrics.get(predictor_stage(key), {})
        found = {score: stage_metrics.get(f"{key}_{score}") for score in SCORES}
        if any(value is not None for value in found.values()):
            scores[key] = found
    return scores


def _predictors(
    db: sqlite3.Connection, ids: Mapping[str, int], metrics: _Metrics, run_id: int
) -> dict:
    """Predictor scores, estimates against observed, the regressor."""
    scores = _scores(metrics)
    _, rows = queries.regions(db, ids, measures=False)
    best = {
        score: (max if score in ("spearman", "r2") else min)(
            (found[score] for found in scores.values() if found[score] is not None),
            default=None,
        )
        for score in SCORES
    }
    figures = {
        score: charts.predictor_bars(scores, score) for score in ("spearman", "mse")
    }
    figures["observed"] = charts.predicted_vs_observed(rows, run_id=run_id)
    folds = []
    if "regress" in ids:
        folds = queries.regressor_folds(db, ids["regress"])
        importances = queries.importances(db, ids["regress"])
        if importances:
            figures["importances"] = charts.importance_bars(importances)
    baselines = metrics.get("baselines", {})
    return {
        "scores": scores,
        "best": best,
        "predictors": PREDICTORS,
        "score_names": SCORES,
        "counts": [
            ("regions", baselines.get("n_regions")),
            ("used", baselines.get("n_regions_used")),
            ("excluded, no test rows", baselines.get("n_excluded_no_test")),
            ("excluded, low support", baselines.get("n_excluded_low_support")),
            ("ATC threshold", baselines.get("atc_threshold")),
        ],
        "folds": folds,
        "figures": figures,
    }


def _partitions(
    db: sqlite3.Connection, ids: Mapping[str, int], metrics: _Metrics, run_id: int
) -> dict:
    """Each arm's operating points and groups; the sweeps' inputs."""
    points = queries.operating_points(db, run_id)
    figures = {
        metric: charts.arm_bars(points, metric) for metric in queries.ARM_METRICS
    }
    groups = queries.partition_groups(db, ids["partitions"])
    figures["bin_counts"] = charts.group_rate_bins(groups, measure="count")
    figures["bin_sizes"] = charts.group_rate_bins(groups, measure="size")
    return {
        "tiles": _tiles(metrics, _TILES["partitions"]),
        "points": points,
        "figures": figures,
        "metrics": queries.PARTITION_METRICS,
        "axes": queries.PARTITION_AXES,
    }


def _overview(ids: Mapping[str, int], metrics: _Metrics) -> dict:
    """The areas whose stage finished, each with its headline tiles."""
    areas = []
    for tab, specs in _OVERVIEW.items():
        if any(stage in ids for stage in TABS[tab]):
            areas.append({"tab": tab, "tiles": _tiles(metrics, specs)})
    return {"areas": areas}


def build(
    db: sqlite3.Connection, run: Mapping, tab: str, root: Path
) -> dict[str, object]:
    """Build the context of one tab of a run.

    Args:
        db: A connection to grace.db.
        run: The run, as `queries.run` returns it.
        tab: A key of `TABS`.
        root: The experiment root, for the artifact files.

    Returns:
        The tab's context; it holds `available` (False when no stage
        that opens the tab is done, and nothing else of the tab then)
        and the config of the stages it shows.
    """
    ids = queries.stage_ids(run)
    context = {
        "tab": tab,
        "ids": ids,
        "needs": TABS[tab],
        "available": not TABS[tab] or any(stage in ids for stage in TABS[tab]),
    }
    if not context["available"]:
        return context
    metrics = queries.metrics_by_stage(db, ids)
    configs = queries.execution_configs(
        db, {stage: ids[stage] for stage in _SETTINGS[tab] if stage in ids}
    )
    context["settings"] = configs
    match tab:
        case "overview":
            context.update(_overview(ids, metrics))
        case "data":
            context.update(_data(db, ids, metrics, configs))
        case "classifier":
            dirs = queries.execution_dirs(run)
            context.update(_classifier(db, ids, metrics, root, dirs))
        case "regions":
            context.update(_regions(db, ids, metrics))
        case "failures":
            context.update(_failures(db, ids, metrics, run["id"]))
        case "predictors":
            context.update(_predictors(db, ids, metrics, run["id"]))
        case "partitions":
            context.update(_partitions(db, ids, metrics, run["id"]))
    return context

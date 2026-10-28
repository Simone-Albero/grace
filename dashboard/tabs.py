"""The context of each tab of a run page: tiles, tables and figures."""

__all__ = ["TABS", "build"]

import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path

from dashboard import artifacts, charts, queries
from dashboard.queries import PREDICTORS, SCORES

# Each tab with the done stages that open it: any of them does.
TABS = {
    "overview": (),
    "data": ("split",),
    "classifier": ("classify",),
    "tree": ("tree",),
    "predictors": ("evaluate",),
}
# The stages whose config a tab shows at its foot.
_SETTINGS = {
    "overview": (),
    "data": ("split", "prepare"),
    "classifier": ("classify",),
    "tree": ("space", "crossfit", "tree"),
    "predictors": ("evaluate",),
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
    "tree": [
        ("tree", "n_leaves", "leaves", "count"),
        ("tree", "beta", "β", "score"),
    ],
    "predictors": [
        ("evaluate", "test_error_rate", "test error rate", "score"),
        ("evaluate", "tree_mse_ratio", "tree MSE ratio", "score"),
        ("evaluate", "tree_spearman", "tree Spearman", "score"),
        ("evaluate", "confidence_mse_ratio", "confidence MSE ratio", "score"),
    ],
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
    "tree": [
        ("space", "n_components", "LDA components", "count"),
        ("crossfit", "oof_error_rate", "out-of-fold error rate", "score"),
        ("tree", "n_nodes", "nodes grown", "count"),
        ("tree", "n_leaves", "leaves", "count"),
        ("tree", "max_depth", "depth of the cut", "count"),
        ("tree", "n_leaves_without_val", "leaves without val rows", "count"),
        ("tree", "beta", "β of the correction", "score"),
        ("tree", "fh_tau2", "τ² of the shrinkage", "score"),
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


def _fit_notes(config: Mapping, *, has_grid: bool) -> list[str]:
    """Say how classify reduced the rows it fit on, from its config."""
    cap = config["data"]["max_per_class"]
    if cap is None:
        notes = ["No cap: every train row is fitted on."]
    else:
        notes = [
            f"Each class is capped at {cap:,} train rows, sampled at random; "
            "a smaller class is kept whole."
        ]
    if config["classifier"]["kind"] == "dl":
        holdout = config["fit"]["training"]["early_stopping"]["holdout"]
        notes.append(
            f"{holdout:.0%} of the rows left are held out, stratified, to "
            "stop the training on."
        )
    elif has_grid and config["grid_search"]["max_samples"] is not None:
        notes.append(
            "The grid search scores its candidates on a stratified sample of "
            f"about {config['grid_search']['max_samples']:,} rows, when the "
            "rows left are more; the refit uses all of them."
        )
    notes.append("The test rows are never capped.")
    return notes


def _classifier(
    db: sqlite3.Connection,
    ids: Mapping[str, int],
    metrics: _Metrics,
    configs: Mapping[str, Mapping],
    root: Path,
    dirs: Mapping[str, str],
) -> dict:
    """Scores, rows fitted, confusion matrix, grid search and loss."""
    classify_id = ids["classify"]
    class_rows = (
        queries.class_metrics(db, ids["prepare"], classify_id)
        if "prepare" in ids
        else []
    )
    cap = configs["classify"]["data"]["max_per_class"]
    for row in class_rows:
        row["n_capped"] = row["n_train"] if cap is None else min(row["n_train"], cap)
    totals = {
        key: sum(row[key] for row in class_rows)
        for key in ("n_train", "n_capped", "n_test")
    }
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
        "totals": totals,
        "fit_notes": _fit_notes(configs["classify"], has_grid=bool(grid)),
        "grid": sorted(grid, key=lambda row: -(row["mean_score"] or 0)),
        "figures": figures,
        "confusion_note": confusion_note,
    }


def _tree(
    db: sqlite3.Connection, ids: Mapping[str, int], metrics: _Metrics, run_id: int
) -> dict:
    """Roots, leaves, their sizes and how their estimates move."""
    leaves = queries.leaves(db, ids)
    return {
        "tiles": _tiles(metrics, _TILES["tree"]),
        "roots": queries.tree_roots(db, ids["tree"], ids["prepare"]),
        "leaves": leaves,
        "figures": {
            "sizes": charts.leaf_size_bars(leaves),
            "shrinkage": charts.shrinkage_scatter(leaves, run_id=run_id),
            "test": charts.test_scatter(leaves, run_id=run_id),
        },
    }


def _scores(metrics: _Metrics) -> dict[str, dict]:
    """Collect each predictor's scores from the stage metrics."""
    scores = {}
    for key in PREDICTORS:
        stage_metrics = metrics.get("evaluate", {})
        found = {score: stage_metrics.get(f"{key}_{score}") for score in SCORES}
        if any(value is not None for value in found.values()):
            scores[key] = found
    return scores


def _predictors(metrics: _Metrics) -> dict:
    """Predictor scores and their charts."""
    scores = _scores(metrics)
    best = {
        score: min(
            (found[score] for found in scores.values() if found[score] is not None),
            default=None,
        )
        for score in ("mse", "mse_ratio")
    }
    best["spearman"] = max(
        (
            found["spearman"]
            for found in scores.values()
            if found["spearman"] is not None
        ),
        default=None,
    )
    figures = {
        score: charts.predictor_bars(scores, score)
        for score in ("mse_ratio", "spearman")
    }
    evaluated = metrics.get("evaluate", {})
    return {
        "scores": scores,
        "best": best,
        "predictors": PREDICTORS,
        "score_names": SCORES,
        "counts": [
            ("test rows", evaluated.get("n_test")),
            ("test error rate", evaluated.get("test_error_rate")),
        ],
        "figures": figures,
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
            context.update(_classifier(db, ids, metrics, configs, root, dirs))
        case "tree":
            context.update(_tree(db, ids, metrics, run["id"]))
        case "predictors":
            context.update(_predictors(metrics))
    return context

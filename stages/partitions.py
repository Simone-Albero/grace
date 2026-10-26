"""Five groupings of val and test that read no val or test label.

`intra` and `inter` cluster train with k-means, within each class and
over all of it, and route every val and test row to its nearest
centroid; `null`, their floor, does the same with random Voronoi cells
at inter's counts. `confidence` and `predicted_class` bin the
classifier's confidence at equal val mass, over all rows and within
each predicted class. Every arm estimates a group's failure rate on
val and is judged against test. DB rows written: `partitions` (one
row per arm, rule, level and draw), `partition_class_levels` (intra's
per-class curve) and `partition_groups` (the groups of every operating
point; of the null arm, the first draw only), plus metrics n_val,
n_test, eval_rows_per_train_row, val_error_rate, test_error_rate,
reference_brier and inter_fit_rows.

Sized for the first round of datasets: every geometric arm embeds and
routes a whole split at once, and a train split past
`clustering.max_fit_samples` ⋅ `clustering.min_cluster_floor` rows or
so grows its group counts, and the null arm's routing cost, with it.
"""

__all__ = ["CONFIG", "INPUTS", "FILES", "OUTPUTS", "run"]

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.clustering import (
    KdnNodes,
    assign_nearest_centroid,
    compute_centroids,
    compute_kdn,
    granularity_loss,
    kmeans_levels,
    level_counts,
    random_voronoi,
)
from src.partitions import (
    assign_bins,
    assign_class_bins,
    class_confidence_edges,
    confidence_edges,
    estimate_group_rates,
)
from src.scores import group_scores, spearman
from src.space import Space, scale_for_metric
from stages import LABEL, Context

logger = logging.getLogger(__name__)

CONFIG = (
    "clustering.min_cluster_floor",
    "clustering.hardness_k",
    "clustering.max_fit_samples",
    "distance",
    "seed",
    "partitions",
)
INPUTS = ("prepare", "regions", "classify")
FILES = ()
# No file is cached against this stage: its DB rows are its only result.
OUTPUTS = ()
_FIXED_BINS = 15
_BIN_RULES = ("fixed15", "cbrt", "per150")


@dataclass(frozen=True)
class _Predictions:
    """The classifier's predictions on one split.

    Attributes:
        predicted: Each row's predicted class.
        confidence: Each row's winning probability.
        error: Whether each row's prediction misses its label, 0 or 1.
    """

    predicted: np.ndarray
    confidence: np.ndarray
    error: np.ndarray


@dataclass(frozen=True)
class _Scoring:
    """What a geometric arm clusters, routes and is scored on.

    Attributes:
        X_train: The train points, scaled for `metric`.
        hardness: Each train row's kDN.
        val_points: The val points, embedded; routing does not need
            them scaled.
        test_points: The test points, embedded, as `val_points`.
        val_error: Each val row's failure, 0 or 1.
        test_error: Each test row's failure, 0 or 1.
        eval_rows_per_train_row: Val rows per train row, the budget
            of the loss's noise term.
        metric: The distance of the clustering and the routing.
        reference: The global val failure rate, the group-blind
            prediction `skill` is measured against.
    """

    X_train: np.ndarray
    hardness: np.ndarray
    val_points: np.ndarray
    test_points: np.ndarray
    val_error: np.ndarray
    test_error: np.ndarray
    eval_rows_per_train_row: float
    metric: str
    reference: float


@dataclass(frozen=True)
class _ClassClustering:
    """One train class, clustered at every granularity level.

    Attributes:
        class_id: The class.
        rows: Its train row positions.
        levels: Its group labels at every level, finest first.
        targets: The groups asked for at every level.
        losses: The granularity loss of every level.
    """

    class_id: int
    rows: np.ndarray
    levels: list[np.ndarray]
    targets: list[int]
    losses: list[float]


def _load_kdn_nodes(
    regions_dir: Path, space: Space, train: pd.DataFrame, metric: str
) -> KdnNodes:
    """Return the regions graph's nodes, where kDN counts neighbours."""
    with np.load(regions_dir / "graph.npz") as stored:
        rows = stored["rows"]
    return KdnNodes(
        X=scale_for_metric(space.embed(train.iloc[rows]), metric),
        y=train[LABEL].to_numpy()[rows],
        row_ids=rows,
    )


def _read_predictions(ctx: Context, split: str) -> _Predictions:
    """Return the classifier's predictions on `split`.

    Raises:
        ValueError: The split's labels and predictions disagree in
            length.
    """
    table = ctx.split_rows("classify", "predictions.parquet", split)
    labels = ctx.labels(split)
    if len(table) != len(labels):
        raise ValueError(
            f"{split.capitalize()} rows disagree: {len(labels)} labels, "
            f"{len(table)} predicted by classify. "
            "Re-run `python -m stages classify --force`."
        )
    proba_columns = [column for column in table.columns if column.startswith("proba_")]
    predicted = table["y_pred"].to_numpy()
    return _Predictions(
        predicted=predicted,
        confidence=table[proba_columns].to_numpy(dtype=np.float64).max(axis=1),
        error=(labels != predicted).astype(np.float64),
    )


def _n_bins(rule: str, n_val: int) -> int:
    """Return the bin count of a confidence rule, at least one."""
    if rule == "fixed15":
        return _FIXED_BINS
    if rule == "cbrt":
        return max(1, round(n_val ** (1 / 3)))
    return max(1, round(n_val / 150))


def _argmin_coarsest(losses: Sequence[float]) -> int:
    """Return the index of least loss, the highest index on a tie."""
    best = min(losses)
    return max(i for i, loss in enumerate(losses) if loss == best)


def _score_groups(
    group_ids: np.ndarray,
    val_group: np.ndarray,
    test_group: np.ndarray,
    val_error: np.ndarray,
    test_error: np.ndarray,
    reference: float,
) -> dict[str, object]:
    """Score a grouping's val-estimated rates against test.

    Returns:
        `n_groups`, `val_p10`, `val_median`, `val_p90`, `share_no_val`
        and every score `group_scores` returns.
    """
    rates = estimate_group_rates(val_group, val_error, group_ids)
    predicted = rates.loc[test_group].to_numpy()
    scores = group_scores(predicted, test_error, test_group, reference=reference)
    p10, median, p90 = np.quantile(rates.to_numpy(), [0.1, 0.5, 0.9])
    return {
        "n_groups": int(group_ids.size),
        "val_p10": float(p10),
        "val_median": float(median),
        "val_p90": float(p90),
        "share_no_val": float(np.isin(test_group, val_group, invert=True).mean()),
        **scores,
    }


def _score_train_grouping(
    train_group: np.ndarray, scoring: _Scoring
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    """Score a grouping of train, routed to val and test.

    Returns:
        Every `partitions` column but `arm`, `rule`, `level`, `draw`
        and `n_target`, then the group id of every val and test row.
    """
    metric = scoring.metric
    n_train = train_group.shape[0]
    loss = granularity_loss(
        train_group,
        scoring.hardness,
        n_class=n_train,
        eval_rows_per_train_row=scoring.eval_rows_per_train_row,
    )
    centroids = compute_centroids(scoring.X_train, train_group, metric=metric)
    val_group = assign_nearest_centroid(scoring.val_points, centroids, metric=metric)
    test_group = assign_nearest_centroid(scoring.test_points, centroids, metric=metric)
    # The routing gap: train rows read their nearest final centroid
    # too, the only way val and test ever get a group.
    routed_group = assign_nearest_centroid(scoring.X_train, centroids, metric=metric)
    routed_loss = granularity_loss(
        routed_group,
        scoring.hardness,
        n_class=n_train,
        eval_rows_per_train_row=scoring.eval_rows_per_train_row,
    )

    summary = _score_groups(
        np.unique(train_group),
        val_group,
        test_group,
        scoring.val_error,
        scoring.test_error,
        scoring.reference,
    )
    mean_kdn = pd.Series(scoring.hardness).groupby(train_group).mean()
    test_rates = pd.Series(scoring.test_error).groupby(test_group).mean()
    common = mean_kdn.index.intersection(test_rates.index)
    summary["loss"] = loss
    summary["routed_loss"] = routed_loss
    summary["kdn_spearman"] = (
        spearman(mean_kdn.loc[common].to_numpy(), test_rates.loc[common].to_numpy())
        if len(common) > 1
        else float("nan")
    )
    return summary, val_group, test_group


def _partition_row(
    arm: str,
    rule: str,
    level: int,
    draw: int,
    n_target: int,
    summary: dict[str, object],
) -> dict[str, object]:
    """Return one row of the `partitions` table."""
    return {
        "arm": arm,
        "rule": rule,
        "level": level,
        "draw": draw,
        "n_target": n_target,
        **summary,
    }


def _group_table_rows(
    arm: str,
    rule: str,
    group_ids: np.ndarray,
    val_group: np.ndarray,
    test_group: np.ndarray,
    val_error: np.ndarray,
    test_error: np.ndarray,
    *,
    train_group: np.ndarray | None = None,
    hardness: np.ndarray | None = None,
) -> list[dict[str, object]]:
    """Return the `partition_groups` rows of one operating point.

    Args:
        arm: The arm name.
        rule: The operating-point rule.
        group_ids: Every group to report, even an empty one.
        val_group: Each val row's group id.
        test_group: Each test row's group id.
        val_error: Each val row's failure, 0 or 1.
        test_error: Each test row's failure, 0 or 1.
        train_group: Each train row's group id; omitted for a bin arm,
            which routes no train row.
        hardness: Each train row's kDN; required with `train_group`.

    Returns:
        One row per id in `group_ids`.
    """
    val_counts = pd.Series(val_group).value_counts()
    val_rates = pd.Series(val_error).groupby(val_group).mean()
    test_counts = pd.Series(test_group).value_counts()
    test_rates = pd.Series(test_error).groupby(test_group).mean()
    empty = pd.Series(dtype=np.float64)
    train_counts = empty
    mean_kdn = empty
    if train_group is not None:
        train_counts = pd.Series(train_group).value_counts()
        mean_kdn = pd.Series(hardness).groupby(train_group).mean()
    return [
        {
            "arm": arm,
            "rule": rule,
            "group_id": group_id,
            "n_train": int(train_counts.get(group_id, 0)),
            "n_val": int(val_counts.get(group_id, 0)),
            "n_test": int(test_counts.get(group_id, 0)),
            "val_rate": float(val_rates.get(group_id, np.nan)),
            "test_rate": float(test_rates.get(group_id, np.nan)),
            "mean_kdn": float(mean_kdn.get(group_id, np.nan)),
        }
        for group_id in group_ids.tolist()
    ]


def _score_levels(
    arm: str,
    levels: Sequence[np.ndarray],
    targets: Sequence[int],
    scoring: _Scoring,
    *,
    draw: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Score each level of a train grouping, then its least-loss one.

    Returns:
        The `partitions` rows, every level's then the selected one's,
        and the `partition_groups` rows of the selected level, at draw
        0 only: the table keeps a single draw of the null arm.
    """
    scored = [_score_train_grouping(group, scoring) for group in levels]
    partition_rows = [
        _partition_row(arm, "level", level, draw, targets[level], summary)
        for level, (summary, _, _) in enumerate(scored)
    ]
    best = _argmin_coarsest([summary["loss"] for summary, _, _ in scored])
    summary, val_group, test_group = scored[best]
    partition_rows.append(
        _partition_row(arm, "selected", -1, draw, targets[best], summary)
    )
    if draw > 0:
        return partition_rows, []
    group_rows = _group_table_rows(
        arm,
        "selected",
        np.unique(levels[best]),
        val_group,
        test_group,
        scoring.val_error,
        scoring.test_error,
        train_group=levels[best],
        hardness=scoring.hardness,
    )
    return partition_rows, group_rows


def _join_class_groups(
    per_class: Sequence[_ClassClustering],
    chosen_levels: Sequence[int],
    n_train: int,
) -> tuple[np.ndarray, int]:
    """Join every class's groups, each at its chosen level.

    Returns:
        The group id of every train row, numbered apart across classes,
        and the groups asked for.
    """
    group = np.empty(n_train, dtype=np.int64)
    offset = 0
    n_target = 0
    for clustering, level in zip(per_class, chosen_levels):
        labels = clustering.levels[level]
        group[clustering.rows] = labels + offset
        offset += int(labels.max()) + 1
        n_target += clustering.targets[level]
    return group, n_target


def _score_intra(
    train_labels: np.ndarray,
    scoring: _Scoring,
    *,
    min_cluster_floor: int,
    max_fit_samples: int,
    random_state: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Score the intra arm: k-means within every class of train.

    A level joins every class at that level; the selected grouping
    joins each class at its own level of least loss.

    Returns:
        The `partitions` rows, the `partition_class_levels` rows and
        the `partition_groups` rows of the selected grouping.
    """
    per_class = []
    for class_id in np.unique(train_labels).tolist():
        rows = np.flatnonzero(train_labels == class_id)
        levels = kmeans_levels(
            scoring.X_train[rows],
            min_cluster_floor=min_cluster_floor,
            max_fit_samples=max_fit_samples,
            metric=scoring.metric,
            random_state=random_state,
        )
        class_hardness = scoring.hardness[rows]
        losses = [
            granularity_loss(
                labels,
                class_hardness,
                n_class=rows.size,
                eval_rows_per_train_row=scoring.eval_rows_per_train_row,
            )
            for labels in levels
        ]
        targets = level_counts(rows.size, min_cluster_floor)
        per_class.append(_ClassClustering(class_id, rows, levels, targets, losses))
    selected_levels = [_argmin_coarsest(clustering.losses) for clustering in per_class]

    n_train = train_labels.shape[0]
    partition_rows = []
    class_level_rows = []
    for level in range(len(per_class[0].levels)):
        for clustering, selected_level in zip(per_class, selected_levels):
            class_level_rows.append(
                {
                    "class_id": clustering.class_id,
                    "level": level,
                    "n_clusters": int(np.unique(clustering.levels[level]).size),
                    "loss": clustering.losses[level],
                    "selected": int(level == selected_level),
                }
            )
        group, n_target = _join_class_groups(
            per_class, [level] * len(per_class), n_train
        )
        summary, _, _ = _score_train_grouping(group, scoring)
        partition_rows.append(
            _partition_row("intra", "level", level, 0, n_target, summary)
        )

    group, n_target = _join_class_groups(per_class, selected_levels, n_train)
    summary, val_group, test_group = _score_train_grouping(group, scoring)
    partition_rows.append(_partition_row("intra", "selected", -1, 0, n_target, summary))
    group_rows = _group_table_rows(
        "intra",
        "selected",
        np.unique(group),
        val_group,
        test_group,
        scoring.val_error,
        scoring.test_error,
        train_group=group,
        hardness=scoring.hardness,
    )
    return partition_rows, class_level_rows, group_rows


def _score_inter(
    scoring: _Scoring,
    *,
    min_cluster_floor: int,
    max_fit_samples: int,
    random_state: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Score the inter arm: k-means over all of train.

    Returns:
        The `partitions` rows and the `partition_groups` rows of the
        selected level.
    """
    n_train = scoring.X_train.shape[0]
    if n_train > max_fit_samples:
        logger.warning(
            "inter: fitting k-means on %s of %s train rows "
            "(clustering.max_fit_samples)",
            max_fit_samples,
            n_train,
        )
    levels = kmeans_levels(
        scoring.X_train,
        min_cluster_floor=min_cluster_floor,
        max_fit_samples=max_fit_samples,
        metric=scoring.metric,
        random_state=random_state,
    )
    targets = level_counts(n_train, min_cluster_floor)
    return _score_levels("inter", levels, targets, scoring, draw=0)


def _score_null(
    scoring: _Scoring,
    *,
    min_cluster_floor: int,
    seed: int,
    null_draws: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Score the null arm: random Voronoi cells, at inter's counts.

    Returns:
        The `partitions` rows of every draw and the `partition_groups`
        rows of the first draw's selected level.
    """
    targets = level_counts(scoring.X_train.shape[0], min_cluster_floor)
    partition_rows = []
    group_rows = []
    for draw in range(null_draws):
        levels = [
            random_voronoi(
                scoring.X_train,
                n_cells,
                min_cluster_floor=min_cluster_floor,
                metric=scoring.metric,
                rng=np.random.default_rng([seed, level, draw]),
            )
            for level, n_cells in enumerate(targets)
        ]
        draw_rows, draw_groups = _score_levels(
            "null", levels, targets, scoring, draw=draw
        )
        partition_rows += draw_rows
        group_rows += draw_groups
    return partition_rows, group_rows


def _score_bins(
    arm: str, val: _Predictions, test: _Predictions, reference: float
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Score a bin arm: equal-mass confidence bins, three bin counts.

    `predicted_class` bins within each predicted class, `confidence`
    over all rows together.

    Returns:
        The `partitions` rows and the `partition_groups` rows of every
        rule.
    """
    n_classes = int(np.unique(val.predicted).size)
    partition_rows = []
    group_rows = []
    for rule in _BIN_RULES:
        n_bins = _n_bins(rule, len(val.error))
        if arm == "predicted_class":
            # Every predicted class needs a bin of its own.
            n_bins = max(n_bins, n_classes)
            edges = class_confidence_edges(val.confidence, val.predicted, n_bins)
            # One call on both splits: assign_class_bins numbers groups
            # from the classes it is given, so routing val and test
            # separately could give the same id to two classes.
            confidence = np.concatenate([val.confidence, test.confidence])
            predicted = np.concatenate([val.predicted, test.predicted])
            group = assign_class_bins(confidence, predicted, edges)
            val_group, test_group = np.split(group, [len(val.confidence)])
            # A class predicted on test but never on val gets an id of
            # its own too, beyond the bins the rule apportioned.
            n_target = n_bins + len(set(predicted.tolist()) - set(edges))
        else:
            edges = confidence_edges(val.confidence, n_bins)
            val_group = assign_bins(val.confidence, edges)
            test_group = assign_bins(test.confidence, edges)
            n_target = n_bins

        group_ids = np.unique(np.concatenate([val_group, test_group]))
        summary = _score_groups(
            group_ids, val_group, test_group, val.error, test.error, reference
        )
        # A bin arm has no geometric loss or train-side kDN.
        summary.update(
            loss=float("nan"), routed_loss=float("nan"), kdn_spearman=float("nan")
        )
        partition_rows.append(_partition_row(arm, rule, -1, 0, n_target, summary))
        group_rows += _group_table_rows(
            arm, rule, group_ids, val_group, test_group, val.error, test.error
        )
    return partition_rows, group_rows


def run(ctx: Context) -> None:
    """Score the five arms' groupings of val and test.

    Args:
        ctx: The stage context.
    """
    cfg = ctx.cfg
    clustering = cfg.clustering
    distance, seed = cfg.distance, cfg.seed
    regions_dir = ctx.inputs["regions"].dir
    prepared = ctx.inputs["prepare"].dir

    space: Space = joblib.load(regions_dir / "space.joblib")
    train = pd.read_parquet(prepared / "train.parquet")
    val = pd.read_parquet(prepared / "val.parquet")
    test = pd.read_parquet(prepared / "test.parquet")
    train_labels = train[LABEL].to_numpy()

    nodes = _load_kdn_nodes(regions_dir, space, train, distance)
    X_train = scale_for_metric(space.embed(train), distance)
    hardness = compute_kdn(
        X_train,
        row_ids=np.arange(len(train)),
        labels=train_labels,
        nodes=nodes,
        k=clustering.hardness_k,
    )
    val_predictions = _read_predictions(ctx, "val")
    test_predictions = _read_predictions(ctx, "test")
    reference = float(val_predictions.error.mean())
    eval_rows_per_train_row = len(val) / len(train)
    scoring = _Scoring(
        X_train=X_train,
        hardness=hardness,
        val_points=space.embed(val),
        test_points=space.embed(test),
        val_error=val_predictions.error,
        test_error=test_predictions.error,
        eval_rows_per_train_row=eval_rows_per_train_row,
        metric=distance,
        reference=reference,
    )

    partition_rows, class_level_rows, group_rows = _score_intra(
        train_labels,
        scoring,
        min_cluster_floor=clustering.min_cluster_floor,
        max_fit_samples=clustering.max_fit_samples,
        random_state=seed,
    )
    other_arms = (
        _score_inter(
            scoring,
            min_cluster_floor=clustering.min_cluster_floor,
            max_fit_samples=clustering.max_fit_samples,
            random_state=seed,
        ),
        _score_null(
            scoring,
            min_cluster_floor=clustering.min_cluster_floor,
            seed=seed,
            null_draws=cfg.partitions.null_draws,
        ),
        _score_bins("confidence", val_predictions, test_predictions, reference),
        _score_bins("predicted_class", val_predictions, test_predictions, reference),
    )
    for arm_rows, arm_groups in other_arms:
        partition_rows += arm_rows
        group_rows += arm_groups

    ctx.write("partitions", pd.DataFrame(partition_rows))
    ctx.write("partition_class_levels", pd.DataFrame(class_level_rows))
    ctx.write("partition_groups", pd.DataFrame(group_rows))
    ctx.save_metrics(
        {
            "n_val": len(val),
            "n_test": len(test),
            "eval_rows_per_train_row": eval_rows_per_train_row,
            "val_error_rate": reference,
            "test_error_rate": float(test_predictions.error.mean()),
            "reference_brier": float(
                np.mean((reference - test_predictions.error) ** 2)
            ),
            "inter_fit_rows": min(len(train), clustering.max_fit_samples),
        }
    )

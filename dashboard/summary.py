"""The experiment tables and figures: what a config's seeds show."""

__all__ = [
    "HEADLINE",
    "experiment_figures",
    "experiment_row",
    "metric_label",
    "metric_order",
    "metric_table",
    "stage_states",
]

from collections.abc import Mapping, Sequence

from dashboard import charts
from dashboard.experiments import Experiment, SeedRun, metric_stage, summarize
from dashboard.queries import PREDICTORS, STAGE_ORDER

# The metrics of the experiments table, with their column label.
HEADLINE = {
    "classify.accuracy": "accuracy",
    "classify.f1_macro": "F1 macro",
    "failures.test_error_rate": "test error rate",
    "regions.n_regions": "regions",
    "baselines.val_spearman": "val Spearman",
    "baselines.val_mse": "val MSE",
}


def stage_states(run: Mapping) -> dict[str, str | None]:
    """Say what happened to each stage in a run.

    Returns:
        By stage: "cached", "done", "running" or "failed"; None if the
        run never reached it.
    """
    states: dict[str, str | None] = dict.fromkeys(STAGE_ORDER)
    for row in run["stages"]:
        done = "cached" if row["cached"] else "done"
        states[row["stage"]] = done if row["status"] == "done" else row["status"]
    return states


def _coverage(experiment: Experiment) -> list[dict]:
    """Count, per stage, the seeds of an experiment that finished it."""
    return [
        {
            "stage": stage,
            "done": sum(stage in seed.executions for seed in experiment.seeds),
            "seeds": len(experiment.seeds),
        }
        for stage in STAGE_ORDER
    ]


def experiment_row(experiment: Experiment) -> dict[str, object]:
    """Summarize an experiment for the experiments table.

    Returns:
        The key, main values, variant, seed count, the latest run, the
        number of running and failed seeds, the stage coverage and the
        `(mean, std, n)` of every headline metric.
    """
    statuses = [seed.run["status"] for seed in experiment.seeds]
    return {
        "key": experiment.key,
        "values": experiment.values,
        "variant": experiment.variant,
        "n_seeds": len(experiment.seeds),
        "latest": max(seed.run["id"] for seed in experiment.seeds),
        "running": statuses.count("running"),
        "failed": statuses.count("failed"),
        "coverage": _coverage(experiment),
        "aggregates": {
            key: summarize(seed.metrics.get(key) for seed in experiment.seeds)
            for key in HEADLINE
        },
    }


def metric_label(key: str) -> str:
    """Name a `<stage>.<name>` metric."""
    stage, name = key.split(".", 1)
    return f"{stage} · {name}"


def metric_order(key: str) -> tuple[int, str]:
    """Order a metric key by its stage's place, then by name."""
    return STAGE_ORDER.index(metric_stage(key)), key


def metric_table(experiment: Experiment) -> list[dict]:
    """List every metric of an experiment with its value per seed.

    Returns:
        One row per metric in stage order, with `values` (one per seed,
        None where it is missing) and the `(mean, std, n)`.
    """
    keys = {key for seed in experiment.seeds for key in seed.metrics}
    return [
        {
            "label": metric_label(key),
            "values": [seed.metrics.get(key) for seed in experiment.seeds],
            "summary": summarize(seed.metrics.get(key) for seed in experiment.seeds),
        }
        for key in sorted(keys, key=metric_order)
    ]


def _series(
    seeds: Sequence[SeedRun], labelled_keys: Mapping[str, str]
) -> dict[str, list[tuple[int, float | None]]]:
    """Pair each seed with its value of each labelled metric."""
    return {
        label: [(seed.seed, seed.metrics.get(key)) for seed in seeds]
        for label, key in labelled_keys.items()
    }


def _has_values(series: Mapping[str, Sequence[tuple[int, float | None]]]) -> bool:
    """Say whether any seed has any value of the series."""
    return any(value is not None for pairs in series.values() for _, value in pairs)


def _predictor_keys(score: str) -> dict[str, str]:
    """Map each predictor's label to its metric key of `score`."""
    return {label: f"baselines.{key}_{score}" for key, label in PREDICTORS.items()}


def experiment_figures(experiment: Experiment) -> list[tuple[str, dict]]:
    """Draw the experiment's seeds, one figure per metric family.

    Returns:
        `(title, figure)` pairs, leaving out a family no seed has.
    """
    families = [
        (
            "Classifier, test",
            "score",
            {
                name: f"classify.{name}"
                for name in ("accuracy", "precision_macro", "recall_macro", "f1_macro")
            },
        ),
        (
            "Failure rate",
            "error rate",
            {
                "val": "failures.val_error_rate",
                "test": "failures.test_error_rate",
            },
        ),
        (
            "Predictors, Spearman with the test rates",
            "Spearman",
            _predictor_keys("spearman"),
        ),
        ("Predictors, MSE against the test rates", "MSE", _predictor_keys("mse")),
    ]
    figures = []
    for title, y_title, keys in families:
        series = _series(experiment.seeds, keys)
        if _has_values(series):
            figures.append((title, charts.seed_strip(series, y_title)))
    return figures

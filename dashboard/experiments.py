"""Group runs into experiments, one per config; aggregate the seeds."""

__all__ = [
    "MAIN_KEYS",
    "Experiment",
    "SeedRun",
    "build",
    "compare_rows",
    "config_key",
    "grouped_values",
    "metric_stage",
    "summarize",
]

import hashlib
import json
import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

# The config keys that do not tell two experiments apart: the seed is
# what an experiment varies, root and runtime are in no cache key, and
# `figure_format` only steers `render`, whose figures are not shown.
_IGNORED = ("seed", "root", "runtime", "figure_format")
# The config values every page shows, each with its place in the config.
MAIN_KEYS = {
    "data": ("groups", "data"),
    "classifier": ("groups", "classifier"),
}


@dataclass(frozen=True)
class SeedRun:
    """The latest run of one seed of an experiment.

    Attributes:
        seed: The seed.
        run: The run, as `queries.runs` lists it.
        metrics: Its metrics by `<stage>.<name>`, those of its done
            executions only.
        executions: The execution id of each of its done stages.
        earlier: The older runs of the same seed, newest first.
    """

    seed: int
    run: dict
    metrics: dict[str, float | None]
    executions: dict[str, int]
    earlier: list[dict]


@dataclass(frozen=True)
class Experiment:
    """The runs of one config, whatever their seed.

    Attributes:
        key: A short stable id of the config, for URLs.
        config: The config without the ignored keys.
        values: The value of each of `MAIN_KEYS`.
        variant: What tells it from the experiments of the same main
            values: the dotted config keys that differ; "" if alone.
        seeds: One entry per seed, ordered by seed.
    """

    key: str
    config: dict
    values: dict[str, str]
    variant: str
    seeds: list[SeedRun]


def _flatten(config: Mapping, prefix: str = "") -> dict[str, str]:
    """Flatten a config into dotted keys; a list is its JSON text."""
    flat = {}
    for name, value in config.items():
        if isinstance(value, Mapping):
            flat.update(_flatten(value, f"{prefix}{name}."))
        else:
            flat[f"{prefix}{name}"] = json.dumps(value)
    return flat


def _main_value(config: Mapping, path: tuple[str, ...]) -> str:
    """Read the value at `path` of the config, as text."""
    value = config
    for name in path:
        value = value[name]
    return str(value)


def _variants(configs: Mapping[str, Mapping]) -> dict[str, str]:
    """Name what tells each config from the others of its main values.

    Args:
        configs: Each experiment's config by key.

    Returns:
        The variant text by key, "" for an experiment alone with its
        main values.
    """
    flats = {key: _flatten(config) for key, config in configs.items()}
    siblings: dict[tuple, list[str]] = {}
    for key, config in configs.items():
        main = tuple(_main_value(config, path) for path in MAIN_KEYS.values())
        siblings.setdefault(main, []).append(key)
    variants = {}
    for keys in siblings.values():
        names = sorted({name for key in keys for name in flats[key]})
        differing = [
            name for name in names if len({flats[key].get(name) for key in keys}) > 1
        ]
        for key in keys:
            parts = [f"{name}={flats[key].get(name, '-')}" for name in differing]
            variants[key] = ", ".join(parts) if len(keys) > 1 else ""
    return variants


def _finite(value: float | None) -> float | None:
    """Return `value`, or None if it is infinite or NaN.

    SQLite keeps an infinity (the ATC threshold can be one), which JSON
    cannot carry and the statistics cannot average.
    """
    return value if value is not None and math.isfinite(value) else None


def config_key(run: Mapping) -> tuple[str, dict]:
    """Identify the experiment a run belongs to.

    Args:
        run: A run, as `queries.runs` lists it.

    Returns:
        The experiment key and its config.
    """
    config = {
        name: value for name, value in run["config"].items() if name not in _IGNORED
    }
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    return digest[:12], config


def _seed_run(
    seed: int,
    seed_runs: Sequence[dict],
    metrics: Mapping[int, Mapping[str, float | None]],
) -> SeedRun:
    """Build the entry of a seed from its runs, newest first."""
    latest = seed_runs[0]
    return SeedRun(
        seed=seed,
        run=latest,
        metrics=dict(metrics.get(latest["id"], {})),
        executions={
            row["stage"]: row["execution_id"]
            for row in latest["stages"]
            if row["status"] == "done"
        },
        earlier=list(seed_runs[1:]),
    )


def build(
    runs: Sequence[dict],
    metric_rows: Iterable[tuple[int, str, int, str, float | None]],
) -> list[Experiment]:
    """Group runs into experiments.

    Args:
        runs: Every run, as `queries.runs` lists them.
        metric_rows: `queries.run_metrics`.

    Returns:
        The experiments, the one with the newest run first.
    """
    metrics: dict[int, dict[str, float | None]] = {}
    for run_id, stage, _, name, value in metric_rows:
        metrics.setdefault(run_id, {})[f"{stage}.{name}"] = _finite(value)

    by_config: dict[str, tuple[dict, dict[int, list[dict]]]] = {}
    for run in sorted(runs, key=lambda run: run["id"], reverse=True):
        key, config = config_key(run)
        _, seeds = by_config.setdefault(key, (config, {}))
        seeds.setdefault(run["config"]["seed"], []).append(run)

    variants = _variants({key: config for key, (config, _) in by_config.items()})
    return [
        Experiment(
            key=key,
            config=config,
            values={
                name: _main_value(config, path) for name, path in MAIN_KEYS.items()
            },
            variant=variants[key],
            seeds=[
                _seed_run(seed, seed_runs, metrics)
                for seed, seed_runs in sorted(seeds.items())
            ],
        )
        for key, (config, seeds) in by_config.items()
    ]


def summarize(values: Iterable[float | None]) -> tuple[float | None, float | None, int]:
    """Aggregate the values of one metric over the seeds.

    Args:
        values: One value per seed; None and NaN are skipped.

    Returns:
        The mean, the sample standard deviation (None below two values)
        and the number of values; the mean is None without any.
    """
    kept = [value for value in values if value is not None and not math.isnan(value)]
    if not kept:
        return None, None, 0
    std = statistics.stdev(kept) if len(kept) > 1 else None
    return statistics.mean(kept), std, len(kept)


def metric_stage(metric: str) -> str:
    """Return the stage of a `<stage>.<name>` metric key."""
    return metric.split(".")[0]


def compare_rows(experiments: Iterable[Experiment]) -> list[dict]:
    """List what Compare plots: one row per seed of every experiment.

    Returns:
        Rows with the main config values, `metrics`, `executions` and
        the `run_id`.
    """
    return [
        {
            **experiment.values,
            "run_id": seed.run["id"],
            "metrics": seed.metrics,
            "executions": seed.executions,
        }
        for experiment in experiments
        for seed in experiment.seeds
    ]


def grouped_values(
    rows: Sequence[Mapping], metric: str, x: str, color: str | None
) -> dict[str, dict[str, list[float]]]:
    """Collect a metric's values by x and color, each execution once.

    Two runs that share the execution holding the metric, such as the
    experiments of a sweep over a later stage, would count it twice in a
    mean; here the first one wins.

    Args:
        rows: The rows of `compare_rows`.
        metric: The `<stage>.<name>` metric.
        x: The main config key of the x axis.
        color: The main config key that splits the traces; None for one.

    Returns:
        The values by color value ("" without a color) and x value.
    """
    stage = metric_stage(metric)
    seen = set()
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        value = row["metrics"].get(metric)
        execution = row["executions"].get(stage)
        if value is None or math.isnan(value):
            continue
        tint = "" if color is None else str(row[color])
        key = (str(row[x]), tint, execution)
        if key in seen:
            continue
        seen.add(key)
        grouped.setdefault(tint, {}).setdefault(str(row[x]), []).append(value)
    return grouped

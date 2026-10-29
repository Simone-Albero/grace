"""Config loading (YAML groups, `k=v` overrides) and validation."""

__all__ = ["CONFIGS", "load_config", "sweep", "validate"]

import itertools
import math
import re
from collections.abc import Sequence
from pathlib import Path

import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import ConfigKeyError, OmegaConfBaseException
from sklearn.metrics import get_scorer_names

from src.dl import LOSSES, MODELS
from src.ml import CLASSIFIERS
from stages import LABEL

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
# The keys a group option sets: changing them would make the `groups`
# label of the results lie.
_PICKED = ("classifier.name", "classifier.kind")


def _group_files(group: str, configs: Path) -> dict[str, Path]:
    """Return the option files of a config group, by name."""
    return {path.stem: path for path in sorted((configs / group).glob("*.yaml"))}


def load_config(overrides: Sequence[str], *, configs: Path = CONFIGS) -> DictConfig:
    """Load config.yaml, its groups and `overrides`, and validate them.

    Args:
        overrides: `key=value` items; `<group>=<option>` picks a group
            option, any other key sets a config value.
        configs: The folder with config.yaml and the group folders.

    Returns:
        The validated, resolved config.

    Raises:
        ValueError: An override is malformed or names an unknown key,
            group or option, or the config fails validation.
    """
    body = OmegaConf.load(configs / "config.yaml")
    groups = dict(body.groups)
    rest = []
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override {item!r} is not key=value.")
        key, value = item.split("=", 1)
        if key in groups:
            groups[key] = value
        else:
            rest.append(item)

    cfg = OmegaConf.create({})
    for group, option in groups.items():
        files = _group_files(group, configs)
        if option not in files:
            raise ValueError(f"Unknown {group} {option!r}. Valid: {sorted(files)}")
        cfg = OmegaConf.merge(cfg, {group: OmegaConf.load(files[option])})
    cfg = OmegaConf.merge(cfg, body, {"groups": groups})

    picked = {key: OmegaConf.select(cfg, key) for key in _PICKED}
    OmegaConf.set_struct(cfg, True)
    for item in rest:
        try:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([item]))
        except ConfigKeyError as error:
            raise ValueError(
                f"Unknown config key in {item!r}: "
                "add it to a YAML under configs/ first."
            ) from error
        except (OmegaConfBaseException, yaml.YAMLError) as error:
            raise ValueError(f"Invalid override {item!r}: {error}") from error
        # The values, not the key text, so `[groups][data]=...` counts.
        _check(
            OmegaConf.to_container(cfg.groups) == groups
            and all(
                OmegaConf.select(cfg, key) == value for key, value in picked.items()
            ),
            f"Override {item!r} changes what a group option sets "
            f"({', '.join(['groups.*', *_PICKED])}); pick it with "
            "<group>=<option>, e.g. classifier=mlp.",
        )
    try:
        cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    except OmegaConfBaseException as error:  # A bad ${interpolation}.
        raise ValueError(f"Invalid config: {error}") from error
    validate(cfg)
    return cfg


def _check(condition: bool, message: str) -> None:
    """Raise `ValueError(message)` unless `condition` holds."""
    if not condition:
        raise ValueError(message)


def _is_number(value: object) -> bool:
    """Return whether `value` is an int or a float (not a bool)."""
    return isinstance(value, int | float) and not isinstance(value, bool)


def _is_int(value: object, minimum: float = 0) -> bool:
    """Return whether `value` is an int (not a bool) >= `minimum`."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _is_subclass(value: object, base: type) -> bool:
    """Return whether `value` derives from `base`, not being `base`."""
    return isinstance(value, type) and issubclass(value, base) and value is not base


def validate(cfg: DictConfig) -> None:
    """Fail fast on a config the stages cannot run with.

    Args:
        cfg: The merged and resolved config.

    Raises:
        ValueError: A value is invalid; the message names the key, the
            value and, where a set exists, the valid ones.
    """
    # A seed must be a valid numpy seed, below 2**32, with a margin.
    _check(
        _is_int(cfg.seed) and cfg.seed < 2**31,
        f"seed {cfg.seed!r} must be an integer in [0, 2**31).",
    )
    _check(
        cfg.figure_format in ("pdf", "png"),
        f"figure_format {cfg.figure_format!r} invalid; valid: ['pdf', 'png'].",
    )
    _check(
        isinstance(cfg.root, str) and bool(cfg.root),
        f"root {cfg.root!r} must be a non-empty folder path.",
    )
    _validate_data(cfg.data)
    _validate_classifier(cfg.classifier)
    _validate_fit(cfg.fit)
    _validate_loss(cfg)
    _validate_grid_search(cfg.grid_search)
    _validate_space(cfg)
    _validate_tree(cfg.tree)
    _validate_runtime(cfg.runtime)


def _validate_data(data: DictConfig) -> None:
    """Check the `data` section."""
    _check(
        str(data.path).endswith(".csv"), f"data.path {data.path!r} must end in '.csv'."
    )
    for key in ("num_cols", "cat_cols"):
        cols = data[key]
        _check(
            OmegaConf.is_list(cols) and all(isinstance(col, str) for col in cols),
            f"data.{key} {cols!r} must be a list of column names.",
        )
        twice = sorted({col for col in cols if list(cols).count(col) > 1})
        _check(not twice, f"data.{key} lists {twice!r} more than once.")
    num_cols, cat_cols = list(data.num_cols), list(data.cat_cols)
    _check(
        not set(num_cols) & set(cat_cols),
        f"data.num_cols and data.cat_cols overlap: "
        f"{sorted(set(num_cols) & set(cat_cols))!r}.",
    )
    _check(
        bool(num_cols or cat_cols),
        "data.num_cols and data.cat_cols are both empty; list at least one feature.",
    )
    _check(
        isinstance(data.label_col, str),
        f"data.label_col {data.label_col!r} must be a str.",
    )
    _check(
        data.label_col not in num_cols + cat_cols,
        f"data.label_col {data.label_col!r} is also a feature; "
        "remove it from num_cols/cat_cols.",
    )
    _check(
        LABEL not in num_cols + cat_cols,
        f"{LABEL!r} is reserved for the class column; rename that feature.",
    )
    fractions = [data.train_frac, data.val_frac, data.test_frac]
    _check(
        all(
            isinstance(fraction, int | float) and 0 < fraction < 1
            for fraction in fractions
        )
        and math.isclose(sum(fractions), 1.0),
        f"data train/val/test fractions {fractions!r} must lie in (0, 1) and sum to 1.",
    )
    for key in ("min_class_count", "top_n", "hash_buckets"):
        _check(_is_int(data[key]), f"data.{key} {data[key]!r} must be an integer >= 0.")
    _check(
        data.max_per_class is None or _is_int(data.max_per_class, 1),
        f"data.max_per_class {data.max_per_class!r} must be null or an integer > 0.",
    )


def _validate_classifier(classifier: DictConfig) -> None:
    """Check the `classifier` section."""
    _check(
        classifier.kind in ("ml", "dl"),
        f"classifier.kind {classifier.kind!r} invalid; valid: ['ml', 'dl'].",
    )
    names = CLASSIFIERS if classifier.kind == "ml" else MODELS
    _check(
        classifier.name in names,
        f"classifier.name {classifier.name!r} invalid for kind {classifier.kind!r}; "
        f"valid: {sorted(names)}.",
    )
    _check(
        OmegaConf.is_dict(classifier.params),
        f"classifier.params {classifier.params!r} must be a mapping ({{}} for none).",
    )
    if classifier.kind == "ml" and "grid" in classifier and classifier.grid:
        grid = classifier.grid
        _check(
            OmegaConf.is_dict(grid)
            and all(
                OmegaConf.is_list(values) and len(values) > 0
                for values in grid.values()
            ),
            f"classifier.grid {OmegaConf.to_container(grid)!r} must map each "
            "parameter to a non-empty list of values.",
        )


def _validate_fit(fit: DictConfig) -> None:
    """Check the `fit` section."""
    training = fit.training
    _check(
        _is_int(training.epochs, 1),
        f"fit.training.epochs {training.epochs!r} must be an integer > 0.",
    )
    patience = training.early_stopping.patience
    _check(
        _is_int(patience, 1),
        f"fit.training.early_stopping.patience {patience!r} must be an integer > 0.",
    )
    min_delta = training.early_stopping.min_delta
    _check(
        _is_number(min_delta) and min_delta >= 0,
        f"fit.training.early_stopping.min_delta {min_delta!r} must be a number >= 0.",
    )
    holdout = training.early_stopping.holdout
    _check(
        _is_number(holdout) and 0 < holdout < 1,
        f"fit.training.early_stopping.holdout {holdout!r} must be a number in (0, 1).",
    )
    max_grad_norm = training.max_grad_norm
    _check(
        _is_number(max_grad_norm) and max_grad_norm > 0,
        f"fit.training.max_grad_norm {max_grad_norm!r} must be a number > 0.",
    )
    batch_size = training.dataloader.batch_size
    _check(
        _is_int(batch_size, 2),
        f"fit.training.dataloader.batch_size {batch_size!r} must be an integer >= 2 "
        "(BatchNorm needs 2 rows per batch).",
    )
    holdout_batch_size = training.early_stopping.dataloader.batch_size
    _check(
        _is_int(holdout_batch_size, 1),
        f"fit.training.early_stopping.dataloader.batch_size {holdout_batch_size!r} "
        "must be an integer > 0.",
    )


def _validate_loss(cfg: DictConfig) -> None:
    """Check the `loss`, `optimizer` and `scheduler` sections."""
    loss = cfg.loss
    _check(
        loss.name in LOSSES,
        f"loss.name {loss.name!r} invalid; valid: {sorted(LOSSES)}.",
    )
    class_weight = loss.params.class_weight
    _check(
        class_weight is None
        or class_weight == "auto"
        or (OmegaConf.is_list(class_weight) and all(map(_is_number, class_weight))),
        f"loss.params.class_weight {class_weight!r} invalid; "
        "valid: 'auto', null or a list with one weight per class.",
    )
    if "label_smoothing" in loss.params:
        smoothing = loss.params.label_smoothing
        _check(
            _is_number(smoothing) and 0 <= smoothing < 1,
            f"loss.params.label_smoothing {smoothing!r} must be a number in [0, 1).",
        )
    if "gamma" in loss.params:
        gamma = loss.params.gamma
        _check(
            _is_number(gamma) and gamma >= 0,
            f"loss.params.gamma {gamma!r} must be a number >= 0.",
        )
    _check(
        _is_subclass(
            getattr(torch.optim, cfg.optimizer.name, None), torch.optim.Optimizer
        ),
        f"optimizer.name {cfg.optimizer.name!r} is not an optimizer of torch.optim.",
    )
    _check(
        # The legacy _LRScheduler base is no scheduler.
        not cfg.scheduler.name.startswith("_")
        and _is_subclass(
            getattr(torch.optim.lr_scheduler, cfg.scheduler.name, None),
            torch.optim.lr_scheduler.LRScheduler,
        ),
        f"scheduler.name {cfg.scheduler.name!r} is not a scheduler of "
        "torch.optim.lr_scheduler.",
    )


def _validate_grid_search(grid_search: DictConfig) -> None:
    """Check the `grid_search` section."""
    _check(
        grid_search.scoring in get_scorer_names(),
        f"grid_search.scoring {grid_search.scoring!r} invalid; "
        "see sklearn.metrics.get_scorer_names().",
    )
    _check(
        _is_int(grid_search.cv, 2),
        f"grid_search.cv {grid_search.cv!r} must be an integer >= 2.",
    )
    _check(
        grid_search.max_samples is None or _is_int(grid_search.max_samples, 1),
        f"grid_search.max_samples {grid_search.max_samples!r} "
        "must be null or an integer > 0.",
    )


def _validate_space(cfg: DictConfig) -> None:
    """Check the space, graph and complexity sections."""
    positive_keys = {
        "graph": ("k", "max_samples"),
        "complexity": tuple(cfg.complexity),
    }
    for group, keys in positive_keys.items():
        for key in keys:
            value = cfg[group][key]
            _check(
                _is_int(value, 1), f"{group}.{key} {value!r} must be an integer > 0."
            )
    _check(
        _is_int(cfg.space.top_k),
        f"space.top_k {cfg.space.top_k!r} must be an integer >= 0.",
    )
    cat_cost = cfg.space.cat_cost
    _check(
        _is_number(cat_cost) and cat_cost > 0,
        f"space.cat_cost {cat_cost!r} must be a number > 0.",
    )


def _validate_tree(tree: DictConfig) -> None:
    """Check the `tree` section."""
    _check(
        _is_int(tree.n_min, 2),
        f"tree.n_min {tree.n_min!r} must be an integer > 1.",
    )
    _check(
        _is_int(tree.max_k, 2),
        f"tree.max_k {tree.max_k!r} must be an integer > 1.",
    )
    _check(
        _is_int(tree.k_neighbours, 1),
        f"tree.k_neighbours {tree.k_neighbours!r} must be an integer > 0.",
    )
    _check(
        _is_number(tree.leaf_cost)
        and math.isfinite(tree.leaf_cost)
        and tree.leaf_cost >= 0,
        f"tree.leaf_cost {tree.leaf_cost!r} must be a number >= 0.",
    )
    _check(
        _is_number(tree.d_weight)
        and math.isfinite(tree.d_weight)
        and tree.d_weight >= 0,
        f"tree.d_weight {tree.d_weight!r} must be a number >= 0.",
    )


def _validate_runtime(runtime: DictConfig) -> None:
    """Check the `runtime` section."""
    _check(
        isinstance(runtime.device, str)
        and re.fullmatch(r"cpu|mps|cuda(:\d+)?", runtime.device) is not None,
        f"runtime.device {runtime.device!r} invalid; "
        "valid: 'cpu', 'cuda', 'cuda:<n>', 'mps'.",
    )
    _check(
        _is_int(runtime.n_jobs, -math.inf) and runtime.n_jobs != 0,
        f"runtime.n_jobs {runtime.n_jobs!r} must be a non-zero integer (-1: all).",
    )
    _check(
        _is_int(runtime.num_workers),
        f"runtime.num_workers {runtime.num_workers!r} must be an integer >= 0.",
    )
    _check(
        isinstance(runtime.pin_memory, bool),
        f"runtime.pin_memory {runtime.pin_memory!r} must be true or false.",
    )


def _split_options(values: str) -> list[str]:
    """Split `values` on the commas outside square brackets."""
    options, depth, start = [], 0, 0
    for i, char in enumerate(values):
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
        elif char == "," and depth == 0:
            options.append(values[start:i])
            start = i + 1
    return [*options, values[start:]]


def sweep(items: Sequence[str]) -> list[list[str]]:
    """Expand `key=v1,v2,...` items into override lists.

    Values are split on the commas outside square brackets, so a list
    value can be swept too: `key=[1,2],[3]`.

    Args:
        items: The `--sweep` items, each `key=value1,value2,...`.

    Returns:
        One list of `key=value` overrides per combination of the swept
        values, the first item varying slowest.

    Raises:
        ValueError: An item has no key or an empty value.
    """
    axes = []
    for item in items:
        key, _, values = item.partition("=")
        options = _split_options(values)
        if not key or "" in options:
            raise ValueError(f"--sweep {item!r} is not key=value1,value2,...")
        axes.append([f"{key}={option}" for option in options])
    return [list(combo) for combo in itertools.product(*axes)]

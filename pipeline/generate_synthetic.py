"""Write the synthetic test dataset, a CSV of labelled rows.

Run `python -m pipeline.generate_synthetic [--rows N]`. The CSV goes to
`resources/raw_data/synthetic/synthetic_test.csv`: 20 numerical and two
categorical features, a `label` and the `true_subgroup` each row was
drawn from. A difficulty ladder of sub-groups makes the classes
overlap by known margins. About 300 Inf and 200 NaN values are
injected. `--rows` sets the total row count (default 69,500).
"""

__all__ = []

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

_OUTPUT_PATH = Path("resources/raw_data/synthetic/synthetic_test.csv")

_DEFAULT_CLASS_SIZES: dict[str, int] = {
    "class_1": 22_000,
    "class_2": 9_000,
    "class_3": 7_000,
    "class_4": 6_000,
    "class_5": 5_000,
    "class_6": 5_000,
    "class_7": 5_000,
    "class_8": 5_000,
    "class_9": 5_000,
    "class_10": 500,
}

_DEFAULT_TOTAL = sum(_DEFAULT_CLASS_SIZES.values())
_RARE_CLASS_SIZE = 500

_CAT1_VALUES = ["A", "B", "C"]
_CAT2_VALUES = ["X", "Y", "Z", "W"]

_N_FEATURES = 20
_BASE_MU = 5.0
_BASE_SIGMA = 1.0
_MIX_SIGMA = 1.5

_SEED = 42
_RNG = np.random.default_rng(_SEED)


@dataclass(frozen=True)
class _ClassSpec:
    """The core of one class: feature means, spread, categoricals.

    Attributes:
        mu: The core mean of the features that differ from the baseline.
        sigma: The core standard deviation of those features.
        cat_1: The probabilities of the `cat_1` values.
        cat_2: The probabilities of the `cat_2` values.
    """

    mu: dict[str, float] = field(default_factory=dict)
    sigma: float = _BASE_SIGMA
    cat_1: tuple[float, ...] = (1 / 3, 1 / 3, 1 / 3)
    cat_2: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)


@dataclass(frozen=True)
class _Rung:
    """One step of the difficulty ladder.

    Attributes:
        name: The sub-group name.
        share: The share of the class rows the step takes.
        margin: The distance from the pair midpoint, in mixing sigmas.
        blend: How far the categoricals move to the pair's mixture.
    """

    name: str
    share: float
    margin: float
    blend: float


_CANONICAL_SHARE = 0.34

_LADDER = (
    _Rung("clear", 0.28, 1.80, 0.25),
    _Rung("evasive", 0.24, 0.90, 0.65),
    _Rung("mimicry", 0.14, 0.25, 1.00),
)

_CLASSES: dict[str, _ClassSpec] = {
    "class_1": _ClassSpec(
        mu={"num_18": 24.0, "num_19": 20.0, "num_20": 18.0},
        cat_1=(0.60, 0.30, 0.10),
        cat_2=(0.40, 0.35, 0.15, 0.10),
    ),
    "class_2": _ClassSpec(
        mu={"num_1": 25.0, "num_2": 22.0, "num_3": 20.0},
        cat_1=(0.80, 0.15, 0.05),
        cat_2=(0.05, 0.10, 0.10, 0.75),
    ),
    "class_3": _ClassSpec(
        mu={"num_1": 24.0, "num_2": 20.0, "num_4": 22.0},
        cat_1=(0.75, 0.20, 0.05),
        cat_2=(0.05, 0.10, 0.35, 0.50),
    ),
    "class_4": _ClassSpec(
        mu={"num_5": 30.0, "num_6": 28.0, "num_7": 25.0},
        cat_1=(0.50, 0.40, 0.10),
        cat_2=(0.50, 0.35, 0.10, 0.05),
    ),
    "class_5": _ClassSpec(
        mu={"num_8": 20.0, "num_9": 18.0, "num_10": 22.0},
        cat_1=(0.40, 0.50, 0.10),
        cat_2=(0.30, 0.50, 0.10, 0.10),
    ),
    "class_6": _ClassSpec(
        mu={"num_8": 22.0, "num_9": 14.0, "num_11": 20.0},
        sigma=2.0,
        cat_1=(0.35, 0.50, 0.15),
        cat_2=(0.30, 0.50, 0.10, 0.10),
    ),
    "class_7": _ClassSpec(
        mu={"num_12": 28.0, "num_13": 25.0, "num_14": 22.0},
        cat_1=(0.55, 0.35, 0.10),
        cat_2=(0.35, 0.30, 0.20, 0.15),
    ),
    "class_8": _ClassSpec(
        mu={"num_16": 11.0, "num_17": 10.0},
        cat_1=(0.70, 0.20, 0.10),
        cat_2=(0.02, 0.03, 0.05, 0.90),
    ),
    "class_9": _ClassSpec(
        mu={"num_1": 18.0, "num_2": 16.0, "num_15": 15.0},
        cat_1=(0.60, 0.30, 0.10),
        cat_2=(0.70, 0.10, 0.10, 0.10),
    ),
    "class_10": _ClassSpec(),
}

_OVERLAPS = (
    ("class_1", "class_2"),
    ("class_1", "class_3"),
    ("class_1", "class_4"),
    ("class_1", "class_7"),
    ("class_1", "class_8"),
    ("class_1", "class_9"),
    ("class_2", "class_3"),
    ("class_5", "class_6"),
)


def _draw_normal(mu: float, sigma: float, n_rows: int) -> np.ndarray:
    """Draw `n_rows` normal samples."""
    return _RNG.normal(loc=mu, scale=sigma, size=n_rows)


def _draw_categorical(
    values: Sequence[str], weights: Sequence[float], n_rows: int
) -> np.ndarray:
    """Draw `n_rows` categorical samples with the given weights."""
    return _RNG.choice(values, size=n_rows, p=weights)


def _baseline_features(n_rows: int) -> dict[str, np.ndarray]:
    """Draw every numerical feature at the shared baseline N(5, 1)."""
    return {
        f"num_{i}": _draw_normal(_BASE_MU, _BASE_SIGMA, n_rows)
        for i in range(1, _N_FEATURES + 1)
    }


def _adversaries(label: str) -> list[str]:
    """Return the classes sharing a difficulty corridor with `label`.

    They come in declaration order.
    """
    return [
        second if first == label else first
        for first, second in _OVERLAPS
        if label in (first, second)
    ]


def _rung_center(
    mu: Mapping[str, float], adversary_mu: Mapping[str, float], margin: float
) -> dict[str, float]:
    """Return means `margin` mixing sigmas from the pair midpoint.

    The means sit on the own-class side.
    """
    keys = sorted(set(mu) | set(adversary_mu))
    own = np.array([mu.get(key, _BASE_MU) for key in keys])
    adversary = np.array([adversary_mu.get(key, _BASE_MU) for key in keys])
    direction = own - adversary
    norm = float(np.linalg.norm(direction))
    offset = margin * _MIX_SIGMA * direction / norm if norm > 0 else 0.0
    return dict(zip(keys, (own + adversary) / 2 + offset))


def _blend_weights(
    weights: Sequence[float], adversary_weights: Sequence[float], blend: float
) -> tuple[float, ...]:
    """Move a categorical profile toward the pair's mixture.

    The profile moves `blend` of the way to the shared 50/50 mixture.
    """
    mixed = [
        (1 - blend / 2) * weight + (blend / 2) * adversary_weight
        for weight, adversary_weight in zip(weights, adversary_weights)
    ]
    total = sum(mixed)
    return tuple(weight / total for weight in mixed)


def _split_counts(n_rows: int, shares: Sequence[float]) -> list[int]:
    """Split `n_rows` across shares, the remainder to the first."""
    total = sum(shares)
    counts = [int(n_rows * share / total) for share in shares]
    counts[0] += n_rows - sum(counts)
    return counts


def _subgroup_frame(
    label: str,
    subgroup: str,
    n_rows: int,
    center: Mapping[str, float],
    sigma: float,
    cat_1: Sequence[float],
    cat_2: Sequence[float],
) -> pd.DataFrame:
    """Draw one sub-group: baseline features, `center` overridden.

    The frame also holds the categoricals, the label and the sub-group.
    """
    features = _baseline_features(n_rows)
    for col, mu in center.items():
        features[col] = _draw_normal(mu, sigma, n_rows)
    dataset = pd.DataFrame(features)
    dataset["cat_1"] = _draw_categorical(_CAT1_VALUES, cat_1, n_rows)
    dataset["cat_2"] = _draw_categorical(_CAT2_VALUES, cat_2, n_rows)
    dataset["label"] = label
    dataset["true_subgroup"] = subgroup
    return dataset


def _generate_class(label: str, n_rows: int) -> pd.DataFrame:
    """Draw a class: its core and a ladder toward each overlap."""
    spec = _CLASSES[label]
    adversaries = _adversaries(label)
    if not adversaries:
        return _subgroup_frame(
            label, "canonical", n_rows, spec.mu, spec.sigma, spec.cat_1, spec.cat_2
        )

    plan = [
        (
            "canonical",
            _CANONICAL_SHARE,
            spec.mu,
            spec.sigma,
            spec.cat_1,
            spec.cat_2,
        )
    ]
    for adversary in adversaries:
        adversary_spec = _CLASSES[adversary]
        for rung in _LADDER:
            plan.append(
                (
                    rung.name,
                    rung.share / len(adversaries),
                    _rung_center(spec.mu, adversary_spec.mu, rung.margin),
                    _MIX_SIGMA,
                    _blend_weights(spec.cat_1, adversary_spec.cat_1, rung.blend),
                    _blend_weights(spec.cat_2, adversary_spec.cat_2, rung.blend),
                )
            )

    counts = _split_counts(n_rows, [share for _, share, *_ in plan])
    return pd.concat(
        [
            _subgroup_frame(label, name, count, center, sigma, cat_1, cat_2)
            for (name, _, center, sigma, cat_1, cat_2), count in zip(plan, counts)
            if count > 0
        ],
        ignore_index=True,
    )


def _inject_edge_cases(dataset: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with ~300 Inf and ~200 NaN in numerical cells."""
    num_cols = [col for col in dataset.columns if col.startswith("num_")]
    dataset = dataset.copy()

    inf_idx = _RNG.choice(dataset.index, size=300, replace=False)
    inf_cols = _RNG.choice(["num_7", "num_8"], size=300)
    for idx, col in zip(inf_idx, inf_cols):
        dataset.at[idx, col] = np.inf

    nan_idx = _RNG.choice(dataset.index, size=200, replace=False)
    nan_cols = _RNG.choice(num_cols, size=200)
    for idx, col in zip(nan_idx, nan_cols):
        dataset.at[idx, col] = np.nan

    return dataset


def _compute_sizes(total_rows: int) -> dict[str, int]:
    """Scale the class sizes to `total_rows`, keeping class_10 rare."""
    scalable_total = _DEFAULT_TOTAL - _RARE_CLASS_SIZE
    scale = (total_rows - _RARE_CLASS_SIZE) / scalable_total
    return {
        label: _RARE_CLASS_SIZE if label == "class_10" else max(1, round(size * scale))
        for label, size in _DEFAULT_CLASS_SIZES.items()
    }


def _build_dataset(total_rows: int) -> pd.DataFrame:
    """Build the full shuffled dataset with edge cases injected.

    Args:
        total_rows: The target row count; class_10 keeps its 500 rows.

    Returns:
        The dataset, with a fresh index.
    """
    sizes = _compute_sizes(total_rows)
    dataset = pd.concat(
        [_generate_class(label, sizes[label]) for label in _CLASSES],
        ignore_index=True,
    )
    dataset = _inject_edge_cases(dataset)
    return dataset.sample(frac=1, random_state=_SEED).reset_index(drop=True)


def main() -> None:
    """Build the synthetic dataset, write it and print its summary."""
    parser = argparse.ArgumentParser(description="Generate synthetic test dataset.")
    parser.add_argument(
        "--rows",
        type=int,
        default=_DEFAULT_TOTAL,
        help=f"Target total row count (default: {_DEFAULT_TOTAL:,}). "
        f"class_10 is always kept at {_RARE_CLASS_SIZE} rows to preserve "
        "the rare-class filter test.",
    )
    args = parser.parse_args()

    print(f"Generating synthetic dataset ({args.rows:,} rows)...")
    dataset = _build_dataset(args.rows)

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(_OUTPUT_PATH, index=False)

    num_cols = [col for col in dataset.columns if col.startswith("num_")]
    inf_count = int((dataset[num_cols] == np.inf).sum().sum())

    print(f"Saved:     {_OUTPUT_PATH}")
    print(f"Shape:     {dataset.shape}")
    print(f"NaN count: {dataset.isna().sum().sum()}")
    print(f"Inf count: {inf_count}")
    print("\nClass distribution:")
    print(dataset["label"].value_counts().sort_index().to_string())
    print("\nSub-group distribution:")
    print(dataset.groupby("label")["true_subgroup"].value_counts().to_string())


if __name__ == "__main__":
    main()

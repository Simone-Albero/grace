"""src.config: every shipped yaml loads, a bad value fails up front."""

import pytest
import yaml
from conftest import ROOT

from src.config import DATASET_DIR, dataset_names, load_config


@pytest.mark.parametrize("dataset", dataset_names())
def test_every_dataset_loads(dataset):
    cfg = load_config(dataset)
    assert cfg.name == dataset
    assert cfg.num_cols


def _method(tmp_path, **changes) -> str:
    method = yaml.safe_load((ROOT / "configs/method/default.yaml").read_text())
    for dotted, value in changes.items():
        section, key = dotted.split("__")
        method[section][key] = value
    path = tmp_path / "method.yaml"
    path.write_text(yaml.safe_dump(method))
    return str(path)


@pytest.mark.parametrize("key", ["name", "params"])
def test_the_dropped_clustering_keys_are_unknown(tmp_path, key):
    with pytest.raises(ValueError, match=f"clustering.{key}"):
        load_config(
            "synthetic_test", method=_method(tmp_path, **{f"clustering__{key}": 1})
        )


def test_the_kdn_reference_is_gone(tmp_path):
    method = yaml.safe_load((ROOT / "configs/method/default.yaml").read_text())
    method["graph"] = {"max_samples": 1000}
    path = tmp_path / "method.yaml"
    path.write_text(yaml.safe_dump(method))
    with pytest.raises(ValueError, match="unknown keys"):
        load_config("synthetic_test", method=str(path))


def test_degenerate_counts_fail(tmp_path):
    with pytest.raises(ValueError, match="min_cluster_floor"):
        load_config(
            "synthetic_test",
            method=_method(tmp_path, clustering__min_cluster_floor=0),
        )
    with pytest.raises(ValueError, match="vote_k"):
        load_config("synthetic_test", method=_method(tmp_path, clustering__vote_k=0))
    with pytest.raises(ValueError, match="min_cluster_floor"):
        load_config(
            "synthetic_test",
            method=_method(tmp_path, clustering__min_cluster_floor=1),
        )
    with pytest.raises(ValueError, match="split_separation"):
        load_config(
            "synthetic_test", method=_method(tmp_path, clustering__split_separation=0)
        )
    with pytest.raises(ValueError, match="stability_repeats"):
        load_config(
            "synthetic_test", method=_method(tmp_path, clustering__stability_repeats=-1)
        )
    with pytest.raises(ValueError, match="overlap_separation"):
        load_config(
            "synthetic_test",
            method=_method(tmp_path, clustering__overlap_separation=0.0),
        )


def test_degenerate_preparation_fails(tmp_path):
    with pytest.raises(ValueError, match="cat_cost"):
        load_config("synthetic_test", method=_method(tmp_path, prepare__cat_cost=0))
    raw = yaml.safe_load((DATASET_DIR / "synthetic_test.yaml").read_text())
    path = tmp_path / "top_n.yaml"
    path.write_text(yaml.safe_dump({**raw, "top_n": 0}))
    with pytest.raises(ValueError, match="top_n"):
        load_config(str(path))


def test_a_missing_yaml_path_is_named_as_given(tmp_path):
    missing = tmp_path / "nope.yaml"
    with pytest.raises(FileNotFoundError, match=f"{missing}$"):
        load_config("synthetic_test", method=str(missing))


def test_unknown_top_level_keys_fail(tmp_path):
    raw = yaml.safe_load((ROOT / "configs/method/default.yaml").read_text())
    path = tmp_path / "old.yaml"
    path.write_text(yaml.safe_dump({**raw, "balance": "undersample"}))
    with pytest.raises(ValueError, match="balance"):
        load_config("synthetic_test", method=str(path))
    raw = yaml.safe_load((DATASET_DIR / "synthetic_test.yaml").read_text())
    path.write_text(yaml.safe_dump({**raw, "split": {"train": 1.0}}))
    with pytest.raises(ValueError, match="split"):
        load_config(str(path))
    path.write_text(yaml.safe_dump({**raw, "hash_buckets": 16}))
    with pytest.raises(ValueError, match="hash_buckets"):
        load_config(str(path))


def test_the_duplicates_column_cannot_be_configured(tmp_path):
    raw = yaml.safe_load((DATASET_DIR / "synthetic_test.yaml").read_text())
    path = tmp_path / "clash.yaml"
    path.write_text(yaml.safe_dump({**raw, "num_cols": ["n_duplicates"]}))
    with pytest.raises(ValueError, match="n_duplicates"):
        load_config(str(path))
    path.write_text(yaml.safe_dump({**raw, "cat_cols": ["n_missing"]}))
    with pytest.raises(ValueError, match="n_missing"):
        load_config(str(path))


def test_quality_values_out_of_range_fail(tmp_path):
    with pytest.raises(ValueError, match="candidate_separation"):
        load_config(
            "synthetic_test",
            method=_method(tmp_path, quality__candidate_separation=0),
        )


@pytest.mark.parametrize("key", ["k", "highlight"])
def test_the_dropped_quality_keys_are_unknown(tmp_path, key):
    with pytest.raises(ValueError, match=f"quality.{key}"):
        load_config(
            "synthetic_test", method=_method(tmp_path, **{f"quality__{key}": 1})
        )


def test_the_quality_thresholds_and_group_are_checked(tmp_path):
    with pytest.raises(ValueError, match="quality.max_group"):
        load_config("synthetic_test", method=_method(tmp_path, quality__max_group=0))
    raw = yaml.safe_load((ROOT / "configs/method/default.yaml").read_text())
    thresholds = raw["quality"]["thresholds"]
    for bad in (
        {**thresholds, "kdn": 0.1},
        {k: v for k, v in thresholds.items() if k != "error"},
        {**thresholds, "error": 0.0},
        {**thresholds, "pdd": 1.5},
    ):
        path = tmp_path / "thresholds.yaml"
        path.write_text(
            yaml.safe_dump({**raw, "quality": {**raw["quality"], "thresholds": bad}})
        )
        with pytest.raises(ValueError, match="quality.thresholds"):
            load_config("synthetic_test", method=str(path))


def test_shortcut_cols_are_unknown(tmp_path):
    raw = yaml.safe_load((DATASET_DIR / "synthetic_test.yaml").read_text())
    path = tmp_path / "shortcuts.yaml"
    path.write_text(yaml.safe_dump({**raw, "shortcut_cols": ["num_1"]}))
    with pytest.raises(ValueError, match="unknown keys"):
        load_config(str(path))

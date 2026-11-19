"""src.config: every shipped yaml loads, a bad value fails up front."""

from datetime import datetime

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


def _dataset(tmp_path, **changes) -> str:
    dataset = yaml.safe_load((DATASET_DIR / "synthetic_test.yaml").read_text())
    dataset.update(changes)
    path = tmp_path / "dataset.yaml"
    path.write_text(yaml.safe_dump(dataset))
    return str(path)


def test_max_class_rows_is_optional_and_not_under_min_class_count(tmp_path):
    assert load_config(_dataset(tmp_path)).max_class_rows is None
    assert load_config(_dataset(tmp_path, max_class_rows=None)).max_class_rows is None
    capped = load_config(_dataset(tmp_path, min_class_count=50, max_class_rows=50))
    assert capped.max_class_rows == 50
    for bad in (49, 0, 1.5):
        with pytest.raises(ValueError, match="max_class_rows"):
            load_config(_dataset(tmp_path, min_class_count=50, max_class_rows=bad))


def test_the_clustering_section_is_gone(tmp_path):
    method = yaml.safe_load((ROOT / "configs/method/default.yaml").read_text())
    method["clustering"] = {"split_separation": 4.0}
    path = tmp_path / "method.yaml"
    path.write_text(yaml.safe_dump(method))
    with pytest.raises(ValueError, match="unknown keys"):
        load_config("synthetic_test", method=str(path))


@pytest.mark.parametrize(
    "key",
    [
        "min_cluster_floor",
        "vote_k",
        "separation_grid",
        "min_stability",
        "stability_repeats",
    ],
)
def test_the_dropped_cell_keys_are_unknown(tmp_path, key):
    with pytest.raises(ValueError, match=f"cells.{key}"):
        load_config("synthetic_test", method=_method(tmp_path, **{f"cells__{key}": 1}))


def test_the_kdn_reference_is_gone(tmp_path):
    method = yaml.safe_load((ROOT / "configs/method/default.yaml").read_text())
    method["graph"] = {"max_samples": 1000}
    path = tmp_path / "method.yaml"
    path.write_text(yaml.safe_dump(method))
    with pytest.raises(ValueError, match="unknown keys"):
        load_config("synthetic_test", method=str(path))


@pytest.mark.parametrize("key", ["cells__max_fit_samples", "quality__rival_rows"])
def test_degenerate_counts_fail(tmp_path, key):
    with pytest.raises(ValueError, match=key.replace("__", ".")):
        load_config("synthetic_test", method=_method(tmp_path, **{key: 0}))


@pytest.mark.parametrize("key", ["max_rows", "min_rows"])
def test_the_cell_sizes_are_gone(tmp_path, key):
    # the cells are as small as the measures allow: tau and confidence say it
    with pytest.raises(ValueError, match=f"cells.{key}"):
        load_config("synthetic_test", method=_method(tmp_path, **{f"cells__{key}": 25}))


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


@pytest.mark.parametrize("value", [0.0, 0.6])
def test_tau_is_a_share_of_rows_under_one_half(tmp_path, value):
    with pytest.raises(ValueError, match="quality.tau"):
        load_config("synthetic_test", method=_method(tmp_path, quality__tau=value))


@pytest.mark.parametrize(
    "key",
    [
        "k",
        "highlight",
        "thresholds",
        "candidate_separation",
        "diversity_pairs",
        "max_rows_per_region",
        "max_group",
        "time_share",
    ],
)
def test_the_dropped_quality_keys_are_unknown(tmp_path, key):
    with pytest.raises(ValueError, match=f"quality.{key}"):
        load_config(
            "synthetic_test", method=_method(tmp_path, **{f"quality__{key}": 1})
        )


def test_the_quality_group_is_checked(tmp_path):
    with pytest.raises(ValueError, match="quality.rival_rows"):
        load_config("synthetic_test", method=_method(tmp_path, quality__rival_rows=0))


def test_the_default_method_loads():
    method = load_config("synthetic_test").method
    # the fewest rows a cell can be measured on: 2 x 25 scored at tau 0.1, 95%
    assert method.min_rows == 50
    assert method.quality.tau == 0.1 and method.quality.rival_rows == 100
    assert method.quality.confidence == 0.95


@pytest.mark.parametrize("value", [0.5, 1.0, 0.3])
def test_confidence_is_a_level_over_one_half(tmp_path, value):
    with pytest.raises(ValueError, match="quality.confidence"):
        load_config(
            "synthetic_test", method=_method(tmp_path, quality__confidence=value)
        )


def test_shortcut_cols_are_unknown(tmp_path):
    raw = yaml.safe_load((DATASET_DIR / "synthetic_test.yaml").read_text())
    path = tmp_path / "shortcuts.yaml"
    path.write_text(yaml.safe_dump({**raw, "shortcut_cols": ["num_1"]}))
    with pytest.raises(ValueError, match="unknown keys"):
        load_config(str(path))


def test_the_provenance_block_is_read_and_optional():
    provenance = load_config("synthetic_test").provenance
    assert provenance.columns() == {
        "src_ip": "src_ip",
        "src_port": "src_port",
        "dst_ip": "dst_ip",
        "dst_port": "dst_port",
        "time": "timestamp",
    }
    assert provenance.time_format == "%Y-%m-%d %H:%M:%S%.f"
    netflow = load_config("cic_2018_v2").provenance
    assert list(netflow.columns()) == ["src_ip", "src_port", "dst_ip", "dst_port"]
    assert load_config("letter_recognition").provenance is None


@pytest.mark.parametrize(
    "block, match",
    [
        ({"time": "timestamp"}, "time_format"),
        ({"src_ip": "src_ip", "time_format": "%Y"}, "time_format"),
        ({"host": "src_ip"}, "provenance.host"),
        ({"src_ip": None}, "names no column"),
    ],
)
def test_a_bad_provenance_block_fails(tmp_path, block, match):
    raw = yaml.safe_load((DATASET_DIR / "synthetic_test.yaml").read_text())
    path = tmp_path / "provenance.yaml"
    path.write_text(yaml.safe_dump({**raw, "provenance": block}))
    with pytest.raises(ValueError, match=match):
        load_config(str(path))


@pytest.mark.parametrize("kind", ["num_cols", "cat_cols"])
@pytest.mark.parametrize("column", ["src_ip", "src_port", "timestamp"])
def test_the_source_and_the_time_are_never_features(tmp_path, kind, column):
    raw = yaml.safe_load((DATASET_DIR / "synthetic_test.yaml").read_text())
    path = tmp_path / "feature.yaml"
    path.write_text(yaml.safe_dump({**raw, kind: [*(raw[kind] or []), column]}))
    with pytest.raises(ValueError, match=column):
        load_config(str(path))


def test_the_destination_may_be_a_feature_too(tmp_path):
    raw = yaml.safe_load((DATASET_DIR / "synthetic_test.yaml").read_text())
    path = tmp_path / "feature.yaml"
    path.write_text(yaml.safe_dump({**raw, "cat_cols": [*raw["cat_cols"], "dst_port"]}))
    cfg = load_config(str(path))
    assert "dst_port" in cfg.cat_cols and cfg.provenance.dst_port == "dst_port"


def test_the_scenario_block_is_read_and_optional():
    scenario = load_config("synthetic_test").scenario
    assert scenario.benign == "class_1"
    assert scenario.background_ports == (53, 123)
    assert list(scenario.attacks) == ["class_4"]
    attack = scenario.attacks["class_4"]
    assert attack.src_ip[0] == "18.3.0.0" and len(attack.src_ip) == 20
    assert attack.time == ((datetime(2018, 2, 20, 11), datetime(2018, 2, 20, 12)),)
    assert load_config("letter_recognition").scenario is None
    for name in ("cic_2018", "cic_2018_broken"):
        assert 53 in load_config(name).scenario.background_ports


def _with_scenario(tmp_path, scenario, provenance=...) -> str:
    raw = yaml.safe_load((DATASET_DIR / "synthetic_test.yaml").read_text())
    raw["scenario"] = scenario
    if provenance is not ...:
        raw["provenance"] = provenance
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump(raw))
    return str(path)


@pytest.mark.parametrize(
    "scenario, provenance, match",
    [
        ({"background_ports": [53]}, ..., "benign"),
        ({"benign": "class_1", "hosts": []}, ..., "scenario.hosts"),
        ({"benign": "class_1", "attacks": {"x": {"port": 1}}}, ..., "attacks.x.port"),
        ({"benign": "class_1", "background_ports": ["dns"]}, ..., "background_ports"),
        (
            {"benign": "class_1", "background_ports": [53]},
            {"src_ip": "src_ip"},
            "dst_port",
        ),
        (
            {"benign": "class_1", "attacks": {"x": {"src_ip": ["1.1.1.1"]}}},
            {"dst_port": "dst_port"},
            "src_ip",
        ),
        (
            {
                "benign": "class_1",
                "attacks": {
                    "x": {"time": [["2018-02-20 11:00:00", "2018-02-20 12:00:00"]]}
                },
            },
            {"src_ip": "src_ip"},
            "time",
        ),
        (
            {
                "benign": "class_1",
                "attacks": {
                    "x": {"time": [["2018-02-20 12:00:00", "2018-02-20 11:00:00"]]}
                },
            },
            ...,
            "before",
        ),
        ({"benign": "class_1", "background_ports": [53]}, None, "provenance"),
    ],
)
def test_a_bad_scenario_block_fails(tmp_path, scenario, provenance, match):
    with pytest.raises(ValueError, match=match):
        load_config(_with_scenario(tmp_path, scenario, provenance))

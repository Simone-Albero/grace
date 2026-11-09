"""src.config: every shipped yaml loads, a bad value fails up front."""

import pytest
import yaml
from conftest import ROOT

from src.config import dataset_names, load_config, model_names


@pytest.mark.parametrize("dataset", dataset_names())
def test_every_dataset_loads(dataset):
    cfg = load_config(dataset)
    assert cfg.name == dataset
    assert cfg.num_cols


@pytest.mark.parametrize("model", model_names())
def test_every_model_loads(model):
    assert load_config("synthetic_test", model).model.name == model


def _method(tmp_path, **changes) -> str:
    method = yaml.safe_load((ROOT / "configs/method/default.yaml").read_text())
    for dotted, value in changes.items():
        section, key = dotted.split("__")
        method[section][key] = value
    path = tmp_path / "method.yaml"
    path.write_text(yaml.safe_dump(method))
    return str(path)


def test_unknown_clustering_param_fails(tmp_path):
    with pytest.raises(ValueError, match="bogus"):
        load_config(
            "synthetic_test", method=_method(tmp_path, clustering__params={"bogus": 1})
        )


def test_unknown_clustering_fails(tmp_path):
    with pytest.raises(ValueError, match="dbscan"):
        load_config(
            "synthetic_test", method=_method(tmp_path, clustering__name="dbscan")
        )


def test_split_shares_must_sum_to_one(tmp_path):
    raw = yaml.safe_load((ROOT / "configs/datasets/synthetic_test.yaml").read_text())
    raw["split"]["test"] = 0.5
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="sum to 1"):
        load_config(str(path))


def test_degenerate_counts_fail(tmp_path):
    with pytest.raises(ValueError, match="min_cluster_floor"):
        load_config(
            "synthetic_test",
            method=_method(tmp_path, clustering__min_cluster_floor=0),
        )
    with pytest.raises(ValueError, match="n_outer_splits"):
        load_config(
            "synthetic_test", method=_method(tmp_path, regressor__n_outer_splits=1)
        )


def test_a_missing_yaml_path_is_named_as_given(tmp_path):
    missing = tmp_path / "nope.yaml"
    with pytest.raises(FileNotFoundError, match=f"{missing}$"):
        load_config("synthetic_test", method=str(missing))

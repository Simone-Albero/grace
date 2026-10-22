"""End-to-end stage tests on the synthetic data."""

import math
from collections.abc import Mapping
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from stages import SPLITS
from stages.runner import STAGES, _config_slice, run
from tests.helpers import fast_cfg, query


def _scalar(root: Path, sql: str) -> object:
    return query(root / "grace.db", sql)[0][0]


def test_outputs(root: Path) -> None:
    """Check that every stage of the full run wrote its outputs."""
    ids = {stage: i for i, stage in enumerate(STAGES, start=1)}
    assert dict(query(root / "grace.db", "SELECT stage, id FROM executions")) == ids
    metrics = {
        (stage, name): value
        for stage, name, value in query(
            root / "grace.db",
            "SELECT stage, name, value FROM metrics "
            "JOIN executions ON executions.id = execution_id",
        )
    }
    n_regions = int(metrics["regions", "n_regions"])
    n_test = int(metrics["split", "n_test"])
    assert _scalar(root, "SELECT COUNT(*) FROM regions") == n_regions
    assert _scalar(root, "SELECT SUM(n_test) FROM regions") == n_test
    centroids = pd.read_parquet(
        root / f"artifacts/regions/{ids['regions']}/centroids.parquet"
    )
    assert len(centroids) == n_regions

    assignments = pd.read_parquet(
        root / f"artifacts/regions/{ids['regions']}/assignments.parquet"
    )
    for split in SPLITS:
        rows = assignments.loc[assignments["split"] == split, "row"]
        assert rows.tolist() == list(range(int(metrics["split", f"n_{split}"])))
    assert assignments["region_id"].isin(centroids["region_id"]).all()

    _check_complexity(root, metrics, n_regions)

    assert _scalar(root, "SELECT COUNT(*) FROM failures") == n_regions
    assert _scalar(root, "SELECT SUM(n_eval) FROM failures") == n_test
    out_of_range = "failure_rate < 0 OR failure_rate > 1"
    assert _scalar(root, f"SELECT COUNT(*) FROM failures WHERE {out_of_range}") == 0
    assert _scalar(root, "SELECT COUNT(*) FROM class_metrics") == 9
    assert ("classify", "accuracy") in metrics
    _check_baselines(root, metrics)
    _check_regress(root, metrics, ids)
    _check_render(root, ids)


def _check_complexity(
    root: Path, metrics: Mapping[tuple[str, str], float], n_regions: int
) -> None:
    measures = {
        scope: query(
            root / "grace.db",
            f"SELECT DISTINCT measure FROM complexity WHERE scope = '{scope}'",
        )
        for scope in ("region", "class")
    }
    assert sorted(measures["region"]) == sorted(measures["class"])
    n_measures = len(measures["region"])
    assert metrics["complexity", "n_measures"] == n_measures
    for scope, count in (("region", n_regions), ("class", 9)):
        sql = f"SELECT COUNT(*) FROM complexity WHERE scope = '{scope}'"
        assert _scalar(root, sql) == count * n_measures


def _check_baselines(root: Path, metrics: Mapping[tuple[str, str], float]) -> None:
    n_regions = int(metrics["regions", "n_regions"])
    assert _scalar(root, "SELECT COUNT(*) FROM baselines") == n_regions
    n_used = int(metrics["baselines", "n_regions_used"])
    assert _scalar(root, "SELECT SUM(used) FROM baselines") == n_used
    outside = "mcp NOT BETWEEN 0 AND 1 OR atc NOT BETWEEN 0 AND 1"
    assert _scalar(root, f"SELECT COUNT(*) FROM baselines WHERE {outside}") == 0
    for predictor in ("mcp", "atc"):
        for score in ("spearman", "r2", "mae", "mse"):
            assert ("baselines", f"{predictor}_{score}") in metrics
    assert ("baselines", "atc_threshold") in metrics


def _check_regress(
    root: Path, metrics: Mapping[tuple[str, str], float], ids: Mapping[str, int]
) -> None:
    n_regions = int(metrics["regions", "n_regions"])
    assert _scalar(root, "SELECT COUNT(*) FROM estimates") == n_regions
    unused = (
        "SELECT COUNT(*) FROM estimates JOIN baselines USING (region_id) "
        f"WHERE estimates.execution_id = {ids['regress']} "
        f"AND baselines.execution_id = {ids['baselines']} "
        "AND (regressor IS NULL) != (used = 0)"
    )
    assert _scalar(root, unused) == 0
    for score in ("spearman", "r2", "mae", "mse"):
        assert ("regress", f"regressor_{score}") in metrics
    folds = int(metrics["regress", "n_outer_splits"]) + 1
    assert _scalar(root, "SELECT COUNT(*) FROM regressor_folds") == folds
    assert _scalar(root, "SELECT COUNT(*) FROM regressor_folds WHERE fold = -1") == 1
    n_features = int(metrics["regress", "n_features"])
    assert _scalar(root, "SELECT COUNT(*) FROM importances") == n_features
    model = joblib.load(root / f"artifacts/regress/{ids['regress']}/model.joblib")
    assert len(model.feature_names_in_) == n_features
    X = pd.DataFrame(np.zeros((2, n_features)), columns=model.feature_names_in_)
    assert model.predict(X).shape == (2,)


def _check_render(root: Path, ids: Mapping[str, int]) -> None:
    expected = {
        "render": {
            "classification/confusion_matrix",
            "baselines/predicted_vs_observed",
            "failures/rate_by_class",
        },
        "render_regress": {"regress/predicted_vs_observed", "regress/scores"},
    }
    for stage, names in expected.items():
        sql = f"SELECT name, file FROM figures WHERE execution_id = {ids[stage]}"
        figures = dict(query(root / "grace.db", sql))
        assert names <= figures.keys()
        folder = root / f"artifacts/{stage}/{ids[stage]}"
        assert all((folder / file).exists() for file in figures.values())
    sql = f"SELECT name FROM figures WHERE execution_id = {ids['render']}"
    render = [name for (name,) in query(root / "grace.db", sql)]
    assert not any(name.startswith("regress/") for name in render)
    # Naive Bayes trains no epochs, so there is no loss history.
    assert "classification/training_loss" not in render


def test_second_run_is_cached(root: Path) -> None:
    """Check that a second identical run links every stage as cached."""
    run_id = run(list(STAGES), fast_cfg(root), force=False)
    cached = query(
        root / "grace.db", f"SELECT cached FROM run_stages WHERE run_id = {run_id}"
    )
    assert cached == [(1,)] * len(STAGES)


def test_figure_format_reruns_render_only(root: Path) -> None:
    """Check that a new figure_format reruns only the render stages."""
    run_id = run(
        ["render", "render_regress"], fast_cfg(root, "figure_format=png"), force=False
    )
    cached = query(
        root / "grace.db",
        f"SELECT stage, cached FROM run_stages WHERE run_id = {run_id}",
    )
    rerun = {stage for stage, was_cached in cached if not was_cached}
    assert rerun == {"render", "render_regress"}
    files = query(
        root / "grace.db",
        "SELECT file FROM figures JOIN run_stages USING (execution_id) "
        f"WHERE run_id = {run_id}",
    )
    assert files and all(file.endswith(".png") for (file,) in files)


def test_classify_key_ignores_dl_and_runtime_keys(root: Path) -> None:
    """Check that the ML classify key skips DL-only and runtime keys."""
    other = fast_cfg(
        root, "loss=cross_entropy", "runtime.n_jobs=2", "fit.training.epochs=3"
    )
    run_id = run(["classify"], other, force=False)
    cached = query(
        root / "grace.db",
        f"SELECT cached FROM run_stages WHERE run_id = {run_id} AND stage = 'classify'",
    )
    assert cached == [(1,)]

    ml = _config_slice("classify", fast_cfg(root))
    assert "loss" not in ml and "runtime" not in ml and "training" not in ml["fit"]
    dl = _config_slice("classify", fast_cfg(root, "classifier=mlp"))
    assert "loss" in dl and "runtime" not in dl and "training" in dl["fit"]


def test_routing_is_label_free(tmp_path: Path) -> None:
    """Check that routing val and test rows ignores their labels."""
    cfg = fast_cfg(tmp_path)
    run(["regions"], cfg, force=False)
    first = pd.read_parquet(tmp_path / "artifacts/regions/3/assignments.parquet")

    rng = np.random.default_rng(0)
    for split in ("val", "test"):
        path = tmp_path / f"artifacts/prepare/2/{split}.parquet"
        rows = pd.read_parquet(path)
        rows["label"] = rng.permutation(rows["label"].to_numpy())
        rows.to_parquet(path, index=False)
    # Else prepare would rerun, undoing the edit.
    (tmp_path / "artifacts/prepare/2/sizes.json").unlink()
    run(["regions"], cfg, force=True)
    second = pd.read_parquet(tmp_path / "artifacts/regions/4/assignments.parquet")
    assert_frame_equal(second, first)


def test_mlp_trains_and_saves_only_model_pt(root: Path) -> None:
    """Check that an MLP trains its epochs and saves only model.pt."""
    run(
        ["classify"],
        fast_cfg(root, "classifier=mlp", "fit.training.epochs=2"),
        force=False,
    )
    execution = _scalar(root, "SELECT MAX(id) FROM executions WHERE stage = 'classify'")
    metrics = dict(
        query(
            root / "grace.db",
            f"SELECT name, value FROM metrics WHERE execution_id = {execution}",
        )
    )
    # Patience 5 cannot stop it within 2 epochs.
    steps = 2 * math.ceil(metrics["n_fit"] / 512)
    history = f"SELECT COUNT(*) FROM history WHERE execution_id = {execution}"
    assert _scalar(root, history) == steps
    model_dir = root / f"artifacts/classify/{execution}/model"
    assert [path.name for path in model_dir.iterdir()] == ["model.pt"]
    n_classes = _scalar(root, "SELECT COUNT(*) FROM classes")
    assert metrics["accuracy"] > 1 / n_classes


def test_mlp_trains_on_a_one_row_last_batch(root: Path) -> None:
    """Check that an MLP trains on a one-row last batch."""
    n_fit = _scalar(
        root,
        "SELECT value FROM metrics JOIN executions ON executions.id = execution_id "
        "WHERE stage = 'classify' AND name = 'n_fit' ORDER BY executions.id LIMIT 1",
    )
    # As n_fit % (n_fit - 1) == 1, the last batch of every epoch has
    # one row.
    batch_size = f"fit.training.dataloader.batch_size={int(n_fit) - 1}"
    cfg = fast_cfg(root, "classifier=mlp", "fit.training.epochs=1", batch_size)
    run(["classify"], cfg, force=False)

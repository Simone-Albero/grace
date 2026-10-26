"""End-to-end stage tests on the synthetic data."""

import json
import math
from collections.abc import Mapping
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf
from pandas.testing import assert_frame_equal

import stages.baselines as baselines_stage
import stages.partitions as partitions_stage
import stages.regress as regress_stage
from pipeline.runner import STAGES, _config_slice, run
from src.regressor import region_descriptors
from stages import SPLITS
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

    n_val = int(metrics["split", "n_val"])
    n_train = int(metrics["split", "n_train"])
    # The granularity noise term reads the val budget, not the test one.
    assert metrics["regions", "eval_rows_per_train_row"] == pytest.approx(
        n_val / n_train
    )
    assert _scalar(root, "SELECT COUNT(*) FROM failures") == n_regions
    assert _scalar(root, "SELECT SUM(n_val) FROM failures") == n_val
    assert _scalar(root, "SELECT SUM(n_test) FROM failures") == n_test
    out_of_range = "val_rate NOT BETWEEN 0 AND 1 OR test_rate NOT BETWEEN 0 AND 1"
    assert _scalar(root, f"SELECT COUNT(*) FROM failures WHERE {out_of_range}") == 0
    assert _scalar(root, "SELECT COUNT(*) FROM class_metrics") == 9
    assert ("classify", "accuracy") in metrics
    _check_baselines(root, metrics)
    _check_regress(root, metrics, ids)
    _check_render(root, ids)
    _check_partitions(root, metrics)


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
    outside = (
        "val NOT BETWEEN 0 AND 1 OR mcp NOT BETWEEN 0 AND 1 OR "
        "atc NOT BETWEEN 0 AND 1"
    )
    assert _scalar(root, f"SELECT COUNT(*) FROM baselines WHERE {outside}") == 0
    assert _scalar(root, "SELECT COUNT(*) FROM baselines WHERE val IS NULL") == 0
    for predictor in ("val", "mcp", "atc"):
        for score in ("spearman", "r2", "mae", "mse"):
            assert ("baselines", f"{predictor}_{score}") in metrics
    assert ("baselines", "atc_threshold") in metrics


def _check_regress(
    root: Path, metrics: Mapping[tuple[str, str], float], ids: Mapping[str, int]
) -> None:
    n_regions = int(metrics["regions", "n_regions"])
    assert _scalar(root, "SELECT COUNT(*) FROM estimates") == n_regions
    # A region is predicted iff it is used, has a val row, or both.
    unpredicted = (
        "SELECT COUNT(*) FROM estimates "
        "JOIN baselines USING (region_id) JOIN failures USING (region_id) "
        f"WHERE estimates.execution_id = {ids['regress']} "
        f"AND baselines.execution_id = {ids['baselines']} "
        f"AND failures.execution_id = {ids['failures']} "
        "AND (regressor IS NULL) != (used = 0 AND n_val = 0)"
    )
    assert _scalar(root, unpredicted) == 0
    # The folds cover exactly the regions with a val row.
    fold_coverage = (
        "SELECT COUNT(*) FROM estimates JOIN failures USING (region_id) "
        f"WHERE estimates.execution_id = {ids['regress']} "
        f"AND failures.execution_id = {ids['failures']} "
        "AND (fold IS NULL) != (n_val = 0)"
    )
    assert _scalar(root, fold_coverage) == 0
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
        "classification/confusion_matrix",
        "baselines/predicted_vs_observed",
        "failures/rate_by_class",
        "regress/predicted_vs_observed",
        "regress/scores",
    }
    sql = f"SELECT name, file FROM figures WHERE execution_id = {ids['render']}"
    figures = dict(query(root / "grace.db", sql))
    assert expected <= figures.keys()
    folder = root / f"artifacts/render/{ids['render']}"
    assert all((folder / file).exists() for file in figures.values())
    # Naive Bayes trains no epochs, so there is no loss history.
    assert "classification/training_loss" not in figures


def _check_partitions(root: Path, metrics: Mapping[tuple[str, str], float]) -> None:
    db_path = root / "grace.db"
    brier_identity = "ABS(brier - (uncertainty - resolution + reliability)) > 1e-6"
    assert _scalar(root, f"SELECT COUNT(*) FROM partitions WHERE {brier_identity}") == 0
    assert (
        _scalar(root, "SELECT COUNT(*) FROM partitions WHERE n_groups > n_target") == 0
    )

    bin_arms = "arm IN ('confidence', 'predicted_class')"
    bins_have_a_spearman = (
        f"SELECT COUNT(*) FROM partitions WHERE {bin_arms} AND kdn_spearman IS NOT NULL"
    )
    assert _scalar(root, bins_have_a_spearman) == 0
    others_lack_one = (
        "SELECT COUNT(*) FROM partitions "
        f"WHERE NOT ({bin_arms}) AND kdn_spearman IS NULL"
    )
    assert _scalar(root, others_lack_one) == 0

    # A selected row's loss is at most the least loss of its arm's own
    # level sweep (intra's per-class pick can do strictly better).
    selected_rows = query(
        db_path,
        "SELECT arm, draw, loss FROM partitions WHERE rule = 'selected' "
        "AND arm IN ('intra', 'inter', 'null')",
    )
    for arm, draw, loss in selected_rows:
        least_level_loss = query(
            db_path,
            "SELECT MIN(loss) FROM partitions "
            f"WHERE arm = '{arm}' AND rule = 'level' AND draw = {draw}",
        )[0][0]
        assert loss <= least_level_loss + 1e-9

    n_val = int(metrics["split", "n_val"])
    n_test = int(metrics["split", "n_test"])
    group_counts = query(
        db_path,
        "SELECT arm, rule, SUM(n_val), SUM(n_test) FROM partition_groups "
        "GROUP BY arm, rule",
    )
    assert group_counts
    for _, _, total_val, total_test in group_counts:
        assert total_val == n_val
        assert total_test == n_test


def test_second_run_is_cached(root: Path) -> None:
    """Check that a second identical run links every stage as cached."""
    run_id = run(list(STAGES), fast_cfg(root), force=False)
    cached = query(
        root / "grace.db", f"SELECT cached FROM run_stages WHERE run_id = {run_id}"
    )
    assert cached == [(1,)] * len(STAGES)


def test_figure_format_reruns_render_only(root: Path) -> None:
    """Check that a new figure_format reruns only render."""
    run_id = run(["render"], fast_cfg(root, "figure_format=png"), force=False)
    cached = query(
        root / "grace.db",
        f"SELECT stage, cached FROM run_stages WHERE run_id = {run_id}",
    )
    rerun = {stage for stage, was_cached in cached if not was_cached}
    assert rerun == {"render"}
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
    prepared = tmp_path / "artifacts/prepare/2"
    sizes = json.loads((prepared / "sizes.json").read_text())
    for split in ("val", "test"):
        path = prepared / f"{split}.parquet"
        rows = pd.read_parquet(path)
        rows["label"] = rng.permutation(rows["label"].to_numpy())
        rows.to_parquet(path, index=False)
        sizes[path.name] = path.stat().st_size
    # Else prepare would rerun, undoing the edit.
    (prepared / "sizes.json").write_text(json.dumps(sizes))
    run(["regions"], cfg, force=True)
    second = pd.read_parquet(tmp_path / "artifacts/regions/4/assignments.parquet")
    assert_frame_equal(second, first)


def test_classify_never_trains_on_val_labels(tmp_path: Path) -> None:
    """Check that every classify prediction ignores val's labels."""
    cfg = fast_cfg(
        tmp_path,
        "classifier=mlp",
        "fit.training.epochs=4",
        "fit.training.early_stopping.patience=1",
    )
    run(["classify"], cfg, force=False)
    first = pd.read_parquet(tmp_path / "artifacts/classify/3/predictions.parquet")

    rng = np.random.default_rng(0)
    prepared = tmp_path / "artifacts/prepare/2"
    sizes = json.loads((prepared / "sizes.json").read_text())
    path = prepared / "val.parquet"
    rows = pd.read_parquet(path)
    rows["label"] = rng.permutation(rows["label"].to_numpy())
    rows.to_parquet(path, index=False)
    sizes[path.name] = path.stat().st_size
    # Else prepare would rerun, undoing the edit.
    (prepared / "sizes.json").write_text(json.dumps(sizes))
    run(["classify"], cfg, force=True)
    second = pd.read_parquet(tmp_path / "artifacts/classify/4/predictions.parquet")
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
    # n_fit is the rows the model actually trained on: for DL, the
    # holdout is already out. Patience 5 cannot stop it within 2 epochs.
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
    holdout_frac = fast_cfg(root).fit.training.early_stopping.holdout
    # A guess at the post-holdout row count, matching
    # train_test_split's own ceil rounding of the held-out share, sized
    # so n_train % batch_size == 1; checked against the real count
    # below instead of trusted, so a rounding drift fails loudly.
    batch_size = int(n_fit) - math.ceil(holdout_frac * n_fit) - 1
    cfg = fast_cfg(
        root,
        "classifier=mlp",
        "fit.training.epochs=1",
        f"fit.training.dataloader.batch_size={batch_size}",
    )
    run(["classify"], cfg, force=False)
    execution = _scalar(root, "SELECT MAX(id) FROM executions WHERE stage = 'classify'")
    n_train = _scalar(
        root,
        f"SELECT value FROM metrics WHERE execution_id = {execution} "
        "AND name = 'n_fit'",
    )
    assert n_train % batch_size == 1


class _FakeContext:
    """A minimal stage Context double: DB tables, an output folder."""

    def __init__(
        self,
        tables: Mapping[tuple[str, str], pd.DataFrame],
        cfg: DictConfig,
        out: Path,
    ) -> None:
        self.cfg = cfg
        self.runtime = OmegaConf.create({"n_jobs": 1})
        self.out = out
        self._tables = tables
        self.pending: list[tuple[str, object]] = []

    def read(self, table: str, stage: str) -> pd.DataFrame:
        return self._tables[(stage, table)].copy()

    def write(self, table: str, frame: pd.DataFrame) -> None:
        self.pending.append((table, frame))

    def save_metrics(self, values: Mapping[str, float]) -> None:
        self.pending.append(("metrics", dict(values)))


def test_baseline_val_rate_falls_back_to_the_global_rate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Check a region with no val row takes the global val rate."""
    failures = pd.DataFrame(
        {
            "region_id": [1, 2, 3],
            "n_val": [10, 0, 20],
            "n_val_error": [1, 0, 6],
            "val_rate": [0.1, None, 0.3],
            "n_test": [50, 50, 50],
            "n_test_error": [5, 5, 5],
            "test_rate": [0.1, 0.1, 0.1],
        }
    )
    monkeypatch.setattr(
        baselines_stage,
        "_rates",
        lambda ctx: (
            0.5,
            pd.Series([0.3, 0.3, 0.3], index=[1, 2, 3]),
            pd.Series([0.4, 0.4, 0.4], index=[1, 2, 3]),
        ),
    )
    cfg = OmegaConf.create({"failure_regressor": {"min_eval_support": 1}})
    ctx = _FakeContext({("failures", "failures"): failures}, cfg, tmp_path)

    baselines_stage.run(ctx)

    table = next(frame for name, frame in ctx.pending if name == "baselines").set_index(
        "region_id"
    )
    # The global rate is (1 + 0 + 6) / (10 + 0 + 20) = 7/30.
    assert table.loc[1, "val"] == pytest.approx(0.1)
    assert table.loc[3, "val"] == pytest.approx(0.3)
    assert table.loc[2, "val"] == pytest.approx(7 / 30)


def _regress_tables() -> dict[tuple[str, str], pd.DataFrame]:
    """Build a four-region toy input for `stages.regress`.

    Region 1: used, has a val row. Region 2: used, no val row (the
    refit-only case). Region 3: has a val row, not used (too few test
    rows). Region 4: used, has a val row.
    """
    regions = pd.DataFrame({"region_id": [1, 2, 3, 4], "class_id": [10, 10, 20, 20]})
    complexity = pd.DataFrame(
        [
            {"scope": "region", "scope_id": 1, "measure": "m1", "value": 1.0},
            {"scope": "region", "scope_id": 2, "measure": "m1", "value": 2.0},
            {"scope": "region", "scope_id": 3, "measure": "m1", "value": 3.0},
            {"scope": "region", "scope_id": 4, "measure": "m1", "value": 4.0},
            {"scope": "class", "scope_id": 10, "measure": "m1", "value": 1.5},
            {"scope": "class", "scope_id": 20, "measure": "m1", "value": 3.5},
        ]
    )
    failures = pd.DataFrame(
        {
            "region_id": [1, 2, 3, 4],
            "n_val": [20, 0, 15, 10],
            "n_val_error": [2, 0, 6, 1],
            "val_rate": [0.1, None, 0.4, 0.1],
            "n_test": [30, 25, 2, 40],
            "n_test_error": [9, 5, 1, 4],
            "test_rate": [0.3, 0.2, 0.5, 0.1],
        }
    )
    baselines = pd.DataFrame({"region_id": [1, 2, 3, 4], "used": [1, 1, 0, 1]})
    metrics = pd.DataFrame(
        {
            "name": [
                f"{predictor}_{score}"
                for predictor in ("val", "mcp", "atc")
                for score in ("spearman", "r2", "mae", "mse")
            ],
            "value": [0.0] * 12,
        }
    )
    return {
        ("regions", "regions"): regions,
        ("complexity", "complexity"): complexity,
        ("failures", "failures"): failures,
        ("baselines", "baselines"): baselines,
        ("baselines", "metrics"): metrics,
    }


def test_regress_predicts_a_used_region_without_a_val_row(tmp_path: Path) -> None:
    """Check the refit, not the CV, predicts a used, val-less region."""
    cfg = OmegaConf.create(
        {
            "failure_regressor": {
                "params": {},
                "param_grid": {"n_estimators": [5]},
                "n_outer_splits": 2,
                "n_inner_splits": 0,
                "n_iter": 1,
            },
            "seed": 0,
        }
    )
    ctx = _FakeContext(_regress_tables(), cfg, tmp_path)

    regress_stage.run(ctx)

    estimates = next(
        frame for name, frame in ctx.pending if name == "estimates"
    ).set_index("region_id")

    # Region 2: used, no val row -> predicted by the refit, fold is NA,
    # and its value is exactly the refit model's own prediction (not,
    # say, a coincidental fallback to some other rate).
    assert pd.isna(estimates.loc[2, "fold"])
    model = joblib.load(tmp_path / "model.joblib")
    tables = _regress_tables()
    regions = tables[("regions", "regions")].set_index("region_id")
    descriptors = region_descriptors(
        tables[("complexity", "complexity")], regions["class_id"]
    )
    expected = np.clip(
        model.predict(descriptors.loc[[2], model.feature_names_in_]), 0.0, 1.0
    )[0]
    assert estimates.loc[2, "regressor"] == pytest.approx(expected)

    # Regions 1 and 4: used, out-of-fold predicted, a real fold index.
    for region in (1, 4):
        assert not pd.isna(estimates.loc[region, "fold"])
        assert 0.0 <= estimates.loc[region, "regressor"] <= 1.0

    # Region 3: has a val row (trained) but is not used: it still gets
    # an out-of-fold estimate, just never scored against test.
    assert not pd.isna(estimates.loc[3, "fold"])
    assert 0.0 <= estimates.loc[3, "regressor"] <= 1.0

    metrics = next(values for name, values in ctx.pending if name == "metrics")
    assert np.isfinite(metrics["regressor_mae"])


def test_predicted_class_n_target_counts_a_class_missing_from_val() -> None:
    """Check n_target widens for a class seen on test but not val.

    Without the widening, `n_groups <= n_target` (the invariant
    `_check_partitions` relies on) would fail whenever a class is
    predicted on test but not on val: `assign_class_bins` still gives
    it a group, uncounted by the bin rule's own `n_bins`.
    """
    val = partitions_stage._Predictions(
        predicted=np.array([0, 0, 2, 2]),
        confidence=np.array([0.6, 0.65, 0.3, 0.35]),
        error=np.array([0.0, 1.0, 0.0, 1.0]),
    )
    test = partitions_stage._Predictions(
        predicted=np.array([0, 1, 2, 2]),
        confidence=np.array([0.62, 0.5, 0.32, 0.4]),
        error=np.array([0.0, 1.0, 1.0, 0.0]),
    )

    rows, _ = partitions_stage._score_bins("predicted_class", val, test, reference=0.5)

    for row in rows:
        assert row["n_groups"] <= row["n_target"]
        # The one test row of the class val never predicts is the only
        # one in a group with no val row. Routing val and test through
        # two separate calls (the id-aliasing bug) instead aliases a
        # val-only class's rows onto it too, raising this to 0.5.
        assert row["share_no_val"] == pytest.approx(0.25)

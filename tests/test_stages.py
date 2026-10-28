"""End-to-end stage tests on the synthetic data."""

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import stages.classify as classify_stage
import stages.crossfit as crossfit_stage
import stages.tree as tree_stage
from pipeline.runner import STAGES, _config_slice, run
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
    assert _scalar(root, "SELECT COUNT(*) FROM class_metrics") == 10
    assert ("classify", "accuracy") in metrics
    _check_space(root, metrics, ids)
    _check_crossfit(root, metrics, ids)
    _check_tree(root, metrics, ids)
    _check_complexity(root, metrics)
    _check_baselines(root, metrics, ids)
    _check_evaluate(root, metrics, ids)
    _check_render(root, ids)


def _check_complexity(root: Path, metrics: Mapping[tuple[str, str], float]) -> None:
    measures = {
        scope: query(
            root / "grace.db",
            f"SELECT DISTINCT measure FROM complexity WHERE scope = '{scope}'",
        )
        for scope in ("leaf", "class")
    }
    assert sorted(measures["leaf"]) == sorted(measures["class"])
    n_measures = len(measures["leaf"])
    assert metrics["complexity", "n_measures"] == n_measures
    sql = "SELECT COUNT(*) FROM complexity WHERE scope = 'class'"
    assert _scalar(root, sql) == 10 * n_measures
    # Every measured leaf is a leaf of the tree; a thin one has no row.
    unknown = (
        "SELECT COUNT(DISTINCT scope_id) FROM complexity WHERE scope = 'leaf' "
        "AND scope_id NOT IN (SELECT node_id FROM tree_nodes WHERE status = 'leaf')"
    )
    assert _scalar(root, unknown) == 0
    assert metrics["complexity", "n_populations"] > 0
    assert metrics["complexity", "n_unmeasured_rows"] >= 0


def _check_space(
    root: Path, metrics: Mapping[tuple[str, str], float], ids: Mapping[str, int]
) -> None:
    projected = pd.read_parquet(root / f"artifacts/space/{ids['space']}/z.parquet")
    for split in SPLITS:
        rows = projected.loc[projected["split"] == split, "row"]
        assert rows.tolist() == list(range(int(metrics["split", f"n_{split}"])))
    coordinates = projected.filter(like="z_")
    # One discriminant direction fewer than the classes.
    assert coordinates.shape[1] == metrics["space", "n_components"] == 9
    assert np.isfinite(coordinates.to_numpy()).all()


def _check_tree(
    root: Path, metrics: Mapping[tuple[str, str], float], ids: Mapping[str, int]
) -> None:
    rows = pd.read_parquet(root / f"artifacts/tree/{ids['tree']}/rows.parquet")
    for split in SPLITS:
        picked = rows[rows["split"] == split]
        assert picked["row"].tolist() == list(
            range(int(metrics["split", f"n_{split}"]))
        )
    assert rows[["q_leaf", "q", "confidence"]].stack().between(0, 1).all()
    for name in ("d_scale", "mu_c", "sigma_c"):
        assert ("tree", name) in metrics
    # The correction's signal is the classifier's own top probability.
    predictions = pd.read_parquet(
        root / f"artifacts/classify/{ids['classify']}/predictions.parquet"
    )
    for split in ("val", "test"):
        top = (
            predictions[predictions["split"] == split]
            .sort_values("row")
            .filter(like="proba_")
            .max(axis=1)
        )
        tree_rows = rows[rows["split"] == split].sort_values("row")
        assert np.allclose(tree_rows["confidence"], top, atol=1e-6)
    leaves = pd.DataFrame(
        query(
            root / "grace.db",
            "SELECT node_id, n_train, n_val, q FROM tree_nodes "
            f"WHERE execution_id = {ids['tree']} AND status = 'leaf'",
        ),
        columns=["node_id", "n_train", "n_val", "q"],
    )
    assert len(leaves) == metrics["tree", "n_leaves"]
    assert leaves["q"].notna().all()
    assert leaves["n_val"].sum() == metrics["split", "n_val"]
    assert leaves["n_train"].sum() == metrics["split", "n_train"]
    assert set(rows.loc[rows["split"] == "test", "leaf"]) <= set(leaves["node_id"])


def _check_crossfit(
    root: Path, metrics: Mapping[tuple[str, str], float], ids: Mapping[str, int]
) -> None:
    oof = pd.read_parquet(root / f"artifacts/crossfit/{ids['crossfit']}/oof.parquet")
    n_train = int(metrics["split", "n_train"])
    assert oof["row"].tolist() == list(range(n_train))
    assert metrics["crossfit", "n_folds"] == oof["fold"].nunique()
    probabilities = oof.filter(like="proba_")
    assert probabilities.shape[1] == 10
    assert np.allclose(probabilities.sum(axis=1), 1, atol=1e-4)
    assert 0 < metrics["crossfit", "oof_error_rate"] < 1


def _check_baselines(
    root: Path, metrics: Mapping[tuple[str, str], float], ids: Mapping[str, int]
) -> None:
    scores = pd.read_parquet(
        root / f"artifacts/baselines/{ids['baselines']}/scores.parquet"
    )
    for split in ("val", "test"):
        rows = scores.loc[scores["split"] == split, "row"]
        assert rows.tolist() == list(range(int(metrics["split", f"n_{split}"])))
    predictors = ["val_rate", "confidence", "binned", "gbm"]
    assert scores[predictors].stack().between(0, 1).all()
    assert scores["val_rate"].nunique() == 1
    assert scores["val_rate"].iloc[0] == pytest.approx(
        metrics["baselines", "val_error_rate"], abs=1e-6
    )
    assert metrics["baselines", "n_gbm_rows"] == metrics["split", "n_train"]


def _check_evaluate(
    root: Path, metrics: Mapping[tuple[str, str], float], ids: Mapping[str, int]
) -> None:
    for predictor in (
        "tree",
        "tree_leaf",
        "val_rate",
        "confidence",
        "binned",
        "gbm",
    ):
        for score in ("mse_ratio", "binned_mse_ratio", "auroc", "ap_lift"):
            assert ("evaluate", f"{predictor}_{score}") in metrics
    # The val rate is every ratio's reference, and a random order.
    assert metrics["evaluate", "val_rate_mse_ratio"] == pytest.approx(1.0)
    assert metrics["evaluate", "val_rate_binned_mse_ratio"] == pytest.approx(1.0)
    assert metrics["evaluate", "val_rate_auroc"] == pytest.approx(0.5)
    assert metrics["evaluate", "val_rate_ap_lift"] == pytest.approx(1.0)
    assert metrics["evaluate", "n_test"] == metrics["split", "n_test"]
    assert (
        _scalar(root, "SELECT COUNT(*) FROM leaf_tests") == metrics["tree", "n_leaves"]
    )
    assert (
        _scalar(root, "SELECT SUM(n_test) FROM leaf_tests")
        == metrics["split", "n_test"]
    )


def _check_render(root: Path, ids: Mapping[str, int]) -> None:
    expected = {
        "classification/confusion_matrix",
        "predictors/mse_ratio",
        "predictors/binned_mse_ratio",
        "predictors/auroc",
        "predictors/ap_lift",
    }
    sql = f"SELECT name, file FROM figures WHERE execution_id = {ids['render']}"
    figures = dict(query(root / "grace.db", sql))
    assert expected <= figures.keys()
    folder = root / f"artifacts/render/{ids['render']}"
    assert all((folder / file).exists() for file in figures.values())
    # Naive Bayes trains no epochs, so there is no loss history.
    assert "classification/training_loss" not in figures


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
    assert "loss" not in ml and "runtime" not in ml and "fit" not in ml
    dl = _config_slice("classify", fast_cfg(root, "classifier=mlp"))
    assert "loss" in dl and "runtime" not in dl and "training" in dl["fit"]


def _shuffle_labels(root: Path, splits: Sequence[str]) -> None:
    """Shuffle the labels of `splits` in place; prepare stays cached."""
    rng = np.random.default_rng(0)
    prepared = root / "artifacts/prepare/2"
    sizes = json.loads((prepared / "sizes.json").read_text())
    for split in splits:
        path = prepared / f"{split}.parquet"
        rows = pd.read_parquet(path)
        rows["label"] = rng.permutation(rows["label"].to_numpy())
        rows.to_parquet(path, index=False)
        sizes[path.name] = path.stat().st_size
    # Else prepare would rerun, undoing the edit.
    (prepared / "sizes.json").write_text(json.dumps(sizes))


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

    _shuffle_labels(tmp_path, ("val",))
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


def test_crossfit_predicts_every_train_row_once(root: Path) -> None:
    """Check every train row has one out-of-fold prediction."""
    execution = _scalar(root, "SELECT id FROM executions WHERE stage = 'crossfit'")
    oof = pd.read_parquet(root / f"artifacts/crossfit/{execution}/oof.parquet")
    n_train = _scalar(
        root,
        "SELECT value FROM metrics JOIN executions ON executions.id = execution_id "
        "WHERE stage = 'split' AND name = 'n_train'",
    )
    assert oof["row"].is_unique and len(oof) == n_train
    sizes = oof["fold"].value_counts()
    assert sizes.max() - sizes.min() <= 10


def test_crossfit_refits_with_the_best_grid_params(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check every fold refits with the grid's chosen parameters."""
    cfg = fast_cfg(tmp_path, "classifier=logistic_regression")
    run(["classify"], cfg, force=False)
    chosen = json.loads(
        _scalar(tmp_path, "SELECT params FROM classifier_grid WHERE best = 1")
    )
    seen = []
    build = crossfit_stage.ml.build_pipeline

    def spy(name, params, num_cols, cat_cols):
        seen.append(dict(params))
        return build(name, params, num_cols, cat_cols)

    monkeypatch.setattr(crossfit_stage.ml, "build_pipeline", spy)
    run(["crossfit"], cfg, force=False)
    assert len(seen) == cfg.crossfit.n_folds
    assert all(params["C"] == chosen["C"] for params in seen)


def test_crossfit_dl_stops_on_its_own_fold_holdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check a DL fold never fits or stops on the rows it predicts."""
    cfg = fast_cfg(tmp_path, "classifier=mlp", "fit.training.epochs=1")
    run(["classify"], cfg, force=False)
    fits, held = [], []
    fit = crossfit_stage.dl.fit_classifier
    predict = crossfit_stage._predict_fold

    def fit_spy(rows, **kwargs):
        fits.append(set(rows.index))
        return fit(rows, **kwargs)

    def predict_spy(ctx, model, rows, **kwargs):
        held.append(set(rows.index))
        return predict(ctx, model, rows, **kwargs)

    monkeypatch.setattr(crossfit_stage.dl, "fit_classifier", fit_spy)
    monkeypatch.setattr(crossfit_stage, "_predict_fold", predict_spy)
    run(["crossfit"], cfg, force=False)
    assert len(fits) == len(held) == cfg.crossfit.n_folds
    assert all(not (fit_rows & held_rows) for fit_rows, held_rows in zip(fits, held))
    assert set().union(*held) == set(range(len(set().union(*held))))


@pytest.mark.parametrize("classifier", ["naive_bayes", "mlp"])
def test_crossfit_keys_cover_classify_keys(classifier: str) -> None:
    """Check crossfit's cache key holds all classify trains on."""
    cfg = fast_cfg(Path("unused"), f"classifier={classifier}")
    classify_keys = set(classify_stage.config_keys(cfg)) - {"grid_search"}
    assert classify_keys <= set(crossfit_stage.config_keys(cfg))


def test_tree_never_reads_test_labels(tmp_path: Path) -> None:
    """Check the tree and its probabilities ignore the test labels."""
    cfg = fast_cfg(tmp_path)
    run(["tree"], cfg, force=False)

    def built() -> tuple[pd.DataFrame, pd.DataFrame]:
        latest = _scalar(tmp_path, "SELECT MAX(id) FROM executions WHERE stage='tree'")
        rows = pd.read_parquet(tmp_path / f"artifacts/tree/{latest}/rows.parquet")
        nodes = pd.DataFrame(
            query(
                tmp_path / "grace.db",
                "SELECT node_id, status, n_val, q FROM tree_nodes "
                f"WHERE execution_id = {latest} ORDER BY node_id",
            )
        )
        return rows, nodes

    first_rows, first_nodes = built()
    _shuffle_labels(tmp_path, ("test",))
    run(["tree"], cfg, force=True)
    second_rows, second_nodes = built()
    assert_frame_equal(second_rows, first_rows)
    assert_frame_equal(second_nodes, first_nodes)


def test_tree_standardizes_with_the_train_statistics() -> None:
    """Check val and test are scaled by train's mean and std alone."""
    values = {
        "train": np.array([1.0, 2.0, 3.0, 4.0]),
        "val": np.array([10.0, 20.0]),
        "test": np.array([0.0]),
    }
    scaled, mean, std = tree_stage._standardize(values)
    assert mean == pytest.approx(2.5) and std == pytest.approx(values["train"].std())
    assert scaled["val"] == pytest.approx((values["val"] - 2.5) / std)
    values["val"] = values["val"] * 7
    assert tree_stage._standardize(values)[1:] == (mean, std)
    flat, _, flat_std = tree_stage._standardize(
        {"train": np.full(4, 3.0), "val": np.array([1.0, 2.0])}
    )
    assert flat_std == 0.0 and (flat["val"] == 0).all()


def test_tree_without_weight_grows_on_the_lda_space(tmp_path: Path) -> None:
    """Check d_weight=0 leaves the disagreement axis out of the tree."""
    run(["tree"], fast_cfg(tmp_path, "tree.d_weight=0"), force=False)
    latest = _scalar(tmp_path, "SELECT MAX(id) FROM executions WHERE stage='tree'")
    fitted = joblib.load(tmp_path / f"artifacts/tree/{latest}/tree.joblib")
    z_file = next((tmp_path / "artifacts/space").glob("*/z.parquet"))
    n_axes = pd.read_parquet(z_file).filter(like="z_").shape[1]
    assert fitted["scale"] == 0.0
    assert fitted["tree"].centroid.shape[1] == n_axes


def test_baselines_never_read_test_labels(tmp_path: Path) -> None:
    """Check the baselines' probabilities ignore the test labels."""
    cfg = fast_cfg(tmp_path)
    run(["baselines"], cfg, force=False)

    def built() -> pd.DataFrame:
        latest = _scalar(
            tmp_path, "SELECT MAX(id) FROM executions WHERE stage = 'baselines'"
        )
        return pd.read_parquet(
            tmp_path / f"artifacts/baselines/{latest}/scores.parquet"
        )

    first = built()
    _shuffle_labels(tmp_path, ("test",))
    run(["baselines"], cfg, force=True)
    assert_frame_equal(built(), first)

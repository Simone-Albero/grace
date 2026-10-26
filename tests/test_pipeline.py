"""Pipeline: config loading, validation, sweeps and stage runs."""

import fcntl
import os
import re
import shutil
from contextlib import closing
from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf

import stages.prepare
from pipeline.config import CONFIGS, load_config, sweep, validate
from pipeline.db import KILLED, RUNS_LOCK, connect, reap
from pipeline.runner import STAGES, _config_slice, plan, run
from stages import SPLITS, Context
from tests.helpers import fast_cfg, query


def _links(root: Path, run_id: int) -> dict[str, tuple[int, int]]:
    """Return {stage: (execution id, cached)} of a run."""
    rows = query(
        root / "grace.db",
        "SELECT stage, execution_id, cached FROM run_stages WHERE run_id = ?",
        (run_id,),
    )
    return {stage: (execution, cached) for stage, execution, cached in rows}


def test_cache_and_invalidation(tmp_path: Path) -> None:
    """Check that stages are cached and a rerun split invalidates."""
    cfg = fast_cfg(tmp_path)
    assert _links(tmp_path, run(["prepare"], cfg, force=False)) == {
        "split": (1, 0),
        "prepare": (2, 0),
    }
    assert _links(tmp_path, run(["prepare"], cfg, force=False)) == {
        "split": (1, 1),
        "prepare": (2, 1),
    }
    top_n = fast_cfg(tmp_path, "data.top_n=64")
    assert _links(tmp_path, run(["prepare"], top_n, force=False)) == {
        "split": (1, 1),
        "prepare": (3, 0),
    }
    assert _links(tmp_path, run(["split"], cfg, force=True)) == {"split": (4, 0)}
    # A re-executed split invalidates prepare through its execution id.
    assert _links(tmp_path, run(["prepare"], cfg, force=False)) == {
        "split": (4, 1),
        "prepare": (5, 0),
    }


def test_lost_outputs_and_forced_downstream(tmp_path: Path) -> None:
    """Check that lost outputs rerun and --force spares the upstream."""
    cfg = fast_cfg(tmp_path)
    run(["prepare"], cfg, force=False)
    # --force on a downstream target leaves the upstream cached.
    assert _links(tmp_path, run(["prepare"], cfg, force=True)) == {
        "split": (1, 1),
        "prepare": (3, 0),
    }
    (tmp_path / "artifacts/split/1/train.parquet").unlink()
    assert _links(tmp_path, run(["prepare"], cfg, force=False)) == {
        "split": (4, 0),
        "prepare": (5, 0),
    }
    # A cut file (same name, other size) is lost too.
    with (tmp_path / "artifacts/prepare/5/train.parquet").open("r+b") as file:
        file.truncate(100)
    assert _links(tmp_path, run(["prepare"], cfg, force=False)) == {
        "split": (4, 1),
        "prepare": (6, 0),
    }
    # Without its sizes.json an execution cannot be checked: lost too.
    (tmp_path / "artifacts/prepare/6/sizes.json").unlink()
    assert _links(tmp_path, run(["prepare"], cfg, force=False)) == {
        "split": (4, 1),
        "prepare": (7, 0),
    }


def test_killed_run_is_failed_by_the_next_run(tmp_path: Path) -> None:
    """Check that a run left `running` by a dead process is failed."""
    cfg = fast_cfg(tmp_path)
    run(["split"], cfg, force=False)
    path = tmp_path / "grace.db"
    with closing(connect(path)) as db:
        db.execute(
            "INSERT INTO runs (targets, config, status, started_at) "
            "VALUES ('[]', '{}', 'running', 'then')"
        )
        db.execute(
            "INSERT INTO executions "
            "(stage, key, config, inputs, dir, status, started_at) "
            "VALUES ('split', 'k', '{}', '{}', '', 'running', 'then')"
        )
        db.commit()
    run(["split"], cfg, force=False)
    assert query(path, "SELECT status, error FROM runs WHERE id = 2") == [
        ("failed", KILLED)
    ]
    assert query(path, "SELECT status, error FROM executions WHERE id = 2") == [
        ("failed", KILLED)
    ]
    assert query(path, "SELECT status FROM runs WHERE id = 3") == [("done",)]


def test_reap_spares_a_live_run(tmp_path: Path) -> None:
    """Check that a run whose process holds runs.lock stays running."""
    with closing(connect(tmp_path / "grace.db")) as db:
        db.execute(
            "INSERT INTO runs (targets, config, status, started_at) "
            "VALUES ('[]', '{}', 'running', 'now')"
        )
        db.commit()
    with (tmp_path / RUNS_LOCK).open("a") as lock:
        # Hold the lock as a second, live process would.
        fcntl.flock(lock, fcntl.LOCK_SH)
        reap(tmp_path)
    assert query(tmp_path / "grace.db", "SELECT status FROM runs") == [("running",)]


def test_csv_without_a_configured_column(tmp_path: Path) -> None:
    """Check that split names the CSV, the column and keys to fix."""
    raw = tmp_path / "raw.csv"
    frame = pd.read_csv(fast_cfg(tmp_path).data.path, nrows=50)
    frame.drop(columns="num_20").to_csv(raw, index=False)
    cfg = fast_cfg(tmp_path, f"data.path={raw}")
    with pytest.raises(ValueError, match=r"raw\.csv.*num_20.*data\.num_cols"):
        run(["split"], cfg, force=False)


def test_root_is_relocatable(tmp_path: Path) -> None:
    """Check that a moved root keeps its cache and relative dirs."""
    run(["prepare"], fast_cfg(tmp_path / "a"), force=False)
    shutil.move(tmp_path / "a", tmp_path / "b")
    run_id = run(["prepare"], fast_cfg(tmp_path / "b"), force=False)
    assert _links(tmp_path / "b", run_id) == {"split": (1, 1), "prepare": (2, 1)}
    rows = query(tmp_path / "b" / "grace.db", "SELECT stage, id, dir FROM executions")
    assert rows == [(stage, i, f"artifacts/{stage}/{i}") for stage, i, _ in rows]


def test_changed_raw_file_reruns_split(tmp_path: Path) -> None:
    """Check that a changed raw file reruns the split."""
    raw = tmp_path / "raw.csv"
    shutil.copy(fast_cfg(tmp_path).data.path, raw)
    cfg = fast_cfg(tmp_path, f"data.path={raw}")
    assert _links(tmp_path, run(["split"], cfg, force=False)) == {"split": (1, 0)}
    assert _links(tmp_path, run(["split"], cfg, force=False)) == {"split": (1, 1)}
    os.utime(raw, ns=(1, 1))
    assert _links(tmp_path, run(["split"], cfg, force=False)) == {"split": (2, 0)}


def test_prepare_outputs(tmp_path: Path) -> None:
    """Check that prepare's classes, splits and features match."""
    run(["prepare"], fast_cfg(tmp_path), force=False)
    with closing(connect(tmp_path / "grace.db")) as db:
        classes = pd.read_sql_query("SELECT * FROM classes WHERE execution_id = 2", db)
        assert classes["class_id"].tolist() == list(range(9))
        metrics = dict(
            db.execute("SELECT name, value FROM metrics WHERE execution_id = 1")
        )
        for name in SPLITS:
            path = tmp_path / "artifacts/prepare/2" / f"{name}.parquet"
            frame = pd.read_parquet(path)
            assert len(frame) == metrics[f"n_{name}"]
            assert frame["label"].isin(classes["class_id"]).all()
            assert classes[f"n_{name}"].sum() == len(frame)
        kinds = db.execute(
            "SELECT kind, COUNT(*) FROM features WHERE execution_id = 1 GROUP BY kind"
        ).fetchall()
        assert dict(kinds) == {"num": 20, "cat": 2}


def _failing_prepare(ctx: Context) -> None:
    """Queue a row, then fail, like a stage that crashes late."""
    ctx.write(
        "classes",
        pd.DataFrame(
            {"class_id": [0], "name": ["x"], "n_train": 1, "n_val": 1, "n_test": 1}
        ),
    )
    raise RuntimeError("boom")


def test_failed_stage_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check that a failed stage rolls back its rows and reruns."""
    cfg = fast_cfg(tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(stages.prepare, "run", _failing_prepare)
        with pytest.raises(RuntimeError, match="boom"):
            run(["prepare"], cfg, force=False)

    db_path = tmp_path / "grace.db"
    rows = query(db_path, "SELECT stage, status FROM executions ORDER BY id")
    assert rows == [("split", "done"), ("prepare", "failed")]
    assert query(db_path, "SELECT status FROM runs") == [("failed",)]
    assert query(db_path, "SELECT COUNT(*) FROM classes") == [(0,)]
    assert _links(tmp_path, run(["prepare"], cfg, force=False)) == {
        "split": (1, 1),
        "prepare": (3, 0),
    }


@pytest.mark.parametrize(
    "overrides, targets",
    [
        (["data=nope"], ["prepare"]),
        (["data.trian_frac=0.5"], ["prepare"]),
        (["distance=manhattan"], ["prepare"]),
        (["groups.data=nb15_v2"], ["prepare"]),
        (["root="], ["prepare"]),
        (["classifier.kind=xx"], ["prepare"]),
        (["classifier.name=nope"], ["prepare"]),
        (["loss.name=nope"], ["prepare"]),
        (["fit.training.epochs=0"], ["prepare"]),
        (["fit.training.early_stopping.min_delta=-1"], ["prepare"]),
        (["fit.training.early_stopping.holdout=0"], ["prepare"]),
        (["fit.training.early_stopping.holdout=1"], ["prepare"]),
        (["fit.training.max_grad_norm=0"], ["prepare"]),
        (["fit.balance=oversample"], ["prepare"]),
        (["runtime.num_workers=-1"], ["prepare"]),
        (["graph.k=0"], ["prepare"]),
        (["failure_regressor.n_outer_splits=1"], ["prepare"]),
        (["failure_regressor.param_grid.max_depth=5"], ["prepare"]),
        (["failure_regressor.n_iter=0"], ["prepare"]),
        ([], ["bogus"]),
    ],
)
def test_config_errors(
    tmp_path: Path, overrides: list[str], targets: list[str]
) -> None:
    """Check that a bad value fails before any stage runs."""
    with pytest.raises(ValueError):
        run(targets, fast_cfg(tmp_path, *overrides), force=False)


@pytest.mark.parametrize(
    "overrides, match",
    [
        (["grid_search.scoring=bogus"], "grid_search.scoring"),
        (
            ["classifier=random_forest", "classifier.grid.n_estimators=5"],
            "classifier.grid",
        ),
        (["optimizer.name=Bogus"], "optimizer.name"),
        (["optimizer.name=Optimizer"], "optimizer.name"),
        (["scheduler.name=Nope"], "scheduler.name"),
        (["scheduler.name=LRScheduler"], "scheduler.name"),
        (["fit.training.dataloader.batch_size=1"], "batch_size"),
        (["fit.training.early_stopping.dataloader.batch_size=0"], "batch_size"),
        (["runtime.device=cuda:x"], "runtime.device"),
        (["clustering.params=5"], "must be a mapping"),
        (
            ["clustering=hdbscan", "clustering.params.min_samples=[]"],
            "at least one",
        ),
        (["clustering.name=birch"], "pick it with <group>=<option>"),
        (["classifier.name=knn"], "pick it with <group>=<option>"),
        (["classifier.kind=ml"], "pick it with <group>=<option>"),
        (["groups.data=nb15_v2"], "pick it with <group>=<option>"),
        (["[groups][data]=nb15_v2"], "pick it with <group>=<option>"),
        (["seed=-1"], "seed"),
        # The regressor's searches take seed + fold, below 2**32.
        ([f"seed={2**31}"], "seed"),
        (["regressor_figures=maybe"], "regressor_figures"),
        (["data.num_cols=[num_1,num_1]"], "more than once"),
        (["loss.params.label_smoothing=1.0"], "label_smoothing"),
        (["loss=focal", "loss.params.gamma=-1"], "gamma"),
    ],
)
def test_config_errors_name_the_key(overrides: list[str], match: str) -> None:
    """Check that a config error names the offending key."""
    with pytest.raises(ValueError, match=re.escape(match)):
        load_config(["data=synthetic_test", *overrides])


def test_render_reads_the_regressor_only_when_asked() -> None:
    """Check that regressor_figures decides what render runs after."""
    with_regressor = load_config(["data=synthetic_test"])
    without_regressor = load_config(["data=synthetic_test", "regressor_figures=false"])
    assert plan(["render"], with_regressor) == list(STAGES)
    skipped = {"complexity", "regress"}
    assert plan(["render"], without_regressor) == [
        stage for stage in STAGES if stage not in skipped
    ]


def test_clustering_params_checked() -> None:
    """Check that pipeline-owned and unknown clustering params fail."""
    cfg = load_config(["data=synthetic_test"])
    OmegaConf.set_struct(cfg, False)
    for params, key in (({"n_clusters": [5]}, "n_clusters"), ({"bogus": 1}, "bogus")):
        cfg.clustering.params = params
        with pytest.raises(ValueError, match=key):
            validate(cfg)
    cfg = load_config(["data=synthetic_test", "clustering=hdbscan"])
    OmegaConf.set_struct(cfg, False)
    cfg.clustering.params = {"min_cluster_size": [5]}
    validate(cfg)


@pytest.mark.parametrize(
    "path", sorted((CONFIGS / "clustering").glob("*.yaml")), ids=lambda path: path.stem
)
def test_clustering_shared_keys(path: Path) -> None:
    """Check that each clustering YAML has a name and shared keys."""
    cfg = load_config(["data=synthetic_test", f"clustering={path.stem}"])
    shared = OmegaConf.load(CONFIGS / "config.yaml").clustering
    assert cfg.clustering.name == path.stem
    for key, value in shared.items():
        assert cfg.clustering[key] == value
    assert "max_regions" in _config_slice("regions", cfg)["clustering"]


@pytest.mark.parametrize(
    "path", sorted((CONFIGS / "data").glob("*.yaml")), ids=lambda path: path.stem
)
def test_every_data_config_validates(path: Path) -> None:
    """Check that every data config loads and validates."""
    load_config([f"data={path.stem}"])


def test_sweep() -> None:
    """Check that a sweep is the product of its values, lists too."""
    combos = sweep(["classifier=a,b", "distance=x,y,z"])
    assert len(combos) == 6
    assert combos[0] == ["classifier=a", "distance=x"]
    assert combos[-1] == ["classifier=b", "distance=z"]
    assert sweep([]) == [[]]
    min_samples = "clustering.params.min_samples"
    assert sweep([f"{min_samples}=[1,5],[15]"]) == [
        [f"{min_samples}=[1,5]"],
        [f"{min_samples}=[15]"],
    ]
    assert len(sweep(["x=[[1,2],[3]],[4]"])) == 2
    cfg = load_config(
        [
            "data=synthetic_test",
            "clustering=hdbscan",
            *sweep([f"{min_samples}=[1,5],[15]"])[0],
        ]
    )
    assert cfg.clustering.params.min_samples == [1, 5]
    for bad in ("classifier", "classifier=a,", "=a,b", "a=[1,2],"):
        with pytest.raises(ValueError, match="--sweep"):
            sweep([bad])

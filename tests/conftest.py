"""Shared fixtures: a small synthetic dataset through every stage."""

__all__ = ["N_DUPLICATES", "ROOT", "STAGE_ORDER", "Workspace", "make_workspace"]

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pytest
import yaml

from main import main

ROOT = Path(__file__).resolve().parents[1]
STAGE_ORDER = ("clean", "prepare", "regions", "complexity")
N_DUPLICATES = 52  # copies synthetic_csv appends


@dataclass(frozen=True)
class Workspace:
    """A dataset and a method yaml in a temp folder, and their outputs.

    Attributes:
        folder: The folder of the yamls, the outputs and the db.
        dataset: The dataset yaml.
        method: The method yaml.
        output_dir: The outputs of every stage.
        db_path: The SQLite file.
    """

    folder: Path
    dataset: Path
    method: Path
    output_dir: Path
    db_path: Path

    def run(self, stage: str) -> int:
        """Run one stage through main, as the CLI does.

        Args:
            stage: A key of STAGES.

        Returns:
            The exit code of main.
        """
        argv = ["--stage", stage, "--dataset", str(self.dataset)]
        argv += ["--method", str(self.method)]
        return main(argv)


def write_dataset_yaml(folder: Path, csv: Path, name: str = "toy") -> Path:
    """Write a dataset yaml of the synthetic CSV; outputs in folder."""
    base = yaml.safe_load((ROOT / "configs/datasets/synthetic_test.yaml").read_text())
    base.update(
        name=name,
        path=str(csv),
        num_cols=[*base["num_cols"], "flat"],
        min_class_count=600,
        top_n=3,  # cat_1 has 3 values, all slotted; cat_2 has 4, one "other"
        output_dir=str(folder / "out"),
        db_path=str(folder / "grace.db"),
    )
    path = folder / f"{name}.yaml"
    path.write_text(yaml.safe_dump(base, sort_keys=False))
    return path


def write_method_yaml(folder: Path) -> Path:
    """Write a method yaml of the defaults, scaled down for speed."""
    method = yaml.safe_load((ROOT / "configs/method/default.yaml").read_text())
    method["graph"]["max_samples"] = 1500
    method["clustering"]["max_regions"] = 120
    method["complexity"]["top_k_clusters"] = 3
    method["complexity"]["max_sample_per_region"] = 50
    method["complexity"]["max_sample_per_class"] = 300
    path = folder / "method.yaml"
    path.write_text(yaml.safe_dump(method, sort_keys=False))
    return path


@pytest.fixture(scope="session")
def synthetic_csv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 12000-row synthetic CSV; class_10 keeps 500 rows, too few.

    A constant column, flat, is added for stage clean to drop, and
    N_DUPLICATES copies at the end: rows 0-49 once more, row 50 twice
    more. Row 51 comes back under another class, which is not a copy.
    """
    csv = tmp_path_factory.mktemp("data") / "synthetic.csv"
    subprocess.run(
        [sys.executable, str(ROOT / "generate_synthetic.py"), "--rows", "12000"]
        + ["--out", str(csv)],
        check=True,
        capture_output=True,
    )
    rows = pl.read_csv(csv, infer_schema=False).with_columns(pl.lit("1").alias("flat"))
    other_class = (
        pl.when(pl.col("label") == "class_1")
        .then(pl.lit("class_2"))
        .otherwise(pl.lit("class_1"))
    )
    relabelled = rows.slice(51, 1).with_columns(other_class.alias("label"))
    copies = [rows.head(50), rows.slice(50, 1), rows.slice(50, 1)]
    pl.concat([rows, *copies, relabelled]).write_csv(csv)
    return csv


def make_workspace(folder: Path, csv: Path) -> Workspace:
    """Write the yamls of a workspace under folder."""
    dataset = write_dataset_yaml(folder, csv)
    return Workspace(
        folder=folder,
        dataset=dataset,
        method=write_method_yaml(folder),
        output_dir=folder / "out",
        db_path=folder / "grace.db",
    )


@pytest.fixture(scope="session")
def done(tmp_path_factory: pytest.TempPathFactory, synthetic_csv: Path) -> Workspace:
    """A workspace where every stage ran once, in order."""
    workspace = make_workspace(tmp_path_factory.mktemp("done"), synthetic_csv)
    for stage in STAGE_ORDER:
        workspace.run(stage)
    return workspace


@pytest.fixture
def copy_of_done(done: Workspace, tmp_path: Path) -> Workspace:
    """A private copy of the done workspace, safe to modify."""
    shutil.copytree(done.output_dir, tmp_path / "out")
    shutil.copy(done.db_path, tmp_path / "grace.db")
    csv = Path(yaml.safe_load(done.dataset.read_text())["path"])
    return make_workspace(tmp_path, csv)

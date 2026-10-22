"""Shared fixtures: a fast config and a root with every stage."""

from pathlib import Path

import pytest

from pipeline.runner import STAGES, run
from tests.helpers import fast_cfg


@pytest.fixture(scope="session")
def root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a root where every stage ran on synthetic data."""
    root = tmp_path_factory.mktemp("stages")
    run(list(STAGES), fast_cfg(root), force=False)
    return root

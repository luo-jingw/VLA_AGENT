"""Shared fixtures: a synthetic dataset without faults, and one S0-S7 run on a dataset with faults."""

from __future__ import annotations

import pytest

from pipeline_chain import ChainRun
from pipeline_chain import run_chain
from synthetic_dataset import SyntheticDataset
from synthetic_dataset import write_pick_place


@pytest.fixture(scope="session")
def plain_dataset(tmp_path_factory: pytest.TempPathFactory) -> SyntheticDataset:
    return write_pick_place(tmp_path_factory.mktemp("plain") / "synth", episodes=4, seed=1, faults=False)


@pytest.fixture(scope="session")
def chain(tmp_path_factory: pytest.TempPathFactory) -> ChainRun:
    return run_chain(tmp_path_factory.mktemp("project"))

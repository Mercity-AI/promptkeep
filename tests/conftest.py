"""Shared pytest fixtures: every test runs against an isolated, fresh DB."""

import pytest

import promptkeep
from promptkeep import config as pm_config
from promptkeep import storage


@pytest.fixture(autouse=True)
def isolated_db(tmp_path):
    """Every test gets a fresh SQLite DB in tmp_path and clean settings.

    write_mode="sync": deterministic assertions matter more than throughput
    in tests — a row must exist the moment the call returns. Writer-specific
    tests opt back into background mode explicitly.
    """
    pm_config.reset()
    storage.reset_caches()
    promptkeep.configure(
        db_path=tmp_path / "prompts.db", enabled=True, strict=False, write_mode="sync"
    )
    yield tmp_path / "prompts.db"
    pm_config.reset()
    storage.reset_caches()

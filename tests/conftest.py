"""Shared pytest fixtures: every test runs against an isolated, fresh DB."""

import pytest

import promptkeep
from promptkeep import config as pm_config


@pytest.fixture(autouse=True)
def isolated_db(tmp_path):
    """Every test gets a fresh SQLite DB in tmp_path and clean settings."""
    pm_config.reset()
    promptkeep.configure(db_path=tmp_path / "prompts.db", enabled=True, strict=False)
    yield tmp_path / "prompts.db"
    pm_config.reset()

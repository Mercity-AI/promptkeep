"""Shared pytest fixtures: every test runs against an isolated, fresh DB."""

import pytest

import prompt_manager
from prompt_manager import config as pm_config


@pytest.fixture(autouse=True)
def isolated_db(tmp_path):
    """Every test gets a fresh SQLite DB in tmp_path and clean settings."""
    pm_config.reset()
    prompt_manager.configure(db_path=tmp_path / "prompts.db", enabled=True, strict=False)
    yield tmp_path / "prompts.db"
    pm_config.reset()

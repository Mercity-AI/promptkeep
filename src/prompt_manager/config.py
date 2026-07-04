"""Global configuration: DB location, tracking on/off, strict rendering."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

DEFAULT_DB_FILENAME = ".prompts.db"

_lock = threading.Lock()
_overrides: dict = {}


@dataclass(frozen=True)
class Settings:
    db_path: Path
    enabled: bool
    strict: bool


def configure(
    db_path: Optional[Union[str, Path]] = None,
    enabled: Optional[bool] = None,
    strict: Optional[bool] = None,
) -> None:
    """Override library settings. Only the arguments you pass are changed.

    - db_path: where the SQLite database lives (default: ./.prompts.db,
      or the PROMPT_MANAGER_DB env var).
    - enabled: turn persistence on/off entirely (default: on, unless
      PROMPT_MANAGER_DISABLED is set). Rendering works either way.
    - strict: raise on missing variables instead of leaving `{name}` literal
      (default: False).
    """
    with _lock:
        if db_path is not None:
            _overrides["db_path"] = Path(db_path)
        if enabled is not None:
            _overrides["enabled"] = bool(enabled)
        if strict is not None:
            _overrides["strict"] = bool(strict)


def get_settings() -> Settings:
    with _lock:
        db_path = _overrides.get("db_path")
        if db_path is None:
            env_path = os.environ.get("PROMPT_MANAGER_DB")
            db_path = Path(env_path) if env_path else Path.cwd() / DEFAULT_DB_FILENAME
        enabled = _overrides.get("enabled")
        if enabled is None:
            disabled = os.environ.get("PROMPT_MANAGER_DISABLED", "").strip().lower()
            enabled = disabled not in ("1", "true", "yes", "on")
        strict = _overrides.get("strict", False)
        return Settings(db_path=db_path, enabled=enabled, strict=strict)


def reset() -> None:
    """Clear all configure() overrides and drop cached DB state. Mainly for tests."""
    with _lock:
        _overrides.clear()
    from . import storage

    storage.reset_caches()

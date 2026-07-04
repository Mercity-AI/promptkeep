"""Global library configuration: DB location, tracking on/off, strict rendering.

Settings are resolved fresh on every access with a simple precedence:
explicit ``configure()`` overrides win, then environment variables
(``PROMPTKEEP_DB``, ``PROMPTKEEP_DISABLED``), then defaults.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

DEFAULT_DB_FILENAME = ".promptkeep.db"

# configure() overrides live here; guarded by a lock since wrapped clients
# may resolve settings from multiple threads.
_lock = threading.Lock()
_overrides: dict = {}


@dataclass(frozen=True)
class Settings:
    """A resolved, immutable snapshot of the library's configuration."""

    db_path: Path
    enabled: bool
    strict: bool


def configure(
    db_path: Optional[Union[str, Path]] = None,
    enabled: Optional[bool] = None,
    strict: Optional[bool] = None,
) -> None:
    """Override library settings. Only the arguments you pass are changed.

    - db_path: where the SQLite database lives (default: ./.promptkeep.db,
      or the PROMPTKEEP_DB env var).
    - enabled: turn persistence on/off entirely (default: on, unless
      PROMPTKEEP_DISABLED is set). Rendering works either way.
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
    """Resolve the current settings: configure() overrides > env vars > defaults."""
    with _lock:
        # DB path: explicit override, then $PROMPTKEEP_DB, then ./.promptkeep.db.
        db_path = _overrides.get("db_path")
        if db_path is None:
            env_path = os.environ.get("PROMPTKEEP_DB")
            db_path = Path(env_path) if env_path else Path.cwd() / DEFAULT_DB_FILENAME

        # Tracking: on by default; $PROMPTKEEP_DISABLED=1/true/yes/on kills it.
        enabled = _overrides.get("enabled")
        if enabled is None:
            disabled = os.environ.get("PROMPTKEEP_DISABLED", "").strip().lower()
            enabled = disabled not in ("1", "true", "yes", "on")

        # Rendering strictness: lenient unless explicitly opted in.
        strict = _overrides.get("strict", False)
        return Settings(db_path=db_path, enabled=enabled, strict=strict)


def reset() -> None:
    """Clear all configure() overrides and drop cached DB state. Mainly for tests."""
    with _lock:
        _overrides.clear()
    from . import storage

    storage.reset_caches()

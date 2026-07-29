"""Global library configuration: DB location, tracking on/off, strict rendering,
and the run-write mode.

Settings are resolved fresh on every access with a simple precedence:
explicit ``configure()`` overrides win, then environment variables
(``PROMPTKEEP_DB``, ``PROMPTKEEP_DISABLED``, ``PROMPTKEEP_WRITE_MODE``),
then defaults.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

DEFAULT_DB_FILENAME = ".promptkeep.db"

_WRITE_MODES = ("background", "sync", "off")

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
    write_mode: str
    queue_size: int
    flush_interval: float
    batch_size: int


def configure(
    db_path: Optional[Union[str, Path]] = None,
    enabled: Optional[bool] = None,
    strict: Optional[bool] = None,
    write_mode: Optional[str] = None,
    queue_size: Optional[int] = None,
    flush_interval: Optional[float] = None,
    batch_size: Optional[int] = None,
) -> None:
    """Override library settings. Only the arguments you pass are changed.

    - db_path: where the SQLite database lives (default: ./.promptkeep.db,
      or the PROMPTKEEP_DB env var).
    - enabled: turn persistence on/off entirely (default: on, unless
      PROMPTKEEP_DISABLED is set). Rendering works either way.
    - strict: raise on missing variables instead of leaving `{name}` literal
      (default: False).
    - write_mode: how run rows are persisted (default: "background", or the
      PROMPTKEEP_WRITE_MODE env var).
        "background" - a worker thread writes batches off the hot path; the
            row may land a moment after the call returns. promptkeep.flush()
            guarantees everything queued is on disk (short-lived scripts get
            an automatic flush at exit).
        "sync" - rows are written before the call returns, like a plain
            insert. Right for tests and read-your-own-write scripts.
        "off" - run telemetry is dropped entirely; version registration
            still works (Prompt.version is a value callers read back).
    - queue_size: background queue bound; when full, the oldest queued run
      is dropped and counted rather than growing without limit (default 10000).
    - flush_interval: seconds the background writer waits for more work
      before writing a partial batch (default 0.5).
    - batch_size: max runs written per transaction (default 100).
    """
    with _lock:
        if db_path is not None:
            _overrides["db_path"] = Path(db_path)
        if enabled is not None:
            _overrides["enabled"] = bool(enabled)
        if strict is not None:
            _overrides["strict"] = bool(strict)
        if write_mode is not None:
            if write_mode not in _WRITE_MODES:
                raise ValueError(f"write_mode must be one of {_WRITE_MODES}, got {write_mode!r}")
            _overrides["write_mode"] = write_mode
        if queue_size is not None:
            if queue_size < 1:
                raise ValueError("queue_size must be a positive integer")
            _overrides["queue_size"] = int(queue_size)
        if flush_interval is not None:
            if flush_interval <= 0:
                raise ValueError("flush_interval must be positive")
            _overrides["flush_interval"] = float(flush_interval)
        if batch_size is not None:
            if batch_size < 1:
                raise ValueError("batch_size must be a positive integer")
            _overrides["batch_size"] = int(batch_size)


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

        # Run persistence mode: background unless overridden; a bad env value
        # falls back to the default rather than breaking every call site.
        write_mode = _overrides.get("write_mode")
        if write_mode is None:
            env_mode = os.environ.get("PROMPTKEEP_WRITE_MODE", "").strip().lower()
            write_mode = env_mode if env_mode in _WRITE_MODES else "background"

        return Settings(
            db_path=db_path,
            enabled=enabled,
            strict=strict,
            write_mode=write_mode,
            queue_size=_overrides.get("queue_size", 10_000),
            flush_interval=_overrides.get("flush_interval", 0.5),
            batch_size=_overrides.get("batch_size", 100),
        )


def reset() -> None:
    """Clear all configure() overrides and drop cached DB state. Mainly for tests."""
    with _lock:
        _overrides.clear()
    from . import storage

    storage.reset_caches()

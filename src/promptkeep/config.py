"""Global library configuration: DB location, tracking on/off, strict rendering,
the run-write mode, and the production controls (sampling, redaction).

Settings are resolved fresh on every access with a simple precedence:
explicit ``configure()`` overrides win, then environment variables
(``PROMPTKEEP_DB``, ``PROMPTKEEP_DISABLED``, ``PROMPTKEEP_WRITE_MODE``,
``PROMPTKEEP_SAMPLE_RATE``), then defaults.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

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
    pre: tuple
    post: tuple
    on_block: str
    sample_rate: float
    redact: Callable[[str], str] | None


def configure(
    db_path: str | Path | None = None,
    enabled: bool | None = None,
    strict: bool | None = None,
    write_mode: str | None = None,
    queue_size: int | None = None,
    flush_interval: float | None = None,
    batch_size: int | None = None,
    pre: list | None = None,
    post: list | None = None,
    on_block: str | None = None,
    sample_rate: float | None = None,
    redact: Callable[[str], str] | None = None,
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
    - pre / post: checks applied to *every* tracked call (global scope). A
      call's checks are its prompt's, plus per-call, plus these.
    - on_block: what a blocking pre-check does — "raise" (default, raises
      PromptBlocked) or "return" a response-shaped stub so a service can
      degrade instead of erroring.
    - sample_rate: the fraction of *uneventful* runs to store, 0.0-1.0
      (default 1.0, or the PROMPTKEEP_SAMPLE_RATE env var). Errors, blocked
      calls and runs with a non-ok verdict are always stored, whatever the
      rate. Inside a conversation the decision is made once per session
      (from the conversation's id, so every process agrees), so a kept
      session is complete rather than full of holes. 0.0 means "store only
      problems". Version registration is never sampled.
    - redact: a function ``str -> str`` applied to every stored text field
      of a run — rendered prompt, input and output, variables and request
      params (as JSON), error text — and of a check verdict (message,
      rewritten text) before it is written. Templates, prompt names and
      conversation metadata are not passed through it. If the hook raises or
      returns a non-string the row is dropped, never stored unredacted.
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
        if pre is not None:
            _overrides["pre"] = tuple(pre)
        if post is not None:
            _overrides["post"] = tuple(post)
        if on_block is not None:
            if on_block not in ("raise", "return"):
                raise ValueError(f"on_block must be 'raise' or 'return', got {on_block!r}")
            _overrides["on_block"] = on_block
        if sample_rate is not None:
            if not _valid_sample_rate(sample_rate):
                raise ValueError(f"sample_rate must be between 0.0 and 1.0, got {sample_rate!r}")
            _overrides["sample_rate"] = float(sample_rate)
        if redact is not None:
            if not callable(redact):
                raise TypeError(
                    f"redact must be callable (str -> str), got {type(redact).__name__}"
                )
            _overrides["redact"] = redact


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

        # Sampling: override, then $PROMPTKEEP_SAMPLE_RATE; anything unparseable
        # or out of range means "keep everything" — the safe direction.
        sample_rate = _overrides.get("sample_rate")
        if sample_rate is None:
            sample_rate = _sample_rate_from_env(os.environ.get("PROMPTKEEP_SAMPLE_RATE"))

        return Settings(
            db_path=db_path,
            enabled=enabled,
            strict=strict,
            write_mode=write_mode,
            queue_size=_overrides.get("queue_size", 10_000),
            flush_interval=_overrides.get("flush_interval", 0.5),
            batch_size=_overrides.get("batch_size", 100),
            pre=_overrides.get("pre", ()),
            post=_overrides.get("post", ()),
            on_block=_overrides.get("on_block", "raise"),
            sample_rate=sample_rate,
            redact=_overrides.get("redact"),
        )


def _valid_sample_rate(value) -> bool:
    """A real number in [0, 1] (bool excluded: True would silently mean 1.0)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return 0.0 <= value <= 1.0


def _sample_rate_from_env(raw: str | None) -> float:
    """Parse $PROMPTKEEP_SAMPLE_RATE; invalid or missing falls back to 1.0."""
    if not raw:
        return 1.0
    try:
        value = float(raw.strip())
    except ValueError:
        return 1.0
    return value if _valid_sample_rate(value) else 1.0


def reset() -> None:
    """Clear every configure() override. Mainly for tests — pair it with
    ``storage.reset_caches()`` to also drop the DB binding and memoized state."""
    with _lock:
        _overrides.clear()

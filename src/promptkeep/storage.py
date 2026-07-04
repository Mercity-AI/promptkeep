"""SQLite persistence via peewee: prompts, versions (lineage), runs.

All write paths that run implicitly (version registration on first render,
run recording inside a wrapped OpenAI call) are exception-shielded: a broken
DB loses telemetry, never a completion. Read paths (history queries) raise
normally.

Schema evolution: the DB carries a schema number in ``PRAGMA user_version``.
``_migrate`` applies forward-only steps for anything below the current
``_SCHEMA_VERSION``; future changes bump the constant and add a step using
``playhouse.migrate`` operations (ships with peewee).
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Tuple

import peewee as pw

logger = logging.getLogger("promptkeep")

_SCHEMA_VERSION = 2

# The models bind to this proxy; _get_db() points it at the configured file.
_proxy = pw.DatabaseProxy()
_db_lock = threading.Lock()
_current_path: Optional[str] = None

# Registration results memoized per (db, name, template-hash) so repeated
# renders of the same prompt cost zero DB round-trips.
_registration_cache: dict = {}
_reg_lock = threading.Lock()

# WAL for concurrent reader/writer access; busy_timeout so contending
# writers wait instead of failing instantly.
_PRAGMAS = {
    "journal_mode": "wal",
    "foreign_keys": 1,
    "busy_timeout": 5000,
}


# --- models -------------------------------------------------------------------


class BaseModel(pw.Model):
    """Base for all tables; binds them to the runtime-configured database."""

    class Meta:
        database = _proxy


class PromptRecord(BaseModel):
    """A prompt's stable identity — one row per unique name."""

    name = pw.TextField(unique=True)
    created_at = pw.TextField()

    class Meta:
        table_name = "prompts"


class PromptVersionRecord(BaseModel):
    """One concrete template text under a prompt name; the unit of lineage."""

    prompt = pw.ForeignKeyField(PromptRecord, column_name="prompt_id", backref="versions")
    version = pw.IntegerField()
    template = pw.TextField()
    template_hash = pw.TextField()
    source = pw.TextField()
    fn_source_hash = pw.TextField(null=True)
    created_at = pw.TextField()

    class Meta:
        table_name = "prompt_versions"
        # Dedup by content, and keep version numbers unique per prompt.
        indexes = (
            (("prompt", "template_hash"), True),
            (("prompt", "version"), True),
        )


class RunRecord(BaseModel):
    """One execution of a prompt version: variables in, model output back."""

    version = pw.ForeignKeyField(PromptVersionRecord, column_name="version_id", backref="runs")
    variables = pw.TextField(null=True)
    rendered_text = pw.TextField()
    provider = pw.TextField()
    model = pw.TextField(null=True)
    request_params = pw.TextField(null=True)
    response_id = pw.TextField(null=True)
    output_text = pw.TextField(null=True)
    prompt_tokens = pw.IntegerField(null=True)
    completion_tokens = pw.IntegerField(null=True)
    total_tokens = pw.IntegerField(null=True)
    latency_ms = pw.IntegerField(null=True)
    status = pw.TextField()
    error = pw.TextField(null=True)
    created_at = pw.TextField()

    class Meta:
        table_name = "runs"
        indexes = ((("version", "created_at"), False),)


_MODELS = [PromptRecord, PromptVersionRecord, RunRecord]


# --- helpers -------------------------------------------------------------------


def _utcnow() -> str:
    """Current UTC time as an ISO-8601 string (how all timestamps are stored)."""
    return datetime.now(timezone.utc).isoformat()


def template_hash(text: str, exact: bool = False) -> str:
    """Content hash used as a template's version identity.

    Default: hashes the *normalized* template (variable names canonicalized
    to positional tokens), so renaming a placeholder — {var1} -> {x} —
    resolves to the same version; only static text and placeholder structure
    matter. With exact=True the raw text is hashed, making placeholder names
    part of the identity.
    """
    if not exact:
        from .rendering import normalize_template

        text = normalize_template(text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_or_none(obj: Any) -> Optional[str]:
    """Serialize to JSON for storage; non-serializable values fall back to repr()."""
    if obj is None:
        return None
    return json.dumps(obj, ensure_ascii=False, default=repr)


# --- database lifecycle ----------------------------------------------------------


def _get_db() -> Optional[pw.DatabaseProxy]:
    """Bind the proxy to the configured DB (once), or None when disabled.

    First-time setup (connect, WAL switch, table creation) runs under a lock:
    concurrent first-connections to a fresh file would otherwise fight over
    the exclusive lock the WAL switch needs. Peewee keeps per-thread
    connection state, so normal queries need no locking here.
    """
    from .config import get_settings

    settings = get_settings()
    if not settings.enabled:
        return None
    path = str(Path(settings.db_path))
    global _current_path
    # Double-checked locking: fast path skips the lock once bound.
    if _current_path != path:
        with _db_lock:
            if _current_path != path:
                parent = Path(path).parent
                if parent and not parent.exists():
                    parent.mkdir(parents=True, exist_ok=True)
                database = pw.SqliteDatabase(path, pragmas=_PRAGMAS, timeout=5)
                _proxy.initialize(database)
                database.connect(reuse_if_open=True)
                _migrate(database)
                _current_path = path
    return _proxy


def _migrate(database: pw.SqliteDatabase) -> None:
    """Apply forward-only schema steps until the DB reaches _SCHEMA_VERSION."""
    (user_version,) = database.execute_sql("PRAGMA user_version").fetchone()
    if user_version < 1:
        database.create_tables(_MODELS, safe=True)
    if 1 <= user_version < 2:
        # v2: template_hash became a hash of the *normalized* template
        # (variable names canonicalized). Recompute stored hashes so old
        # rows keep deduping correctly against new registrations.
        for row in PromptVersionRecord.select():
            new_hash = template_hash(row.template)
            if new_hash != row.template_hash:
                try:
                    PromptVersionRecord.update(template_hash=new_hash).where(
                        PromptVersionRecord.id == row.id
                    ).execute()
                except pw.IntegrityError:
                    # Two old versions differing only in variable names now
                    # collide; keep the older row normalized, leave this one.
                    pass
    # Future steps: `if user_version < 3:` apply playhouse.migrate operations.
    if user_version < _SCHEMA_VERSION:
        database.execute_sql(f"PRAGMA user_version = {_SCHEMA_VERSION}")


def reset_caches() -> None:
    """Drop memoized registrations and close the DB binding. Mainly for tests."""
    global _current_path
    with _reg_lock:
        _registration_cache.clear()
    with _db_lock:
        if _current_path is not None:
            try:
                _proxy.close()
            except Exception:
                pass
            _current_path = None


# --- writes (shielded) -------------------------------------------------------


def register_version(
    name: str,
    template: str,
    source: str = "literal",
    fn_source_hash: Optional[str] = None,
    exact_match: bool = False,
) -> Optional[Tuple[int, int]]:
    """Idempotently record (name, template) and return (version_id, version).

    Deduplicated by template content hash (normalized by default; raw text
    when exact_match=True): re-registering matching text returns the existing
    version. Returns None when tracking is disabled or the write fails
    (never raises).
    """
    from .config import get_settings

    settings = get_settings()
    if not settings.enabled:
        return None

    # Cheap path: this exact template was already registered this process.
    content_hash = template_hash(template, exact=exact_match)
    cache_key = (str(settings.db_path), name, content_hash)
    with _reg_lock:
        cached = _registration_cache.get(cache_key)
    if cached is not None:
        return cached

    # Slow path: hit the DB, shielded so a broken DB can't break rendering.
    try:
        if _get_db() is None:
            return None
        result = _register(name, template, content_hash, source, fn_source_hash)
    except Exception:
        logger.warning("promptkeep: failed to register version for prompt %r", name, exc_info=True)
        return None
    with _reg_lock:
        _registration_cache[cache_key] = result
    return result


def _register(name, template, content_hash, source, fn_source_hash) -> Tuple[int, int]:
    """Insert the prompt/version rows, deduping and racing safely.

    Retry: two writers can race on the same next-version number; the unique
    (prompt, version) index rejects the loser, who re-reads. BEGIN IMMEDIATE
    takes the write lock up front — a deferred transaction would read first
    and SQLite refuses to wait on read->write upgrades.
    """
    now = _utcnow()
    for _ in range(5):
        try:
            with _proxy.atomic("IMMEDIATE"):
                # Ensure the identity row exists.
                prompt_row, _created = PromptRecord.get_or_create(
                    name=name, defaults={"created_at": now}
                )
                # Same template text already registered? Return that version.
                existing = (
                    PromptVersionRecord.select(PromptVersionRecord.id, PromptVersionRecord.version)
                    .where(
                        (PromptVersionRecord.prompt == prompt_row)
                        & (PromptVersionRecord.template_hash == content_hash)
                    )
                    .first()
                )
                if existing is not None:
                    return (existing.id, existing.version)
                # New text: claim the next sequential version number.
                max_version = (
                    PromptVersionRecord.select(pw.fn.MAX(PromptVersionRecord.version))
                    .where(PromptVersionRecord.prompt == prompt_row)
                    .scalar()
                ) or 0
                row = PromptVersionRecord.create(
                    prompt=prompt_row,
                    version=max_version + 1,
                    template=template,
                    template_hash=content_hash,
                    source=source,
                    fn_source_hash=fn_source_hash,
                    created_at=now,
                )
                return (row.id, row.version)
        except (pw.IntegrityError, pw.OperationalError):
            continue
    raise RuntimeError(f"could not register a version for prompt {name!r} after retries")


def record_run(
    *,
    version_id: int,
    variables: Optional[dict],
    rendered_text: str,
    provider: str,
    model: Optional[str] = None,
    request_params: Optional[dict] = None,
    response_id: Optional[str] = None,
    output_text: Optional[str] = None,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    latency_ms: Optional[int] = None,
    status: str = "ok",
    error: Optional[str] = None,
) -> None:
    """Insert one run row for a prompt version. Shielded: never raises."""
    try:
        if _get_db() is None:
            return
        RunRecord.create(
            version=version_id,
            variables=_json_or_none(variables),
            rendered_text=rendered_text,
            provider=provider,
            model=model,
            request_params=_json_or_none(request_params),
            response_id=response_id,
            output_text=output_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            status=status,
            error=error,
            created_at=_utcnow(),
        )
    except Exception:
        logger.warning("promptkeep: failed to record run", exc_info=True)


# --- reads (raise on real errors) ---------------------------------------------


def fetch_versions(name: str) -> list:
    """All version rows for a prompt name as dicts, oldest first."""
    if _get_db() is None:
        return []
    query = (
        PromptVersionRecord.select(
            PromptVersionRecord.version,
            PromptVersionRecord.template,
            PromptVersionRecord.template_hash,
            PromptVersionRecord.source,
            PromptVersionRecord.fn_source_hash,
            PromptVersionRecord.created_at,
        )
        .join(PromptRecord)
        .where(PromptRecord.name == name)
        .order_by(PromptVersionRecord.version)
        .dicts()
    )
    return list(query)


def fetch_runs(name: str, version: Optional[int] = None, limit: int = 50) -> list:
    """Run rows for a prompt (optionally one version) as dicts, newest first."""
    if _get_db() is None:
        return []
    # Join through versions to prompts so callers filter by name, not ids.
    query = (
        RunRecord.select(
            RunRecord.id,
            PromptRecord.name.alias("prompt_name"),
            PromptVersionRecord.version.alias("version"),
            RunRecord.variables,
            RunRecord.rendered_text,
            RunRecord.provider,
            RunRecord.model,
            RunRecord.request_params,
            RunRecord.response_id,
            RunRecord.output_text,
            RunRecord.prompt_tokens,
            RunRecord.completion_tokens,
            RunRecord.total_tokens,
            RunRecord.latency_ms,
            RunRecord.status,
            RunRecord.error,
            RunRecord.created_at,
        )
        .join(PromptVersionRecord)
        .join(PromptRecord)
        .where(PromptRecord.name == name)
    )
    if version is not None:
        query = query.where(PromptVersionRecord.version == version)
    query = query.order_by(RunRecord.id.desc()).limit(limit).dicts()
    return list(query)

"""SQLite persistence: prompts, versions (lineage), runs.

All write paths that run implicitly (version registration on first render,
run recording inside a wrapped OpenAI call) are exception-shielded: a broken
DB loses telemetry, never a completion. Read paths (history queries) raise
normally.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Tuple

logger = logging.getLogger("prompt_manager")

_local = threading.local()
_registration_cache: dict = {}
_reg_lock = threading.Lock()
_connect_lock = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS prompts (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS prompt_versions (
    id             INTEGER PRIMARY KEY,
    prompt_id      INTEGER NOT NULL REFERENCES prompts(id),
    version        INTEGER NOT NULL,
    template       TEXT NOT NULL,
    template_hash  TEXT NOT NULL,
    source         TEXT NOT NULL,
    fn_source_hash TEXT,
    created_at     TEXT NOT NULL,
    UNIQUE (prompt_id, template_hash),
    UNIQUE (prompt_id, version)
);

CREATE TABLE IF NOT EXISTS runs (
    id                INTEGER PRIMARY KEY,
    version_id        INTEGER NOT NULL REFERENCES prompt_versions(id),
    variables         TEXT,
    rendered_text     TEXT NOT NULL,
    provider          TEXT NOT NULL,
    model             TEXT,
    request_params    TEXT,
    response_id       TEXT,
    output_text       TEXT,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    total_tokens      INTEGER,
    latency_ms        INTEGER,
    status            TEXT NOT NULL,
    error             TEXT,
    created_at        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runs_version ON runs(version_id, created_at);
"""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def template_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_or_none(obj: Any) -> Optional[str]:
    if obj is None:
        return None
    return json.dumps(obj, ensure_ascii=False, default=repr)


def _connect(path: Path) -> sqlite3.Connection:
    conns = getattr(_local, "conns", None)
    if conns is None:
        conns = _local.conns = {}
    key = str(path)
    conn = conns.get(key)
    if conn is None:
        # Serialize setup: switching a fresh DB into WAL needs an exclusive
        # lock, which concurrent first-connections would fight over.
        with _connect_lock:
            if path.parent and not path.parent.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(key, timeout=5.0)
            conn.row_factory = sqlite3.Row
            try:
                conn.execute("PRAGMA journal_mode=WAL")
            except sqlite3.OperationalError:
                pass  # WAL is an optimization; default journal mode works too
            conn.execute("PRAGMA foreign_keys=ON")
            _migrate(conn)
        conns[key] = conn
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    (user_version,) = conn.execute("PRAGMA user_version").fetchone()
    if user_version < 1:
        conn.executescript(_SCHEMA)
        conn.execute("PRAGMA user_version = 1")
        conn.commit()


def connection() -> Optional[sqlite3.Connection]:
    """Connection for the configured DB, or None when tracking is disabled."""
    from .config import get_settings

    settings = get_settings()
    if not settings.enabled:
        return None
    return _connect(Path(settings.db_path))


def reset_caches() -> None:
    with _reg_lock:
        _registration_cache.clear()
    conns = getattr(_local, "conns", None)
    if conns:
        for conn in conns.values():
            try:
                conn.close()
            except Exception:
                pass
        conns.clear()


# --- writes (shielded) -------------------------------------------------------


def register_version(
    name: str,
    template: str,
    source: str = "literal",
    fn_source_hash: Optional[str] = None,
) -> Optional[Tuple[int, int]]:
    """Idempotently record (name, template) and return (version_id, version).

    Deduplicated by template content hash: re-registering identical text
    returns the existing version. Returns None when tracking is disabled or
    the write fails (never raises).
    """
    from .config import get_settings

    settings = get_settings()
    if not settings.enabled:
        return None
    content_hash = template_hash(template)
    cache_key = (str(settings.db_path), name, content_hash)
    with _reg_lock:
        cached = _registration_cache.get(cache_key)
    if cached is not None:
        return cached
    try:
        conn = _connect(Path(settings.db_path))
        result = _register(conn, name, template, content_hash, source, fn_source_hash)
    except Exception:
        logger.warning(
            "prompt_manager: failed to register version for prompt %r", name, exc_info=True
        )
        return None
    with _reg_lock:
        _registration_cache[cache_key] = result
    return result


def _register(conn, name, template, content_hash, source, fn_source_hash) -> Tuple[int, int]:
    now = _utcnow()
    # Retry: two writers can race on the same next-version number; the
    # UNIQUE(prompt_id, version) constraint rejects the loser, who re-reads.
    for _ in range(5):
        try:
            with conn:
                conn.execute(
                    "INSERT OR IGNORE INTO prompts (name, created_at) VALUES (?, ?)",
                    (name, now),
                )
                (prompt_id,) = conn.execute(
                    "SELECT id FROM prompts WHERE name = ?", (name,)
                ).fetchone()
                row = conn.execute(
                    "SELECT id, version FROM prompt_versions"
                    " WHERE prompt_id = ? AND template_hash = ?",
                    (prompt_id, content_hash),
                ).fetchone()
                if row is not None:
                    return (row["id"], row["version"])
                (next_version,) = conn.execute(
                    "SELECT COALESCE(MAX(version), 0) + 1 FROM prompt_versions WHERE prompt_id = ?",
                    (prompt_id,),
                ).fetchone()
                cursor = conn.execute(
                    "INSERT INTO prompt_versions"
                    " (prompt_id, version, template, template_hash, source,"
                    "  fn_source_hash, created_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (prompt_id, next_version, template, content_hash, source, fn_source_hash, now),
                )
                return (cursor.lastrowid, next_version)
        except sqlite3.IntegrityError:
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
    from .config import get_settings

    settings = get_settings()
    if not settings.enabled:
        return
    try:
        conn = _connect(Path(settings.db_path))
        with conn:
            conn.execute(
                "INSERT INTO runs (version_id, variables, rendered_text, provider, model,"
                " request_params, response_id, output_text, prompt_tokens,"
                " completion_tokens, total_tokens, latency_ms, status, error, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    version_id,
                    _json_or_none(variables),
                    rendered_text,
                    provider,
                    model,
                    _json_or_none(request_params),
                    response_id,
                    output_text,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    latency_ms,
                    status,
                    error,
                    _utcnow(),
                ),
            )
    except Exception:
        logger.warning("prompt_manager: failed to record run", exc_info=True)


# --- reads (raise on real errors) ---------------------------------------------


def fetch_versions(name: str) -> list:
    conn = connection()
    if conn is None:
        return []
    return conn.execute(
        "SELECT v.version, v.template, v.template_hash, v.source, v.fn_source_hash,"
        " v.created_at"
        " FROM prompt_versions v JOIN prompts p ON v.prompt_id = p.id"
        " WHERE p.name = ? ORDER BY v.version",
        (name,),
    ).fetchall()


def fetch_runs(name: str, version: Optional[int] = None, limit: int = 50) -> list:
    conn = connection()
    if conn is None:
        return []
    query = (
        "SELECT r.id, p.name AS prompt_name, v.version, r.variables, r.rendered_text,"
        " r.provider, r.model, r.request_params, r.response_id, r.output_text,"
        " r.prompt_tokens, r.completion_tokens, r.total_tokens, r.latency_ms,"
        " r.status, r.error, r.created_at"
        " FROM runs r"
        " JOIN prompt_versions v ON r.version_id = v.id"
        " JOIN prompts p ON v.prompt_id = p.id"
        " WHERE p.name = ?"
    )
    params: list = [name]
    if version is not None:
        query += " AND v.version = ?"
        params.append(version)
    query += " ORDER BY r.id DESC LIMIT ?"
    params.append(limit)
    return conn.execute(query, params).fetchall()

"""SQLite persistence: the database binding and every write path.

The tables live in ``models``, schema upgrades in ``migrations``, the read
side in ``history``. This module owns the connection and the writes: version
registration, conversation resolution, run and verdict recording, span
upserts, the batch insert the background writer drains into, and the
retention sweep.

All write paths that run implicitly (version registration on first render,
run recording inside a wrapped call) are exception-shielded: a broken DB
loses telemetry, never a completion.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import sqlite3
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import peewee as pw

from . import controls, writer
from .config import Settings, _destination, get_settings
from .migrations import migrate
from .models import (
    CheckRecord,
    ConversationRecord,
    ImportedSessionRecord,
    ImportedUsageRecord,
    PromptRecord,
    PromptVersionRecord,
    RunRecord,
    SpanRecord,
    db_proxy,
    new_run_key,
)
from .normalization import merge, payload
from .rendering import template_hash

logger = logging.getLogger("promptkeep")


# --- the database binding -------------------------------------------------------

_db_lock = threading.RLock()
_current_path: str | None = None

# WAL for concurrent reader/writer access; busy_timeout so contending
# writers wait instead of failing instantly.
_PRAGMAS = {
    "journal_mode": "wal",
    "foreign_keys": 1,
    "busy_timeout": 5000,
}


class _Database(pw.SqliteDatabase):
    """Close failed connection attempts and retry transient WAL initialization races."""

    def _add_conn_hooks(self, connection):
        """Peewee cannot close a connection whose initialization hook raised."""
        try:
            return super()._add_conn_hooks(connection)
        except BaseException:
            connection.close()
            raise

    def _connect(self):
        """A fresh-file WAL switch may fail immediately despite busy_timeout."""
        deadline = time.monotonic() + 5
        while True:
            try:
                return super()._connect()
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or time.monotonic() >= deadline:
                    raise
                time.sleep(0.01)


def _open_database(path):
    """Serialize initial WAL/schema setup for all package-owned connections."""
    database = _Database(str(path), pragmas=_PRAGMAS, timeout=5)
    with _db_lock:
        try:
            database.connect()
            with database.atomic("IMMEDIATE"):
                migrate(database)
        except BaseException:
            if not database.is_closed():
                database.close()
            raise
    return database


@contextmanager
def connection_context(path: str | None = None):
    """Pin one operation's destination and close only its own connection."""
    token = _destination.set(path or _destination.get() or str(get_settings().db_path))
    previous = db_proxy.obj
    was_open = previous is not None and not previous.is_closed()
    try:
        yield
    finally:
        current = db_proxy.obj
        if current is not None and (current is not previous or not was_open):
            current.close()
        _destination.reset(token)


def connection_scope(function):
    """Bound synchronous storage/read work without closing outer-owned resources."""

    @functools.wraps(function)
    def scoped(*args, **kwargs):
        """Release connections on the same thread, including exception paths."""
        with connection_context():
            return function(*args, **kwargs)

    return scoped


def get_db() -> pw.DatabaseProxy | None:
    """Bind the proxy to the configured DB (once), or None when disabled.

    First-time setup (connect, WAL switch, table creation) runs under a lock:
    concurrent first-connections to a fresh file would otherwise fight over
    the exclusive lock the WAL switch needs. Peewee keeps per-thread
    connection state, so normal queries need no locking here.
    """
    settings = get_settings()
    if not settings.enabled:
        return None
    path = str(Path(settings.db_path))

    # Double-checked locking: the fast path skips the lock once bound.
    global _current_path
    if db_proxy.obj is None or db_proxy.obj.database != path:
        with _db_lock:
            if db_proxy.obj is None or db_proxy.obj.database != path:
                parent = Path(path).parent
                if parent and not parent.exists():
                    parent.mkdir(parents=True, exist_ok=True)
                database = _open_database(path)
                db_proxy.initialize(database)
                _current_path = path
    return db_proxy


def reset_caches() -> None:
    """Drop memoized registrations, conversation/turn caches, anything still
    queued in the background writer, and the DB binding. Mainly for tests."""
    global _current_path
    writer.reset()
    with _reg_lock:
        _registration_cache.clear()
    with _prune_lock:
        _next_prune.clear()
    with _convo_lock:
        _conversation_cache.clear()
        _turn_counters.clear()
        _response_index.clear()
    with _db_lock:
        if db_proxy.obj is not None:
            try:
                db_proxy.close()
            finally:
                db_proxy.initialize(None)
        _current_path = None


# --- helpers ---------------------------------------------------------------------


def _utcnow() -> str:
    """Current UTC time as an ISO-8601 string (how all timestamps are stored)."""
    return datetime.now(UTC).isoformat()


def _json_or_none(obj: Any) -> str | None:
    """Serialize to JSON for storage; non-serializable values fall back to repr()."""
    if obj is None:
        return None
    return json.dumps(obj, ensure_ascii=False, default=repr)


# --- version registration ---------------------------------------------------------

# Registration results memoized per (db, name, template-hash) so repeated
# renders of the same prompt cost zero DB round-trips.
_registration_cache: dict[tuple[str, str, str], tuple[int, int]] = {}
_reg_lock = threading.Lock()


@connection_scope
def register_version(
    name: str,
    template: str,
    source: str = "literal",
    fn_source_hash: str | None = None,
    exact_match: bool = False,
) -> tuple[int, int] | None:
    """Idempotently record (name, template) and return (version_id, version).

    Deduplicated by template content hash (normalized by default; raw text
    when exact_match=True): re-registering matching text returns the existing
    version. Returns None when tracking is disabled or the write fails
    (never raises).
    """
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
        if get_db() is None:
            return None
        result = _register(name, template, content_hash, source, fn_source_hash)
    except Exception:
        logger.warning("promptkeep: failed to register version for prompt %r", name, exc_info=True)
        return None
    with _reg_lock:
        _registration_cache[cache_key] = result
    return result


def _register(
    name: str, template: str, content_hash: str, source: str, fn_source_hash: str | None
) -> tuple[int, int]:
    """Insert the prompt/version rows, deduping and racing safely.

    Retry: two writers can race on the same next-version number; the unique
    (prompt, version) index rejects the loser, who re-reads. BEGIN IMMEDIATE
    takes the write lock up front — a deferred transaction would read first
    and SQLite refuses to wait on read->write upgrades.
    """
    now = _utcnow()
    for _ in range(5):
        try:
            with db_proxy.atomic("IMMEDIATE"):
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


@connection_scope
def version_rows(name: str) -> list[dict[str, Any]]:
    """A prompt's stored versions, oldest first, as plain rows (id included).

    The one lineage query, shared by its two readers: ``history.versions``
    (rows -> VersionInfo) and ``Prompt.variants`` (rows -> Prompt objects).
    It lives here rather than in ``history`` because ``prompts`` sits below
    ``history`` in the module graph and needs it too. An explicit read, so it
    raises normally; empty when tracking is disabled or the name is unknown.
    """
    if get_db() is None:
        return []
    query = (
        PromptVersionRecord.select(
            PromptVersionRecord.id,
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


# --- conversations ----------------------------------------------------------------

# Conversations memoized per (db, external_id), and turn numbers handed out
# from in-process counters per (db, conversation_id): the DB's MAX(turn_index)
# goes stale the moment rows sit in the background write queue, so within a
# process the counter is the source of truth. (Two *processes* driving one
# conversation concurrently can still race — that caveat is cross-process
# only.)
_conversation_cache: dict[tuple[str, str], int] = {}
_turn_counters: dict[tuple[str, int], int] = {}
_convo_lock = threading.Lock()


@connection_scope
def get_or_create_conversation(
    external_id: str,
    title: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> int | None:
    """Idempotently resolve a conversation by its caller-supplied id.

    Creates the row on first sight (the "never break the caller" rule
    extends to conversations: an unrecognized id is not an error) and
    memoizes the result, so a long conversation costs one DB round-trip,
    not one per turn. Returns None when tracking is disabled or the write
    fails.
    """
    settings = get_settings()
    if not settings.enabled:
        return None

    # Memoized per process: one round-trip per conversation, not per turn.
    cache_key = (str(settings.db_path), external_id)
    with _convo_lock:
        cached = _conversation_cache.get(cache_key)
    if cached is not None:
        return cached

    # First sight of this id: create or reuse the row, shielded.
    try:
        if get_db() is None:
            return None
        now = _utcnow()
        row, _created = ConversationRecord.get_or_create(
            external_id=external_id,
            defaults={
                "title": title,
                "metadata": _json_or_none(metadata),
                "created_at": now,
                "updated_at": now,
            },
        )
        with _convo_lock:
            _conversation_cache[cache_key] = row.id
        return row.id
    except Exception:
        logger.warning("promptkeep: failed to resolve conversation %r", external_id, exc_info=True)
        return None


@connection_scope
def reserve_turn_index(conversation_id: int) -> int:
    """Claim this conversation's next turn number (0 for its first turn).

    Backed by an in-process counter seeded from the DB on first use: the
    DB's MAX(turn_index) can't be trusted directly while rows sit in the
    background write queue, and the counter also closes the read-then-write
    race two threads had in sync mode. Each call *reserves* — calling twice
    claims two turns.
    """
    key = (str(get_settings().db_path), conversation_id)
    with _convo_lock:
        current = _turn_counters.get(key)
        if current is None:
            max_turn = (
                RunRecord.select(pw.fn.MAX(RunRecord.turn_index))
                .where(RunRecord.conversation == conversation_id)
                .scalar()
            )
            current = 0 if max_turn is None else max_turn + 1
        _turn_counters[key] = current + 1
        return current


# --- chaining by response id ------------------------------------------------------

# The run that produced each recently recorded response, per (db, response_id):
# (its conversation or None, its run key). The database has the same answer —
# once the row lands. In background mode a chained call can arrive while its
# predecessor is still in the writer queue, so within a process this index is
# what is consulted first. Bounded: a chain is followed up promptly or not at
# all, and anything that has aged out is on disk by then.
_response_index: OrderedDict[tuple[str, str], tuple[int | None, str]] = OrderedDict()
_RESPONSE_INDEX_SIZE = 4096


def _index_response(response_id: str, conversation_id: int | None, run_key: str) -> None:
    """Remember which run (and conversation) produced a recorded response.

    A call carrying several Prompts records several rows for one response. The
    first is its primary run — the one its verdicts are filed under — and it
    stays the run a successor points back to; later rows of the same call
    leave the key alone.
    """
    key = (str(get_settings().db_path), response_id)
    with _convo_lock:
        known = _response_index.get(key)
        if known is not None:
            run_key = known[1]
        _response_index[key] = (conversation_id, run_key)
        _response_index.move_to_end(key)
        while len(_response_index) > _RESPONSE_INDEX_SIZE:
            _response_index.popitem(last=False)


def _lookup_response(response_id: str) -> tuple[int | None, str] | None:
    """(conversation or None, run key) of the run that produced ``response_id``
    — the in-process index first (its row may still be queued), then the
    database (another process, or an earlier session). None when no recorded
    run produced it. The caller has checked that there is a database."""
    key = (str(get_settings().db_path), response_id)
    with _convo_lock:
        known = _response_index.get(key)
    if known is not None:
        return known
    row = (
        RunRecord.select(RunRecord.conversation, RunRecord.run_key)
        .where(RunRecord.response_id == response_id)
        .order_by(RunRecord.id)
        .tuples()
        .first()
    )
    return None if row is None else (row[0], row[1])


def _recording() -> bool:
    """Whether run rows are being written at all — tracking on, a write mode
    other than "off", and a database to write to."""
    settings = get_settings()
    return settings.enabled and settings.write_mode != "off" and get_db() is not None


@connection_scope
def predecessor(previous_response_id: str) -> str | None:
    """The key of the run that produced ``previous_response_id`` — the parent
    of a call that continues it — or None when promptkeep never recorded that
    response (or isn't recording now). Raises like any storage call; the
    caller shields."""
    if not _recording():
        return None
    found = _lookup_response(previous_response_id)
    return None if found is None else found[1]


@connection_scope
def chain_conversation(previous_response_id: str) -> int | None:
    """The conversation a call continuing ``previous_response_id`` belongs in.

    Providers that chain server-side (OpenAI Responses) name a call's
    predecessor in the request, so the conversation can be followed with no
    user code: whatever conversation the run that produced that response is
    in, this call joins. When that run is in none — it was the first call of
    the chain, which had nothing to point back at — one is started here,
    keyed ``response:<id>``, and the earlier run is adopted into it as turn 0.

    Returns None when no recorded run produced that response: a chain is
    followed only from a call promptkeep tracked, never inferred. Raises like
    any storage call; the caller shields.
    """
    # Nothing is being recorded, so there is nothing to group: don't leave a
    # conversation row behind for a run that will never be written.
    if not _recording():
        return None
    found = _lookup_response(previous_response_id)
    if found is None:
        return None
    conversation_id, run_key = found
    if conversation_id is not None:
        return conversation_id

    # First link of a chain: start its conversation. The adopted run takes
    # turn 0, so the counter starts past it — the adoption may still be
    # queued, which the DB's MAX(turn_index) would not see.
    conversation_id = get_or_create_conversation(f"response:{previous_response_id}")
    if conversation_id is None:
        return None
    settings = get_settings()
    counter_key = (str(settings.db_path), conversation_id)
    with _convo_lock:
        if counter_key not in _turn_counters:
            max_turn = (
                RunRecord.select(pw.fn.MAX(RunRecord.turn_index))
                .where(RunRecord.conversation == conversation_id)
                .scalar()
            )
            _turn_counters[counter_key] = max(1, 0 if max_turn is None else max_turn + 1)

    # The adoption takes the same road as the run it touches: queued behind
    # it in background mode (so it finds the row), applied directly in sync.
    adoption = {
        "_kind": "adopt",
        "response_id": previous_response_id,
        "conversation": conversation_id,
    }
    if settings.write_mode == "background":
        writer.submit(adoption)
    else:
        _adopt_run(adoption)
    _index_response(previous_response_id, conversation_id, run_key)
    return conversation_id


# --- runs and verdicts ------------------------------------------------------------


@connection_scope
def record_run(
    *,
    run_key: str | None = None,
    version_id: int | None = None,
    variables: dict[str, Any] | None = None,
    rendered_text: str | None = None,
    provider: str,
    model: str | None = None,
    request_params: dict[str, Any] | None = None,
    response_id: str | None = None,
    output_text: str | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    cost_usd: float | None = None,
    latency_ms: int | None = None,
    status: str = "ok",
    error: str | None = None,
    conversation_id: int | None = None,
    turn_index: int | None = None,
    input_text: str | None = None,
    original_input_text: str | None = None,
    parent_run_key: str | None = None,
    span_key: str | None = None,
    cohort: bool | None = None,
    checks: list[dict[str, Any]] | None = None,
    usage_details: dict[str, Any] | None = None,
) -> str | None:
    """Record one run row, honoring the configured write_mode. Never raises.

    Returns the run's key — the identity everything else references it by
    (a RunHandle, history.checks(), a verdict landing later) — or None when
    nothing was recorded (tracking disabled, write_mode "off", sampled out,
    a failed redaction, or the write failed). The key is minted here when
    the caller didn't bring one, and it is valid the moment this returns: in
    background mode the row may still be in the queue, but the key already
    names it. ``span_key`` links this row to its model span when the call
    happened inside an explicit agent trace; ``cohort`` is that trace's
    already-chosen successful-span decision, so a call outside a conversation
    does not flip a second independent coin.

    "sync" inserts before returning; "background" builds the complete row
    (timestamps and turn number included — they must reflect *call* time,
    not whenever the writer thread gets to it) and hands it to the writer
    queue; "off" drops it. Version registration is unaffected by the mode.
    Bundled ``checks`` — verdicts already known at record time — travel with
    the row and land in the same transaction, whichever mode.

    version_id is optional: a conversation turn with no wrapped Prompt still
    gets a row (there's simply nothing to attach to a lineage). Nothing is
    enforced here regardless — failing loudly would violate the shielding
    this function exists to provide.

    turn_index is normally left unset and reserved here. Callers that write
    more than one row for the same physical API call (a message with more
    than one tracked Prompt in it) should reserve one via
    reserve_turn_index() up front and pass it to every row from that call —
    otherwise each row would claim its own turn, splitting one exchange into
    several.

    parent_run_key names the run this one branched from; leave it unset for a
    run that simply follows the one before it.
    """
    try:
        settings = get_settings()
        if not settings.enabled or settings.write_mode == "off":
            return None

        # Build the complete row now: its identity, its timestamp and its
        # turn number all reflect call time, whatever the write mode.
        if run_key is None:
            run_key = new_run_key()
        if turn_index is None and conversation_id is not None:
            turn_index = reserve_turn_index(conversation_id)
        row: dict[str, Any] = {
            "run_key": run_key,
            "version": version_id,
            "variables": _json_or_none(variables),
            "rendered_text": rendered_text,
            "provider": provider,
            "model": model,
            "request_params": _json_or_none(request_params),
            "response_id": response_id,
            "output_text": output_text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost_usd": cost_usd,
            "usage_details": _json_or_none(usage_details),
            "latency_ms": latency_ms,
            "status": status,
            "error": error,
            "created_at": _utcnow(),
            "conversation": conversation_id,
            "turn_index": turn_index,
            "input_text": input_text,
            "original_input_text": original_input_text,
            "parent_run_key": parent_run_key,
            "span_key": span_key,
        }
        if checks:
            row["_checks"] = list(checks)

        # The production controls, before the row can reach a queue or a disk.
        if not controls.keep_run(settings, row, conversation_id, cohort=cohort):
            return None
        if settings.redact is not None:
            try:
                row = controls.redact_run(settings.redact, row)
            except Exception:
                logger.warning(
                    "promptkeep: redact hook failed — run dropped rather than stored unredacted",
                    exc_info=True,
                )
                return None

        # A kept run is one a later call can chain onto by its response id.
        if response_id is not None:
            _index_response(response_id, conversation_id, run_key)

        # Retention rides the write path: a sweep, if one is due.
        _schedule_prune(settings)

        # Persist: hand off to the writer thread, or insert right here.
        if settings.write_mode == "background":
            writer.submit(row)
            return run_key
        if get_db() is None:
            return None
        _insert_run(row)
        return run_key
    except Exception:
        logger.warning("promptkeep: failed to record run", exc_info=True)
        return None


@connection_scope
def record_check(run_key: str | None, check_row: dict[str, Any]) -> None:
    """Record one late verdict — an async post-check that finished after its
    run was recorded — honoring the configured write_mode. Never raises.

    It takes the same road as the run row it belongs to: queued behind it in
    background mode (so flush() covers it), inserted directly in sync mode,
    dropped in "off". run_key=None means the run itself was never recorded
    (tracking disabled), so there is nothing to attach the verdict to.
    """
    try:
        if run_key is None:
            return
        settings = get_settings()
        if not settings.enabled or settings.write_mode == "off":
            return
        item = {"_kind": "check", "run_key": run_key, "created_at": _utcnow(), **check_row}

        # Redaction, then the same persist choice as a run row.
        if settings.redact is not None:
            try:
                item = controls.redact_check(settings.redact, item)
            except Exception:
                logger.warning(
                    "promptkeep: redact hook failed — verdict dropped rather than stored unredacted",
                    exc_info=True,
                )
                return
        if settings.write_mode == "background":
            writer.submit(item)
            return
        if get_db() is None:
            return
        _insert_check(item)
    except Exception:
        logger.warning("promptkeep: failed to record check", exc_info=True)


@connection_scope
def record_span(
    *,
    span_key: str,
    trace_key: str,
    parent_span_key: str | None,
    kind: str,
    name: str,
    started_at: str,
    ended_at: str | None = None,
    duration_ms: int | None = None,
    status: str = "running",
    input_json: str | None = None,
    output_json: str | None = None,
    error: str | None = None,
    sampled: bool | None = None,
) -> None:
    """Record one span, honoring the configured write_mode. Never raises.

    The row is prepared here, redacted before it can reach the queue, and
    then either handed to the existing writer or inserted directly. A span
    upsert is keyed by ``span_key``: a root's final row can restore a start
    row the queue dropped, and a terminal status can never regress to
    ``running`` (see ``_insert_span``). A redaction failure drops the span
    rather than storing its payload unredacted.
    """
    try:
        settings = get_settings()
        if not settings.enabled or settings.write_mode == "off":
            return
        item: dict[str, Any] = {
            "_kind": "span",
            "span_key": span_key,
            "trace_key": trace_key,
            "parent_span_key": parent_span_key,
            "kind": kind,
            "name": name,
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_ms": duration_ms,
            "status": status,
            "input_json": input_json,
            "output_json": output_json,
            "error": error,
            "sampled": sampled,
        }

        # The production redaction hook, before the span can be queued.
        if settings.redact is not None:
            try:
                item = controls.redact_span(settings.redact, item)
            except Exception:
                logger.warning(
                    "promptkeep: redact hook failed — span dropped rather than stored unredacted",
                    exc_info=True,
                )
                return

        # Retention rides the write path, exactly as it does for runs.
        _schedule_prune(settings)

        if settings.write_mode == "background":
            writer.submit(item)
            return
        if get_db() is None:
            return
        _insert_span(item)
    except Exception:
        logger.warning("promptkeep: failed to record span", exc_info=True)


# --- retention --------------------------------------------------------------------

# When the next sweep is due, per database (a time.monotonic() deadline). A
# sweep is cheap when there is nothing to delete, but it is still a write
# transaction, so it runs at most this often per process.
_PRUNE_INTERVAL = 3600.0
# Rows deleted per transaction: a first sweep over a year of history must not
# hold the write lock long enough to time out every other writer.
_PRUNE_CHUNK = 500
_next_prune: dict[str, float] = {}
_prune_lock = threading.Lock()


def _schedule_prune(settings: Settings) -> None:
    """Run a retention sweep if one is due — the first on the first run this
    process records, then hourly. Queued behind the rows ahead of it in
    background mode (so it runs on the writer thread, off the request path);
    inline in sync mode. Called from the shielded write path."""
    if settings.retention_days is None:
        return
    path = str(settings.db_path)
    now = time.monotonic()
    with _prune_lock:
        if now < _next_prune.get(path, 0.0):
            return
        _next_prune[path] = now + _PRUNE_INTERVAL
    if settings.write_mode == "background":
        writer.submit({"_kind": "prune", "retention_days": settings.retention_days})
    else:
        _sweep(settings.retention_days)


def _sweep(retention_days: float) -> None:
    """Run ``prune`` shielded. A sweep is housekeeping riding someone else's
    write — a run row in sync mode, a batch in background mode — and a
    failure (a locked file, say) must cost neither; the next sweep is an
    hour away and retries."""
    try:
        prune(retention_days)
    except Exception:
        logger.warning("promptkeep: retention sweep failed", exc_info=True)


@connection_scope
def prune(retention_days: float) -> int:
    """Delete recorded history older than ``retention_days``; return how many
    runs went.

    A conversation is the unit, as it is for sampling: it goes whole — its
    runs, their verdicts and feedback, the conversation row — once its *last*
    activity is older than the cutoff, so a long-running session never loses
    its opening turns while it is still in use. A run outside any
    conversation (or whose conversation no longer exists) goes by its own
    timestamp. Agent traces expire whole, based on their latest span or linked
    run activity, so a recent child keeps its siblings and an abandoned
    running root does not live forever. Prompts and versions are never
    deleted: they are the code's history, not the traffic's, and they hold no
    user data.

    Chunked, one IMMEDIATE transaction per chunk, so the write lock is never
    held for long. Deleting frees pages for SQLite to reuse; the file itself
    does not shrink (run ``VACUUM`` for that). Called by the scheduled sweep;
    raises like any storage call.
    """
    if get_db() is None:
        return 0
    cutoff = (datetime.now(UTC) - timedelta(days=retention_days)).isoformat()
    deleted = 0

    # Conversations idle since before the cutoff, a chunk at a time. The
    # in-process caches forget them too, or the next turn of a resumed
    # session would point at a row that no longer exists.
    while True:
        with db_proxy.atomic("IMMEDIATE"):
            expired = [
                cid
                for (cid,) in ConversationRecord.select(ConversationRecord.id)
                .where(ConversationRecord.updated_at < cutoff)
                .limit(_PRUNE_CHUNK)
                .tuples()
            ]
            if not expired:
                break
            their_runs = RunRecord.select(RunRecord.id).where(RunRecord.conversation.in_(expired))
            CheckRecord.delete().where(CheckRecord.run.in_(their_runs)).execute()
            deleted += RunRecord.delete().where(RunRecord.conversation.in_(expired)).execute()
            ConversationRecord.delete().where(ConversationRecord.id.in_(expired)).execute()
        _forget_conversations(expired)

    # Whole traces whose latest span or linked-run activity is older than the
    # cutoff. This runs after conversation pruning: a conversation may take
    # its model runs first, and the remaining spans stay readable until their
    # own trace expires.
    deleted += _expire_traces(cutoff)

    # Runs that belong to no (surviving) conversation, by their own age. A
    # run still linked to a surviving span is left for that trace's expiry,
    # so ordinary pruning cannot cut a trace into pieces. A run whose span
    # was never persisted (an orphan span key) is still eligible by age.
    surviving_spans = SpanRecord.select(SpanRecord.span_key)
    unattached = (
        RunRecord.conversation.is_null()
        | RunRecord.conversation.not_in(ConversationRecord.select(ConversationRecord.id))
    ) & (RunRecord.span_key.is_null() | RunRecord.span_key.not_in(surviving_spans))
    while True:
        with db_proxy.atomic("IMMEDIATE"):
            expired = [
                rid
                for (rid,) in RunRecord.select(RunRecord.id)
                .where(unattached & (RunRecord.created_at < cutoff))
                .limit(_PRUNE_CHUNK)
                .tuples()
            ]
            if not expired:
                break
            CheckRecord.delete().where(CheckRecord.run.in_(expired)).execute()
            deleted += RunRecord.delete().where(RunRecord.id.in_(expired)).execute()
    return deleted


def _expire_traces(cutoff: str) -> int:
    """Delete whole trace groups whose latest recorded activity is old.

    A trace's activity is the newest span start/end or linked-run timestamp,
    so a running root is not pinned forever and a recent model call keeps its
    older siblings alive. Trace keys are walked in bounded chunks; each chunk
    deletes its spans, linked runs and those runs' verdicts in one IMMEDIATE
    transaction. Returns the number of run rows deleted, preserving prune()'s
    run-count meaning.
    """
    deleted = 0
    last_key: str | None = None
    while True:
        with db_proxy.atomic("IMMEDIATE"):
            # One bounded page of trace keys, ordered so the walk advances
            # even when none of them are old enough to expire.
            query = SpanRecord.select(SpanRecord.trace_key).distinct()
            if last_key is not None:
                query = query.where(SpanRecord.trace_key > last_key)
            trace_keys = [
                key for (key,) in query.order_by(SpanRecord.trace_key).limit(_PRUNE_CHUNK).tuples()
            ]
            if not trace_keys:
                break
            last_key = trace_keys[-1]

            # The newest span activity and linked-run activity per trace.
            span_activity = {
                trace_key: timestamp
                for trace_key, timestamp in SpanRecord.select(
                    SpanRecord.trace_key,
                    pw.fn.MAX(
                        pw.fn.COALESCE(
                            SpanRecord.ended_at, SpanRecord.started_at, SpanRecord.observed_at
                        )
                    ).alias("last"),
                )
                .where(SpanRecord.trace_key.in_(trace_keys))
                .group_by(SpanRecord.trace_key)
                .tuples()
            }
            run_activity = {
                trace_key: timestamp
                for trace_key, timestamp in RunRecord.select(
                    SpanRecord.trace_key,
                    pw.fn.MAX(RunRecord.created_at).alias("last"),
                )
                .join(SpanRecord, on=(RunRecord.span_key == SpanRecord.span_key))
                .where(SpanRecord.trace_key.in_(trace_keys))
                .group_by(SpanRecord.trace_key)
                .tuples()
            }

            # A trace expires when both its spans and its runs are old.
            expired = []
            for trace_key in trace_keys:
                activity = span_activity.get(trace_key)
                run_timestamp = run_activity.get(trace_key)
                if run_timestamp is not None and (activity is None or run_timestamp > activity):
                    activity = run_timestamp
                if activity is not None and activity < cutoff:
                    expired.append(trace_key)
            if not expired:
                continue

            # Delete the trace's model runs and their verdicts, then its spans.
            linked_span_keys = SpanRecord.select(SpanRecord.span_key).where(
                SpanRecord.trace_key.in_(expired)
            )
            run_ids = [
                run_id
                for (run_id,) in RunRecord.select(RunRecord.id)
                .where(RunRecord.span_key.in_(linked_span_keys))
                .tuples()
            ]
            for start in range(0, len(run_ids), _PRUNE_CHUNK):
                chunk = run_ids[start : start + _PRUNE_CHUNK]
                CheckRecord.delete().where(CheckRecord.run.in_(chunk)).execute()
                deleted += RunRecord.delete().where(RunRecord.id.in_(chunk)).execute()
            imported_ids = ImportedSessionRecord.select(ImportedSessionRecord.id).where(
                ImportedSessionRecord.trace_key.in_(expired)
            )
            ImportedUsageRecord.delete().where(
                ImportedUsageRecord.session.in_(imported_ids)
            ).execute()
            ImportedSessionRecord.delete().where(
                ImportedSessionRecord.trace_key.in_(expired)
            ).execute()
            for table, column in (("external_identities", "span_key"), ("span_links", "span_key")):
                db_proxy.execute_sql(
                    f"DELETE FROM {table} WHERE {column} IN "
                    f"(SELECT span_key FROM spans WHERE trace_key IN ({','.join('?' for _ in expired)}))",
                    expired,
                )
            SpanRecord.delete().where(SpanRecord.trace_key.in_(expired)).execute()
    return deleted


def _forget_conversations(conversation_ids: list[int]) -> None:
    """Drop deleted conversations from this process's caches: the id memo,
    the turn counters and the response index all name them by row id."""
    path = str(get_settings().db_path)
    gone = set(conversation_ids)
    with _convo_lock:
        cached = [k for k, cid in _conversation_cache.items() if k[0] == path and cid in gone]
        for external in cached:
            del _conversation_cache[external]
        counters = [k for k in _turn_counters if k[0] == path and k[1] in gone]
        for counter in counters:
            del _turn_counters[counter]
        indexed = [k for k, v in _response_index.items() if k[0] == path and v[0] in gone]
        for response in indexed:
            del _response_index[response]


# --- the batch path (what the writer thread drains into) ----------------------------


def write_batch(items: list[dict[str, Any]]) -> None:
    """Persist queued items in one transaction (called from the writer thread).

    Raises on failure — the writer shields and logs, keeping the whole batch
    as the unit of loss rather than half-writing it. The transaction is
    IMMEDIATE, not deferred: ``_insert_span`` and ``_insert_check`` read
    before they write, and SQLite refuses a read→write lock upgrade outright
    when another connection commits in between. A deferred batch would then
    lose *every* item it carried, regular run rows included. A queued
    retention sweep is not part of that unit: it runs after the batch, in
    transactions of its own (see ``prune``).
    """
    if not items:
        return
    groups = []
    for item in items:
        path = item.get("_db_path", str(get_settings().db_path))
        row = {k: v for k, v in item.items() if k != "_db_path"}
        if groups and groups[-1][0] == path:
            groups[-1][1].append(row)
        else:
            groups.append((path, [row]))
    for path, rows in groups:
        with connection_context(path):
            _write_destination_batch(rows)


def _write_destination_batch(items: list[dict[str, Any]]) -> None:
    """Persist one pinned destination, retaining IMMEDIATE batch semantics."""
    if get_db() is None:
        return
    sweeps = [item for item in items if item.get("_kind") == "prune"]
    rows = [item for item in items if item.get("_kind") != "prune"]
    if rows:
        with notification_transaction(db_proxy):
            for item in rows:
                _persist(item)
    for sweep in sweeps:
        _sweep(sweep["retention_days"])


def _persist(item: dict[str, Any]) -> None:
    """Apply one queued item: a run row (the default), a late check verdict,
    a run's adoption into a conversation, or a span upsert.

    These are the dicts record_run/record_check/chain_conversation hand to the
    writer; in sync mode the same functions apply them directly. A ``_kind``
    marker tells the shapes apart — everything else in the dict is column data.
    """
    kind = item.get("_kind")
    if kind == "event":
        _insert_event(item)
    elif kind == "check":
        _insert_check(item)
    elif kind == "adopt":
        _adopt_run(item)
    elif kind == "span":
        _insert_span(item)
    else:
        _insert_run(item)


def _insert_run(row: dict[str, Any]) -> int:
    """Insert one prepared run row plus any bundled check rows; return its row id.

    A row may carry ``_checks`` — verdicts known at record time (all
    pre-checks, plus blocking post-checks). They're written in the same
    transaction as the run so they can never orphan. The input dict is left
    untouched; the writer keeps its batch as the unit of loss and must be
    free to log or drop it intact.
    """
    row = dict(row)
    checks = row.pop("_checks", None)
    try:
        run = RunRecord.create(**row)
    except pw.IntegrityError:
        # Retention can delete a conversation while one of its turns is on
        # the way — queued here, or cached by another process. Record the
        # turn outside any conversation rather than lose it; forgetting the
        # conversation lets the next turn start it afresh.
        conversation_id = row.get("conversation")
        if conversation_id is None or ConversationRecord.get_or_none(id=conversation_id):
            raise
        _forget_conversations([conversation_id])
        logger.warning(
            "promptkeep: conversation %s was pruned; run %s recorded outside it",
            conversation_id,
            row["run_key"],
        )
        row.update(conversation=None, turn_index=None)
        run = RunRecord.create(**row)

    # Bundled verdicts ride in the same transaction as their run.
    if checks:
        now = _utcnow()
        CheckRecord.insert_many([{**c, "run": run.id, "created_at": now} for c in checks]).execute()

    # A conversation's "last active" is its newest turn.
    if row.get("conversation") is not None:
        ConversationRecord.update(updated_at=_utcnow()).where(
            ConversationRecord.id == row["conversation"]
        ).execute()
    return run.id


def _insert_check(item: dict[str, Any]) -> None:
    """Insert one late verdict against its run, located by run_key.

    The run row always precedes its verdicts in the queue, so by the time this
    runs it is on disk — or earlier in this same transaction. When it isn't
    (the run was evicted on queue overflow, or its batch failed) the verdict
    has nothing to attach to and is skipped with a warning; one orphaned
    verdict must not fail the whole batch it rode in on.
    """
    run_id = RunRecord.select(RunRecord.id).where(RunRecord.run_key == item["run_key"]).scalar()
    if run_id is None:
        logger.warning(
            "promptkeep: dropping verdict %r — its run %s was never persisted",
            item.get("name"),
            item["run_key"],
        )
        return
    fields = {k: v for k, v in item.items() if k not in ("_kind", "run_key")}
    CheckRecord.create(run=run_id, **fields)


def _insert_span(item: dict[str, Any]) -> None:
    """Insert or update one span, never downgrading a terminal row to running
    or clearing a payload the start row already stored.

    A root's start row may be missing (the queue dropped it, or the span was
    sampled out until a problem occurred); the final row then creates it.
    If an out-of-order running row arrives after a terminal one, it is
    ignored: a completed span must not start running again. The final row of
    a span deliberately omits its input when the start row already carried it
    (one large payload is queued and redacted once), so a missing value here
    never clears a stored one.
    """
    row = {key: value for key, value in item.items() if key != "_kind"}
    existing = SpanRecord.get_or_none(span_key=row["span_key"])
    if existing is None:
        SpanRecord.create(**row)
        _notify_span(row)
        return
    if existing.status != "running" and row.get("status") == "running":
        enrich = {
            k: v
            for k, v in row.items()
            if k not in ("span_key", "status") and getattr(existing, k) is None and v is not None
        }
        if enrich:
            SpanRecord.update(**enrich).where(SpanRecord.span_key == row["span_key"]).execute()
        return
    fields = {key: value for key, value in row.items() if key != "span_key"}
    if fields.get("input_json") is None:
        fields.pop("input_json", None)
    SpanRecord.update(**fields).where(SpanRecord.span_key == row["span_key"]).execute()
    _notify_span(SpanRecord.get(SpanRecord.span_key == row["span_key"]).__data__)


def _adopt_run(item: dict[str, Any]) -> None:
    """File the run(s) that produced a response under a conversation, as its
    turn 0 — the first call of a chain, recorded before there was a chain.

    By response id rather than run key: a call carrying several Prompts wrote
    several rows for the one response, and they are one turn. Only rows still
    outside any conversation are touched, so replaying an adoption (or racing
    another process to it) changes nothing.
    """
    RunRecord.update(conversation=item["conversation"], turn_index=0).where(
        (RunRecord.response_id == item["response_id"]) & RunRecord.conversation.is_null()
    ).execute()


def _drain_to_db(items: list[dict[str, Any]]) -> None:
    """The writer's sink: what its daemon thread calls with each batch. Looks
    ``write_batch`` up at call time so a test can swap it out."""
    write_batch(items)


# The writer is a generic queue that knows nothing about SQLite; this is the
# one place that tells it where batches go. No I/O and no thread at import.
writer.set_sink(_drain_to_db)


class _ImportRedactionError(RuntimeError):
    """Explicit import redaction failed; abort rather than storing plaintext."""


class _ImportStage:
    """Bounded-payload staging with redaction before any temporary disk write."""

    def __init__(self, path: str, settings: Settings) -> None:
        """Use a private temporary database, never the global model proxy."""
        self.db = _Database(path)
        self.db.connect()
        self.settings = settings
        self.dirty = {"spans": set(), "runs": set(), "usage": set()}
        self.db.execute_sql("CREATE TABLE spans (key TEXT PRIMARY KEY, data TEXT NOT NULL)")
        self.db.execute_sql("CREATE TABLE usage (key TEXT PRIMARY KEY, data TEXT NOT NULL)")
        self.db.execute_sql("CREATE TABLE runs (key TEXT PRIMARY KEY, data TEXT NOT NULL)")

    def close(self) -> None:
        """Release the temporary database before its directory is removed."""
        self.db.close()

    def get(self, kind: str, key: str) -> dict | None:
        """Read one staged operation; table names are internal constants."""
        row = self.db.execute_sql(f"SELECT data FROM {kind} WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, kind: str, key: str, values: dict) -> None:
        """Merge one normalized record, retaining previously observed payloads."""
        existing = self.get(kind, key) or {}
        # All delivery modes prepare payloads through the same pure normalizer.
        values = dict(values)
        if kind == "spans":
            for field in ("input_json", "output_json", "error"):
                if field in values and values[field] is not None:
                    try:
                        values[field] = payload(
                            values[field], self.settings, serialized=field == "error"
                        )
                    except Exception:
                        raise _ImportRedactionError("Trace payload redaction failed") from None
        merged = merge(existing, values)
        if merged == existing:
            return
        existing = merged
        self.dirty[kind].add(key)
        self.db.execute_sql(
            f"INSERT INTO {kind}(key,data) VALUES (?,?) "
            "ON CONFLICT(key) DO UPDATE SET data=excluded.data",
            (key, json.dumps(existing)),
        )

    def append_text(self, key: str, part_id: str, text: str) -> None:
        """Assemble immutable transcript parts, redacting each part only once."""
        row = self.get("spans", key)
        parts = json.loads(row.get("content_parts_json") or "{}")
        fingerprint = hashlib.sha256(text.encode()).hexdigest()
        if part_id in parts and parts[part_id][0] == fingerprint:
            return
        value = text
        if self.settings.redact:
            try:
                value = self.settings.redact(value)
                if not isinstance(value, str):
                    raise TypeError("redact must return a string")
            except Exception:
                raise _ImportRedactionError("Claude text redaction failed") from None
        parts[part_id] = (fingerprint, value)
        budget = 65536
        truncated = False
        for identity, (digest, content) in list(parts.items()):
            encoded_part = content.encode("utf-8")
            if len(encoded_part) > budget:
                content = encoded_part[:budget].decode("utf-8", errors="ignore")
                truncated = True
            parts[identity] = (digest, content)
            budget -= len(content.encode("utf-8"))
        combined = "".join(part[1] for part in parts.values())
        if truncated:
            combined += "\n[promptkeep truncated]"
        encoded = combined.encode("utf-8")
        if len(encoded) > 65536:
            combined = encoded[:65536].decode("utf-8", errors="ignore") + "\n[promptkeep truncated]"
        row.update(content_parts_json=json.dumps(parts), output_json=json.dumps(combined))
        self.db.execute_sql("UPDATE spans SET data=? WHERE key=?", (json.dumps(row), key))
        self.dirty["spans"].add(key)

    def delete_prefix(self, kind, prefix):
        """Remove obsolete staged observations and mark deletions for checkpoint commits."""
        for key, _ in self.rows(kind):
            if key.startswith(prefix):
                self.db.execute_sql(f"DELETE FROM {kind} WHERE key=?", (key,))
                self.dirty[kind].add(key)

    def event(self, item):
        """Reduce an already prepared normalized event without repeating controls."""
        if item.get("links"):
            item["span"]["_link_keys"] = item["links"]
        self.put("spans", item["span"]["span_key"], item["span"])
        if item.get("run") is not None:
            self.put("runs", item["run"]["run_key"], item["run"])

    def rows(self, kind: str):
        """Stream staged normalized rows without materializing their payloads."""
        cursor = self.db.execute_sql(f"SELECT key,data FROM {kind} ORDER BY key")
        for key, data in cursor:
            yield key, json.loads(data)


def persist_import(
    stage: _ImportStage,
    *,
    db_path: str,
    session_id: str,
    trace_key: str,
    digest: str,
    revision: int,
    summary: dict,
    source: str = "codex",
    checkpoint: dict | None = None,
) -> str:
    """Atomically replace one import on an independent, pinned connection.

    Explicit archival writes raise and bypass the lossy live-telemetry queue.
    Models remain bound to the application's database throughout the import.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    database = _open_database(path)
    try:
        # Migration and import both use the same concrete destination.
        with database.atomic("IMMEDIATE"):
            migrate(database)
            existing = database.execute_sql(
                "SELECT id,source_digest,parser_revision,trace_key FROM imported_sessions "
                "WHERE source=? AND external_session_id=?",
                (source, session_id),
            ).fetchone()
            if (
                existing
                and existing[1:3] == (digest, revision)
                and not (checkpoint and checkpoint.get("reconcile"))
            ):
                if checkpoint is not None:
                    _save_checkpoint(database, checkpoint)
                return "unchanged"
            if existing:
                database.execute_sql(
                    "DELETE FROM imported_usage WHERE session_id=?", (existing[0],)
                )
                # Only imported rows with this reserved trace identity are owned here.
                owned_trace = existing[3]
                keys = "SELECT span_key FROM spans WHERE trace_key=?"
                database.execute_sql(
                    f"DELETE FROM checks WHERE run_id IN (SELECT id FROM runs WHERE span_key IN ({keys}))",
                    (owned_trace,),
                )
                for table in ("span_links", "external_identities"):
                    database.execute_sql(
                        f"DELETE FROM {table} WHERE span_key IN ({keys})", (owned_trace,)
                    )
                database.execute_sql(f"DELETE FROM runs WHERE span_key IN ({keys})", (owned_trace,))
                database.execute_sql("DELETE FROM spans WHERE trace_key=?", (owned_trace,))
                database.execute_sql("DELETE FROM imported_sessions WHERE id=?", (existing[0],))
            cursor = database.execute_sql(
                "INSERT INTO imported_sessions(source,external_session_id,trace_key,source_digest,"
                "parser_revision,imported_at,summary_json) VALUES (?,?,?,?,?,?,?)",
                (source, session_id, trace_key, digest, revision, _utcnow(), json.dumps(summary)),
            )
            imported_id = cursor.lastrowid
            for _, row in stage.rows("spans"):
                links = row.get("_link_keys", ())
                for linked in links:
                    database.execute_sql(
                        "INSERT INTO span_links(span_key,linked_span_key) VALUES (?,?) ON CONFLICT DO NOTHING",
                        (row["span_key"], linked),
                    )
                row = {k: v for k, v in row.items() if not k.startswith("_")}
                columns = list(row)
                database.execute_sql(
                    f"INSERT INTO spans ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
                    tuple(row.values()),
                )
            for _, row in stage.rows("runs"):
                columns = list(row)
                database.execute_sql(
                    f"INSERT INTO runs ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
                    tuple(row.values()),
                )
            if checkpoint is not None:
                _save_checkpoint(database, checkpoint)
            for key, row in stage.rows("usage"):
                database.execute_sql(
                    "INSERT INTO imported_usage(usage_key,session_id,turn_id,response_id,span_key,"
                    "scope,observed_at,usage_json) VALUES (?,?,?,?,?,?,?,?)",
                    (
                        f"{trace_key}:{key}",
                        imported_id,
                        row.get("turn_id"),
                        row.get("response_id"),
                        row.get("span_key"),
                        row["scope"],
                        row.get("observed_at"),
                        json.dumps(row["usage"]),
                    ),
                )
        return "replaced" if existing else "created"
    finally:
        if not database.is_closed():
            database.close()


_notifications = threading.local()
_span_observers: list = []
_observer_lock = threading.RLock()


def observe_spans(callback):
    """Subscribe to completed, redacted live spans; return an unsubscribe callback."""
    with _observer_lock:
        _span_observers.append(callback)

    def unsubscribe():
        """Remove only this installation's observer, idempotently."""
        with _observer_lock:
            if callback in _span_observers:
                _span_observers.remove(callback)

    return unsubscribe


@contextmanager
def notification_transaction(database):
    """Publish committed observations only; rollback discards pending exports."""
    outer = getattr(_notifications, "buffer", None)
    if outer is not None:
        start = len(outer)
        try:
            with database.atomic("IMMEDIATE"):
                yield
        except BaseException:
            del outer[start:]
            raise
        return
    _notifications.buffer = []
    try:
        with database.atomic("IMMEDIATE"):
            yield
        ready = _notifications.buffer
    except BaseException:
        raise
    else:
        _notifications.buffer = None
        for row in ready:
            _notify_span(row)
    finally:
        _notifications.buffer = None


def _notify_span(row: dict) -> None:
    """Observers enqueue exports only; their failures cannot break persistence."""
    if row.get("status") == "running":
        return
    buffer = getattr(_notifications, "buffer", None)
    if buffer is not None:
        buffer.append(dict(row))
        return
    with _observer_lock:
        callbacks = tuple(_span_observers)
    for callback in callbacks:
        try:
            callback(dict(row))
        except Exception:
            logger.warning("promptkeep: span observer failed", exc_info=True)


def _insert_event(item: dict) -> None:
    """Upsert one source-scoped event and its model run exactly once."""
    span = item["span"]
    source, trace_id, external_id = item["identity"]
    db_proxy.execute_sql(
        "INSERT INTO external_identities(source,external_trace_id,external_span_id,span_key) "
        "VALUES (?,?,?,?) ON CONFLICT(source,external_trace_id,external_span_id) DO NOTHING",
        (source, trace_id, external_id, span["span_key"]),
    )
    for linked in item.get("links", ()):
        db_proxy.execute_sql(
            "INSERT INTO span_links(span_key,linked_span_key) VALUES (?,?) ON CONFLICT DO NOTHING",
            (span["span_key"], linked),
        )
    run = item.get("run")
    if run is not None:
        # Provider wrappers queue their enriched rows before framework end events.
        existing = RunRecord.select().where(RunRecord.span_key == span["span_key"]).exists()
        if existing:
            for column in (
                "model",
                "response_id",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "cost_usd",
                "usage_details",
            ):
                if run.get(column) is not None:
                    db_proxy.execute_sql(
                        f"UPDATE runs SET {column}=? WHERE span_key=? AND {column} IS NULL",
                        (run[column], span["span_key"]),
                    )
        else:
            columns = list(run)
            db_proxy.execute_sql(
                f"INSERT INTO runs ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)}) "
                "ON CONFLICT(run_key) DO NOTHING",
                tuple(run.values()),
            )
    _insert_span(span)


def _save_checkpoint(database, checkpoint: dict) -> None:
    """Commit a watcher cursor in the same transaction as its ingested events."""
    database.execute_sql(
        "INSERT INTO watch_checkpoints(source_key,generation,byte_offset,parser_revision,state_json) "
        "VALUES (?,?,?,?,?) ON CONFLICT(source_key) DO UPDATE SET generation=excluded.generation,"
        "byte_offset=excluded.byte_offset,parser_revision=excluded.parser_revision,state_json=excluded.state_json",
        (
            checkpoint["source_key"],
            checkpoint["generation"],
            checkpoint["byte_offset"],
            checkpoint["parser_revision"],
            json.dumps(checkpoint["state"]),
        ),
    )


def watch_snapshot(db_path: str, source_key: str) -> tuple[dict | None, dict]:
    """Load committed parser state and redacted owned rows on a concrete connection."""
    database = _open_database(db_path)
    try:
        with database.atomic("IMMEDIATE"):
            migrate(database)
        row = database.execute_sql(
            "SELECT generation,byte_offset,parser_revision,state_json "
            "FROM watch_checkpoints WHERE source_key=?",
            (source_key,),
        ).fetchone()
        if row is None:
            return None, {}
        checkpoint = dict(
            generation=row[0], byte_offset=row[1], parser_revision=row[2], state=json.loads(row[3])
        )
        trace = checkpoint["state"].get("trace")
        data = {}
        for table, sql, parameters in (
            ("spans", "SELECT * FROM spans WHERE trace_key=?", (trace,)),
            (
                "runs",
                "SELECT * FROM runs WHERE span_key IN (SELECT span_key FROM spans WHERE trace_key=?)",
                (trace,),
            ),
        ):
            cursor = database.execute_sql(sql, parameters)
            names = [column[0] for column in cursor.description]
            data[table] = [{k: v for k, v in zip(names, values) if k != "id"} for values in cursor]
        cursor = database.execute_sql(
            "SELECT usage_key,turn_id,response_id,span_key,scope,observed_at,usage_json "
            "FROM imported_usage WHERE session_id IN (SELECT id FROM imported_sessions WHERE trace_key=?)",
            (trace,),
        )
        data["usage"] = [
            (
                key.split(":", 1)[1],
                dict(
                    turn_id=turn,
                    response_id=response,
                    span_key=span,
                    scope=scope,
                    observed_at=observed,
                    usage=json.loads(usage),
                ),
            )
            for key, turn, response, span, scope, observed, usage in cursor
        ]
        return checkpoint, data
    finally:
        database.close()


def persist_watch(
    stage: _ImportStage,
    *,
    db_path: str,
    source: str,
    session_id: str,
    trace_key: str,
    summary: dict,
    checkpoint: dict,
) -> None:
    """Upsert changed observations and their cursor atomically; never delete on rotation."""
    if checkpoint.get("reconcile"):
        persist_import(
            stage,
            db_path=db_path,
            source=source,
            session_id=session_id,
            trace_key=trace_key,
            digest="watch:" + str(checkpoint["byte_offset"]),
            revision=checkpoint["parser_revision"],
            summary=summary,
            checkpoint=checkpoint,
        )
        checkpoint.pop("reconcile", None)
        for keys in stage.dirty.values():
            keys.clear()
        return
    database = _open_database(db_path)
    try:
        with database.atomic("IMMEDIATE"):
            migrate(database)
            database.execute_sql(
                "INSERT INTO imported_sessions(source,external_session_id,trace_key,source_digest,"
                "parser_revision,imported_at,summary_json) VALUES (?,?,?,?,?,?,?) "
                "ON CONFLICT(source,external_session_id) DO UPDATE SET summary_json=excluded.summary_json,"
                "imported_at=excluded.imported_at,source_digest=excluded.source_digest,parser_revision=excluded.parser_revision",
                (
                    source,
                    session_id,
                    trace_key,
                    "watch:" + str(checkpoint["byte_offset"]),
                    checkpoint["parser_revision"],
                    _utcnow(),
                    json.dumps(summary),
                ),
            )
            imported_id = database.execute_sql(
                "SELECT id FROM imported_sessions WHERE source=? AND external_session_id=?",
                (source, session_id),
            ).fetchone()[0]
            for kind, key_column in (("spans", "span_key"), ("runs", "run_key")):
                for key in stage.dirty[kind]:
                    staged = stage.get(kind, key)
                    for linked in staged.get("_link_keys", ()):
                        database.execute_sql(
                            "INSERT INTO span_links(span_key,linked_span_key) VALUES (?,?) ON CONFLICT DO NOTHING",
                            (key, linked),
                        )
                    row = {k: v for k, v in staged.items() if not k.startswith("_")}
                    columns = list(row)
                    assignments = ",".join(f"{k}=excluded.{k}" for k in columns if k != key_column)
                    database.execute_sql(
                        f"INSERT INTO {kind} ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)}) "
                        f"ON CONFLICT({key_column}) DO UPDATE SET {assignments}",
                        tuple(row.values()),
                    )
            for key in stage.dirty["usage"]:
                row = stage.get("usage", key)
                if row is None:
                    database.execute_sql(
                        "DELETE FROM imported_usage WHERE usage_key=?", (trace_key + ":" + key,)
                    )
                    continue
                database.execute_sql(
                    "INSERT INTO imported_usage(usage_key,session_id,turn_id,response_id,span_key,scope,observed_at,usage_json) "
                    "VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(usage_key) DO UPDATE SET usage_json=excluded.usage_json,observed_at=excluded.observed_at",
                    (
                        trace_key + ":" + key,
                        imported_id,
                        row.get("turn_id"),
                        row.get("response_id"),
                        row.get("span_key"),
                        row["scope"],
                        row.get("observed_at"),
                        json.dumps(row["usage"]),
                    ),
                )
            _save_checkpoint(database, checkpoint)
        for keys in stage.dirty.values():
            keys.clear()
    finally:
        database.close()

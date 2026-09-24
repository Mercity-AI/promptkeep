"""SQLite persistence: the database binding and every write path.

The tables live in ``models``, schema upgrades in ``migrations``, the read
side in ``history``. This module owns the connection and the writes: version
registration, conversation resolution, run and verdict recording, and the
batch insert the background writer drains into.

All write paths that run implicitly (version registration on first render,
run recording inside a wrapped call) are exception-shielded: a broken DB
loses telemetry, never a completion.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import peewee as pw

from . import controls, writer
from .config import get_settings
from .migrations import migrate
from .models import (
    CheckRecord,
    ConversationRecord,
    PromptRecord,
    PromptVersionRecord,
    RunRecord,
    db_proxy,
    new_run_key,
)
from .rendering import template_hash

logger = logging.getLogger("promptkeep")


# --- the database binding -------------------------------------------------------

_db_lock = threading.Lock()
_current_path: str | None = None

# WAL for concurrent reader/writer access; busy_timeout so contending
# writers wait instead of failing instantly.
_PRAGMAS = {
    "journal_mode": "wal",
    "foreign_keys": 1,
    "busy_timeout": 5000,
}


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
    if _current_path != path:
        with _db_lock:
            if _current_path != path:
                parent = Path(path).parent
                if parent and not parent.exists():
                    parent.mkdir(parents=True, exist_ok=True)
                database = pw.SqliteDatabase(path, pragmas=_PRAGMAS, timeout=5)
                db_proxy.initialize(database)
                database.connect(reuse_if_open=True)
                migrate(database)
                _current_path = path
    return db_proxy


def reset_caches() -> None:
    """Drop memoized registrations, conversation/turn caches, anything still
    queued in the background writer, and the DB binding. Mainly for tests."""
    global _current_path
    writer.reset()
    with _reg_lock:
        _registration_cache.clear()
    with _convo_lock:
        _conversation_cache.clear()
        _turn_counters.clear()
        _response_index.clear()
    with _db_lock:
        if _current_path is not None:
            try:
                db_proxy.close()
            except Exception:
                pass
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


def predecessor(previous_response_id: str) -> str | None:
    """The key of the run that produced ``previous_response_id`` — the parent
    of a call that continues it — or None when promptkeep never recorded that
    response (or isn't recording now). Raises like any storage call; the
    caller shields."""
    if not _recording():
        return None
    found = _lookup_response(previous_response_id)
    return None if found is None else found[1]


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
    checks: list[dict[str, Any]] | None = None,
) -> str | None:
    """Record one run row, honoring the configured write_mode. Never raises.

    Returns the run's key — the identity everything else references it by
    (a RunHandle, history.checks(), a verdict landing later) — or None when
    nothing was recorded (tracking disabled, write_mode "off", sampled out,
    a failed redaction, or the write failed). The key is minted here when
    the caller didn't bring one, and it is valid the moment this returns: in
    background mode the row may still be in the queue, but the key already
    names it.

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
            "latency_ms": latency_ms,
            "status": status,
            "error": error,
            "created_at": _utcnow(),
            "conversation": conversation_id,
            "turn_index": turn_index,
            "input_text": input_text,
            "original_input_text": original_input_text,
            "parent_run_key": parent_run_key,
        }
        if checks:
            row["_checks"] = list(checks)

        # The production controls, before the row can reach a queue or a disk.
        if not controls.keep_run(settings, row, conversation_id):
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


# --- the batch path (what the writer thread drains into) ----------------------------


def write_batch(items: list[dict[str, Any]]) -> None:
    """Persist queued items in one transaction (called from the writer thread).

    Raises on failure — the writer shields and logs, keeping the whole batch
    as the unit of loss rather than half-writing it.
    """
    if not items or get_db() is None:
        return
    with db_proxy.atomic():
        for item in items:
            _persist(item)


def _persist(item: dict[str, Any]) -> None:
    """Apply one queued item: a run row (the default), a late check verdict,
    or a run's adoption into a conversation.

    These are the dicts record_run/record_check/chain_conversation hand to the
    writer; in sync mode the same functions apply them directly. A ``_kind``
    marker tells the shapes apart — everything else in the dict is column data.
    """
    kind = item.get("_kind")
    if kind == "check":
        _insert_check(item)
    elif kind == "adopt":
        _adopt_run(item)
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

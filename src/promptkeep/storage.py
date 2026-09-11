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
import random
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import peewee as pw

logger = logging.getLogger("promptkeep")

_SCHEMA_VERSION = 5

# The models bind to this proxy; _get_db() points it at the configured file.
_proxy = pw.DatabaseProxy()
_db_lock = threading.Lock()
_current_path: str | None = None

# Registration results memoized per (db, name, template-hash) so repeated
# renders of the same prompt cost zero DB round-trips.
_registration_cache: dict = {}
_reg_lock = threading.Lock()

# Conversations memoized per (db, external_id) the same way, and turn numbers
# handed out from in-process counters per (db, conversation_id): the DB's
# MAX(turn_index) goes stale the moment rows sit in the background write
# queue, so within a process the counter is the source of truth. (Two
# *processes* driving one conversation concurrently can still race — same
# caveat as before, now confined to the cross-process case.)
_conversation_cache: dict = {}
_turn_counters: dict = {}
_convo_lock = threading.Lock()

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


class ConversationRecord(BaseModel):
    """One multi-turn session; its runs are the ordered turns within it."""

    external_id = pw.TextField(unique=True)
    title = pw.TextField(null=True)
    metadata = pw.TextField(null=True)
    created_at = pw.TextField()
    updated_at = pw.TextField()

    class Meta:
        table_name = "conversations"


class RunRecord(BaseModel):
    """One execution: a tracked prompt version and/or a conversation turn.

    run_key is the run's public identity — a UUID minted by the caller at call
    time (see new_run_key), before the row exists. Everything that references
    a run (a RunHandle, a late check verdict, history.checks) does so by key,
    which is what lets the row itself travel through the background writer:
    nothing has to wait for SQLite to assign the integer id. The id stays the
    primary key for joins and foreign keys; it is not part of the public API.

    version is nullable because a conversation turn need not involve a
    wrapped Prompt (a plain follow-up message still gets a row so the whole
    session can be replayed). Usually a run has a version, a conversation, or
    both; the one exception is a checked call outside any conversation with no
    wrapped Prompt, whose row exists solely to anchor its check verdicts.

    original_input_text is set only when a pre-check rewrote the outgoing turn:
    input_text is then what was actually sent, and this column preserves what
    the caller originally passed, so the audit trail shows both sides.
    """

    run_key = pw.TextField(unique=True)
    version = pw.ForeignKeyField(
        PromptVersionRecord, column_name="version_id", backref="runs", null=True
    )
    variables = pw.TextField(null=True)
    rendered_text = pw.TextField(null=True)
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
    conversation = pw.ForeignKeyField(
        ConversationRecord, column_name="conversation_id", backref="runs", null=True
    )
    turn_index = pw.IntegerField(null=True)
    input_text = pw.TextField(null=True)
    original_input_text = pw.TextField(null=True)

    class Meta:
        table_name = "runs"
        indexes = (
            (("version", "created_at"), False),
            (("conversation", "turn_index"), False),
        )


class CheckRecord(BaseModel):
    """One check verdict against a run — the label store for optimization.

    phase is 'pre' | 'post'; status is 'ok' | 'warn' | 'block' | 'error'.
    An async post-check's row is written when its verdict lands, so a run can
    accumulate check rows after it was itself recorded. rewritten holds the
    replacement text a rewriting pre-check produced (None for every other
    verdict), so the audit trail names which check changed the turn and to
    what — the run's original_input_text holds what it changed *from*.
    """

    run = pw.ForeignKeyField(RunRecord, column_name="run_id", backref="checks")
    name = pw.TextField()
    phase = pw.TextField()
    status = pw.TextField()
    score = pw.FloatField(null=True)
    message = pw.TextField(null=True)
    latency_ms = pw.IntegerField(null=True)
    rewritten = pw.TextField(null=True)
    created_at = pw.TextField()

    class Meta:
        table_name = "checks"
        indexes = ((("run", "name"), False),)


_MODELS = [PromptRecord, PromptVersionRecord, ConversationRecord, RunRecord, CheckRecord]


# --- helpers -------------------------------------------------------------------


def _utcnow() -> str:
    """Current UTC time as an ISO-8601 string (how all timestamps are stored)."""
    return datetime.now(UTC).isoformat()


def new_run_key() -> str:
    """Mint a run's identity: a random UUID, assigned *before* anything is written.

    Runs are referenced by this key rather than by their integer row id
    precisely so the identity exists at call time. The background writer may
    not have inserted the row yet when the caller needs to point at it (a
    RunHandle on the response, an async post-check's verdict landing later),
    and a key minted client-side never has to wait for the database — the
    same reason tracing systems mint trace ids in the client.
    """
    return str(uuid.uuid4())


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


def _json_or_none(obj: Any) -> str | None:
    """Serialize to JSON for storage; non-serializable values fall back to repr()."""
    if obj is None:
        return None
    return json.dumps(obj, ensure_ascii=False, default=repr)


# --- database lifecycle ----------------------------------------------------------


def _get_db() -> pw.DatabaseProxy | None:
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
    """Apply forward-only schema steps until the DB reaches _SCHEMA_VERSION.

    Steps compose regardless of which version a real DB starts at (e.g. 1 -> 3
    runs both the v2 and v3 steps) — each guard tests the version the DB began
    at. Every step runs in its own transaction that also stamps the new
    `user_version`, so a step and its version bump commit together: an
    interrupted step rolls back whole and re-runs from the same version rather
    than replaying half of it into a "duplicate column" error. A brand-new DB
    is created straight from `_MODELS` at the latest shape, so it skips the
    legacy fix-up steps meant for pre-existing rows.
    """
    (user_version,) = database.execute_sql("PRAGMA user_version").fetchone()
    if user_version < 1:
        with database.atomic():
            database.create_tables(_MODELS, safe=True)
            database.execute_sql(f"PRAGMA user_version = {_SCHEMA_VERSION}")
        return
    if user_version < 2:
        # v2: template_hash became a hash of the *normalized* template
        # (variable names canonicalized). Recompute stored hashes so old
        # rows keep deduping correctly against new registrations.
        with database.atomic():
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
            database.execute_sql("PRAGMA user_version = 2")
    if user_version < 3:
        # v3: conversations. runs.version_id/rendered_text drop NOT NULL
        # because a conversation turn with no wrapped Prompt still gets a
        # row (it has no version to attach to), and the new columns link a
        # run to its conversation and turn position.
        from playhouse.migrate import SqliteMigrator, migrate

        with database.atomic():
            database.create_tables([ConversationRecord], safe=True)
            migrator = SqliteMigrator(database)
            migrate(
                migrator.add_column("runs", "conversation_id", pw.IntegerField(null=True)),
                migrator.add_column("runs", "turn_index", pw.IntegerField(null=True)),
                migrator.add_column("runs", "input_text", pw.TextField(null=True)),
                migrator.drop_not_null("runs", "version_id"),
                migrator.drop_not_null("runs", "rendered_text"),
            )
            database.execute_sql("PRAGMA user_version = 3")
    if user_version < 4:
        # v4: the checks table — one verdict per (run, check).
        with database.atomic():
            database.create_tables([CheckRecord], safe=True)
            database.execute_sql("PRAGMA user_version = 4")
    if user_version < 5:
        # v5: runs gain a caller-minted identity (run_key) so a run can be
        # referenced before its row lands — checked calls go through the
        # background writer like everything else, and late verdicts find
        # their run by key. Plus the rewrite audit trail: the turn as the
        # caller passed it (runs.original_input_text) and what the rewriting
        # check produced (checks.rewritten).
        from playhouse.migrate import SqliteMigrator, migrate

        with database.atomic():
            migrator = SqliteMigrator(database)
            operations = [
                migrator.add_column("runs", "run_key", pw.TextField(null=True)),
                migrator.add_column("runs", "original_input_text", pw.TextField(null=True)),
            ]
            # A DB that started below v4 just had its checks table created
            # from the current model (rewritten included) in the step above;
            # one that started at v4 has the older shape and needs the column.
            if not _has_column(database, "checks", "rewritten"):
                operations.append(
                    migrator.add_column("checks", "rewritten", pw.TextField(null=True))
                )
            migrate(*operations)
            # Backfill so every pre-existing run has a key: the column is the
            # run's identity from here on, and readers rely on it being set.
            for (row_id,) in database.execute_sql("SELECT id FROM runs").fetchall():
                database.execute_sql(
                    "UPDATE runs SET run_key = ? WHERE id = ?", (new_run_key(), row_id)
                )
            migrate(migrator.add_index("runs", ("run_key",), True))
            database.execute_sql("PRAGMA user_version = 5")


def _has_column(database: pw.SqliteDatabase, table: str, column: str) -> bool:
    """Whether ``table`` already has ``column`` — for migration steps whose
    starting shape depends on which earlier steps ran in the same pass."""
    return any(col.name == column for col in database.get_columns(table))


def reset_caches() -> None:
    """Drop memoized registrations, conversation/turn caches, anything still
    queued in the background writer, and the DB binding. Mainly for tests."""
    global _current_path
    from . import writer

    writer.reset()
    with _reg_lock:
        _registration_cache.clear()
    with _convo_lock:
        _conversation_cache.clear()
        _turn_counters.clear()
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
    fn_source_hash: str | None = None,
    exact_match: bool = False,
) -> tuple[int, int] | None:
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


def _register(name, template, content_hash, source, fn_source_hash) -> tuple[int, int]:
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


def get_or_create_conversation(
    external_id: str,
    title: str | None = None,
    metadata: dict | None = None,
) -> int | None:
    """Idempotently resolve a conversation by its caller-supplied id.

    Creates the row on first sight (the "never break the caller" rule
    extends to conversations: an unrecognized id is not an error) and
    memoizes the result, so a long conversation costs one DB round-trip,
    not one per turn. Returns None when tracking is disabled or the write
    fails.
    """
    from .config import get_settings

    settings = get_settings()
    if not settings.enabled:
        return None
    cache_key = (str(settings.db_path), external_id)
    with _convo_lock:
        cached = _conversation_cache.get(cache_key)
    if cached is not None:
        return cached
    try:
        if _get_db() is None:
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
    from .config import get_settings

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


def record_run(
    *,
    run_key: str | None = None,
    version_id: int | None = None,
    variables: dict | None = None,
    rendered_text: str | None = None,
    provider: str,
    model: str | None = None,
    request_params: dict | None = None,
    response_id: str | None = None,
    output_text: str | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    latency_ms: int | None = None,
    status: str = "ok",
    error: str | None = None,
    conversation_id: int | None = None,
    turn_index: int | None = None,
    input_text: str | None = None,
    original_input_text: str | None = None,
    checks: list | None = None,
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
    gets a row (there's simply nothing to attach to a lineage). A row usually
    has a version_id, a conversation_id, or both; the deliberate exception is a
    checked call that has neither, whose row exists only to anchor its check
    verdicts. Nothing is enforced here regardless — failing loudly would
    violate the shielding this function exists to provide.

    turn_index is normally left unset and reserved here. Callers that write
    more than one row for the same physical API call (a message with more
    than one tracked Prompt in it) should reserve one via
    reserve_turn_index() up front and pass it to every row from that call —
    otherwise each row would claim its own turn, splitting one exchange into
    several.
    """
    try:
        from .config import get_settings

        settings = get_settings()
        if not settings.enabled or settings.write_mode == "off":
            return None
        if run_key is None:
            run_key = new_run_key()
        if turn_index is None and conversation_id is not None:
            turn_index = reserve_turn_index(conversation_id)
        row = {
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
            "latency_ms": latency_ms,
            "status": status,
            "error": error,
            "created_at": _utcnow(),
            "conversation": conversation_id,
            "turn_index": turn_index,
            "input_text": input_text,
            "original_input_text": original_input_text,
        }
        if checks:
            row["_checks"] = list(checks)
        # The production controls, applied at the one point every run row
        # passes through, before it can reach a queue or a disk.
        if not _keep_run(settings, row, conversation_id):
            return None
        if settings.redact is not None:
            try:
                row = _redact_run(settings.redact, row)
            except Exception:
                logger.warning(
                    "promptkeep: redact hook failed — run dropped rather than stored unredacted",
                    exc_info=True,
                )
                return None
        if settings.write_mode == "background":
            from . import writer

            writer.submit(row)
            return run_key
        if _get_db() is None:
            return None
        _insert_run(row)
        return run_key
    except Exception:
        logger.warning("promptkeep: failed to record run", exc_info=True)
        return None


# --- production controls: sampling and redaction --------------------------------

# Run-row fields that can carry user data, and the check-row ones. Everything
# else on a row is an id, a number, a status or a timestamp.
_REDACTED_RUN_FIELDS = (
    "rendered_text",
    "variables",
    "request_params",
    "output_text",
    "error",
    "input_text",
    "original_input_text",
)
_REDACTED_CHECK_FIELDS = ("message", "rewritten")


def _keep_run(settings, row: dict, conversation_id: int | None) -> bool:
    """The sampling decision for one run.

    Anything worth investigating is always kept: a call that failed or was
    blocked, or one carrying a verdict that isn't plain ok (a warning
    included — ``Verdict.from_score`` below its threshold is a warn). Only an
    uneventful run is subject to ``sample_rate``.

    Inside a conversation the decision is derived from the conversation's id
    rather than drawn per turn, so one session is kept whole or dropped whole
    (no holes in a replay) and every process writing to the same file makes
    the same call. Async post-check verdicts land after this decision and
    can't rescue a dropped run — use ``mode="blocking"`` for a check whose
    failures must always be kept.
    """
    rate = settings.sample_rate
    if rate >= 1.0:
        return True
    if row["status"] != "ok":
        return True
    if any(check.get("status") != "ok" for check in row.get("_checks", ())):
        return True
    if conversation_id is None:
        return random.random() < rate
    digest = hashlib.sha256(str(conversation_id).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < rate


def _redact_fields(redact, row: dict, fields: tuple) -> dict:
    """A copy of ``row`` with each string-valued field in ``fields`` passed
    through the hook. Raises if the hook does — the caller decides what a
    failed redaction means (it always means "don't store")."""
    redacted = dict(row)
    for field in fields:
        value = redacted.get(field)
        if isinstance(value, str):
            result = redact(value)
            if not isinstance(result, str):
                raise TypeError(f"redact hook must return a str, got {type(result).__name__}")
            redacted[field] = result
    return redacted


def _redact_run(redact, row: dict) -> dict:
    """Redact a run row and any check rows bundled with it."""
    redacted = _redact_fields(redact, row, _REDACTED_RUN_FIELDS)
    if redacted.get("_checks"):
        redacted["_checks"] = [
            _redact_fields(redact, check, _REDACTED_CHECK_FIELDS) for check in redacted["_checks"]
        ]
    return redacted


def record_check(run_key: str | None, check_row: dict) -> None:
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
        from .config import get_settings

        settings = get_settings()
        if not settings.enabled or settings.write_mode == "off":
            return
        item = {"_kind": "check", "run_key": run_key, "created_at": _utcnow(), **check_row}
        if settings.redact is not None:
            try:
                item = _redact_fields(settings.redact, item, _REDACTED_CHECK_FIELDS)
            except Exception:
                logger.warning(
                    "promptkeep: redact hook failed — verdict dropped rather than stored unredacted",
                    exc_info=True,
                )
                return
        if settings.write_mode == "background":
            from . import writer

            writer.submit(item)
            return
        if _get_db() is None:
            return
        _insert_check(item)
    except Exception:
        logger.warning("promptkeep: failed to record check", exc_info=True)


def write_batch(items: list) -> None:
    """Persist queued items in one transaction (called from the writer thread).

    Raises on failure — the writer shields and logs, keeping the whole batch
    as the unit of loss rather than half-writing it.
    """
    if not items or _get_db() is None:
        return
    with _proxy.atomic():
        for item in items:
            _persist(item)


def _persist(item: dict) -> None:
    """Apply one queued item: a run row (the default) or a late check verdict.

    These are the dicts record_run/record_check hand to the writer; in sync
    mode the same two functions insert directly. A ``_kind`` marker tells the
    shapes apart — everything else in the dict is column data.
    """
    if item.get("_kind") == "check":
        _insert_check(item)
    else:
        _insert_run(item)


def _insert_run(row: dict) -> int:
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
    if checks:
        now = _utcnow()
        CheckRecord.insert_many([{**c, "run": run.id, "created_at": now} for c in checks]).execute()
    if row.get("conversation") is not None:
        ConversationRecord.update(updated_at=_utcnow()).where(
            ConversationRecord.id == row["conversation"]
        ).execute()
    return run.id


def _insert_check(item: dict) -> None:
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


# --- reads (raise on real errors) ---------------------------------------------

# The run/turn projection, defined once. Every run-shaped read selects these
# same columns and differs only in joins/filters/order; listing them per
# function meant a new `runs` column had to be threaded into three places by
# hand. Reusing aliased column objects across queries is safe — building a
# query never mutates them. history._run_info_from_row reads results by key,
# so column order here is irrelevant to callers.
_RUN_COLUMNS = (
    RunRecord.id,
    RunRecord.run_key,
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
    RunRecord.turn_index,
    RunRecord.input_text,
    RunRecord.original_input_text,
)
# Appended only where the caller filters across conversations and needs to
# know which one each run belongs to (fetch_conversation_turns already knows).
_CONVERSATION_ID_COLUMN = ConversationRecord.external_id.alias("conversation_id")


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


def fetch_runs(name: str, version: int | None = None, limit: int = 50) -> list:
    """Run rows for a prompt (optionally one version) as dicts, newest first."""
    if _get_db() is None:
        return []
    # Join through versions to prompts so callers filter by name, not ids.
    query = (
        RunRecord.select(*_RUN_COLUMNS, _CONVERSATION_ID_COLUMN)
        .join(PromptVersionRecord)
        .join(PromptRecord)
        .switch(RunRecord)
        .join(ConversationRecord, pw.JOIN.LEFT_OUTER)
        .where(PromptRecord.name == name)
    )
    if version is not None:
        query = query.where(PromptVersionRecord.version == version)
    query = query.order_by(RunRecord.id.desc()).limit(limit).dicts()
    return list(query)


def fetch_all_runs(limit: int = 100) -> list:
    """Every run row regardless of prompt (or with none), newest first.

    Left-joined throughout: a conversation-only turn has no version/prompt
    to join to, and a run outside any conversation has no conversation to
    join to. Both cases surface as NULLs rather than dropping the row.
    """
    if _get_db() is None:
        return []
    query = (
        RunRecord.select(*_RUN_COLUMNS, _CONVERSATION_ID_COLUMN)
        .join(PromptVersionRecord, pw.JOIN.LEFT_OUTER)
        .join(PromptRecord, pw.JOIN.LEFT_OUTER)
        .switch(RunRecord)
        .join(ConversationRecord, pw.JOIN.LEFT_OUTER)
        .order_by(RunRecord.id.desc())
        .limit(limit)
        .dicts()
    )
    return list(query)


def fetch_prompt_summaries() -> list:
    """One row per prompt with its version and run counts, name-sorted.

    COUNT(DISTINCT ...) is required here: joining both versions and runs off
    the same prompt fans out into a cross product, so a plain COUNT would
    double-count whichever side has more rows.
    """
    if _get_db() is None:
        return []
    query = (
        PromptRecord.select(
            PromptRecord.name,
            PromptRecord.created_at,
            pw.fn.COUNT(pw.fn.DISTINCT(PromptVersionRecord.id)).alias("version_count"),
            pw.fn.COUNT(pw.fn.DISTINCT(RunRecord.id)).alias("run_count"),
        )
        .join(PromptVersionRecord, pw.JOIN.LEFT_OUTER)
        .join(RunRecord, pw.JOIN.LEFT_OUTER, on=(RunRecord.version == PromptVersionRecord.id))
        .group_by(PromptRecord.id)
        .order_by(PromptRecord.name)
        .dicts()
    )
    return list(query)


def fetch_conversation_summaries(
    limit: int = 100, prompt: str | None = None, version: int | None = None
) -> list:
    """One row per conversation with its turn count, most recently active first.

    With ``prompt`` (and optionally ``version``), only conversations that have
    at least one run driven by that prompt version are returned. The filter is
    a separate EXISTS-style subquery rather than a condition on the counting
    join, so turn_count stays the conversation's full length.
    """
    if _get_db() is None:
        return []
    query = (
        ConversationRecord.select(
            ConversationRecord.external_id,
            ConversationRecord.title,
            ConversationRecord.created_at,
            ConversationRecord.updated_at,
            pw.fn.COUNT(RunRecord.id).alias("turn_count"),
        )
        .join(RunRecord, pw.JOIN.LEFT_OUTER)
        .group_by(ConversationRecord.id)
        .order_by(ConversationRecord.updated_at.desc())
        .limit(limit)
    )
    if prompt is not None:
        driven_by = (
            RunRecord.select(RunRecord.conversation)
            .join(PromptVersionRecord)
            .join(PromptRecord)
            .where(PromptRecord.name == prompt)
        )
        if version is not None:
            driven_by = driven_by.where(PromptVersionRecord.version == version)
        query = query.where(ConversationRecord.id.in_(driven_by))
    return list(query.dicts())


def fetch_conversation(external_id: str) -> dict | None:
    """A conversation's own row (not its turns) as a dict, or None if unknown."""
    if _get_db() is None:
        return None
    return (
        ConversationRecord.select()
        .where(ConversationRecord.external_id == external_id)
        .dicts()
        .first()
    )


def fetch_conversation_turns(external_id: str) -> list:
    """All turns (runs) of a conversation as dicts, oldest first.

    A turn's version fields are NULL when that turn had no wrapped Prompt —
    e.g. a plain follow-up message with no system-prompt change.
    """
    if _get_db() is None:
        return []
    # No conversation_id column: the caller already knows which conversation.
    query = (
        RunRecord.select(*_RUN_COLUMNS)
        .join(PromptVersionRecord, pw.JOIN.LEFT_OUTER)
        .join(PromptRecord, pw.JOIN.LEFT_OUTER)
        .switch(RunRecord)
        .join(ConversationRecord)
        .where(ConversationRecord.external_id == external_id)
        .order_by(RunRecord.turn_index)
        .dicts()
    )
    return list(query)


def fetch_checks(run_key: str) -> list:
    """All check verdicts recorded against one run (by key), as dicts, oldest first."""
    if _get_db() is None:
        return []
    query = (
        CheckRecord.select(
            CheckRecord.name,
            CheckRecord.phase,
            CheckRecord.status,
            CheckRecord.score,
            CheckRecord.message,
            CheckRecord.latency_ms,
            CheckRecord.rewritten,
            CheckRecord.created_at,
        )
        .join(RunRecord)
        .where(RunRecord.run_key == run_key)
        .order_by(CheckRecord.id)
        .dicts()
    )
    return list(query)

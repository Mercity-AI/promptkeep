"""The SQLite tables, as peewee models.

Four entities: a prompt's identity, its versions (the lineage), the runs
that executed a version and/or belong to a conversation, and the check
verdicts recorded against a run. The models bind to ``db_proxy``, which
``storage.get_db()`` points at the configured file on first use — nothing
here touches a database at import time.
"""

from __future__ import annotations

import uuid

import peewee as pw

# Bound to the real SqliteDatabase by storage.get_db(); models reference it
# so the file can be chosen at runtime (configure / PROMPTKEEP_DB).
db_proxy = pw.DatabaseProxy()


class BaseModel(pw.Model):
    """Base for all tables; binds them to the runtime-configured database."""

    class Meta:
        """Table binding shared by every model."""

        database = db_proxy


class PromptRecord(BaseModel):
    """A prompt's stable identity — one row per unique name."""

    name = pw.TextField(unique=True)
    created_at = pw.TextField()

    class Meta:
        """Table name."""

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
        """Table name and the two uniqueness rules of a lineage."""

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
        """Table name."""

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

    cost_usd is what the provider *said* the call cost, in US dollars — read
    off the response's usage block by the adapter (OpenRouter reports it on
    every response). It is never estimated from a price table: NULL means the
    provider didn't say, which is the honest answer for one that doesn't.

    parent_run_key names the run this one continues from, when that isn't
    simply the previous turn: a regeneration, a branch, a retry, a sub-agent
    call. It is what makes a conversation a tree rather than a list. A key
    rather than a foreign key, for the same reason run_key exists — the
    parent's row may still be in the write queue when its child is recorded
    — so nothing guarantees the parent is (still) on disk; readers treat a
    missing parent as the root of a branch.
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
    cost_usd = pw.FloatField(null=True)
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
    parent_run_key = pw.TextField(null=True)

    class Meta:
        """Table name and the indexes reads depend on."""

        table_name = "runs"
        indexes = (
            (("version", "created_at"), False),
            (("conversation", "turn_index"), False),
            # A chained call finds its predecessor by the provider's response id.
            (("response_id",), False),
            # A run's children, for walking a conversation tree downwards.
            (("parent_run_key",), False),
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
        """Table name and the lookup index."""

        table_name = "checks"
        indexes = ((("run", "name"), False),)


# Creation order: parents before children, so a fresh file can be built in
# one pass.
MODELS = [PromptRecord, PromptVersionRecord, ConversationRecord, RunRecord, CheckRecord]


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

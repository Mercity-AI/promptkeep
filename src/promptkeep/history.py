"""The read side: lineage, runs, conversations and verdicts as frozen
dataclasses.

Every query lives here, next to the type it produces. Reads raise normally —
a broken query is a bug you want to see — but against a disabled tracker
they return nothing rather than opening a file.
"""

from __future__ import annotations

import difflib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import groupby
from typing import Any

import peewee as pw

from . import storage
from .models import CheckRecord, ConversationRecord, PromptRecord, PromptVersionRecord, RunRecord


@dataclass(frozen=True)
class VersionInfo:
    """One version of a prompt's template, as recorded in the lineage."""

    version: int
    template: str
    template_hash: str
    source: str
    fn_source_hash: str | None
    created_at: str


@dataclass(frozen=True)
class RunInfo:
    """One recorded execution: which version ran, with what, and what came back.

    run_key is the run's identity — what response.promptkeep.run_key carries
    and what history.checks() takes. id is the row number, kept for display.

    version/prompt_name are None for a conversation turn that involved no
    wrapped Prompt (a plain follow-up message) — there's simply no lineage
    to attach it to. conversation_id/turn_index/input_text are None for a
    run recorded outside any conversation. original_input_text is set only
    when a pre-check rewrote the turn: input_text is then what was sent, and
    this is what the caller originally passed. cost_usd is the cost the
    provider reported for the call, None when it reported none.
    parent_run_key names the run this one branched from (a regeneration, a
    retry, a chained Responses call); None means it followed the turn before.
    """

    id: int
    run_key: str
    prompt_name: str | None
    version: int | None
    variables: dict[str, Any] | None
    rendered_text: str | None
    provider: str
    model: str | None
    request_params: dict[str, Any] | None
    response_id: str | None
    output_text: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    latency_ms: int | None
    status: str
    error: str | None
    created_at: str
    conversation_id: str | None = None
    turn_index: int | None = None
    input_text: str | None = None
    original_input_text: str | None = None
    cost_usd: float | None = None
    parent_run_key: str | None = None


@dataclass(frozen=True)
class ConversationInfo:
    """One conversation: its own metadata plus every turn, oldest first.

    ``turns`` holds one RunInfo per run row. A single API call that carried
    more than one tracked Prompt produced several rows sharing a turn_index;
    the derived views below treat those as one turn.

    A conversation is a tree. Each turn continues from one earlier turn: the
    run named by its ``parent_run_key`` when it has one (a regeneration, a
    branch, a chained Responses call), otherwise the turn just before it. A
    session nobody branched is a single line, and every view below reads it
    exactly as before; once it branches, ``leaves`` are the tips of its
    branches and ``path()`` / ``replay(upto=...)`` follow one of them.
    """

    external_id: str
    title: str | None
    metadata: dict[str, Any] | None
    created_at: str
    updated_at: str
    turns: list[RunInfo]

    @property
    def versions_used(self) -> dict[str, list[int]]:
        """Which prompt versions drove this conversation, e.g.
        ``{"REVIEW_SYSTEM": [4, 5], "SUMMARIZE": [2]}`` — each name's versions
        in order of first use. Turns without a tracked Prompt contribute nothing."""
        used: dict[str, list[int]] = {}
        for turn in self.turns:
            if turn.prompt_name is None or turn.version is None:
                continue
            versions = used.setdefault(turn.prompt_name, [])
            if turn.version not in versions:
                versions.append(turn.version)
        return used

    @property
    def total_tokens(self) -> int:
        """Tokens across every call, from the providers' reported usage. A call
        with no usage (an error, a stream without a usage chunk) counts as 0."""
        return sum(call.total_tokens or 0 for call in self._calls())

    @property
    def total_cost(self) -> float | None:
        """Dollars across every call, from the providers' reported cost — or
        None when no call reported one, so "free" and "unknown" stay apart."""
        known = [call.cost_usd for call in self._calls() if call.cost_usd is not None]
        return sum(known) if known else None

    @property
    def duration(self) -> float:
        """Wall-clock seconds from the start of the first turn to the end of the
        last one; 0.0 for an empty conversation.

        A run's timestamp is taken when it is recorded — at the end of its
        call — so a turn's start is its timestamp less its latency. Turns whose
        timestamp can't be parsed are ignored.
        """
        starts, ends = [], []
        for turn in self.turns:
            end = _parse_timestamp(turn.created_at)
            if end is None:
                continue
            ends.append(end)
            starts.append(end - timedelta(milliseconds=turn.latency_ms or 0))
        if not ends:
            return 0.0
        return max(0.0, (max(ends) - min(starts)).total_seconds())

    @property
    def leaves(self) -> list[RunInfo]:
        """The tip of every branch — each turn no later turn continues from —
        oldest first. One for a session that never branched: its last turn."""
        continued = set(self._predecessors().values())
        return [rows[0] for index, rows in self._turn_rows().items() if index not in continued]

    @property
    def forks(self) -> dict[int | None, int | None]:
        """Every turn that doesn't continue the turn just before it, mapped to
        the turn it does continue — None when that is a run outside this
        conversation. Empty for a session that never branched."""
        predecessors = self._predecessors()
        found: dict[int | None, int | None] = {}
        previous: int | None = None
        for index in self._turn_rows():
            if predecessors[index] != previous:
                found[index] = predecessors[index]
            previous = index
        return found

    def path(self, run_key: str) -> list[RunInfo]:
        """The branch that led to a run: every turn from the start of the
        conversation to the one holding ``run_key``, following each turn back
        to the one it continued from. Rows of the same API call stay together.
        A branch whose parent lives outside this conversation (or was never
        persisted) starts at that turn. Raises ValueError for a run that isn't
        in this conversation."""
        turns = self._turn_rows()
        index = next((t.turn_index for t in self.turns if t.run_key == run_key), None)
        if index is None:
            raise ValueError(f"run {run_key!r} is not in conversation {self.external_id!r}")

        # Walk back to the root; the visited set guards against a cycle that
        # only corrupt parent keys could make.
        predecessors = self._predecessors()
        chain: list[int | None] = []
        while index is not None and index not in chain:
            chain.append(index)
            index = predecessors.get(index)
        return [row for index in reversed(chain) for row in turns[index]]

    def replay(self, system: Any = None, upto: str | None = None) -> list[dict[str, Any]]:
        """Rebuild the conversation as a chat ``messages`` list, ready to send
        back to a provider — the raw material of every eval and re-run.

        A replay follows one branch: the one ending at ``upto`` (a run key),
        or by default the one ending at the latest turn — for a session that
        never branched, simply every turn. Turns on an abandoned branch (a
        reply that was regenerated) are not part of the history the next
        turn saw, so they are left out.

        Completed turns (status "ok") are walked in order. Each contributes the
        system prompt that was in play (the tracked Prompt's rendered text —
        emitted only when it differs from the previous turn's, so an unchanged
        system prompt appears once), then the user message as it was actually
        sent (``input_text``, after any pre-check rewrite) and the assistant's
        reply (``output_text``). Blocked calls and provider errors never gave
        the model anything to build on, so they are left out. Content that was
        multi-part when recorded was flattened to its text; replay is text-only.

        To re-run the session against a different prompt, pass ``system=``: it
        goes first as the one system message (a str, or a Prompt object — a
        wrapped client tracks the latter like any other) and the stored system
        prompts are dropped.
        """
        messages: list[dict[str, Any]] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        previous_prompts: set = set()
        if upto is None and self.turns:
            upto = self.turns[-1].run_key
        branch = self.path(upto) if upto is not None else []
        for rows in _completed_turns(branch):
            head = rows[0]
            # A Prompt that *was* the user turn is already the input; it is
            # not a system prompt, so don't emit it twice.
            prompts = [r.rendered_text for r in rows if r.rendered_text]
            prompts = [t for t in prompts if t != head.input_text]
            if system is None:
                for text in prompts:
                    if text not in previous_prompts:
                        messages.append({"role": "system", "content": text})
            previous_prompts = set(prompts)
            if head.input_text is not None:
                messages.append({"role": "user", "content": head.input_text})
            if head.output_text is not None:
                messages.append({"role": "assistant", "content": head.output_text})
        return messages

    def _calls(self) -> Iterator[RunInfo]:
        """One row per physical API call. Rows sharing a turn_index came from
        the same call and repeat its usage and cost, so summing every row
        would count that call once per tracked Prompt."""
        for _index, rows in groupby(self.turns, key=lambda t: t.turn_index):
            yield next(rows)

    def _turn_rows(self) -> dict[int | None, list[RunInfo]]:
        """The rows of each turn, by turn_index, in turn order — rows sharing
        a turn_index came from the same API call. Relies on ``turns`` being
        ordered by turn_index, which the storage read guarantees."""
        return {index: list(rows) for index, rows in groupby(self.turns, lambda t: t.turn_index)}

    def _predecessors(self) -> dict[int | None, int | None]:
        """Each turn's predecessor, by turn_index: the turn holding its
        parent run if it names one, else the turn before it. None for a turn
        that starts a branch — the first turn, or one whose parent isn't here."""
        turn_of = {t.run_key: t.turn_index for t in self.turns}
        found: dict[int | None, int | None] = {}
        previous: int | None = None
        for index, rows in self._turn_rows().items():
            parent = rows[0].parent_run_key
            found[index] = previous if parent is None else turn_of.get(parent)
            previous = index
        return found


def _completed_turns(rows: list[RunInfo]) -> Iterator[list[RunInfo]]:
    """The turns among ``rows`` that completed, each as its rows (grouped by
    turn_index — rows of one API call), in order."""
    for _index, group in groupby(rows, key=lambda t: t.turn_index):
        completed = [r for r in group if r.status == "ok"]
        if completed:
            yield completed


@dataclass(frozen=True)
class PromptSummary:
    """One prompt's row in an overview listing: name plus counts."""

    name: str
    version_count: int
    run_count: int
    created_at: str


@dataclass(frozen=True)
class VersionStats:
    """How one version of a prompt has performed — a row of ``stats()``.

    Every average is over the runs (or labels) that reported the number, and
    None when none did: an unknown cost is not a zero cost. check_pass_rate
    is the share of *checked* runs whose every verdict was ok; avg_score
    averages the scores checks gave, avg_feedback the scores given through
    ``promptkeep.feedback()`` — kept apart because one is automatic and the
    other is a person's.
    """

    version: int
    created_at: str
    runs: int
    errors: int
    blocked: int
    avg_latency_ms: float | None
    avg_tokens: float | None
    total_cost: float | None
    checked_runs: int = 0
    check_pass_rate: float | None = None
    avg_score: float | None = None
    feedback_count: int = 0
    avg_feedback: float | None = None


@dataclass(frozen=True)
class ConversationSummary:
    """One conversation's row in an overview listing: id plus turn count."""

    external_id: str
    title: str | None
    turn_count: int
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class CheckInfo:
    """One label recorded against a run: a check verdict, or a piece of
    ``promptkeep.feedback()`` (phase 'feedback', where name is the feedback's
    label and message its comment). rewritten is the replacement text a
    rewriting pre-check produced; None for everything else."""

    name: str
    phase: str  # 'pre' | 'post' | 'feedback'
    status: str  # 'ok' | 'warn' | 'block' | 'error'
    score: float | None
    message: str | None
    rewritten: str | None = None


def checks(run_key: str) -> list[CheckInfo]:
    """Every label on a run (by its key), oldest first: its check verdicts
    and any ``promptkeep.feedback()`` given on it — told apart by ``phase``."""
    if not _ready():
        return []
    query = (
        CheckRecord.select(
            CheckRecord.name,
            CheckRecord.phase,
            CheckRecord.status,
            CheckRecord.score,
            CheckRecord.message,
            CheckRecord.rewritten,
        )
        .join(RunRecord)
        .where(RunRecord.run_key == run_key)
        .order_by(CheckRecord.id)
        .dicts()
    )
    return [CheckInfo(**row) for row in query]


def verdict(run_status: str, check_infos: list[CheckInfo]) -> str | None:
    """The one-word headline for a run's checks, for a badge in the UI.

    'blocked' if the call was gated, else 'failed' if any check errored,
    'warn' if any warned, 'ok' if checks ran and all passed, None if the run
    had no checks at all. Feedback rows are labels, not checks: they never
    move the headline.
    """
    if run_status == "blocked":
        return "blocked"
    statuses = {c.status for c in check_infos if c.phase != "feedback"}
    if not statuses:
        return None
    if "block" in statuses or "error" in statuses:
        return "failed"
    if "warn" in statuses:
        return "warn"
    return "ok"


def format_cost(cost_usd: float | None) -> str:
    """A cost for display: dollars with as many decimals as it takes — a
    single call is often a fraction of a cent, so two fixed places would
    round most of them to $0.00. None (the provider reported nothing) is a
    dash, never a zero."""
    if cost_usd is None:
        return "—"
    whole, _, decimals = f"{cost_usd:.6f}".partition(".")
    return f"${whole}.{decimals.rstrip('0').ljust(2, '0')}"


# --- row helpers -------------------------------------------------------------------


def _ready() -> bool:
    """Whether there is a database to read: False when tracking is disabled."""
    return storage.get_db() is not None


def _parse_timestamp(value: str) -> datetime | None:
    """A stored ISO-8601 timestamp as a datetime; None when unparseable."""
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def _load_json(value: str | None) -> Any:
    """Decode a stored JSON column; malformed/missing data becomes None."""
    if value is None:
        return None
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return None


# The run projection, defined once: every run-shaped read selects these same
# columns (aliased to RunInfo's field names) and differs only in joins,
# filters and order. Reusing aliased column objects across queries is safe —
# building a query never mutates them.
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
    RunRecord.cost_usd,
    RunRecord.parent_run_key,
)
# Added only where the caller reads across conversations and needs to know
# which one each run belongs to (a conversation's own turns already know).
_CONVERSATION_ID_COLUMN = ConversationRecord.external_id.alias("conversation_id")


def _count_status(status: str) -> pw.Node:
    """How many runs in a group have this status (0, not NULL, for an empty group)."""
    matches = pw.Case(None, [(RunRecord.status == status, 1)], 0)
    return pw.fn.COALESCE(pw.fn.SUM(matches), 0)


def _run_info(row: dict[str, Any]) -> RunInfo:
    """One projected run row as a RunInfo; the two JSON columns are decoded."""
    return RunInfo(
        **{
            **row,
            "variables": _load_json(row.get("variables")),
            "request_params": _load_json(row.get("request_params")),
        }
    )


# --- reads -------------------------------------------------------------------------


def versions(name: str) -> list[VersionInfo]:
    """All versions of a prompt, oldest first."""
    rows = storage.version_rows(name)
    return [VersionInfo(**{k: v for k, v in row.items() if k != "id"}) for row in rows]


def diff(name: str, old: int, new: int) -> str:
    """Unified diff between two versions of a prompt's template."""
    # Load the lineage once and validate both requested versions exist.
    by_number = {v.version: v for v in versions(name)}
    for wanted in (old, new):
        if wanted not in by_number:
            raise ValueError(f"prompt {name!r} has no version {wanted}")

    lines = difflib.unified_diff(
        by_number[old].template.splitlines(),
        by_number[new].template.splitlines(),
        fromfile=f"{name} v{old}",
        tofile=f"{name} v{new}",
        lineterm="",
    )
    return "\n".join(lines)


def runs(name: str, version: int | None = None, limit: int | None = 50) -> list[RunInfo]:
    """Recorded runs for a prompt (optionally one version), newest first.
    ``limit=None`` returns every one."""
    if not _ready():
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
    return [_run_info(row) for row in query]


def all_runs(limit: int | None = 100) -> list[RunInfo]:
    """Every recorded run regardless of prompt (or with none), newest first
    (``limit=None``: all of them).

    Left-joined throughout: a conversation-only turn has no version/prompt
    to join to, and a run outside any conversation has no conversation to
    join to. Both cases surface as NULLs rather than dropping the row.
    """
    if not _ready():
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
    return [_run_info(row) for row in query]


def stats(name: str) -> list[VersionStats]:
    """Every version of a prompt with how its runs went, oldest first — the
    table that answers "did the change actually help?".

    Counts, latency, tokens and cost come from the runs; pass rate and scores
    from the labels on them. A version with no runs yet still gets a row.
    Three grouped queries rather than one: joining runs to their labels fans
    out, which would double-count every run that has more than one verdict.
    """
    if not _ready():
        return []

    # The runs: one row per version, versions without runs included.
    per_version = (
        PromptVersionRecord.select(
            PromptVersionRecord.version,
            PromptVersionRecord.created_at,
            pw.fn.COUNT(RunRecord.id).alias("runs"),
            _count_status("error").alias("errors"),
            _count_status("blocked").alias("blocked"),
            pw.fn.AVG(RunRecord.latency_ms).alias("avg_latency_ms"),
            pw.fn.AVG(RunRecord.total_tokens).alias("avg_tokens"),
            pw.fn.SUM(RunRecord.cost_usd).alias("total_cost"),
        )
        .join(PromptRecord)
        .switch(PromptVersionRecord)
        .join(RunRecord, pw.JOIN.LEFT_OUTER)
        .where(PromptRecord.name == name)
        .group_by(PromptVersionRecord.id)
        .order_by(PromptVersionRecord.version)
        .dicts()
    )

    # The labels, by version and kind: check scores apart from feedback scores.
    is_feedback = CheckRecord.phase == "feedback"
    labels = (
        CheckRecord.select(
            PromptVersionRecord.version,
            is_feedback.alias("feedback"),
            pw.fn.COUNT(CheckRecord.id).alias("count"),
            pw.fn.AVG(CheckRecord.score).alias("score"),
        )
        .join(RunRecord)
        .join(PromptVersionRecord)
        .join(PromptRecord)
        .where(PromptRecord.name == name)
        .group_by(PromptVersionRecord.version, is_feedback)
        .tuples()
    )
    by_kind = {
        (version, bool(feedback)): (count, score) for version, feedback, count, score in labels
    }

    # The pass rate: a run passes when every one of its verdicts is ok, so
    # this is a fold per run first, then a share per version.
    all_ok = pw.fn.MIN(pw.Case(None, [(CheckRecord.status == "ok", 1)], 0))
    verdicts = (
        CheckRecord.select(PromptVersionRecord.version, all_ok.alias("passed"))
        .join(RunRecord)
        .join(PromptVersionRecord)
        .join(PromptRecord)
        .where((PromptRecord.name == name) & ~is_feedback)
        .group_by(PromptVersionRecord.version, RunRecord.id)
        .tuples()
    )
    passed: dict[int, list[int]] = {}
    for version, run_passed in verdicts:
        passed.setdefault(version, []).append(run_passed)

    # One VersionStats per version, the three reads folded together.
    result = []
    for row in per_version:
        checked = passed.get(row["version"], [])
        _count, avg_score = by_kind.get((row["version"], False), (0, None))
        feedback_count, avg_feedback = by_kind.get((row["version"], True), (0, None))
        result.append(
            VersionStats(
                **row,
                checked_runs=len(checked),
                check_pass_rate=sum(checked) / len(checked) if checked else None,
                avg_score=avg_score,
                feedback_count=feedback_count,
                avg_feedback=avg_feedback,
            )
        )
    return result


def list_prompts() -> list[PromptSummary]:
    """Every prompt with its version/run counts, for an overview listing.

    COUNT(DISTINCT ...) is required here: joining both versions and runs off
    the same prompt fans out into a cross product, so a plain COUNT would
    double-count whichever side has more rows.
    """
    if not _ready():
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
    return [PromptSummary(**row) for row in query]


def list_conversations(
    limit: int = 100, prompt: str | None = None, version: int | None = None
) -> list[ConversationSummary]:
    """Every conversation with its turn count, most recently active first.

    ``prompt=`` keeps only conversations in which that prompt drove at least
    one turn; ``version=`` (which needs ``prompt=``) narrows to one version of
    it — "every session where REVIEW_SYSTEM v4 was live". The turn count is
    always the conversation's full length, not just the matching turns.
    """
    if version is not None and prompt is None:
        raise ValueError("list_conversations(version=...) requires prompt= as well")
    if not _ready():
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

    # The prompt filter is a separate subquery rather than a condition on the
    # counting join, so turn_count stays the conversation's full length.
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
    return [ConversationSummary(**row) for row in query.dicts()]


def conversation(external_id: str) -> ConversationInfo:
    """A conversation's full transcript: its metadata plus every turn.

    Turns are ordered oldest-first (turn_index ascending) — read input then
    output off each turn, in order, to replay the whole session. Raises
    ValueError if no conversation was ever recorded under this id.
    """
    row = None
    if _ready():
        row = (
            ConversationRecord.select()
            .where(ConversationRecord.external_id == external_id)
            .dicts()
            .first()
        )
    if row is None:
        raise ValueError(f"no conversation found for external_id {external_id!r}")

    # The turns: version fields are NULL where a turn had no wrapped Prompt.
    turn_rows = (
        RunRecord.select(*_RUN_COLUMNS)
        .join(PromptVersionRecord, pw.JOIN.LEFT_OUTER)
        .join(PromptRecord, pw.JOIN.LEFT_OUTER)
        .switch(RunRecord)
        .join(ConversationRecord)
        .where(ConversationRecord.external_id == external_id)
        .order_by(RunRecord.turn_index)
        .dicts()
    )
    return ConversationInfo(
        external_id=row["external_id"],
        title=row["title"],
        metadata=_load_json(row["metadata"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        turns=[_run_info(r) for r in turn_rows],
    )

"""The read side: lineage, runs, conversations, verdicts and agent traces as
frozen dataclasses.

Every query lives here, next to the type it produces. Reads raise normally —
a broken query is a bug you want to see — but against a disabled tracker
they return nothing rather than opening a file.
"""

from __future__ import annotations

import difflib
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from itertools import groupby
from typing import Any

import peewee as pw

from . import storage
from .models import (
    CheckRecord,
    ConversationRecord,
    PromptRecord,
    PromptVersionRecord,
    RunRecord,
    SpanRecord,
)


class _Unset:
    """The "payload was never recorded" sentinel: distinct from a recorded
    JSON null, and printable so a span shows ``<unset>`` rather than an
    object address."""

    def __repr__(self) -> str:
        """A stable, readable stand-in for the sentinel."""
        return "<unset>"


# A span payload that was never recorded (SQLite NULL), as opposed to an
# explicit JSON null. Kept private; SpanInfo exposes it via has_input/has_output.
_UNSET = _Unset()


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
class SpanInfo:
    """One recorded span: an agent scope, a tool scope, or a model call.

    ``input``/``output`` use a private sentinel for "never recorded", which is
    different from an explicitly recorded JSON ``null``; ``has_input`` and
    ``has_output`` answer the question without touching the sentinel. A
    payload that the redaction hook made unparsable is returned as the
    redacted raw string, so it stays inspectable instead of silently becoming
    empty. ``runs`` holds the existing run rows linked to an ``llm`` span; a
    missing run row is shown by ``run_unavailable`` rather than invented.
    ``sampled`` is this span's successful-span decision; for a root it is the
    trace cohort.
    """

    span_key: str
    trace_key: str
    parent_span_key: str | None
    kind: str
    name: str | None
    started_at: str
    ended_at: str | None
    duration_ms: int | None
    status: str
    error: str | None = None
    input: Any = _UNSET
    output: Any = _UNSET
    sampled: bool | None = None
    runs: list[RunInfo] = field(default_factory=list)
    checks: list[CheckInfo] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float | None:
        """Elapsed seconds, or None while the span has no recorded duration."""
        return None if self.duration_ms is None else self.duration_ms / 1000

    @property
    def has_input(self) -> bool:
        """Whether an input payload was actually recorded for this span."""
        return self.input is not _UNSET

    @property
    def has_output(self) -> bool:
        """Whether an output payload was recorded — omitted output is not null."""
        return self.output is not _UNSET

    @property
    def primary_run(self) -> RunInfo | None:
        """The run that carries this model span's checks and shared metrics."""
        return self.runs[0] if self.runs else None

    @property
    def run_unavailable(self) -> bool:
        """Whether an llm span's linked run row is missing (pruned or dropped)."""
        return self.kind == "llm" and not self.runs


@dataclass(frozen=True)
class TraceInfo:
    """One trace's ordered spans plus the linked run details.

    ``root`` is None when the root start/final row is missing but children
    survived; the spans are still returned so the partial execution is
    inspectable. Model metrics count each llm span once, using its primary
    run, so several Prompt rows from one physical call cannot multiply
    tokens or cost. ``sampled`` is the root's cohort decision; False means the
    root's cohort was dropped, which is typical of a problem-only trace but
    can also happen when an explicit conversation override supplies a kept
    model step.
    """

    trace_key: str
    root: SpanInfo | None
    spans: list[SpanInfo]

    @property
    def status(self) -> str | None:
        """The root's own exit status, or None when the root row is missing."""
        return self.root.status if self.root is not None else None

    @property
    def sampled(self) -> bool | None:
        """The root cohort decision, or None when the root row is missing."""
        return self.root.sampled if self.root is not None else None

    @property
    def duration(self) -> float:
        """Observed wall interval from the earliest start to the latest end.

        A running root with no completed children contributes its start only,
        so an open trace is not reported as an ever-growing duration. This is
        an interval, not the sum of overlapping child durations.
        """
        starts = [
            parsed
            for span in self.spans
            if (parsed := _parse_timestamp(span.started_at)) is not None
        ]
        ends = [
            parsed
            for span in self.spans
            if span.ended_at is not None and (parsed := _parse_timestamp(span.ended_at)) is not None
        ]
        if not starts:
            return 0.0
        return max(0.0, (max(ends) - min(starts)).total_seconds() if ends else 0.0)

    @property
    def model_calls(self) -> int:
        """How many physical model calls the trace observed."""
        return sum(1 for span in self.spans if span.kind == "llm")

    @property
    def total_tokens(self) -> int:
        """Tokens from each model span's primary run, counted once per call."""
        total = 0
        for span in self.spans:
            if span.kind != "llm" or span.primary_run is None:
                continue
            total += span.primary_run.total_tokens or 0
        return total

    @property
    def total_cost(self) -> float | None:
        """Known provider-reported dollars, or None when no call reported any."""
        known = [
            span.primary_run.cost_usd
            for span in self.spans
            if span.kind == "llm"
            and span.primary_run is not None
            and span.primary_run.cost_usd is not None
        ]
        return sum(known) if known else None

    @property
    def cost_complete(self) -> bool:
        """Whether every observed model call has a known provider-reported cost."""
        model_spans = [span for span in self.spans if span.kind == "llm"]
        return bool(model_spans) and all(
            span.primary_run is not None and span.primary_run.cost_usd is not None
            for span in model_spans
        )

    @property
    def error_count(self) -> int:
        """Child failures and blocks, separate from the root's exit status."""
        return sum(
            1
            for span in self.spans
            if span.span_key != self.trace_key and span.status in ("error", "blocked")
        )

    @property
    def versions_used(self) -> dict[str, list[int]]:
        """Prompt versions driving this trace, in first-use order per name."""
        used: dict[str, list[int]] = {}
        for span in self.spans:
            if span.kind != "llm":
                continue
            for run in span.runs:
                if run.prompt_name is None or run.version is None:
                    continue
                versions = used.setdefault(run.prompt_name, [])
                if run.version not in versions:
                    versions.append(run.version)
        return used


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


def labels(run_keys: list[str]) -> dict[str, list[CheckInfo]]:
    """The labels on many runs at once — ``checks()`` for a batch, in one
    query per few hundred runs rather than one per run. Every key asked for
    is in the result, oldest label first; a run with none maps to []."""
    found: dict[str, list[CheckInfo]] = {key: [] for key in run_keys}
    if not run_keys or not _ready():
        return found

    # Chunked: SQLite caps the number of bound parameters in one statement.
    for start in range(0, len(run_keys), _KEYS_PER_QUERY):
        chunk = run_keys[start : start + _KEYS_PER_QUERY]
        query = (
            CheckRecord.select(
                RunRecord.run_key,
                CheckRecord.name,
                CheckRecord.phase,
                CheckRecord.status,
                CheckRecord.score,
                CheckRecord.message,
                CheckRecord.rewritten,
            )
            .join(RunRecord)
            .where(RunRecord.run_key.in_(chunk))
            .order_by(CheckRecord.id)
            .dicts()
        )
        for row in query:
            found[row.pop("run_key")].append(CheckInfo(**row))
    return found


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
# Run keys bound per query by batch reads — well under SQLite's parameter cap.
_KEYS_PER_QUERY = 500
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


def _span_info(
    row: dict[str, Any],
    runs: list[RunInfo] | None = None,
    checks: list[CheckInfo] | None = None,
) -> SpanInfo:
    """One span row as a SpanInfo, decoding payloads without losing raw text."""
    return SpanInfo(
        span_key=row["span_key"],
        trace_key=row["trace_key"],
        parent_span_key=row.get("parent_span_key"),
        kind=row["kind"],
        name=row.get("name"),
        started_at=row["started_at"],
        ended_at=row.get("ended_at"),
        duration_ms=row.get("duration_ms"),
        status=row["status"],
        error=row.get("error"),
        input=_decode_span_payload(row.get("input_json")),
        output=_decode_span_payload(row.get("output_json")),
        sampled=None if row.get("sampled") is None else bool(row["sampled"]),
        runs=runs or [],
        checks=checks or [],
    )


def _decode_span_payload(value: str | None) -> Any:
    """Decode a stored span payload; NULL is unset, unparsable text is kept."""
    if value is None:
        return _UNSET
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return value


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


def list_traces(limit: int = 100) -> list[SpanInfo]:
    """The newest root agent spans, bounded by ``limit``.

    Only roots are fetched, and their payload columns are deliberately left
    unloaded: a list of 100 roots must not read 6 MiB of tool output. Use
    ``trace(trace_key)`` when the root payload is needed. A trace whose root
    row was dropped but whose children survived is intentionally absent here;
    ``trace(trace_key)`` can still load it by key.
    """
    if not _ready() or limit <= 0:
        return []
    # Everything SpanInfo needs except the two potentially large payloads.
    query = (
        SpanRecord.select(
            SpanRecord.span_key,
            SpanRecord.trace_key,
            SpanRecord.parent_span_key,
            SpanRecord.kind,
            SpanRecord.name,
            SpanRecord.started_at,
            SpanRecord.ended_at,
            SpanRecord.duration_ms,
            SpanRecord.status,
            SpanRecord.error,
            SpanRecord.sampled,
        )
        .where((SpanRecord.kind == "agent") & (SpanRecord.span_key == SpanRecord.trace_key))
        .order_by(SpanRecord.started_at.desc(), SpanRecord.span_key.desc())
        .limit(limit)
        .dicts()
    )
    return [_span_info(row) for row in query]


def trace(trace_key: str) -> TraceInfo:
    """One trace's ordered spans and linked model runs; ValueError if unknown.

    Three bounded read phases: spans for the trace, linked runs in key-sized
    batches, and check labels in batches. Missing parents, a missing root, or
    a pruned linked run are tolerated rather than reconstructed. The returned
    spans are flat and ordered by start time with a stable key tie-breaker.
    """
    if not _ready():
        raise ValueError(f"no trace found for trace_key {trace_key!r}")
    span_rows = list(
        SpanRecord.select()
        .where(SpanRecord.trace_key == trace_key)
        .order_by(SpanRecord.started_at, SpanRecord.span_key)
        .dicts()
    )
    if not span_rows:
        raise ValueError(f"no trace found for trace_key {trace_key!r}")

    # Linked model runs, grouped by span. The run projection is shared with
    # the existing run reads; span_key is selected only to group and removed
    # before RunInfo is built. Chunked for the same SQLite parameter limit as
    # labels(), rather than one query per span.
    span_keys = [row["span_key"] for row in span_rows]
    runs_by_span: dict[str, list[RunInfo]] = {}
    for start in range(0, len(span_keys), _KEYS_PER_QUERY):
        chunk = span_keys[start : start + _KEYS_PER_QUERY]
        run_rows = (
            RunRecord.select(*_RUN_COLUMNS, _CONVERSATION_ID_COLUMN, RunRecord.span_key)
            .join(PromptVersionRecord, pw.JOIN.LEFT_OUTER)
            .join(PromptRecord, pw.JOIN.LEFT_OUTER)
            .switch(RunRecord)
            .join(ConversationRecord, pw.JOIN.LEFT_OUTER)
            .where(RunRecord.span_key.in_(chunk))
            .order_by(RunRecord.id)
            .dicts()
        )
        for row in run_rows:
            span_key = row.pop("span_key")
            runs_by_span.setdefault(span_key, []).append(_run_info(row))

    # Every run's labels in batched queries, then only the primary run's
    # labels attach to its model span (checks stay filed under that run).
    labels_by_run = labels([run.run_key for runs in runs_by_span.values() for run in runs])
    spans = []
    for row in span_rows:
        runs = runs_by_span.get(row["span_key"], [])
        primary_checks = labels_by_run.get(runs[0].run_key, []) if runs else []
        spans.append(_span_info(row, runs=runs, checks=primary_checks))

    # The root is an agent whose trace key is its own span key. Its absence
    # is shown as None, not invented from the oldest child.
    root = next(
        (span for span in spans if span.kind == "agent" and span.span_key == span.trace_key),
        None,
    )
    return TraceInfo(trace_key=trace_key, root=root, spans=spans)


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

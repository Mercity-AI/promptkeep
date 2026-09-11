"""Query the lineage and run history of a prompt by name.

The read-side API: storage returns raw dict rows; this module shapes them
into typed, immutable dataclasses that are pleasant to work with.
"""

from __future__ import annotations

import difflib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import groupby
from typing import Any

from . import storage


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
    this is what the caller originally passed.
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


@dataclass(frozen=True)
class ConversationInfo:
    """One conversation: its own metadata plus every turn, oldest first.

    ``turns`` holds one RunInfo per run row. A single API call that carried
    more than one tracked Prompt produced several rows sharing a turn_index;
    the derived views below treat those as one turn.
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
        """Tokens across every turn, from the providers' reported usage. A turn
        with no usage (an error, a stream without a usage chunk) counts as 0."""
        return sum(turn.total_tokens or 0 for turn in self.turns)

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

    def replay(self, system: Any = None) -> list[dict[str, Any]]:
        """Rebuild the conversation as a chat ``messages`` list, ready to send
        back to a provider — the raw material of every eval and re-run.

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
        for _index, rows in self._completed_turns():
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

    def _completed_turns(self) -> Iterator[tuple[int | None, list[RunInfo]]]:
        """Turns that completed, as (turn_index, rows) — rows sharing a
        turn_index came from the same API call. Relies on ``turns`` being
        ordered by turn_index, which the storage read guarantees."""
        for index, group in groupby(self.turns, key=lambda t: t.turn_index):
            rows = [r for r in group if r.status == "ok"]
            if rows:
                yield index, rows


@dataclass(frozen=True)
class PromptSummary:
    """One prompt's row in an overview listing: name plus counts."""

    name: str
    version_count: int
    run_count: int
    created_at: str


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
    """One check verdict recorded against a run. rewritten is the replacement
    text a rewriting pre-check produced; None for every other verdict."""

    name: str
    phase: str  # 'pre' | 'post'
    status: str  # 'ok' | 'warn' | 'block' | 'error'
    score: float | None
    message: str | None
    rewritten: str | None = None


def checks(run_key: str) -> list[CheckInfo]:
    """Every check verdict for a run (by its key), oldest first."""
    return [
        CheckInfo(
            name=row["name"],
            phase=row["phase"],
            status=row["status"],
            score=row["score"],
            message=row["message"],
            rewritten=row["rewritten"],
        )
        for row in storage.fetch_checks(run_key)
    ]


def verdict(run_status: str, check_infos: list[CheckInfo]) -> str | None:
    """The one-word headline for a run's checks, for a badge in the UI.

    'blocked' if the call was gated, else 'failed' if any check errored,
    'warn' if any warned, 'ok' if checks ran and all passed, None if the run
    had no checks at all.
    """
    if run_status == "blocked":
        return "blocked"
    if not check_infos:
        return None
    statuses = {c.status for c in check_infos}
    if "block" in statuses or "error" in statuses:
        return "failed"
    if "warn" in statuses:
        return "warn"
    return "ok"


def _parse_timestamp(value: str) -> datetime | None:
    """A stored ISO-8601 timestamp as a datetime; None when unparseable."""
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def _load_json(value: str | None):
    """Decode a stored JSON column; malformed/missing data becomes None."""
    if value is None:
        return None
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return None


def versions(name: str) -> list[VersionInfo]:
    """All versions of a prompt, oldest first."""
    return [
        VersionInfo(
            version=row["version"],
            template=row["template"],
            template_hash=row["template_hash"],
            source=row["source"],
            fn_source_hash=row["fn_source_hash"],
            created_at=row["created_at"],
        )
        for row in storage.fetch_versions(name)
    ]


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


def _run_info_from_row(row: dict[str, Any]) -> RunInfo:
    """Shape one raw run/turn dict row (from either fetch_runs or
    fetch_conversation_turns) into a RunInfo. Missing optional columns
    (older query shapes) default to None via dict.get."""
    return RunInfo(
        id=row["id"],
        run_key=row["run_key"],
        prompt_name=row.get("prompt_name"),
        version=row.get("version"),
        variables=_load_json(row.get("variables")),
        rendered_text=row.get("rendered_text"),
        provider=row["provider"],
        model=row["model"],
        request_params=_load_json(row.get("request_params")),
        response_id=row["response_id"],
        output_text=row["output_text"],
        prompt_tokens=row["prompt_tokens"],
        completion_tokens=row["completion_tokens"],
        total_tokens=row["total_tokens"],
        latency_ms=row["latency_ms"],
        status=row["status"],
        error=row["error"],
        created_at=row["created_at"],
        conversation_id=row.get("conversation_id"),
        turn_index=row.get("turn_index"),
        input_text=row.get("input_text"),
        original_input_text=row.get("original_input_text"),
    )


def runs(name: str, version: int | None = None, limit: int = 50) -> list[RunInfo]:
    """Recorded runs for a prompt (optionally one version), newest first."""
    return [
        _run_info_from_row(row) for row in storage.fetch_runs(name, version=version, limit=limit)
    ]


def all_runs(limit: int = 100) -> list[RunInfo]:
    """Every recorded run regardless of prompt (or with none), newest first."""
    return [_run_info_from_row(row) for row in storage.fetch_all_runs(limit=limit)]


def list_prompts() -> list[PromptSummary]:
    """Every prompt with its version/run counts, for an overview listing."""
    return [
        PromptSummary(
            name=row["name"],
            version_count=row["version_count"],
            run_count=row["run_count"],
            created_at=row["created_at"],
        )
        for row in storage.fetch_prompt_summaries()
    ]


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
    return [
        ConversationSummary(
            external_id=row["external_id"],
            title=row["title"],
            turn_count=row["turn_count"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
        for row in storage.fetch_conversation_summaries(limit=limit, prompt=prompt, version=version)
    ]


def conversation(external_id: str) -> ConversationInfo:
    """A conversation's full transcript: its metadata plus every turn.

    Turns are ordered oldest-first (turn_index ascending) — read input then
    output off each turn, in order, to replay the whole session. Raises
    ValueError if no conversation was ever recorded under this id.
    """
    row = storage.fetch_conversation(external_id)
    if row is None:
        raise ValueError(f"no conversation found for external_id {external_id!r}")
    turns = [_run_info_from_row(r) for r in storage.fetch_conversation_turns(external_id)]
    return ConversationInfo(
        external_id=row["external_id"],
        title=row["title"],
        metadata=_load_json(row["metadata"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        turns=turns,
    )

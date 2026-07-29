"""Query the lineage and run history of a prompt by name.

The read-side API: storage returns raw dict rows; this module shapes them
into typed, immutable dataclasses that are pleasant to work with.
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from . import storage


@dataclass(frozen=True)
class VersionInfo:
    """One version of a prompt's template, as recorded in the lineage."""

    version: int
    template: str
    template_hash: str
    source: str
    fn_source_hash: Optional[str]
    created_at: str


@dataclass(frozen=True)
class RunInfo:
    """One recorded execution: which version ran, with what, and what came back.

    version/prompt_name are None for a conversation turn that involved no
    wrapped Prompt (a plain follow-up message) — there's simply no lineage
    to attach it to. conversation_id/turn_index/input_text are None for a
    run recorded outside any conversation.
    """

    id: int
    prompt_name: Optional[str]
    version: Optional[int]
    variables: Optional[Dict[str, Any]]
    rendered_text: Optional[str]
    provider: str
    model: Optional[str]
    request_params: Optional[Dict[str, Any]]
    response_id: Optional[str]
    output_text: Optional[str]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    latency_ms: Optional[int]
    status: str
    error: Optional[str]
    created_at: str
    conversation_id: Optional[str] = None
    turn_index: Optional[int] = None
    input_text: Optional[str] = None


@dataclass(frozen=True)
class ConversationInfo:
    """One conversation: its own metadata plus every turn, oldest first."""

    external_id: str
    title: Optional[str]
    metadata: Optional[Dict[str, Any]]
    created_at: str
    updated_at: str
    turns: List[RunInfo]


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
    title: Optional[str]
    turn_count: int
    created_at: str
    updated_at: str


def _load_json(value: Optional[str]):
    """Decode a stored JSON column; malformed/missing data becomes None."""
    if value is None:
        return None
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return None


def versions(name: str) -> List[VersionInfo]:
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


def _run_info_from_row(row: Dict[str, Any]) -> RunInfo:
    """Shape one raw run/turn dict row (from either fetch_runs or
    fetch_conversation_turns) into a RunInfo. Missing optional columns
    (older query shapes) default to None via dict.get."""
    return RunInfo(
        id=row["id"],
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
    )


def runs(name: str, version: Optional[int] = None, limit: int = 50) -> List[RunInfo]:
    """Recorded runs for a prompt (optionally one version), newest first."""
    return [
        _run_info_from_row(row) for row in storage.fetch_runs(name, version=version, limit=limit)
    ]


def all_runs(limit: int = 100) -> List[RunInfo]:
    """Every recorded run regardless of prompt (or with none), newest first."""
    return [_run_info_from_row(row) for row in storage.fetch_all_runs(limit=limit)]


def list_prompts() -> List[PromptSummary]:
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


def list_conversations(limit: int = 100) -> List[ConversationSummary]:
    """Every conversation with its turn count, most recently active first."""
    return [
        ConversationSummary(
            external_id=row["external_id"],
            title=row["title"],
            turn_count=row["turn_count"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
        for row in storage.fetch_conversation_summaries(limit=limit)
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

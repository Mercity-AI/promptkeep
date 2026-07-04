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
    """One recorded execution: which version ran, with what, and what came back."""

    id: int
    prompt_name: str
    version: int
    variables: Optional[Dict[str, Any]]
    rendered_text: str
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


def runs(name: str, version: Optional[int] = None, limit: int = 50) -> List[RunInfo]:
    """Recorded runs for a prompt (optionally one version), newest first."""
    return [
        RunInfo(
            id=row["id"],
            prompt_name=row["prompt_name"],
            version=row["version"],
            variables=_load_json(row["variables"]),
            rendered_text=row["rendered_text"],
            provider=row["provider"],
            model=row["model"],
            request_params=_load_json(row["request_params"]),
            response_id=row["response_id"],
            output_text=row["output_text"],
            prompt_tokens=row["prompt_tokens"],
            completion_tokens=row["completion_tokens"],
            total_tokens=row["total_tokens"],
            latency_ms=row["latency_ms"],
            status=row["status"],
            error=row["error"],
            created_at=row["created_at"],
        )
        for row in storage.fetch_runs(name, version=version, limit=limit)
    ]

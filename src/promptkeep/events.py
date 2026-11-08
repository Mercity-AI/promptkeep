"""Internal source-neutral events and execution ownership; no SDK dependencies."""

from __future__ import annotations

import contextvars
import uuid
from dataclasses import dataclass, field
from typing import Any

from .config import get_settings

_NAMESPACE = uuid.UUID("7e08da92-e6ab-49a6-9071-3ff052b33cc0")


def key(source: str, trace_id: str, span_id: str) -> str:
    """Mint stable, source-scoped identities without depending on arrival order."""
    return str(uuid.uuid5(_NAMESPACE, repr((source, trace_id, span_id))))


@dataclass(frozen=True)
class Event:
    """Observed operation data; missing timing and usage are never estimated."""

    source: str
    trace_id: str
    span_id: str
    name: str
    kind: str = "operation"
    parent_id: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    observed_at: str | None = None
    status: str = "ok"
    input: Any = None
    output: Any = None
    error: str | None = None
    model: str | None = None
    response_id: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    links: tuple[str, ...] = ()
    timing_quality: str = "explicit"
    provider: str | None = None
    cost_usd: float | None = None
    link_keys: tuple[str, ...] = ()
    destination: str | None = None

    @property
    def span_key(self) -> str:
        """A root's span and trace key coincide."""
        return key(self.source, self.trace_id, self.span_id)

    @property
    def trace_key(self) -> str:
        """The external trace container uses its trace ID as its root ID."""
        return key(self.source, self.trace_id, self.trace_id)


active: contextvars.ContextVar[Event | None] = contextvars.ContextVar(
    "promptkeep_external_span", default=None
)


_live: dict[str, Event] = {}


def enter(event: Event) -> Event | None:
    """Activate a callback-owned scope without assuming callbacks share tasks."""
    previous = current()
    _live[event.span_key] = event
    active.set(event)
    return previous


def leave(event: Event, previous: Event | None) -> None:
    """Invalidate inherited snapshots globally; restore only this context's scope."""
    _live.pop(event.span_key, None)
    local = active.get()
    if local is not None and local.span_key == event.span_key:
        active.set(previous if previous and previous.span_key in _live else None)


def current() -> Event | None:
    """Closed callback scopes inherited by another task are no longer attached."""
    event = active.get()
    return (
        event
        if (
            event is not None
            and event.span_key in _live
            and (event.destination is None or event.destination == str(get_settings().db_path))
        )
        else None
    )

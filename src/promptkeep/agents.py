"""Explicit agent and tool scopes for tracing application work.

An ``agent(...)`` scope is the root of one trace; nested agents and
``tool(...)`` scopes are children in that same trace. A wrapped model call
inside any of them records one ``llm`` span linked to the existing run rows.
A tool may name the provider tool-call id it is executing (``call_id=``), and
then becomes a child of the model span that requested it. The context is a
plain ``contextvars`` value, so it follows normal async task inheritance and
never leaks into a thread that did not inherit it.

The scope API is deliberately small: enter it, do the application work, call
``output(value)`` if there is a result, ``fail(message)`` when a normal return
is a failure, and leave. The context manager never invokes a tool or decides
whether to await anything. Telemetry failures are shielded; the only
exceptions here are explicit usage errors (a bad name, re-entering a closed
handle, or output/fail on a closed handle). A tool entered with no active
agent records nothing and warns instead of raising, so helper code that runs
in a forked worker stays safe to call.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from . import checks, controls, storage
from .config import Settings, get_settings
from .conversation import current as current_conversation
from .models import new_run_key

logger = logging.getLogger("promptkeep")

# One serialized payload is capped so a large tool result cannot silently
# dominate a database. The marker is intentionally visible and keeps the
# truncated bytes inspectable as redacted text in history.
_SPAN_PAYLOAD_LIMIT = 64 * 1024

# Provider tool-call ids seen on model responses, resolved to the model span
# that requested them (see register_tool_calls). In-process and bounded, like
# the response-id index: a tool executed in another process, or an id evicted
# under memory pressure, simply keeps its ambient parent span.
_tool_call_index: OrderedDict[str, tuple[str, str, str]] = OrderedDict()
_tool_call_lock = threading.Lock()
_TOOL_CALL_INDEX_LIMIT = 2048


@dataclass
class _Trace:
    """State shared by every span in one trace.

    ``sampled`` is the one successful-span cohort decision; ``problem`` flips
    when a child fails or carries a non-ok verdict; ``recorded`` flips once a
    span is kept despite a sampled-out cohort. Together they explain why a
    trace may contain a problem step without its earlier successful siblings.
    """

    sampled: bool
    conversation_id: int | None
    db_path: str
    pid: int
    problem: bool = False
    recorded: bool = False


@dataclass(frozen=True)
class _Frame:
    """The immutable snapshot an active scope pushes into the contextvar.

    The database and process identity live on the shared ``trace``; a frame
    only carries what is per-scope.
    """

    kind: str
    name: str
    span_key: str
    trace_key: str
    parent_span_key: str | None
    started_at: str
    started: float
    input_json: str | None
    trace: _Trace
    tracked: bool


_current: contextvars.ContextVar[_Frame | None] = contextvars.ContextVar(
    "promptkeep_agent", default=None
)


def current() -> _Frame | None:
    """The active scope snapshot, or None outside a valid scope.

    A frame inherited by a forked child is deliberately ignored: its trace's
    process id no longer matches, and a child must start its own explicit
    scopes rather than share the parent's lifecycle.
    """
    frame = _current.get()
    if frame is None or frame.trace.pid != os.getpid():
        return None
    return frame


def agent(name: str, *, input: Any = None) -> Span:
    """Start an agent scope: the root of a trace, or a child of the active one.

    Use it as ``with`` or ``async with``. The returned handle exposes
    ``span_key`` immediately and ``trace_key`` after entry; ``output(value)``
    records an optional result when the scope closes.
    """
    _validate_name(name, "agent")
    return Span("agent", name, input)


def tool(name: str, *, input: Any = None, call_id: str | None = None) -> Span:
    """Start an application tool scope inside an active agent.

    A tool records what the application explicitly did around it; it never
    infers tool calls from provider responses. ``call_id`` is the provider's
    id for the tool call this scope is executing (a chat completion's
    ``tool_call.id``, a Responses item's ``call_id``); when it names a model
    call recorded in this trace, the tool span becomes that model span's
    child instead of the active scope's. An unknown or foreign id keeps the
    ambient parent. Entering a tool with no active agent records nothing and
    warns; it never raises.
    """
    _validate_name(name, "tool")
    _validate_call_id(call_id)
    return Span("tool", name, input, call_id=call_id)


def finalize_model_span(
    context: _Frame,
    *,
    span_key: str,
    name: str,
    started_at: str,
    duration_ms: int,
    status: str,
    error: str | None,
    sampled: bool,
    checks: list[dict[str, Any]] | None = None,
) -> None:
    """Record the one model span for a physical wrapped call.

    ``context`` is the frame snapshotted when the call started, never the
    ambient context at stream completion. The span is kept when the trace
    cohort was sampled, when a problem makes it worth keeping, or when the
    trace already has a kept span (so a partial trace stays coherent). Its
    end is derived from its start and duration — both stamped around the
    provider interaction — so pre-checks and the writes that follow are not
    part of its recorded interval. A database change during the call drops
    only the span association; ordinary run recording follows current
    configuration.
    """
    try:
        if not attached(context):
            if context.tracked:
                logger.warning(
                    "promptkeep: trace detached by a database or process change; model span not recorded"
                )
            return

        problem = status != "ok" or any(check.get("status") != "ok" for check in (checks or ()))
        if problem:
            context.trace.problem = True
        # A kept trace keeps its shape even when this call was explicitly
        # moved to a sampled-out conversation: the span is structural, while
        # its run row follows that conversation's sampling decision.
        if not (
            sampled
            or context.trace.sampled
            or problem
            or context.trace.problem
            or context.trace.recorded
        ):
            return

        storage.record_span(
            span_key=span_key,
            trace_key=context.trace_key,
            parent_span_key=context.span_key,
            kind="llm",
            name=name,
            started_at=started_at,
            ended_at=_ended_at(started_at, duration_ms),
            duration_ms=duration_ms,
            status=status,
            input_json=None,
            output_json=None,
            error=error,
            sampled=sampled,
        )
        context.trace.recorded = True
    except Exception:
        logger.warning("promptkeep: failed to record model span", exc_info=True)


class Span:
    """The handle ``agent()`` and ``tool()`` return: one scope's lifecycle.

    Enter it once (sync or async), do the work, optionally call ``output()``
    with a result and ``fail()`` when the operation failed without raising,
    and leave. ``span_key`` is available as soon as the handle exists;
    ``trace_key`` after entry.
    """

    def __init__(self, kind: str, name: str, input: Any, call_id: str | None = None) -> None:
        """Prepare a handle; keys are allocated before entry, not at exit."""
        self._kind = kind
        self._name = name
        self._input = input
        self._call_id = call_id
        self._span_key = new_run_key()
        self._trace_key: str | None = None
        self._frame: _Frame | None = None
        self._token: contextvars.Token | None = None
        self._entered = False
        self._closed = False
        self._start_recorded = False
        self._failure: str | None = None
        self._output_set = False
        self._output_json: str | None = None

    @property
    def span_key(self) -> str:
        """This scope's identity, available as soon as the handle exists.

        A key only names an attempt: the writer may drop it, redaction may
        reject it, or tracking may be off.
        """
        return self._span_key

    @property
    def trace_key(self) -> str | None:
        """The trace root's identity, known once the scope has been entered."""
        return self._trace_key

    def output(self, value: Any) -> None:
        """Set the optional result recorded when this scope closes.

        The value is serialized immediately, so mutating it afterwards does
        not rewrite history. Repeated calls keep the last value; an explicit
        ``None`` is recorded as JSON null. Omitted output stays absent.
        """
        if not self._entered or self._closed:
            raise RuntimeError("output() must be called on an open agent/tool scope")
        self._output_json = _encode_payload(value)
        self._output_set = True

    def fail(self, message: str) -> None:
        """Mark the scope failed while still exiting normally.

        A tool can fail on a normal return — a non-zero exit code, an
        ``is_error`` result, unusable output — and should not have to raise
        for the span to say so. The final status becomes ``error`` with
        ``message`` as the error text; an exception escaping the block keeps
        its own message.
        """
        if not self._entered or self._closed:
            raise RuntimeError("fail() must be called on an open agent/tool scope")
        self._failure = message

    def __enter__(self) -> Span:
        """Enter the scope, allocate its trace identity, and persist a root start."""
        if self._entered or self._closed:
            raise RuntimeError("agent/tool scopes cannot be re-entered; create a new scope")
        self._entered = True
        try:
            self._start()
        except BaseException:
            if self._token is not None:
                self._restore()
            else:
                self._entered = False
            raise
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        """Finalize the scope without ever replacing the application outcome."""
        try:
            self._finish(exc_type, exc)
        except Exception:
            logger.warning("promptkeep: failed to finalize agent scope", exc_info=True)
        finally:
            self._restore()
        return False

    async def __aenter__(self) -> Span:
        """Async twin of ``__enter__``."""
        return self.__enter__()

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        """Async twin of ``__exit__``; cancellation becomes status ``cancelled``."""
        return self.__exit__(exc_type, exc, tb)

    def _start(self) -> None:
        """Build the immutable frame and push it for the duration of the scope."""
        settings = get_settings()
        parent = current()

        # A tool with no agent in context records nothing: a forked worker or
        # shared helper can call it, so this warns rather than raising. The
        # handle still works as a context manager and output() is a no-op.
        if self._kind == "tool" and parent is None:
            logger.warning(
                "promptkeep: tool(%r) entered outside an agent scope; the span is not recorded",
                self._name,
            )
            return

        started_at = storage._utcnow()
        started = time.perf_counter()
        db_path = str(settings.db_path)
        pid = os.getpid()
        tracked = settings.enabled and settings.write_mode != "off" and not checks.suppressed()

        # A nested scope inherits the trace and its cohort. A database change
        # mid-scope detaches only telemetry; application code still runs.
        if parent is not None:
            trace = parent.trace
            self._trace_key = parent.trace_key
            parent_span_key = parent.span_key

            # A tool that names the provider call it executes parents under
            # that model span instead of the ambient scope.
            if self._call_id is not None:
                parent_span_key = (
                    _resolve_tool_call(
                        self._call_id, trace_key=parent.trace_key, db_path=trace.db_path
                    )
                    or parent_span_key
                )
            if parent.tracked and db_path != trace.db_path:
                logger.warning(
                    "promptkeep: database changed during an active trace; child span not recorded"
                )
            tracked = tracked and parent.tracked and db_path == trace.db_path
        else:
            self._trace_key = self._span_key
            parent_span_key = None
            trace = _new_trace(settings, tracked, db_path, pid)

        # Snapshot the input now; a caller mutating it later must not rewrite
        # history. Serialization failures lose the payload, never the scope.
        input_json = _encode_payload(self._input) if tracked else None
        frame = _Frame(
            kind=self._kind,
            name=self._name,
            span_key=self._span_key,
            trace_key=self._trace_key,
            parent_span_key=parent_span_key,
            started_at=started_at,
            started=started,
            input_json=input_json,
            trace=trace,
            tracked=tracked,
        )
        self._frame = frame
        self._token = _current.set(frame)

        # A sampled-in scope gets a running row immediately; a sampled-out one
        # waits for its final row, which can restore a missing start if a
        # problem turns up later. Only this row carries the input: the final
        # row omits it, so one large payload is queued and redacted once.
        if tracked and trace.sampled:
            self._start_recorded = True
            storage.record_span(
                span_key=frame.span_key,
                trace_key=frame.trace_key,
                parent_span_key=frame.parent_span_key,
                kind=frame.kind,
                name=frame.name,
                started_at=frame.started_at,
                status="running",
                input_json=frame.input_json,
                sampled=trace.sampled,
            )

    def _finish(self, exc_type: Any, exc: Any) -> None:
        """Write the final row when the trace's sampling and problem state allow it."""
        frame = self._frame
        if frame is None:
            return
        status, error = _outcome(exc_type, exc)

        # fail() marks a normal return as a failure; a real exception keeps
        # its own message.
        if self._failure is not None and status == "ok":
            status, error = "error", self._failure
        still_attached = attached(frame)

        # A suppressed or detached child shares context but cannot affect the
        # parent's sampling state or promote its other spans into storage.
        if status != "ok" and still_attached:
            frame.trace.problem = True
        keep = frame.trace.sampled or frame.trace.problem or frame.trace.recorded or status != "ok"
        if not keep or not still_attached:
            return
        storage.record_span(
            span_key=frame.span_key,
            trace_key=frame.trace_key,
            parent_span_key=frame.parent_span_key,
            kind=frame.kind,
            name=frame.name,
            started_at=frame.started_at,
            ended_at=storage._utcnow(),
            duration_ms=int(round((time.perf_counter() - frame.started) * 1000)),
            status=status,
            input_json=None if self._start_recorded else frame.input_json,
            output_json=self._output_json if self._output_set else None,
            error=error,
            sampled=frame.trace.sampled,
        )
        frame.trace.recorded = True

    def _restore(self) -> None:
        """Pop the frame and close the handle, even if finalization failed."""
        token = self._token
        self._token = None
        self._entered = False
        self._closed = True
        if token is not None:
            try:
                _current.reset(token)
            except Exception:
                logger.warning("promptkeep: could not restore agent context", exc_info=True)


def attached(frame: _Frame) -> bool:
    """Whether this frame still belongs to the configured database and process.

    A scope captures its identity at entry. A later configure() or a forked
    child must not attach new telemetry to the old trace; the application work
    itself is unaffected. One shared helper, so the scope exit and the
    wrapped-call paths in ``integrations.core`` cannot drift apart.
    """
    if not frame.tracked or frame.trace.pid != os.getpid():
        return False
    return str(get_settings().db_path) == frame.trace.db_path


def register_tool_calls(
    call_ids: tuple[str, ...], *, span_key: str, trace_key: str, db_path: str
) -> None:
    """Remember which model span asked for these provider tool-call ids.

    Called by the wrapper when a response names tool calls; ``tool(call_id=)``
    resolves against this in-process index. It is bounded — oldest ids are
    evicted first — and a tool executed in another process simply falls back
    to its ambient parent.
    """
    if not call_ids:
        return
    with _tool_call_lock:
        for call_id in call_ids:
            _tool_call_index[call_id] = (span_key, trace_key, db_path)
            _tool_call_index.move_to_end(call_id)
        while len(_tool_call_index) > _TOOL_CALL_INDEX_LIMIT:
            _tool_call_index.popitem(last=False)


def _resolve_tool_call(call_id: str, *, trace_key: str, db_path: str) -> str | None:
    """The model span that requested ``call_id`` in this trace, if known.

    An id recorded for another trace or database is not this scope's parent:
    one trace stays one containment tree rather than linking across them.
    """
    with _tool_call_lock:
        entry = _tool_call_index.get(call_id)
        if entry is None:
            return None
        span_key, entry_trace, entry_db = entry
        if entry_trace != trace_key or entry_db != db_path:
            return None
        _tool_call_index.move_to_end(call_id)
        return span_key


def _ended_at(started_at: str, duration_ms: int) -> str:
    """The wall-clock end matching a duration measured from ``started_at``.

    Used for model spans: their duration is measured in-process around the
    provider interaction, and stamping the end at write time would grow the
    recorded interval to include the checks and database work that followed.
    """
    return (datetime.fromisoformat(started_at) + timedelta(milliseconds=duration_ms)).isoformat()


def _new_trace(settings: Settings, tracked: bool, db_path: str, pid: int) -> _Trace:
    """Choose the trace cohort, aligning with a conversation when one is active.

    Resolving the conversation here reuses the existing sampler's conversation
    id. The row is created by the same helper a model call would use; a trace
    with no model call still has a trace record, which is its own work.
    """
    conversation_id = None
    sampled = False
    if tracked:
        try:
            active = current_conversation()
            if active is not None:
                conversation_id = storage.get_or_create_conversation(
                    active.external_id, active.title, active.metadata
                )
            sampled = controls.keep_run(settings, {"status": "ok"}, conversation_id)
        except Exception:
            logger.warning("promptkeep: failed to prepare trace sampling", exc_info=True)
            sampled = False
    return _Trace(
        sampled=sampled,
        conversation_id=conversation_id,
        db_path=db_path,
        pid=pid,
    )


def _validate_name(name: Any, kind: str) -> None:
    """Reject a missing or blank explicit name before any scope state changes."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{kind}() requires a non-empty string name")


def _validate_call_id(call_id: Any) -> None:
    """Reject a non-string tool call id before any scope state changes."""
    if call_id is not None and not isinstance(call_id, str):
        raise ValueError("tool() call_id must be a string or None")


def _encode_payload(value: Any) -> str | None:
    """Serialize one span payload, capped with a visible byte-count marker.

    The existing JSON conventions are reused: values are JSON, unsupported
    values fall back to ``repr``, and a custom serializer/repr failure loses
    only the payload. The cap is applied after serialization, so it bounds
    stored bytes rather than the cost of serializing an arbitrary object.
    """
    try:
        encoded = json.dumps(value, ensure_ascii=False, default=repr).encode("utf-8")
    except Exception:
        logger.warning("promptkeep: failed to serialize span payload", exc_info=True)
        return None
    if len(encoded) <= _SPAN_PAYLOAD_LIMIT:
        return encoded.decode("utf-8")
    preview = encoded[:_SPAN_PAYLOAD_LIMIT].decode("utf-8", errors="ignore")
    return f"{preview}\n…[promptkeep truncated: {len(encoded)} bytes]"


def _outcome(exc_type: Any, exc: Any) -> tuple[str, str | None]:
    """Map a scope exit to a status and optional error text.

    Cancellation is inspected before ordinary exceptions because
    ``asyncio.CancelledError`` is a ``BaseException``. The original exception
    is never wrapped; the context manager returns false and lets it continue.
    """
    if exc_type is None:
        return "ok", None
    if isinstance(exc, asyncio.CancelledError) or (
        isinstance(exc_type, type) and issubclass(exc_type, asyncio.CancelledError)
    ):
        return "cancelled", "cancelled"
    if exc is not None:
        return "error", repr(exc)
    return "error", getattr(exc_type, "__name__", "error")


__all__ = ["Span", "agent", "tool"]

"""Transparent instrumentation of the OpenAI SDK.

We never monkey-patch the `openai` module — only the client object the user
explicitly passed to `wrap()` gets its `chat.completions.create` replaced by
a tracking closure. The interceptor:

1. renders any Prompt objects found in `messages` (the API receives plain
   strings — the request payload is byte-for-byte what an unwrapped client
   would send with `prompt.text`),
2. calls the real `create()`,
3. records one run per tracked prompt: version, variables, rendered text,
   model, output, token usage, latency.

Tracking can never break the user's call: recording is exception-shielded,
and errors from the API are re-raised unchanged (after recording a failed
run). Sync/async and streaming/non-streaming are all supported.
"""

from __future__ import annotations

import functools
import inspect
import logging
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from ..conversation import current as current_conversation
from ..prompts import Prompt, RenderedText
from ..tracking import record_conversation_turn, record_prompt_run

logger = logging.getLogger("promptkeep")

# (prompt, variables used, rendered text) — everything a run row needs
# from the request side.
TrackedPrompt = Tuple[Prompt, Dict[str, Any], str]


# --- wrapping entry points ------------------------------------------------------


def wrap_openai_class(cls):
    """Subclass the client class so every instance self-instruments on init."""

    class WrappedClient(cls):
        """The user's client class plus tracking; behaves identically otherwise."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            instrument_client(self)

    # Keep the wrapper indistinguishable in reprs, logs, and debuggers.
    WrappedClient.__name__ = cls.__name__
    WrappedClient.__qualname__ = cls.__qualname__
    WrappedClient.__doc__ = cls.__doc__
    return WrappedClient


def wrap_openai_instance(client):
    """Instrument a live client in place and hand it back."""
    instrument_client(client)
    return client


def instrument_client(client) -> None:
    """Replace client.chat.completions.create with a tracking interceptor.

    Idempotent: wrapping an already-wrapped client is a no-op. Picks the
    sync or async interceptor based on the original method.
    """
    try:
        completions = client.chat.completions
        original = completions.create
    except AttributeError:
        raise TypeError(
            f"{type(client).__name__} does not expose chat.completions.create —"
            " is this an OpenAI client?"
        ) from None
    if getattr(completions, "_pm_instrumented", False):
        return
    if inspect.iscoroutinefunction(original):
        completions.create = _make_async_create(original)
    else:
        completions.create = _make_sync_create(original)
    completions._pm_instrumented = True


# --- message processing --------------------------------------------------------


def _resolve_text(value) -> Optional[Tuple[str, List[TrackedPrompt]]]:
    """If value is a Prompt or provenance-carrying string, return
    (plain string for the API, tracked prompts). Otherwise None."""
    if isinstance(value, Prompt):
        rendered = value.text
        return str(rendered), [(value, rendered.variables, str(rendered))]
    if isinstance(value, RenderedText):
        # A bare RenderedText (constructed without a prompt) is just a string.
        tracked = (
            [(value._pm_prompt, dict(value._pm_variables), str(value))]
            if value._pm_prompt is not None
            else []
        )
        return str(value), tracked
    return None


def _process_messages(messages):
    """Replace Prompt/RenderedText content with plain strings.

    Returns (tracked prompts, new messages). Original message dicts are
    never mutated. Handles both string content and content-block lists.
    """
    tracked: List[TrackedPrompt] = []
    if not isinstance(messages, (list, tuple)):
        return tracked, messages
    new_messages = []
    for message in messages:
        if isinstance(message, dict) and "content" in message:
            content = message["content"]
            # Simple case: content is itself a Prompt / RenderedText.
            resolved = _resolve_text(content)
            if resolved is not None:
                text, found = resolved
                tracked.extend(found)
                message = {**message, "content": text}
            # Multi-part case: content is a list of blocks; check text blocks.
            elif isinstance(content, list):
                new_blocks, changed = [], False
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        block_resolved = _resolve_text(block["text"])
                        if block_resolved is not None:
                            text, found = block_resolved
                            tracked.extend(found)
                            block = {**block, "text": text}
                            changed = True
                    new_blocks.append(block)
                if changed:
                    message = {**message, "content": new_blocks}
        new_messages.append(message)
    return tracked, new_messages


# --- conversation resolution ------------------------------------------------------


def _resolve_conversation(kwargs) -> Tuple[Optional[str], Optional[str], dict]:
    """Which conversation (if any) this call belongs to.

    The explicit per-call kwarg always wins over the ambient `with
    conversation(...)` block — it's popped here so it never reaches the real
    API. Returns (external_id, title, metadata); external_id is None when
    neither path is active.
    """
    explicit = kwargs.pop("promptkeep_conversation", None)
    if explicit is not None:
        return explicit, None, {}
    active = current_conversation()
    if active is not None:
        return active.external_id, active.title, active.metadata
    return None, None, {}


def _extract_input_text(messages) -> Optional[str]:
    """The newest message's text content — what's actually new at this turn.

    Earlier turns (including the model's own prior reply) already live in
    the conversation as earlier rows; the system prompt, if any, is captured
    by the tracked Prompt's own version lineage instead of being repeated
    here. Multimodal content blocks are flattened to their text parts.
    """
    if not messages or not isinstance(messages, (list, tuple)):
        return None
    last = messages[-1]
    # "developer" is the newer-model spelling of the same system-level role.
    if not isinstance(last, dict) or last.get("role") in ("system", "developer"):
        return None
    content = last.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block["text"]
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        ]
        return "\n".join(parts) if parts else None
    return None


def _prepare_conversation(
    external_id, title, metadata, messages
) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    """Resolve/create the conversation row and reserve this call's turn number.

    One reservation per physical API call, reused for every run row it
    produces — a call with two tracked prompts in one message list must not
    split into two turns. Shielded like every other implicit write path.
    """
    if external_id is None:
        return None, None, None
    try:
        from .. import storage

        conversation_id = storage.get_or_create_conversation(external_id, title, metadata)
        if conversation_id is None:
            return None, None, None
        turn_index = storage.next_turn_index(conversation_id)
        return conversation_id, turn_index, _extract_input_text(messages)
    except Exception:
        logger.warning("promptkeep: failed to prepare conversation %r", external_id, exc_info=True)
        return None, None, None


# --- run recording ---------------------------------------------------------------


def _ms(start: float) -> int:
    """Elapsed milliseconds since a perf_counter() start mark."""
    return int(round((time.perf_counter() - start) * 1000))


def _record_runs(
    tracked,
    kwargs,
    response,
    latency_ms,
    status="ok",
    error=None,
    conversation_id=None,
    turn_index=None,
    input_text=None,
) -> None:
    """Write one run row per tracked prompt, sharing the response metadata.

    When no Prompt was tracked but a conversation is active, still write one
    untracked turn row — otherwise plain follow-up messages would be
    invisible to conversation playback even though they're part of the
    session.
    """
    if not tracked and conversation_id is None:
        return
    # Pull metadata defensively — response may be None (errors) or a
    # synthetic stream summary; missing fields just become NULLs.
    try:
        model = getattr(response, "model", None) or kwargs.get("model")
        response_id = getattr(response, "id", None)
        usage = getattr(response, "usage", None)
        output_text = None
        choices = getattr(response, "choices", None)
        if choices:
            message = getattr(choices[0], "message", None)
            content = getattr(message, "content", None)
            if isinstance(content, str):
                output_text = content
        request_params = {k: v for k, v in kwargs.items() if k != "messages"}
    except Exception:
        logger.warning("promptkeep: failed to extract response metadata", exc_info=True)
        return
    if tracked:
        for prompt_obj, variables, rendered in tracked:
            record_prompt_run(
                prompt_obj,
                variables,
                rendered,
                provider="openai",
                model=model,
                request_params=request_params,
                response_id=response_id,
                output_text=output_text,
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                completion_tokens=getattr(usage, "completion_tokens", None),
                total_tokens=getattr(usage, "total_tokens", None),
                latency_ms=latency_ms,
                status=status,
                error=error,
                conversation_id=conversation_id,
                turn_index=turn_index,
                input_text=input_text,
            )
    else:
        record_conversation_turn(
            provider="openai",
            model=model,
            request_params=request_params,
            response_id=response_id,
            output_text=output_text,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
            latency_ms=latency_ms,
            status=status,
            error=error,
            conversation_id=conversation_id,
            turn_index=turn_index,
            input_text=input_text,
        )


# --- interceptors -------------------------------------------------------------------


def _make_sync_create(original):
    """Build the sync replacement for chat.completions.create."""

    @functools.wraps(original)
    def create(*args, **kwargs):
        """Substitute prompts, call the real API, record the outcome."""
        external_id, title, metadata = _resolve_conversation(kwargs)
        tracked, messages = _process_messages(kwargs.get("messages"))
        if "messages" in kwargs:
            kwargs["messages"] = messages
        conversation_id, turn_index, input_text = _prepare_conversation(
            external_id, title, metadata, messages
        )
        start = time.perf_counter()
        try:
            response = original(*args, **kwargs)
        except Exception as exc:
            # Record the failure, then surface the original error untouched.
            _record_runs(
                tracked,
                kwargs,
                None,
                _ms(start),
                status="error",
                error=repr(exc),
                conversation_id=conversation_id,
                turn_index=turn_index,
                input_text=input_text,
            )
            raise
        # Streaming: defer recording until the stream is exhausted.
        if kwargs.get("stream") and (tracked or conversation_id is not None):
            return _SyncStreamProxy(
                response,
                _StreamRecorder(tracked, kwargs, start, conversation_id, turn_index, input_text),
            )
        _record_runs(
            tracked,
            kwargs,
            response,
            _ms(start),
            conversation_id=conversation_id,
            turn_index=turn_index,
            input_text=input_text,
        )
        return response

    return create


def _make_async_create(original):
    """Build the async replacement for chat.completions.create (AsyncOpenAI)."""

    @functools.wraps(original)
    async def create(*args, **kwargs):
        """Async twin of the sync interceptor: substitute, await, record."""
        external_id, title, metadata = _resolve_conversation(kwargs)
        tracked, messages = _process_messages(kwargs.get("messages"))
        if "messages" in kwargs:
            kwargs["messages"] = messages
        conversation_id, turn_index, input_text = _prepare_conversation(
            external_id, title, metadata, messages
        )
        start = time.perf_counter()
        try:
            response = await original(*args, **kwargs)
        except Exception as exc:
            _record_runs(
                tracked,
                kwargs,
                None,
                _ms(start),
                status="error",
                error=repr(exc),
                conversation_id=conversation_id,
                turn_index=turn_index,
                input_text=input_text,
            )
            raise
        if kwargs.get("stream") and (tracked or conversation_id is not None):
            return _AsyncStreamProxy(
                response,
                _StreamRecorder(tracked, kwargs, start, conversation_id, turn_index, input_text),
            )
        _record_runs(
            tracked,
            kwargs,
            response,
            _ms(start),
            conversation_id=conversation_id,
            turn_index=turn_index,
            input_text=input_text,
        )
        return response

    return create


# --- streaming --------------------------------------------------------------------


class _StreamRecorder:
    """Accumulates streamed deltas; writes the run once when the stream ends."""

    def __init__(
        self, tracked, kwargs, start, conversation_id=None, turn_index=None, input_text=None
    ):
        """Hold the request context; content/usage fill in as chunks arrive."""
        self.tracked = tracked
        self.kwargs = kwargs
        self.start = start
        self.conversation_id = conversation_id
        self.turn_index = turn_index
        self.input_text = input_text
        self.parts: List[str] = []
        self.model = None
        self.response_id = None
        self.usage = None
        self.recorded = False

    def absorb(self, chunk) -> None:
        """Fold one chunk in: capture ids/usage, append any delta content."""
        self.model = getattr(chunk, "model", None) or self.model
        self.response_id = getattr(chunk, "id", None) or self.response_id
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            self.usage = usage
        choices = getattr(chunk, "choices", None)
        if choices:
            delta = getattr(choices[0], "delta", None)
            content = getattr(delta, "content", None)
            if isinstance(content, str) and content:
                self.parts.append(content)

    def finish(self, status: str = "ok", error: Optional[str] = None) -> None:
        """Write the run exactly once, from a synthetic response-shaped summary."""
        if self.recorded:
            return
        self.recorded = True
        # Mimic a non-streaming response so _record_runs handles both paths.
        response = SimpleNamespace(
            model=self.model,
            id=self.response_id,
            usage=self.usage,
            choices=[SimpleNamespace(message=SimpleNamespace(content="".join(self.parts) or None))],
        )
        _record_runs(
            self.tracked,
            self.kwargs,
            response,
            _ms(self.start),
            status,
            error,
            conversation_id=self.conversation_id,
            turn_index=self.turn_index,
            input_text=self.input_text,
        )


class _SyncStreamProxy:
    """Wraps a sync stream: passes chunks through, records the run at the end."""

    def __init__(self, stream, recorder: _StreamRecorder):
        self._stream = stream
        self._recorder = recorder
        self._iterator = None

    def __iter__(self):
        return self

    def __next__(self):
        """Yield the next chunk, absorbing it; finish the run on exhaustion/error."""
        if self._iterator is None:
            self._iterator = iter(self._stream)
        try:
            chunk = next(self._iterator)
        except StopIteration:
            self._recorder.finish()
            raise
        except Exception as exc:
            self._recorder.finish(status="error", error=repr(exc))
            raise
        self._recorder.absorb(chunk)
        return chunk

    def __enter__(self):
        """Support `with client...create(stream=True) as stream:` usage."""
        enter = getattr(self._stream, "__enter__", None)
        if enter is not None:
            enter()
        return self

    def __exit__(self, exc_type, exc, tb):
        """Record on context exit (even if the loop broke early), then delegate."""
        self._recorder.finish(
            status="error" if exc_type else "ok",
            error=repr(exc) if exc_type else None,
        )
        exit_ = getattr(self._stream, "__exit__", None)
        if exit_ is not None:
            return exit_(exc_type, exc, tb)
        return False

    def __getattr__(self, name):
        """Everything else (close(), response, ...) delegates to the real stream."""
        return getattr(self._stream, name)


class _AsyncStreamProxy:
    """Async twin of _SyncStreamProxy for AsyncOpenAI streams."""

    def __init__(self, stream, recorder: _StreamRecorder):
        self._stream = stream
        self._recorder = recorder
        self._iterator = None

    def __aiter__(self):
        return self

    async def __anext__(self):
        """Yield the next chunk, absorbing it; finish the run on exhaustion/error."""
        if self._iterator is None:
            self._iterator = self._stream.__aiter__()
        try:
            chunk = await self._iterator.__anext__()
        except StopAsyncIteration:
            self._recorder.finish()
            raise
        except Exception as exc:
            self._recorder.finish(status="error", error=repr(exc))
            raise
        self._recorder.absorb(chunk)
        return chunk

    async def __aenter__(self):
        """Support `async with ... as stream:` usage."""
        enter = getattr(self._stream, "__aenter__", None)
        if enter is not None:
            await enter()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """Record on context exit (even if the loop broke early), then delegate."""
        self._recorder.finish(
            status="error" if exc_type else "ok",
            error=repr(exc) if exc_type else None,
        )
        exit_ = getattr(self._stream, "__aexit__", None)
        if exit_ is not None:
            return await exit_(exc_type, exc, tb)
        return False

    def __getattr__(self, name):
        """Everything else delegates to the real stream."""
        return getattr(self._stream, name)

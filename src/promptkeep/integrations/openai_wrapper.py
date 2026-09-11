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
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from ..checks import (
    CheckContext,
    PromptBlocked,
    RunHandle,
    run_pre_checks,
    schedule_async,
    suppressed,
)
from ..conversation import current as current_conversation
from ..prompts import Prompt, RenderedText
from ..storage import new_run_key
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
        turn_index = storage.reserve_turn_index(conversation_id)
        return conversation_id, turn_index, _extract_input_text(messages)
    except Exception:
        logger.warning("promptkeep: failed to prepare conversation %r", external_id, exc_info=True)
        return None, None, None


class _RunContext(NamedTuple):
    """What identifies a checked call's run row before anything is written.

    run_key is minted up front (see storage.new_run_key) so the RunHandle and
    any late verdict can name the run while its row is still in the write
    queue. The conversation slot and turn text come from _prepare_conversation.
    original_input_text is filled in only when a pre-check rewrote the turn:
    input_text then holds what was sent, this holds what the caller passed.
    """

    run_key: str
    conversation_id: Optional[int]
    turn_index: Optional[int]
    input_text: Optional[str]
    original_input_text: Optional[str] = None


# --- run recording ---------------------------------------------------------------


def _ms(start: float) -> int:
    """Elapsed milliseconds since a perf_counter() start mark."""
    return int(round((time.perf_counter() - start) * 1000))


def _response_fields(response, kwargs) -> dict:
    """The run-row fields pulled from a response and its request kwargs.

    Shared by the checked and unchecked record paths so there's one place that
    knows how to read a response. Defensive by construction — response may be
    None (an error) or a synthetic stream summary, so every missing attribute
    becomes a NULL rather than raising.
    """
    usage = getattr(response, "usage", None)
    return dict(
        model=getattr(response, "model", None) or kwargs.get("model"),
        request_params={k: v for k, v in kwargs.items() if k != "messages"},
        response_id=getattr(response, "id", None),
        output_text=_extract_output_text(response),
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
        total_tokens=getattr(usage, "total_tokens", None),
    )


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
    try:
        fields = _response_fields(response, kwargs)
    except Exception:
        logger.warning("promptkeep: failed to extract response metadata", exc_info=True)
        return
    common = dict(
        provider="openai",
        latency_ms=latency_ms,
        status=status,
        error=error,
        conversation_id=conversation_id,
        turn_index=turn_index,
        input_text=input_text,
        **fields,
    )
    if tracked:
        for prompt_obj, variables, rendered in tracked:
            record_prompt_run(prompt_obj, variables, rendered, **common)
    else:
        record_conversation_turn(**common)


# --- checks -----------------------------------------------------------------------


def _pop_check_kwargs(kwargs):
    """Strip and return per-call checks (never reach the provider)."""
    pre = kwargs.pop("promptkeep_pre", None) or ()
    post = kwargs.pop("promptkeep_post", None) or ()
    return tuple(pre), tuple(post)


def _collect_checks(tracked, per_call_pre, per_call_post):
    """Merge checks from all three scopes: global < prompt < per-call.

    Deduplicated by name (most specific wins), preserving order. A tracked
    prompt contributes its own pre/post; global comes from configure().
    """
    from ..config import get_settings

    settings = get_settings()
    pre, post = [], []
    for prompt_obj, _vars, _rendered in tracked:
        pre.extend(prompt_obj.pre)
        post.extend(prompt_obj.post)
    pre = list(settings.pre) + pre + list(per_call_pre)
    post = list(settings.post) + post + list(per_call_post)
    return _dedupe_checks(pre), _dedupe_checks(post)


def _dedupe_checks(checks):
    """Keep the last check registered under each name (most specific scope)."""
    by_name = {}
    for chk in checks:
        by_name[chk.name] = chk
    return list(by_name.values())


def _joined_text(messages):
    """All outgoing text as one string — what a pre-check scans."""
    if not isinstance(messages, (list, tuple)):
        return ""
    parts = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(
                b["text"] for b in content if isinstance(b, dict) and isinstance(b.get("text"), str)
            )
    return "\n".join(parts)


def _apply_rewrite(messages, new_text):
    """Replace the current turn's text with a pre-check's rewrite.

    Targets exactly the message ``_extract_input_text`` reads as
    ``ctx.last_text`` — the newest non-system message — so a rewrite built
    from ``last_text`` lands where the check meant it to. A trailing
    system/developer message has no user turn to rewrite, so this is a no-op;
    multi-part (image) content keeps its non-text blocks and its rewritten
    text goes into the first text block.
    """
    if not isinstance(messages, (list, tuple)) or not messages:
        return messages
    new_messages = list(messages)
    last = new_messages[-1]
    if not isinstance(last, dict) or last.get("role") in ("system", "developer"):
        return new_messages
    content = last.get("content")
    if isinstance(content, str):
        new_messages[-1] = {**last, "content": new_text}
    elif isinstance(content, list):
        new_blocks, replaced = [], False
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                if not replaced:  # the joined text folds into the first block
                    new_blocks.append({**block, "text": new_text})
                    replaced = True
            else:
                new_blocks.append(block)
        if replaced:
            new_messages[-1] = {**last, "content": new_blocks}
    return new_messages


def _blocked_stub(blocked, kwargs):
    """A response-shaped object returned when on_block='return'."""
    return SimpleNamespace(
        id=None,
        model=kwargs.get("model"),
        usage=None,
        choices=[],
        promptkeep_blocked=SimpleNamespace(check=blocked.name, message=blocked.message),
    )


def _attach_handle(response, handle):
    """Attach the RunHandle as response.promptkeep, tolerating frozen objects."""
    try:
        object.__setattr__(response, "promptkeep", handle)
    except Exception:
        try:
            response.promptkeep = handle
        except Exception:
            logger.warning("promptkeep: could not attach run handle to response", exc_info=True)


def _record_checked(tracked, kwargs, response, latency_ms, status, error, conv, check_rows):
    """Record the run for a checked call (checks bundled), returning its run_key
    — or None when nothing was recorded (tracking disabled, write_mode "off").

    Mirrors _record_runs but returns the primary run's key and attaches the
    verdicts. For multiple tracked prompts, checks bundle onto the first —
    the run whose key the RunHandle carries; the rest record normally under
    keys of their own.
    """
    usage_kw = dict(
        provider="openai",
        latency_ms=latency_ms,
        status=status,
        error=error,
        conversation_id=conv.conversation_id,
        turn_index=conv.turn_index,
        input_text=conv.input_text,
        original_input_text=conv.original_input_text,
        **_response_fields(response, kwargs),
    )
    if tracked:
        run_key = record_prompt_run(
            tracked[0][0],
            tracked[0][1],
            tracked[0][2],
            run_key=conv.run_key,
            checks=check_rows,
            **usage_kw,
        )
        for prompt_obj, variables, rendered in tracked[1:]:
            record_prompt_run(prompt_obj, variables, rendered, **usage_kw)
        return run_key
    return record_conversation_turn(run_key=conv.run_key, checks=check_rows, **usage_kw)


def _extract_output_text(response):
    """The assistant's text reply, or None (errors, empty, non-string)."""
    choices = getattr(response, "choices", None)
    if choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
    return None


# The checked-call flow is split into pure-sync phases (run pre-checks, handle
# a block, run post-checks + record) so the sync and async orchestrators can
# share every bit of logic — the async one just runs each phase via
# asyncio.to_thread so a blocking check or DB write never stalls the loop.


def _checked_pre(tracked, messages, kwargs, pre_checks):
    """Run pre-checks; return (outcome, pre_ctx, version).

    pre_ctx is the exact context the checks saw. The later phases derive the
    post-check context from it (via dataclasses.replace) instead of rebuilding
    it from a positional tuple — one source of truth for the request-side
    fields, and the place a rewrite gets folded in so the audit and the
    recorded row see the rewritten turn, not the original.
    """
    # When several prompts ride one call, checks and the RunHandle refer to
    # the first — the same run the verdicts are filed under (see
    # _record_checked) — so ctx.prompt/version and the recorded row agree.
    prompt_obj = tracked[0][0] if tracked else None
    variables = tracked[0][1] if tracked else None
    version = prompt_obj.version if prompt_obj is not None else None
    pre_ctx = CheckContext(
        rendered=_joined_text(messages),
        messages=messages,
        prompt=prompt_obj,
        variables=variables,
        model=kwargs.get("model"),
        # The current turn — skips a trailing system prompt and flattens image
        # content blocks, so a PII/rewrite check reads the user's real message,
        # not whatever last happened to be a plain string.
        last_text=_extract_input_text(messages),
    )
    outcome = run_pre_checks(list(pre_checks), pre_ctx)
    return outcome, pre_ctx, version


def _apply_pre_rewrite(outcome, kwargs, messages, pre_ctx, conv):
    """Fold a pre-check rewrite into everything downstream, or pass through.

    A rewrite has to reach three places, not just the wire: the outgoing
    request (so the provider sees it), the post-check context (so the audit
    grades what was actually sent), and the recorded ``input_text`` (so the
    stored turn is what actually went out). The turn as the caller passed it
    is kept alongside — ``original_input_text`` on the row, ``original_text``
    on the context — so the record shows both sides of the change and which
    check made it (its verdict row carries the rewritten text). Anything
    that must never be stored in the clear belongs to redaction, which runs
    on every stored field, not to a rewrite. Returns the updated
    (messages, pre_ctx, conv).
    """
    if outcome.rewritten is None:
        return messages, pre_ctx, conv
    original = pre_ctx.last_text
    messages = _apply_rewrite(messages, outcome.rewritten)
    kwargs["messages"] = messages
    last_text = _extract_input_text(messages)
    pre_ctx = replace(
        pre_ctx,
        rendered=_joined_text(messages),
        last_text=last_text,
        messages=messages,
        original_text=original,
    )
    conv = conv._replace(input_text=last_text, original_input_text=original)
    return messages, pre_ctx, conv


def _checked_block(outcome, tracked, kwargs, conv, version):
    """If a pre-check blocked: record the blocked run, then raise PromptBlocked
    or return a response-shaped stub. Returns None when nothing blocked."""
    if outcome.blocked is None:
        return None
    from ..config import get_settings

    pre_rows = [r.to_row() for r in outcome.results]
    run_key = _record_checked(
        tracked,
        kwargs,
        None,
        0,
        "blocked",
        f"blocked by check {outcome.blocked.name!r}",
        conv,
        pre_rows,
    )
    if get_settings().on_block == "raise":
        raise PromptBlocked(outcome.blocked.name, outcome.blocked.message)
    stub = _blocked_stub(outcome.blocked, kwargs)
    _attach_handle(stub, RunHandle(run_key, version, outcome.results))
    return stub


def _checked_record_error(tracked, kwargs, conv, outcome, error_repr, latency_ms):
    """Record a failed checked call (provider raised), keeping pre verdicts."""
    pre_rows = [r.to_row() for r in outcome.results]
    _record_checked(tracked, kwargs, None, latency_ms, "error", error_repr, conv, pre_rows)


def _checked_post(
    response, tracked, kwargs, conv, outcome, pre_ctx, version, post_checks, latency, handle=None
):
    """Run post-checks, record the run with all known verdicts, and surface the
    RunHandle. If `handle` is given (streaming: it's already on the proxy),
    populate it in place; otherwise create one and attach it to `response`.
    Async post-checks land later via their futures.

    The post context is the pre context with the response fields filled in, so
    any pre-check rewrite (already folded into pre_ctx) is what the audit sees.
    Async post-checks run whether or not the run was persisted — the handle's
    wait() still collects their verdicts; only the DB write is skipped then.
    """
    post_ctx = replace(
        pre_ctx,
        model=getattr(response, "model", None) or pre_ctx.model,
        output_text=_extract_output_text(response),
        response=response,
    )
    blocking = [c for c in post_checks if c.mode != "async"]
    asyncs = [c for c in post_checks if c.mode == "async"]
    blocking_results = [c.run(post_ctx) for c in blocking]

    known_results = list(outcome.results) + blocking_results
    check_rows = [r.to_row() for r in known_results]
    run_key = _record_checked(tracked, kwargs, response, latency, "ok", None, conv, check_rows)

    futures = [schedule_async(c, post_ctx, run_key) for c in asyncs]
    if handle is None:
        _attach_handle(response, RunHandle(run_key, version, known_results, futures))
    else:
        handle.run_key = run_key
        handle.checks = list(known_results)
        handle._futures = futures
    return response


def _run_checked_create(original, args, kwargs, tracked, messages, conv, pre_checks, post_checks):
    """Checked call on the sync path: pre-gate, call, post-audit, RunHandle."""
    outcome, pre_ctx, version = _checked_pre(tracked, messages, kwargs, pre_checks)
    stub = _checked_block(outcome, tracked, kwargs, conv, version)
    if stub is not None:
        return stub
    messages, pre_ctx, conv = _apply_pre_rewrite(outcome, kwargs, messages, pre_ctx, conv)
    start = time.perf_counter()
    try:
        response = original(*args, **kwargs)
    except Exception as exc:
        _checked_record_error(tracked, kwargs, conv, outcome, repr(exc), _ms(start))
        raise
    return _checked_post(
        response, tracked, kwargs, conv, outcome, pre_ctx, version, post_checks, _ms(start)
    )


async def _run_checked_create_async(
    original, args, kwargs, tracked, messages, conv, pre_checks, post_checks
):
    """Checked call on the async path: same phases, each run off the event loop
    via asyncio.to_thread so a slow check or DB write never blocks it."""
    import asyncio

    outcome, pre_ctx, version = await asyncio.to_thread(
        _checked_pre, tracked, messages, kwargs, pre_checks
    )
    # _checked_block raises PromptBlocked through to_thread when on_block='raise'.
    stub = await asyncio.to_thread(_checked_block, outcome, tracked, kwargs, conv, version)
    if stub is not None:
        return stub
    messages, pre_ctx, conv = _apply_pre_rewrite(outcome, kwargs, messages, pre_ctx, conv)
    start = time.perf_counter()
    try:
        response = await original(*args, **kwargs)
    except Exception as exc:
        await asyncio.to_thread(
            _checked_record_error, tracked, kwargs, conv, outcome, repr(exc), _ms(start)
        )
        raise
    return await asyncio.to_thread(
        _checked_post,
        response,
        tracked,
        kwargs,
        conv,
        outcome,
        pre_ctx,
        version,
        post_checks,
        _ms(start),
    )


# --- interceptors -------------------------------------------------------------------


def _make_sync_create(original):
    """Build the sync replacement for chat.completions.create."""

    @functools.wraps(original)
    def create(*args, **kwargs):
        """Substitute prompts, call the real API, record the outcome."""
        # A call made from inside a check (e.g. an LLM judge) is a passthrough:
        # no tracking, no nested checks — just strip our kwargs and delegate.
        if suppressed():
            _pop_check_kwargs(kwargs)
            kwargs.pop("promptkeep_conversation", None)
            return original(*args, **kwargs)

        external_id, title, metadata = _resolve_conversation(kwargs)
        per_call_pre, per_call_post = _pop_check_kwargs(kwargs)
        tracked, messages = _process_messages(kwargs.get("messages"))
        if "messages" in kwargs:
            kwargs["messages"] = messages
        conversation_id, turn_index, input_text = _prepare_conversation(
            external_id, title, metadata, messages
        )

        pre_checks, post_checks = _collect_checks(tracked, per_call_pre, per_call_post)
        if pre_checks or post_checks:
            conv = _RunContext(new_run_key(), conversation_id, turn_index, input_text)
            if kwargs.get("stream"):
                return _run_checked_stream(
                    original, args, kwargs, tracked, messages, conv, pre_checks, post_checks
                )
            return _run_checked_create(
                original, args, kwargs, tracked, messages, conv, pre_checks, post_checks
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
        if suppressed():
            _pop_check_kwargs(kwargs)
            kwargs.pop("promptkeep_conversation", None)
            return await original(*args, **kwargs)
        external_id, title, metadata = _resolve_conversation(kwargs)
        per_call_pre, per_call_post = _pop_check_kwargs(kwargs)
        tracked, messages = _process_messages(kwargs.get("messages"))
        if "messages" in kwargs:
            kwargs["messages"] = messages
        conversation_id, turn_index, input_text = _prepare_conversation(
            external_id, title, metadata, messages
        )

        pre_checks, post_checks = _collect_checks(tracked, per_call_pre, per_call_post)
        if pre_checks or post_checks:
            conv = _RunContext(new_run_key(), conversation_id, turn_index, input_text)
            if kwargs.get("stream"):
                return await _run_checked_stream_async(
                    original, args, kwargs, tracked, messages, conv, pre_checks, post_checks
                )
            return await _run_checked_create_async(
                original, args, kwargs, tracked, messages, conv, pre_checks, post_checks
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


class _CheckedStreamRecorder(_StreamRecorder):
    """A stream recorder that also runs post-checks once the stream ends —
    output only exists then — and populates the RunHandle already on the proxy.

    Pre-checks already ran (before the stream started); their verdicts live in
    `outcome`. finish() adds the post verdicts and records the run with both.
    """

    def __init__(
        self, tracked, kwargs, start, conv, outcome, pre_ctx, version, post_checks, handle
    ):
        super().__init__(
            tracked, kwargs, start, conv.conversation_id, conv.turn_index, conv.input_text
        )
        self.conv = conv
        self.outcome = outcome
        self.pre_ctx = pre_ctx
        self.version = version
        self.post_checks = post_checks
        self.handle = handle

    def finish(self, status: str = "ok", error: Optional[str] = None) -> None:
        """Record once: post-checks on success, or a checked error run."""
        if self.recorded:
            return
        self.recorded = True
        if error is not None:
            _checked_record_error(
                self.tracked, self.kwargs, self.conv, self.outcome, error, _ms(self.start)
            )
            return
        response = SimpleNamespace(
            model=self.model,
            id=self.response_id,
            usage=self.usage,
            choices=[SimpleNamespace(message=SimpleNamespace(content="".join(self.parts) or None))],
        )
        _checked_post(
            response,
            self.tracked,
            self.kwargs,
            self.conv,
            self.outcome,
            self.pre_ctx,
            self.version,
            self.post_checks,
            _ms(self.start),
            handle=self.handle,
        )


def _run_checked_stream(original, args, kwargs, tracked, messages, conv, pre_checks, post_checks):
    """Checked streaming (sync): pre-gate before the stream, post-audit after it
    drains. Returns a proxy carrying the (progressively filled) RunHandle."""
    outcome, pre_ctx, version = _checked_pre(tracked, messages, kwargs, pre_checks)
    stub = _checked_block(outcome, tracked, kwargs, conv, version)
    if stub is not None:
        return stub
    messages, pre_ctx, conv = _apply_pre_rewrite(outcome, kwargs, messages, pre_ctx, conv)
    start = time.perf_counter()
    try:
        stream = original(*args, **kwargs)
    except Exception as exc:
        _checked_record_error(tracked, kwargs, conv, outcome, repr(exc), _ms(start))
        raise
    handle = RunHandle(None, version, list(outcome.results))
    recorder = _CheckedStreamRecorder(
        tracked, kwargs, start, conv, outcome, pre_ctx, version, post_checks, handle
    )
    proxy = _SyncStreamProxy(stream, recorder)
    _attach_handle(proxy, handle)
    return proxy


async def _run_checked_stream_async(
    original, args, kwargs, tracked, messages, conv, pre_checks, post_checks
):
    """Checked streaming (async): same shape, blocking phases off the loop."""
    import asyncio

    outcome, pre_ctx, version = await asyncio.to_thread(
        _checked_pre, tracked, messages, kwargs, pre_checks
    )
    stub = await asyncio.to_thread(_checked_block, outcome, tracked, kwargs, conv, version)
    if stub is not None:
        return stub
    messages, pre_ctx, conv = _apply_pre_rewrite(outcome, kwargs, messages, pre_ctx, conv)
    start = time.perf_counter()
    try:
        stream = await original(*args, **kwargs)
    except Exception as exc:
        await asyncio.to_thread(
            _checked_record_error, tracked, kwargs, conv, outcome, repr(exc), _ms(start)
        )
        raise
    handle = RunHandle(None, version, list(outcome.results))
    recorder = _CheckedStreamRecorder(
        tracked, kwargs, start, conv, outcome, pre_ctx, version, post_checks, handle
    )
    proxy = _AsyncStreamProxy(stream, recorder)
    _attach_handle(proxy, handle)
    return proxy


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
        self._finalized = False

    def __aiter__(self):
        return self

    async def _finish(self, status: str = "ok", error: Optional[str] = None) -> None:
        """Finalize off the event loop.

        recorder.finish() runs post-checks and a synchronous DB insert. On the
        async path those must not run on the loop thread — a blocking
        post-check would stall every other task, and even a plain insert can
        wait on the write lock (busy_timeout). Offload to a worker thread,
        mirroring the non-streaming async path.

        The write-once claim is taken *here*, synchronously, before the await:
        two finalize points (StopAsyncIteration and __aexit__) can both reach
        this across the yield, and flipping the flag before offloading is what
        keeps it single-shot. asyncio.shield keeps the record from being lost
        if the consuming task is cancelled mid-finalize.
        """
        import asyncio

        if self._finalized:
            return
        self._finalized = True
        await asyncio.shield(asyncio.to_thread(self._recorder.finish, status, error))

    async def __anext__(self):
        """Yield the next chunk, absorbing it; finish the run on exhaustion/error."""
        if self._iterator is None:
            self._iterator = self._stream.__aiter__()
        try:
            chunk = await self._iterator.__anext__()
        except StopAsyncIteration:
            await self._finish()
            raise
        except Exception as exc:
            await self._finish(status="error", error=repr(exc))
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
        await self._finish(
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

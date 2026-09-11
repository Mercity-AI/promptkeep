"""Provider-agnostic instrumentation: the one interceptor every adapter shares.

``instrument()`` installs a tracking replacement for each call method an
adapter locates on a client. The replacement:

1. resolves the active conversation and strips promptkeep's own kwargs,
2. asks the adapter to parse the request (Prompts substituted by plain
   strings, the current turn picked out),
3. runs pre-checks, calls the real method, runs post-checks,
4. records one run per tracked prompt (or one bare turn inside a
   conversation) through ``tracking`` -> ``storage``,
5. attaches a RunHandle to the response when checks ran.

Sync/async and streaming/non-streaming are all handled here, once: the
async orchestrators run each blocking phase via ``asyncio.to_thread``, and
the stream proxies defer recording until the stream ends. Tracking can never
break the user's call: recording is exception-shielded, adapter failures are
logged and lose telemetry, and provider errors are re-raised unchanged after
a failed run is recorded.
"""

from __future__ import annotations

import functools
import inspect
import logging
import time
from dataclasses import replace
from typing import Any, Iterable, List, NamedTuple, Optional, Tuple

from ..checks import (
    CheckContext,
    PromptBlocked,
    RunHandle,
    run_pre_checks,
    schedule_async,
    suppressed,
)
from ..conversation import current as current_conversation
from ..storage import new_run_key
from ..tracking import record_conversation_turn, record_prompt_run
from .base import ProviderAdapter, Request, ResponseFields, Target

logger = logging.getLogger("promptkeep")

# The kwargs promptkeep understands and strips before the request goes out.
_PROMPTKEEP_KWARGS = ("promptkeep_conversation", "promptkeep_pre", "promptkeep_post")


# --- instrumentation -----------------------------------------------------------------


def instrument(client: Any, adapters: Iterable[ProviderAdapter]) -> int:
    """Instrument every provider surface the adapters find on ``client``.

    Returns how many surfaces were found (already-instrumented ones count —
    wrapping twice is a no-op, not an error). Zero means no adapter
    recognized the object; the caller decides whether that is a TypeError.
    """
    found = 0
    for adapter in adapters:
        target = adapter.locate(client)
        if target is None:
            continue
        found += 1
        _instrument_target(target, adapter)
    return found


def is_instrumented(target: Target) -> bool:
    """Whether a located surface already carries the tracking replacement."""
    return bool(getattr(target.owner, "_pm_instrumented", False))


def _instrument_target(target: Target, adapter: ProviderAdapter) -> None:
    """Replace ``target``'s method with the tracking interceptor, once. Picks
    the sync or async interceptor based on the original method."""
    if is_instrumented(target):
        return
    original = getattr(target.owner, target.attribute)
    if inspect.iscoroutinefunction(original):
        setattr(target.owner, target.attribute, _make_async_create(original, adapter))
    else:
        setattr(target.owner, target.attribute, _make_sync_create(original, adapter))
    target.owner._pm_instrumented = True


# --- conversation resolution --------------------------------------------------------


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


def _prepare_conversation(external_id, title, metadata) -> Tuple[Optional[int], Optional[int]]:
    """Resolve/create the conversation row and reserve this call's turn number.

    One reservation per physical API call, reused for every run row it
    produces — a call with two tracked prompts in one message list must not
    split into two turns. Shielded like every other implicit write path.
    """
    if external_id is None:
        return None, None
    try:
        from .. import storage

        conversation_id = storage.get_or_create_conversation(external_id, title, metadata)
        if conversation_id is None:
            return None, None
        return conversation_id, storage.reserve_turn_index(conversation_id)
    except Exception:
        logger.warning("promptkeep: failed to prepare conversation %r", external_id, exc_info=True)
        return None, None


class _RunContext(NamedTuple):
    """What identifies a checked call's run row before anything is written.

    run_key is minted up front (see storage.new_run_key) so the RunHandle and
    any late verdict can name the run while its row is still in the write
    queue. original_input_text is filled in only when a pre-check rewrote the
    turn: input_text then holds what was sent, this holds what the caller
    passed.
    """

    run_key: str
    conversation_id: Optional[int]
    turn_index: Optional[int]
    input_text: Optional[str]
    original_input_text: Optional[str] = None


# --- adapter calls, shielded ---------------------------------------------------------


def _ms(start: float) -> int:
    """Elapsed milliseconds since a perf_counter() start mark."""
    return int(round((time.perf_counter() - start) * 1000))


def _read_response(adapter: ProviderAdapter, response: Any) -> ResponseFields:
    """The adapter's reading of a response; a failure there loses the
    response's metadata, never the call (or the run row)."""
    try:
        return adapter.read_response(response)
    except Exception:
        logger.warning(
            "promptkeep: %s adapter failed to read response", adapter.provider, exc_info=True
        )
        return ResponseFields()


def _row_fields(request: Request, fields: ResponseFields) -> dict:
    """The response-side columns of a run row. The request's model is the
    fallback when the response (or an error) didn't name one."""
    return dict(
        model=fields.model or request.kwargs.get("model"),
        request_params=request.request_params,
        response_id=fields.response_id,
        output_text=fields.output_text,
        prompt_tokens=fields.prompt_tokens,
        completion_tokens=fields.completion_tokens,
        total_tokens=fields.total_tokens,
    )


# --- run recording ---------------------------------------------------------------


def _record_runs(
    adapter: ProviderAdapter,
    request: Request,
    fields: ResponseFields,
    latency_ms: int,
    status: str = "ok",
    error: Optional[str] = None,
    conversation_id: Optional[int] = None,
    turn_index: Optional[int] = None,
    input_text: Optional[str] = None,
) -> None:
    """Write one run row per tracked prompt, sharing the response metadata.

    When no Prompt was tracked but a conversation is active, still write one
    untracked turn row — otherwise plain follow-up messages would be
    invisible to conversation playback even though they're part of the
    session.
    """
    if not request.tracked and conversation_id is None:
        return
    common = dict(
        provider=adapter.provider,
        latency_ms=latency_ms,
        status=status,
        error=error,
        conversation_id=conversation_id,
        turn_index=turn_index,
        input_text=input_text,
        **_row_fields(request, fields),
    )
    if request.tracked:
        for prompt_obj, variables, rendered in request.tracked:
            record_prompt_run(prompt_obj, variables, rendered, **common)
    else:
        record_conversation_turn(**common)


def _record_checked(adapter, request, fields, latency_ms, status, error, conv, check_rows):
    """Record the run for a checked call (checks bundled), returning its run_key
    — or None when nothing was recorded (tracking disabled, write_mode "off",
    sampled out).

    Mirrors _record_runs but returns the primary run's key and attaches the
    verdicts. For multiple tracked prompts, checks bundle onto the first —
    the run whose key the RunHandle carries; the rest record normally under
    keys of their own.
    """
    common = dict(
        provider=adapter.provider,
        latency_ms=latency_ms,
        status=status,
        error=error,
        conversation_id=conv.conversation_id,
        turn_index=conv.turn_index,
        input_text=conv.input_text,
        original_input_text=conv.original_input_text,
        **_row_fields(request, fields),
    )
    tracked = request.tracked
    if tracked:
        run_key = record_prompt_run(
            tracked[0][0],
            tracked[0][1],
            tracked[0][2],
            run_key=conv.run_key,
            checks=check_rows,
            **common,
        )
        for prompt_obj, variables, rendered in tracked[1:]:
            record_prompt_run(prompt_obj, variables, rendered, **common)
        return run_key
    return record_conversation_turn(run_key=conv.run_key, checks=check_rows, **common)


# --- checks -----------------------------------------------------------------------


def _pop_check_kwargs(kwargs):
    """Strip and return per-call checks (never reach the provider)."""
    pre = kwargs.pop("promptkeep_pre", None) or ()
    post = kwargs.pop("promptkeep_post", None) or ()
    return tuple(pre), tuple(post)


def _strip_promptkeep_kwargs(kwargs) -> None:
    """Drop every promptkeep kwarg — for the untracked passthrough path."""
    for key in _PROMPTKEEP_KWARGS:
        kwargs.pop(key, None)


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


def _attach_handle(response, handle):
    """Attach the RunHandle as response.promptkeep, tolerating frozen objects."""
    try:
        object.__setattr__(response, "promptkeep", handle)
    except Exception:
        try:
            response.promptkeep = handle
        except Exception:
            logger.warning("promptkeep: could not attach run handle to response", exc_info=True)


# The checked-call flow is split into pure-sync phases (run pre-checks, handle
# a block, run post-checks + record) so the sync and async orchestrators can
# share every bit of logic — the async one just runs each phase via
# asyncio.to_thread so a blocking check or DB write never stalls the loop.


def _checked_pre(adapter, request, pre_checks):
    """Run pre-checks; return (outcome, pre_ctx, version).

    pre_ctx is the exact context the checks saw. The later phases derive the
    post-check context from it (via dataclasses.replace) instead of rebuilding
    it — one source of truth for the request-side fields, and the place a
    rewrite gets folded in so the audit and the recorded row see the
    rewritten turn, not the original.
    """
    # When several prompts ride one call, checks and the RunHandle refer to
    # the first — the same run the verdicts are filed under (see
    # _record_checked) — so ctx.prompt/version and the recorded row agree.
    prompt_obj = request.tracked[0][0] if request.tracked else None
    variables = request.tracked[0][1] if request.tracked else None
    version = prompt_obj.version if prompt_obj is not None else None
    pre_ctx = CheckContext(
        rendered=request.joined_text,
        messages=request.payload,
        prompt=prompt_obj,
        variables=variables,
        model=request.kwargs.get("model"),
        provider=adapter.provider,
        last_text=request.input_text,
    )
    outcome = run_pre_checks(list(pre_checks), pre_ctx)
    return outcome, pre_ctx, version


def _apply_pre_rewrite(adapter, outcome, request, pre_ctx, conv):
    """Fold a pre-check rewrite into everything downstream, or pass through.

    A rewrite has to reach three places, not just the wire: the outgoing
    request (so the provider sees it), the post-check context (so the audit
    grades what was actually sent), and the recorded ``input_text`` (so the
    stored turn is what actually went out). The turn as the caller passed it
    is kept alongside — ``original_input_text`` on the row, ``original_text``
    on the context — so the record shows both sides of the change and which
    check made it (its verdict row carries the rewritten text). Returns the
    updated (request, pre_ctx, conv).
    """
    if outcome.rewritten is None:
        return request, pre_ctx, conv
    original = pre_ctx.last_text
    request = adapter.apply_rewrite(request, outcome.rewritten)
    pre_ctx = replace(
        pre_ctx,
        rendered=request.joined_text,
        last_text=request.input_text,
        messages=request.payload,
        original_text=original,
    )
    conv = conv._replace(input_text=request.input_text, original_input_text=original)
    return request, pre_ctx, conv


def _checked_block(adapter, outcome, request, conv, version):
    """If a pre-check blocked: record the blocked run, then raise PromptBlocked
    or return a response-shaped stub. Returns None when nothing blocked."""
    if outcome.blocked is None:
        return None
    from ..config import get_settings

    pre_rows = [r.to_row() for r in outcome.results]
    run_key = _record_checked(
        adapter,
        request,
        ResponseFields(),
        0,
        "blocked",
        f"blocked by check {outcome.blocked.name!r}",
        conv,
        pre_rows,
    )
    if get_settings().on_block == "raise":
        raise PromptBlocked(outcome.blocked.name, outcome.blocked.message)
    stub = adapter.blocked_stub(request, outcome.blocked)
    _attach_handle(stub, RunHandle(run_key, version, outcome.results))
    return stub


def _checked_record_error(adapter, request, conv, outcome, error_repr, latency_ms):
    """Record a failed checked call (provider raised), keeping pre verdicts."""
    pre_rows = [r.to_row() for r in outcome.results]
    _record_checked(
        adapter, request, ResponseFields(), latency_ms, "error", error_repr, conv, pre_rows
    )


def _checked_post(
    adapter,
    response,
    fields,
    request,
    conv,
    outcome,
    pre_ctx,
    version,
    post_checks,
    latency,
    handle=None,
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
        model=fields.model or pre_ctx.model,
        output_text=fields.output_text,
        response=response,
    )
    blocking = [c for c in post_checks if c.mode != "async"]
    asyncs = [c for c in post_checks if c.mode == "async"]
    blocking_results = [c.run(post_ctx) for c in blocking]

    known_results = list(outcome.results) + blocking_results
    check_rows = [r.to_row() for r in known_results]
    run_key = _record_checked(adapter, request, fields, latency, "ok", None, conv, check_rows)

    futures = [schedule_async(c, post_ctx, run_key) for c in asyncs]
    if handle is None:
        _attach_handle(response, RunHandle(run_key, version, known_results, futures))
    else:
        handle.run_key = run_key
        handle.checks = list(known_results)
        handle._futures = futures
    return response


def _run_checked_create(adapter, original, args, request, conv, pre_checks, post_checks):
    """Checked call on the sync path: pre-gate, call, post-audit, RunHandle."""
    outcome, pre_ctx, version = _checked_pre(adapter, request, pre_checks)
    stub = _checked_block(adapter, outcome, request, conv, version)
    if stub is not None:
        return stub
    request, pre_ctx, conv = _apply_pre_rewrite(adapter, outcome, request, pre_ctx, conv)
    start = time.perf_counter()
    try:
        response = original(*args, **request.kwargs)
    except Exception as exc:
        _checked_record_error(adapter, request, conv, outcome, repr(exc), _ms(start))
        raise
    fields = _read_response(adapter, response)
    return _checked_post(
        adapter, response, fields, request, conv, outcome, pre_ctx, version, post_checks, _ms(start)
    )


async def _run_checked_create_async(
    adapter, original, args, request, conv, pre_checks, post_checks
):
    """Checked call on the async path: same phases, each run off the event loop
    via asyncio.to_thread so a slow check or DB write never blocks it."""
    import asyncio

    outcome, pre_ctx, version = await asyncio.to_thread(_checked_pre, adapter, request, pre_checks)
    # _checked_block raises PromptBlocked through to_thread when on_block='raise'.
    stub = await asyncio.to_thread(_checked_block, adapter, outcome, request, conv, version)
    if stub is not None:
        return stub
    request, pre_ctx, conv = _apply_pre_rewrite(adapter, outcome, request, pre_ctx, conv)
    start = time.perf_counter()
    try:
        response = await original(*args, **request.kwargs)
    except Exception as exc:
        await asyncio.to_thread(
            _checked_record_error, adapter, request, conv, outcome, repr(exc), _ms(start)
        )
        raise
    fields = _read_response(adapter, response)
    return await asyncio.to_thread(
        _checked_post,
        adapter,
        response,
        fields,
        request,
        conv,
        outcome,
        pre_ctx,
        version,
        post_checks,
        _ms(start),
    )


# --- interceptors -------------------------------------------------------------------


def _make_sync_create(original, adapter: ProviderAdapter):
    """Build the sync replacement for a provider's call method."""

    @functools.wraps(original)
    def create(*args, **kwargs):
        """Substitute prompts, call the real API, record the outcome."""
        # A call made from inside a check (e.g. an LLM judge) is a passthrough:
        # no tracking, no nested checks — just strip our kwargs and delegate.
        if suppressed():
            _strip_promptkeep_kwargs(kwargs)
            return original(*args, **kwargs)

        external_id, title, metadata = _resolve_conversation(kwargs)
        per_call_pre, per_call_post = _pop_check_kwargs(kwargs)
        request = adapter.parse_request(kwargs)
        conversation_id, turn_index = _prepare_conversation(external_id, title, metadata)

        pre_checks, post_checks = _collect_checks(request.tracked, per_call_pre, per_call_post)
        if pre_checks or post_checks:
            conv = _RunContext(new_run_key(), conversation_id, turn_index, request.input_text)
            if adapter.is_streaming(request):
                return _run_checked_stream(
                    adapter, original, args, request, conv, pre_checks, post_checks
                )
            return _run_checked_create(
                adapter, original, args, request, conv, pre_checks, post_checks
            )

        start = time.perf_counter()
        try:
            response = original(*args, **request.kwargs)
        except Exception as exc:
            # Record the failure, then surface the original error untouched.
            _record_runs(
                adapter,
                request,
                ResponseFields(),
                _ms(start),
                status="error",
                error=repr(exc),
                conversation_id=conversation_id,
                turn_index=turn_index,
                input_text=request.input_text,
            )
            raise
        # Streaming: defer recording until the stream is exhausted.
        if adapter.is_streaming(request) and (request.tracked or conversation_id is not None):
            return _SyncStreamProxy(
                response,
                _StreamRecorder(adapter, request, start, conversation_id, turn_index),
            )
        _record_runs(
            adapter,
            request,
            _read_response(adapter, response),
            _ms(start),
            conversation_id=conversation_id,
            turn_index=turn_index,
            input_text=request.input_text,
        )
        return response

    return create


def _make_async_create(original, adapter: ProviderAdapter):
    """Build the async replacement for a provider's call method."""

    @functools.wraps(original)
    async def create(*args, **kwargs):
        """Async twin of the sync interceptor: substitute, await, record."""
        if suppressed():
            _strip_promptkeep_kwargs(kwargs)
            return await original(*args, **kwargs)
        external_id, title, metadata = _resolve_conversation(kwargs)
        per_call_pre, per_call_post = _pop_check_kwargs(kwargs)
        request = adapter.parse_request(kwargs)
        conversation_id, turn_index = _prepare_conversation(external_id, title, metadata)

        pre_checks, post_checks = _collect_checks(request.tracked, per_call_pre, per_call_post)
        if pre_checks or post_checks:
            conv = _RunContext(new_run_key(), conversation_id, turn_index, request.input_text)
            if adapter.is_streaming(request):
                return await _run_checked_stream_async(
                    adapter, original, args, request, conv, pre_checks, post_checks
                )
            return await _run_checked_create_async(
                adapter, original, args, request, conv, pre_checks, post_checks
            )

        start = time.perf_counter()
        try:
            response = await original(*args, **request.kwargs)
        except Exception as exc:
            _record_runs(
                adapter,
                request,
                ResponseFields(),
                _ms(start),
                status="error",
                error=repr(exc),
                conversation_id=conversation_id,
                turn_index=turn_index,
                input_text=request.input_text,
            )
            raise
        if adapter.is_streaming(request) and (request.tracked or conversation_id is not None):
            return _AsyncStreamProxy(
                response,
                _StreamRecorder(adapter, request, start, conversation_id, turn_index),
            )
        _record_runs(
            adapter,
            request,
            _read_response(adapter, response),
            _ms(start),
            conversation_id=conversation_id,
            turn_index=turn_index,
            input_text=request.input_text,
        )
        return response

    return create


# --- streaming --------------------------------------------------------------------


class _StreamRecorder:
    """Feeds streamed chunks to the adapter's absorber; writes the run once
    when the stream ends."""

    def __init__(self, adapter, request, start, conversation_id=None, turn_index=None):
        """Hold the request context; the absorber fills in as chunks arrive."""
        self.adapter = adapter
        self.request = request
        self.start = start
        self.conversation_id = conversation_id
        self.turn_index = turn_index
        self.absorber = adapter.stream_absorber()
        self.recorded = False

    def absorb(self, chunk) -> None:
        """Fold one chunk in. An absorber bug loses telemetry, never a chunk."""
        try:
            self.absorber.absorb(chunk)
        except Exception:
            logger.warning(
                "promptkeep: %s adapter failed to absorb a stream chunk",
                self.adapter.provider,
                exc_info=True,
            )

    def summary(self) -> ResponseFields:
        """What the absorber gathered, shielded like read_response."""
        try:
            return self.absorber.summary()
        except Exception:
            logger.warning(
                "promptkeep: %s adapter failed to summarize a stream",
                self.adapter.provider,
                exc_info=True,
            )
            return ResponseFields()

    def finish(self, status: str = "ok", error: Optional[str] = None) -> None:
        """Write the run exactly once."""
        if self.recorded:
            return
        self.recorded = True
        _record_runs(
            self.adapter,
            self.request,
            self.summary(),
            _ms(self.start),
            status,
            error,
            conversation_id=self.conversation_id,
            turn_index=self.turn_index,
            input_text=self.request.input_text,
        )


class _CheckedStreamRecorder(_StreamRecorder):
    """A stream recorder that also runs post-checks once the stream ends —
    output only exists then — and populates the RunHandle already on the proxy.

    Pre-checks already ran (before the stream started); their verdicts live in
    `outcome`. finish() adds the post verdicts and records the run with both.
    Post-checks receive the stream's ResponseFields summary as ``ctx.response``
    — there is no single provider response object for a stream.
    """

    def __init__(
        self, adapter, request, start, conv, outcome, pre_ctx, version, post_checks, handle
    ):
        super().__init__(adapter, request, start, conv.conversation_id, conv.turn_index)
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
                self.adapter, self.request, self.conv, self.outcome, error, _ms(self.start)
            )
            return
        fields = self.summary()
        _checked_post(
            self.adapter,
            fields,
            fields,
            self.request,
            self.conv,
            self.outcome,
            self.pre_ctx,
            self.version,
            self.post_checks,
            _ms(self.start),
            handle=self.handle,
        )


def _run_checked_stream(adapter, original, args, request, conv, pre_checks, post_checks):
    """Checked streaming (sync): pre-gate before the stream, post-audit after it
    drains. Returns a proxy carrying the (progressively filled) RunHandle."""
    outcome, pre_ctx, version = _checked_pre(adapter, request, pre_checks)
    stub = _checked_block(adapter, outcome, request, conv, version)
    if stub is not None:
        return stub
    request, pre_ctx, conv = _apply_pre_rewrite(adapter, outcome, request, pre_ctx, conv)
    start = time.perf_counter()
    try:
        stream = original(*args, **request.kwargs)
    except Exception as exc:
        _checked_record_error(adapter, request, conv, outcome, repr(exc), _ms(start))
        raise
    handle = RunHandle(None, version, list(outcome.results))
    recorder = _CheckedStreamRecorder(
        adapter, request, start, conv, outcome, pre_ctx, version, post_checks, handle
    )
    proxy = _SyncStreamProxy(stream, recorder)
    _attach_handle(proxy, handle)
    return proxy


async def _run_checked_stream_async(
    adapter, original, args, request, conv, pre_checks, post_checks
):
    """Checked streaming (async): same shape, blocking phases off the loop."""
    import asyncio

    outcome, pre_ctx, version = await asyncio.to_thread(_checked_pre, adapter, request, pre_checks)
    stub = await asyncio.to_thread(_checked_block, adapter, outcome, request, conv, version)
    if stub is not None:
        return stub
    request, pre_ctx, conv = _apply_pre_rewrite(adapter, outcome, request, pre_ctx, conv)
    start = time.perf_counter()
    try:
        stream = await original(*args, **request.kwargs)
    except Exception as exc:
        await asyncio.to_thread(
            _checked_record_error, adapter, request, conv, outcome, repr(exc), _ms(start)
        )
        raise
    handle = RunHandle(None, version, list(outcome.results))
    recorder = _CheckedStreamRecorder(
        adapter, request, start, conv, outcome, pre_ctx, version, post_checks, handle
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
    """Async twin of _SyncStreamProxy for async streams."""

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


__all__: List[str] = ["instrument", "is_instrumented"]

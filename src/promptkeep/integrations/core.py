"""Provider-agnostic instrumentation: the one interceptor every adapter shares.

``instrument()`` installs a tracking replacement for each call method an
adapter locates on a client. The replacement builds a ``_Call`` — one
object holding everything known about the call in flight — and drives it:

1. promptkeep's own kwargs come off and the active conversation is resolved,
2. the adapter parses the request (Prompts substituted by plain strings, the
   current turn picked out),
3. pre-checks run and may block or rewrite,
4. the real method is called,
5. post-checks run, one run row per tracked prompt (or one bare turn) is
   recorded through ``tracking`` -> ``storage``, and a RunHandle is attached
   to the response when checks ran.

Sync/async and streaming/non-streaming are all handled here, once: the
async interceptor runs each blocking phase via ``asyncio.to_thread``, and
the stream proxies defer the last phase until the stream ends. Tracking can
never break the user's call: recording is exception-shielded, adapter
failures are logged and lose telemetry, and provider errors are re-raised
unchanged after a failed run is recorded.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import time
from collections.abc import Iterable
from dataclasses import replace
from typing import Any

from .. import storage
from ..checks import (
    Check,
    CheckContext,
    CheckResult,
    PreOutcome,
    PromptBlocked,
    RunHandle,
    run_pre_checks,
    schedule_async,
    suppressed,
)
from ..config import get_settings
from ..conversation import current as current_conversation
from ..models import new_run_key
from ..tracking import record_prompt_run
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


# --- the interceptors ------------------------------------------------------------------


def _make_sync_create(original: Any, adapter: ProviderAdapter) -> Any:
    """Build the sync replacement for a provider's call method."""

    @functools.wraps(original)
    def create(*args: Any, **kwargs: Any) -> Any:
        """Substitute prompts, gate, call the real API, audit, record."""
        # A call made from inside a check (e.g. an LLM judge) is a passthrough:
        # no tracking, no nested checks — just strip our kwargs and delegate.
        if suppressed():
            _strip_promptkeep_kwargs(kwargs)
            return original(*args, **kwargs)

        # Everything known before the provider is called, in one place.
        call = _Call(adapter, kwargs)
        if call.checked:
            call.run_pre()
            blocked = call.blocked_response()
            if blocked is not None:
                return blocked

        # The real call. A provider error is recorded, then re-raised untouched.
        start = time.perf_counter()
        try:
            response = original(*args, **call.request.kwargs)
        except Exception as exc:
            call.record_error(repr(exc), _ms(start))
            raise

        # Streams are recorded when they end; everything else right now.
        if call.streaming:
            return _wrap_stream(call, response, start, _SyncStreamProxy)
        return call.finish(response, _read_response(adapter, response), _ms(start))

    return create


def _make_async_create(original: Any, adapter: ProviderAdapter) -> Any:
    """Build the async replacement for a provider's call method: the same
    phases, each blocking one run via asyncio.to_thread so a slow check or a
    DB write never stalls the event loop."""

    @functools.wraps(original)
    async def create(*args: Any, **kwargs: Any) -> Any:
        """Async twin of the sync interceptor."""
        if suppressed():
            _strip_promptkeep_kwargs(kwargs)
            return await original(*args, **kwargs)

        call = _Call(adapter, kwargs)
        if call.checked:
            await asyncio.to_thread(call.run_pre)
            # PromptBlocked raises straight through to_thread when on_block="raise".
            blocked = await asyncio.to_thread(call.blocked_response)
            if blocked is not None:
                return blocked

        start = time.perf_counter()
        try:
            response = await original(*args, **call.request.kwargs)
        except Exception as exc:
            await asyncio.to_thread(call.record_error, repr(exc), _ms(start))
            raise

        if call.streaming:
            return _wrap_stream(call, response, start, _AsyncStreamProxy)
        fields = _read_response(adapter, response)
        return await asyncio.to_thread(call.finish, response, fields, _ms(start))

    return create


def _wrap_stream(call: _Call, stream: Any, start: float, proxy_cls: type) -> Any:
    """Hand a stream back through a proxy that records the call once the
    stream ends. An untracked stream (no Prompt, no conversation, no checks)
    is returned as-is — there is nothing to record."""
    if not call.records:
        return stream
    proxy = proxy_cls(stream, _StreamRecorder(call, start))
    # A checked stream carries its handle from the start; finish() fills it.
    if call.checked:
        _attach_handle(proxy, call.handle)
    return proxy


# --- one call in flight ------------------------------------------------------------------


class _Call:
    """One tracked call in flight: what the interceptor knows before, during
    and after the provider call, and the four things it does with it.

    Built from the raw kwargs: promptkeep's own kwargs come off, the adapter
    parses the rest, the conversation slot is reserved, and checks are
    collected across the three scopes. A call is *checked* when any check
    applies; checked calls mint their ``run_key`` up front (see
    ``models.new_run_key``) so the RunHandle and late verdicts can name the
    run before its row exists.
    """

    def __init__(self, adapter: ProviderAdapter, kwargs: dict[str, Any]) -> None:
        """Parse the call and reserve everything it needs. No provider I/O."""
        # promptkeep's kwargs never reach the provider; take them off first.
        external_id, title, metadata = _resolve_conversation(kwargs)
        per_call_pre, per_call_post = _pop_check_kwargs(kwargs)

        # The adapter turns what's left into a Request (Prompts substituted).
        self.adapter = adapter
        self.request: Request = adapter.parse_request(kwargs)

        # The conversation slot: one turn number per physical API call.
        self.conversation_id, self.turn_index = _prepare_conversation(external_id, title, metadata)

        # Checks across scopes; a checked call gets its identity now.
        self.pre_checks, self.post_checks = _collect_checks(
            self.request.tracked, per_call_pre, per_call_post
        )
        self.checked = bool(self.pre_checks or self.post_checks)
        self.run_key: str | None = new_run_key() if self.checked else None

        # Filled in by run_pre() for a checked call.
        self.outcome = PreOutcome()
        self.ctx: CheckContext | None = None
        self.version: int | None = None
        self.handle: RunHandle | None = None
        # Set only when a pre-check rewrote the turn: what the caller passed.
        self.original_input_text: str | None = None

    @property
    def records(self) -> bool:
        """Whether this call produces a run row at all: a tracked Prompt, a
        conversation turn, or a checked call (whose row anchors its verdicts)."""
        return bool(self.request.tracked) or self.conversation_id is not None or self.checked

    @property
    def streaming(self) -> bool:
        """Whether the provider will return a stream for this call."""
        return self.adapter.is_streaming(self.request)

    # --- phase 1: the gate --------------------------------------------------

    def run_pre(self) -> None:
        """Run the pre-checks in order, then fold any rewrite into the request.

        The context the checks saw is kept: the post-check context is derived
        from it later, so a rewrite reaches the outgoing request, the audit
        and the recorded row alike. When several prompts ride one call, the
        checks and the RunHandle refer to the first — the run the verdicts
        are filed under (see _write) — so ctx.prompt and the row agree.
        """
        tracked = self.request.tracked
        prompt_obj = tracked[0][0] if tracked else None
        self.version = prompt_obj.version if prompt_obj is not None else None
        self.ctx = CheckContext(
            rendered=self.request.joined_text,
            messages=self.request.payload,
            prompt=prompt_obj,
            variables=tracked[0][1] if tracked else None,
            model=self.request.kwargs.get("model"),
            provider=self.adapter.provider,
            last_text=self.request.input_text,
        )
        self.outcome = run_pre_checks(list(self.pre_checks), self.ctx)
        self.handle = RunHandle(None, self.version, list(self.outcome.results))

        # A rewrite has to reach three places: the wire, the audit, the row.
        if self.outcome.rewritten is not None:
            original = self.ctx.last_text
            self.request = self.adapter.apply_rewrite(self.request, self.outcome.rewritten)
            self.ctx = replace(
                self.ctx,
                rendered=self.request.joined_text,
                last_text=self.request.input_text,
                messages=self.request.payload,
                original_text=original,
            )
            self.original_input_text = original

    def blocked_response(self) -> Any | None:
        """If a pre-check blocked: record the blocked run, then raise
        PromptBlocked or return a response-shaped stub carrying the handle.
        None when nothing blocked."""
        blocked = self.outcome.blocked
        if blocked is None:
            return None
        run_key = self._write(ResponseFields(), 0, "blocked", f"blocked by check {blocked.name!r}")
        if get_settings().on_block == "raise":
            raise PromptBlocked(blocked.name, blocked.message)
        stub = self.adapter.blocked_stub(self.request, blocked)
        assert self.handle is not None
        self.handle.run_key = run_key
        _attach_handle(stub, self.handle)
        return stub

    # --- phase 2: the outcome -----------------------------------------------

    def record_error(
        self, error_repr: str, latency_ms: int, fields: ResponseFields | None = None
    ) -> None:
        """The provider raised (or a stream broke): record the failed run,
        keeping any pre verdicts and whatever a stream had produced so far."""
        self._write(fields or ResponseFields(), latency_ms, "error", error_repr)

    def finish(
        self, response: Any, fields: ResponseFields, latency_ms: int, attach: bool = True
    ) -> Any:
        """Record the completed call and return the response.

        Unchecked: one row (per tracked prompt, or a bare turn) and nothing
        else. Checked: the blocking post-checks run first, the row goes down
        with every verdict known so far, the async post-checks are scheduled
        (their verdicts land later, by run_key), and the RunHandle is filled
        in — and attached to the response unless it already rides on a
        stream proxy. For a stream, ``response`` is the ResponseFields summary
        (there is no single provider object), and that is what a post-check
        sees as ``ctx.response``.
        """
        if not self.checked:
            self._write(fields, latency_ms, "ok", None)
            return response

        # The audit context: the pre context plus the response.
        assert self.ctx is not None and self.handle is not None
        post_ctx = replace(
            self.ctx,
            model=fields.model or self.ctx.model,
            output_text=fields.output_text,
            response=response,
        )
        blocking = [c for c in self.post_checks if c.mode != "async"]
        asyncs = [c for c in self.post_checks if c.mode == "async"]
        results = list(self.outcome.results) + [c.run(post_ctx) for c in blocking]

        # The row, then the verdicts still to come, then the handle.
        run_key = self._write(fields, latency_ms, "ok", None, results)
        futures = [schedule_async(c, post_ctx, run_key) for c in asyncs]
        self.handle.run_key = run_key
        self.handle.checks = list(results)
        self.handle._futures = futures
        if attach:
            _attach_handle(response, self.handle)
        return response

    # --- the write -------------------------------------------------------------

    def _write(
        self,
        fields: ResponseFields,
        latency_ms: int,
        status: str,
        error: str | None,
        results: list[CheckResult] | None = None,
    ) -> str | None:
        """Write this call's run row(s); return the primary run's key.

        One row per tracked prompt, sharing the response metadata; a call
        with no Prompt still gets one bare row when it belongs to a
        conversation (so follow-up turns replay) or is checked (so its
        verdicts have a run to hang off). The verdicts — ``results`` if
        given, else the pre verdicts — and the minted run_key go with the
        first prompt's row; the rest record plainly under keys of their own.
        Shielded: nothing here can raise into the request path.
        """
        if not self.records:
            return None
        verdicts = self.outcome.results if results is None else results
        check_rows = [r.to_row() for r in verdicts] or None
        common: dict[str, Any] = dict(
            provider=self.adapter.provider,
            latency_ms=latency_ms,
            status=status,
            error=error,
            conversation_id=self.conversation_id,
            turn_index=self.turn_index,
            input_text=self.request.input_text,
            original_input_text=self.original_input_text,
            **_row_fields(self.request, fields),
        )
        tracked = self.request.tracked
        if not tracked:
            return _record_bare(run_key=self.run_key, checks=check_rows, **common)
        (prompt_obj, variables, rendered), *rest = tracked
        run_key = record_prompt_run(
            prompt_obj, variables, rendered, run_key=self.run_key, checks=check_rows, **common
        )
        for prompt_obj, variables, rendered in rest:
            record_prompt_run(prompt_obj, variables, rendered, **common)
        return run_key


# --- request-side helpers ------------------------------------------------------------


def _resolve_conversation(kwargs: dict[str, Any]) -> tuple[str | None, str | None, dict[str, Any]]:
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


def _prepare_conversation(
    external_id: str | None, title: str | None, metadata: dict[str, Any]
) -> tuple[int | None, int | None]:
    """Resolve/create the conversation row and reserve this call's turn number.

    One reservation per physical API call, reused for every run row it
    produces — a call with two tracked prompts in one message list must not
    split into two turns. Shielded like every other implicit write path.
    """
    if external_id is None:
        return None, None
    try:
        conversation_id = storage.get_or_create_conversation(external_id, title, metadata)
        if conversation_id is None:
            return None, None
        return conversation_id, storage.reserve_turn_index(conversation_id)
    except Exception:
        logger.warning("promptkeep: failed to prepare conversation %r", external_id, exc_info=True)
        return None, None


def _pop_check_kwargs(kwargs: dict[str, Any]) -> tuple[tuple[Check, ...], tuple[Check, ...]]:
    """Strip and return per-call checks (never reach the provider)."""
    pre = kwargs.pop("promptkeep_pre", None) or ()
    post = kwargs.pop("promptkeep_post", None) or ()
    return tuple(pre), tuple(post)


def _strip_promptkeep_kwargs(kwargs: dict[str, Any]) -> None:
    """Drop every promptkeep kwarg — for the untracked passthrough path."""
    for key in _PROMPTKEEP_KWARGS:
        kwargs.pop(key, None)


def _collect_checks(
    tracked: list, per_call_pre: tuple[Check, ...], per_call_post: tuple[Check, ...]
) -> tuple[list[Check], list[Check]]:
    """Merge checks from all three scopes: global < prompt < per-call.

    Deduplicated by name (most specific wins), preserving order. A tracked
    prompt contributes its own pre/post; global comes from configure().
    """
    settings = get_settings()
    pre: list[Check] = []
    post: list[Check] = []
    for prompt_obj, _vars, _rendered in tracked:
        pre.extend(prompt_obj.pre)
        post.extend(prompt_obj.post)
    pre = list(settings.pre) + pre + list(per_call_pre)
    post = list(settings.post) + post + list(per_call_post)
    return _dedupe_checks(pre), _dedupe_checks(post)


def _dedupe_checks(checks: list[Check]) -> list[Check]:
    """Keep the last check registered under each name (most specific scope)."""
    by_name: dict[str, Check] = {}
    for chk in checks:
        by_name[chk.name] = chk
    return list(by_name.values())


# --- response-side helpers ------------------------------------------------------------


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


def _row_fields(request: Request, fields: ResponseFields) -> dict[str, Any]:
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


def _record_bare(**fields: Any) -> str | None:
    """A run row with no Prompt — a bare conversation turn, or the anchor a
    checked call's verdicts hang off — shielded like every write made from
    the request path."""
    try:
        return storage.record_run(**fields)
    except Exception:
        logger.warning("promptkeep: failed to record run", exc_info=True)
        return None


def _attach_handle(response: Any, handle: RunHandle | None) -> None:
    """Attach the RunHandle as response.promptkeep, tolerating frozen objects."""
    try:
        object.__setattr__(response, "promptkeep", handle)
    except Exception:
        try:
            response.promptkeep = handle
        except Exception:
            logger.warning("promptkeep: could not attach run handle to response", exc_info=True)


# --- streaming --------------------------------------------------------------------


class _StreamRecorder:
    """Feeds streamed chunks to the adapter's absorber; finishes the call once
    when the stream ends. Shared by the sync and async proxies."""

    def __init__(self, call: _Call, start: float) -> None:
        """Hold the call; the absorber fills in as chunks arrive."""
        self.call = call
        self.start = start
        self.absorber = call.adapter.stream_absorber()
        self.recorded = False

    def absorb(self, chunk: Any) -> None:
        """Fold one chunk in. An absorber bug loses telemetry, never a chunk."""
        try:
            self.absorber.absorb(chunk)
        except Exception:
            logger.warning(
                "promptkeep: %s adapter failed to absorb a stream chunk",
                self.call.adapter.provider,
                exc_info=True,
            )

    def summary(self) -> ResponseFields:
        """What the absorber gathered, shielded like read_response."""
        try:
            return self.absorber.summary()
        except Exception:
            logger.warning(
                "promptkeep: %s adapter failed to summarize a stream",
                self.call.adapter.provider,
                exc_info=True,
            )
            return ResponseFields()

    def finish(self, status: str = "ok", error: str | None = None) -> None:
        """Record exactly once: a failed run if the stream broke, otherwise the
        completed call (post-checks included) with the accumulated summary
        standing in for the response."""
        if self.recorded:
            return
        self.recorded = True
        fields = self.summary()
        if error is not None:
            self.call.record_error(error, _ms(self.start), fields)
            return
        self.call.finish(fields, fields, _ms(self.start), attach=False)


class _SyncStreamProxy:
    """Wraps a sync stream: passes chunks through, records the run at the end."""

    def __init__(self, stream: Any, recorder: _StreamRecorder) -> None:
        """Hold the real stream and the recorder that finishes it."""
        self._stream = stream
        self._recorder = recorder
        self._iterator: Any = None

    def __iter__(self) -> _SyncStreamProxy:
        """The proxy is its own iterator."""
        return self

    def __next__(self) -> Any:
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

    def __enter__(self) -> _SyncStreamProxy:
        """Support `with client...create(stream=True) as stream:` usage."""
        enter = getattr(self._stream, "__enter__", None)
        if enter is not None:
            enter()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        """Record on context exit (even if the loop broke early), then delegate."""
        self._recorder.finish(
            status="error" if exc_type else "ok",
            error=repr(exc) if exc_type else None,
        )
        exit_ = getattr(self._stream, "__exit__", None)
        if exit_ is not None:
            return bool(exit_(exc_type, exc, tb))
        return False

    def __getattr__(self, name: str) -> Any:
        """Everything else (close(), response, ...) delegates to the real stream."""
        return getattr(self._stream, name)


class _AsyncStreamProxy:
    """Async twin of _SyncStreamProxy for async streams."""

    def __init__(self, stream: Any, recorder: _StreamRecorder) -> None:
        """Hold the real stream and the recorder that finishes it."""
        self._stream = stream
        self._recorder = recorder
        self._iterator: Any = None
        self._finalized = False

    def __aiter__(self) -> _AsyncStreamProxy:
        """The proxy is its own async iterator."""
        return self

    async def _finish(self, status: str = "ok", error: str | None = None) -> None:
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
        if self._finalized:
            return
        self._finalized = True
        await asyncio.shield(asyncio.to_thread(self._recorder.finish, status, error))

    async def __anext__(self) -> Any:
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

    async def __aenter__(self) -> _AsyncStreamProxy:
        """Support `async with ... as stream:` usage."""
        enter = getattr(self._stream, "__aenter__", None)
        if enter is not None:
            await enter()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        """Record on context exit (even if the loop broke early), then delegate."""
        await self._finish(
            status="error" if exc_type else "ok",
            error=repr(exc) if exc_type else None,
        )
        exit_ = getattr(self._stream, "__aexit__", None)
        if exit_ is not None:
            return bool(await exit_(exc_type, exc, tb))
        return False

    def __getattr__(self, name: str) -> Any:
        """Everything else delegates to the real stream."""
        return getattr(self._stream, name)


__all__ = ["instrument", "is_instrumented"]

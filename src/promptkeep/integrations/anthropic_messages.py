"""Anthropic Messages instrumentation, including lazy SDK stream helpers.

Only client-owned methods are replaced. The SDK continues to accumulate helper
messages itself; a recorder underneath its helper observes raw events once.
"""

from __future__ import annotations

import asyncio
import functools
import importlib
import inspect
import json
import threading
import time
from collections.abc import Mapping
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

from ..checks import suppressed
from .base import ProviderAdapter, Request, ResponseFields, StreamAbsorber, Target
from .core import (
    _AsyncStreamProxy,
    _attach_handle,
    _Call,
    _make_async_create,
    _make_sync_create,
    _ms,
    _read_response,
    _StreamRecorder,
    _strip_promptkeep_kwargs,
    _SyncStreamProxy,
)
from .openai_wrapper import _resolve_text


def _content(value: Any, tracked: list) -> Any:
    """Resolve text provenance recursively without touching opaque blocks."""
    resolved = _resolve_text(value)
    if resolved is not None:
        text, found = resolved
        tracked.extend(found)
        return text
    if isinstance(value, str) or value is None:
        return value
    if not isinstance(value, (list, tuple)):
        value = list(value)
    blocks = []
    for block in value:
        if isinstance(block, Mapping):
            block = dict(block)
            if block.get("type") == "text":
                block["text"] = _content(block.get("text"), tracked)
            elif block.get("type") == "tool_result":
                block["content"] = _content(block.get("content"), tracked)
        blocks.append(block)
    return blocks


def _text(value: Any) -> str:
    """Visible request text, including tool-result text but never binary data."""
    if isinstance(value, str):
        return value
    if not isinstance(value, (list, tuple)):
        return ""
    parts = []
    for block in value:
        if isinstance(block, Mapping):
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append(block["text"])
            elif block.get("type") == "tool_result":
                parts.append(_text(block.get("content")))
    return "\n".join(parts)


def _usage(value: Any) -> dict[str, Any]:
    """Preserve numeric provider fields and nested cache breakdowns."""
    if value is None:
        return {}
    data = value if isinstance(value, dict) else vars(value)
    return {
        k: (_usage(v) if isinstance(v, (dict, SimpleNamespace)) or hasattr(v, "model_dump") else v)
        for k, v in data.items()
        if v is not None
        and not isinstance(v, bool)
        and (isinstance(v, (int, float, dict, SimpleNamespace)) or hasattr(v, "model_dump"))
    }


def _fields(
    model: str | None, response_id: str | None, text: str | None, usage: dict, ids: tuple[str, ...]
) -> ResponseFields:
    """Normalize cached input once, retaining the original usage separately."""
    incoming = usage.get("input_tokens")
    if incoming is not None:
        incoming += usage.get("cache_read_input_tokens") or 0
        incoming += usage.get("cache_creation_input_tokens") or 0
    outgoing = usage.get("output_tokens")
    return ResponseFields(
        model=model,
        response_id=response_id,
        output_text=text,
        prompt_tokens=incoming,
        completion_tokens=outgoing,
        total_tokens=incoming + outgoing if incoming is not None and outgoing is not None else None,
        usage_details=usage or None,
        tool_call_ids=ids,
    )


class AnthropicMessagesAdapter(ProviderAdapter):
    """The stable Messages API on Anthropic and AsyncAnthropic clients."""

    provider = "anthropic"

    def locate(self, client: Any) -> Target | None:
        """Recognize the Messages resource without importing the optional SDK."""
        messages = getattr(client, "messages", None)
        if messages is not None and callable(getattr(messages, "create", None)):
            return Target(messages, "create")
        return None

    def locate_targets(self, client: Any) -> list[Target]:
        """The helper bypasses create, so instrument both independently."""
        target = self.locate(client)
        if target is None:
            return []
        result = [target]
        if callable(getattr(target.owner, "stream", None)):
            result.append(Target(target.owner, "stream", "manager"))
        if callable(getattr(target.owner, "parse", None)):
            result.append(Target(target.owner, "parse"))
        for resource, style in (
            ("with_raw_response", "raw_call"),
            ("with_streaming_response", "raw_manager"),
        ):
            owner = getattr(target.owner, resource, None)
            if owner is not None:
                result.append(Target(owner, "create", style))
        return result

    def is_deferred_response(self, response: Any) -> bool:
        """Raw API response objects expose parse and http_response."""
        return callable(getattr(response, "parse", None)) and hasattr(response, "http_response")

    def defer_response(self, response: Any, call: Any, start: float, asynchronous: bool) -> Any:
        """Patch instance-owned parse/close methods, preserving headers and native type."""
        return _DeferredResponse(response, call, start, asynchronous).install()

    def wrap_raw_call(self, original: Any) -> Any:
        """Expose the SDK's raw-response mode before executing pre-checks."""
        instrumented = (
            _make_async_create
            if inspect.iscoroutinefunction(inspect.unwrap(original))
            else _make_sync_create
        )(original, self)

        @functools.wraps(original)
        def create(*args: Any, **kwargs: Any) -> Any:
            """The same SDK header is forwarded, without a second model call."""
            kwargs["extra_headers"] = {
                **(kwargs.get("extra_headers") or {}),
                "X-Stainless-Raw-Response": "raw",
            }
            return instrumented(*args, **kwargs)

        return create

    def wrap_raw_manager(self, original: Any) -> Any:
        """Raw HTTP response helpers remain lazy until context entry."""

        @functools.wraps(original)
        def create(*args: Any, **kwargs: Any) -> Any:
            """Suppressed check calls delegate directly to the SDK."""
            if suppressed():
                _strip_promptkeep_kwargs(kwargs)
                return original(*args, **kwargs)
            return _RawManager(self, original, args, kwargs)

        return create

    def parse_request(self, kwargs: dict[str, Any]) -> Request:
        """Resolve system and message text while preserving all non-text blocks."""
        tracked: list = []
        outgoing = dict(kwargs)
        if "system" in kwargs:
            outgoing["system"] = _content(kwargs["system"], tracked)
        messages = []
        for message in kwargs.get("messages", ()):
            message = dict(message)
            message["content"] = _content(message.get("content"), tracked)
            messages.append(message)
        outgoing["messages"] = messages
        payload = {"system": outgoing.get("system"), "messages": messages}
        latest = next((m for m in reversed(messages) if m.get("role") == "user"), {})
        return Request(
            kwargs=outgoing,
            tracked=tracked,
            payload=payload,
            input_text=_text(latest.get("content")) or None,
            joined_text="\n".join(
                [_text(outgoing.get("system"))] + [_text(m.get("content")) for m in messages]
            ),
            request_params={k: v for k, v in outgoing.items() if k not in ("messages", "system")},
        )

    def apply_rewrite(self, request: Request, text: str) -> Request:
        """Replace eligible user text, retaining tool results and opaque blocks."""
        messages = list(request.kwargs["messages"])
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                content = text
            elif isinstance(content, list):
                blocks, done = [], False
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        if not done:
                            blocks.append({**block, "text": text})
                            done = True
                    else:
                        blocks.append(block)
                if not done:
                    return request
                content = blocks
            else:
                return request
            messages[index] = {**message, "content": content}
            parsed = self.parse_request({**request.kwargs, "messages": messages})
            return replace(parsed, tracked=request.tracked, asynchronous=request.asynchronous)
        return request

    def read_response(self, response: Any) -> ResponseFields:
        """Read visible text, tool-use IDs and reported usage."""
        if isinstance(response, str):
            try:
                response = json.loads(response)
            except ValueError:
                return ResponseFields()

        def value(obj, name, default=None):
            """Raw parse(to=dict) and native messages expose the same content."""
            return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)

        content = value(response, "content", ()) or ()
        text = "".join(
            value(block, "text", "") for block in content if value(block, "type") == "text"
        )
        ids = tuple(value(block, "id") for block in content if value(block, "type") == "tool_use")
        return _fields(
            value(response, "model"),
            value(response, "id"),
            text or None,
            _usage(value(response, "usage")),
            ids,
        )

    def stream_absorber(self) -> StreamAbsorber:
        """One accumulator per physical Messages stream."""
        return _MessagesAbsorber()

    def blocked_stub(self, request: Request, blocked: Any) -> Any:
        """Construct a native empty message only when the SDK is actually used."""
        types = importlib.import_module("anthropic.types")
        response = types.Message.model_validate(
            {
                "id": "promptkeep_blocked",
                "type": "message",
                "role": "assistant",
                "model": request.kwargs.get("model", "unknown"),
                "content": [],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }
        )
        object.__setattr__(response, "promptkeep_blocked", blocked)
        if (request.kwargs.get("extra_headers") or {}).get("X-Stainless-Raw-Response") in (
            "raw",
            "stream",
        ):
            return (_AsyncBlockedRaw if request.asynchronous else _BlockedRaw)(response)
        if request.kwargs.get("stream"):
            return _EmptyStream(response)
        return response

    def wrap_manager(self, original: Any) -> Any:
        """Return a lazy manager: checks and HTTP begin on context entry."""

        @functools.wraps(original)
        def stream(*args: Any, **kwargs: Any) -> Any:
            """Do not create an awaitable or start telemetry before entry."""
            if suppressed():
                _strip_promptkeep_kwargs(kwargs)
                return original(*args, **kwargs)
            return _MessagesManager(self, original, args, kwargs)

        return stream


class _MessagesAbsorber(StreamAbsorber):
    """Fold SSE events; token deltas contain cumulative snapshots."""

    def __init__(self) -> None:
        """Initialize only observed fields; absent usage remains unknown."""
        self.model = None
        self.response_id = None
        self.parts: list[str] = []
        self.usage: dict = {}
        self.ids: list[str] = []

    def absorb(self, chunk: Any) -> None:
        """Observe starts, visible text deltas, tool IDs and usage updates."""
        kind = getattr(chunk, "type", None)
        if kind == "message_start":
            message = chunk.message
            self.model, self.response_id = message.model, message.id
            self.usage.update(_usage(getattr(message, "usage", None)))
        elif kind == "content_block_start":
            block = chunk.content_block
            if block.type == "tool_use":
                self.ids.append(block.id)
            elif block.type == "text" and block.text:
                self.parts.append(block.text)
        elif kind == "content_block_delta" and getattr(chunk.delta, "type", None) == "text_delta":
            self.parts.append(chunk.delta.text)
        elif kind == "message_delta":
            self.usage.update(_usage(getattr(chunk, "usage", None)))

    def summary(self) -> ResponseFields:
        """Return observed provider data without estimating missing fields."""
        return _fields(
            self.model, self.response_id, "".join(self.parts) or None, self.usage, tuple(self.ids)
        )


class _EmptyStream:
    """An empty, native-event stream for a blocked Messages request."""

    def __init__(self, message: Any) -> None:
        """Create the events needed by the SDK's own message accumulator."""
        types = importlib.import_module("anthropic.types")
        self.events = [
            types.RawMessageStartEvent(type="message_start", message=message),
            types.RawMessageDeltaEvent.model_validate(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 0},
                }
            ),
            types.RawMessageStopEvent(type="message_stop"),
        ]
        self.promptkeep = getattr(message, "promptkeep", None)
        self.response = SimpleNamespace(headers={})

    def __iter__(self):
        """Yield native events without a network request."""
        return iter(self.events)

    def __aiter__(self):
        """Yield the same blocked events asynchronously."""
        return self._iterate()

    async def _iterate(self):
        """The asynchronous event generator."""
        for event in self.events:
            yield event

    def close(self) -> Any:
        """An immediate close that can also be awaited by an async client."""
        return _CompletedClose()

    def __enter__(self) -> Any:
        """Support ordinary blocked synchronous stream contexts."""
        return self

    def __exit__(self, *args: Any) -> bool:
        """A blocked stream owns no resource and suppresses no exception."""
        return False

    async def __aenter__(self) -> Any:
        """Support ordinary blocked asynchronous stream contexts."""
        return self

    async def __aexit__(self, *args: Any) -> bool:
        """A blocked stream owns no resource and suppresses no exception."""
        return False


class _CompletedClose:
    """Awaitable no-op without allocating an unawaited coroutine for sync users."""

    def __await__(self):
        """Complete immediately in either stream protocol."""
        if False:
            yield
        return None


class _AsyncEmptyStream(_EmptyStream):
    """Blocked stream with the asynchronous SDK close protocol."""

    async def close(self) -> None:
        """No network resource exists to close."""


def _attach_helpers(stream: Any, handle: Any) -> None:
    """Expose the same handle on native helper streams and final messages."""
    original = stream.get_final_message
    if inspect.iscoroutinefunction(original):

        @functools.wraps(original)
        async def async_final(*args: Any, **kwargs: Any) -> Any:
            """Return the SDK's original final message with its tracking handle."""
            message = await original(*args, **kwargs)
            _attach_handle(message, handle, expected=False)
            return message

        stream.get_final_message = async_final
    else:

        @functools.wraps(original)
        def sync_final(*args: Any, **kwargs: Any) -> Any:
            """Return the SDK's original final message with its tracking handle."""
            message = original(*args, **kwargs)
            _attach_handle(message, handle, expected=False)
            return message

        stream.get_final_message = sync_final
    _attach_handle(stream, handle)


class _MessagesManager:
    """Keep the native helper and observe the raw stream it consumes."""

    def __init__(self, adapter: Any, original: Any, args: tuple, kwargs: dict) -> None:
        """Defer all work until the user enters the context."""
        self.adapter, self.original, self.args, self.kwargs = adapter, original, args, kwargs
        self.manager = None
        self.proxy = None
        self.entered = False

    def _prepare(self, asynchronous=False) -> tuple[_Call, Any]:
        """Create one call and execute its gate, without provider I/O."""
        if self.entered:
            raise RuntimeError("a stream manager cannot be entered twice")
        self.entered = True
        call = _Call(self.adapter, {**self.kwargs, "stream": True})
        call.request = replace(call.request, asynchronous=asynchronous)
        if call.checked:
            call.run_pre()
            blocked = call.blocked_response()
            if blocked is not None:
                return call, blocked
        return call, None

    def __enter__(self) -> Any:
        """Enter the real helper and place observation beneath its accumulator."""
        call, blocked = self._prepare()
        if blocked is not None:
            sdk = importlib.import_module("anthropic.lib.streaming")
            constants = importlib.import_module("anthropic")
            self.stream = _native_helper(sdk.MessageStream, blocked, constants.NOT_GIVEN)
            _attach_helpers(self.stream, call.handle)
            return self.stream
        call.begin_model_call()
        start = time.perf_counter()
        try:
            kwargs = {k: v for k, v in call.request.kwargs.items() if k != "stream"}
            self.manager = self.original(*self.args, **kwargs)
            self.stream = self.manager.__enter__()
        except Exception as exc:
            call.record_error(repr(exc), _ms(start))
            raise
        self.proxy = _SyncStreamProxy(self.stream._raw_stream, _StreamRecorder(call, start))
        self.stream._raw_stream = self.proxy
        _attach_helpers(self.stream, call.handle)
        return self.stream

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        """Finalize before native cleanup; cleanup always runs."""
        result = False
        try:
            if self.proxy is not None:
                self.proxy._recorder.finish(
                    "error" if exc_type else "ok", repr(exc) if exc_type else None
                )
        finally:
            if self.manager is not None:
                result = bool(self.manager.__exit__(exc_type, exc, tb))
        return result

    async def __aenter__(self) -> Any:
        """Run checks off-loop, then await the native helper's request."""
        call, blocked = await asyncio.to_thread(self._prepare, True)
        if blocked is not None:
            sdk = importlib.import_module("anthropic.lib.streaming")
            constants = importlib.import_module("anthropic")
            raw = _AsyncEmptyStream(blocked.events[0].message)
            self.stream = _native_helper(sdk.AsyncMessageStream, raw, constants.NOT_GIVEN)
            _attach_helpers(self.stream, call.handle)
            return self.stream
        call.begin_model_call()
        start = time.perf_counter()
        try:
            kwargs = {k: v for k, v in call.request.kwargs.items() if k != "stream"}
            self.manager = self.original(*self.args, **kwargs)
            self.stream = await self.manager.__aenter__()
        except asyncio.CancelledError:
            await asyncio.shield(
                asyncio.to_thread(call._write, ResponseFields(), _ms(start), "cancelled", None)
            )
            raise
        except Exception as exc:
            await asyncio.to_thread(call.record_error, repr(exc), _ms(start))
            raise
        self.proxy = _AsyncStreamProxy(self.stream._raw_stream, _StreamRecorder(call, start))
        self.stream._raw_stream = self.proxy
        _attach_helpers(self.stream, call.handle)
        return self.stream

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        """Finalize cancellation and failures before native asynchronous cleanup."""
        result = False
        try:
            if self.proxy is not None:
                status = (
                    "cancelled"
                    if exc_type and issubclass(exc_type, asyncio.CancelledError)
                    else "error"
                    if exc_type
                    else "ok"
                )
                await self.proxy._finish(status, repr(exc) if exc_type else None)
        finally:
            if self.manager is not None:
                result = bool(await self.manager.__aexit__(exc_type, exc, tb))
        return result


def _native_helper(factory, raw, output_format):
    """Older stable SDK helpers do not accept output_format."""
    if "output_format" in inspect.signature(factory).parameters:
        return factory(raw, output_format=output_format)
    return factory(raw)


class _DeferredResponse:
    """Observe native raw bodies exactly once, only after parsing or explicit close."""

    def __init__(self, response, call, start, asynchronous):
        """Capture original methods without reading the HTTP body."""
        self.response, self.call, self.start = response, call, start
        self.asynchronous = asynchronous
        self.parse = response.parse
        self.close = response.close
        self._lock = threading.Lock()
        self._recorded = False
        self._stream = None

    def _stream_value(self, value):
        """Cached SDK parsing returns the same observed stream and finalization claim."""
        with self._lock:
            if self._stream is None:
                self._recorded = True
                cls = _AsyncStreamProxy if self.asynchronous else _SyncStreamProxy
                self._stream = cls(value, _StreamRecorder(self.call, self.start))
            return self._stream

    def _claim(self):
        """Competing parse/close paths share one finalization claim."""
        with self._lock:
            if self._recorded:
                return False
            self._recorded = True
            return True

    def _finish(self, parsed=None, error=None, status="incomplete"):
        """Successful parses audit normally; unread closes retain unknown output."""
        if not self._claim():
            return
        if parsed is not None:
            self.call.finish(parsed, _read_response(self.call.adapter, parsed), _ms(self.start))
        else:
            key = self.call._write(
                ResponseFields(), _ms(self.start), status, str(error) if error else None
            )
            if self.call.handle:
                self.call.handle.run_key = key

    def install(self):
        """Return the same native object with observed instance methods."""
        if self.asynchronous:

            async def parse(*args, **kwargs):
                """Await the SDK parser and preserve its typed parsed object."""
                try:
                    value = await self.parse(*args, **kwargs)
                except BaseException as exc:
                    await asyncio.shield(
                        asyncio.to_thread(
                            self._finish,
                            error=exc,
                            status="cancelled"
                            if isinstance(exc, asyncio.CancelledError)
                            else "error",
                        )
                    )
                    raise
                if self.call.streaming:
                    return self._stream_value(value)
                await asyncio.to_thread(self._finish, value)
                return value

            async def close():
                """Always close the native transport before recording missing output."""
                try:
                    return await self.close()
                finally:
                    await asyncio.shield(asyncio.to_thread(self._finish))
        else:

            def parse(*args, **kwargs):
                """Parse without changing native body/headers or consuming extra data."""
                try:
                    value = self.parse(*args, **kwargs)
                except BaseException as exc:
                    self._finish(error=exc, status="error")
                    raise
                if self.call.streaming:
                    return self._stream_value(value)
                self._finish(value)
                return value

            def close():
                """An unread raw response is incomplete, not an empty successful reply."""
                try:
                    return self.close()
                finally:
                    self._finish()

        self.response.parse = parse
        self.response.close = close
        _attach_handle(self.response, self.call.handle, expected=False)
        return self.response


class _RawManager(_MessagesManager):
    """The native raw-response manager opens HTTP without parsing the body."""

    def _prepare(self, asynchronous=False):
        """Raw-response contexts are not provider event streams by default."""
        if self.entered:
            raise RuntimeError("a raw response manager cannot be entered twice")
        self.entered = True
        kwargs = {
            **self.kwargs,
            "extra_headers": {
                **(self.kwargs.get("extra_headers") or {}),
                "X-Stainless-Raw-Response": "stream",
            },
        }
        call = _Call(self.adapter, kwargs)
        call.request = replace(call.request, asynchronous=asynchronous)
        if call.checked:
            call.run_pre()
            blocked = call.blocked_response()
            if blocked is not None:
                return call, blocked
        return call, None

    def __enter__(self):
        """Open the native context and observe parse/close lazily."""
        call, blocked = self._prepare()
        if blocked is not None:
            return blocked
        call.begin_model_call()
        start = time.perf_counter()
        try:
            self.manager = self.original(*self.args, **call.request.kwargs)
            raw = self.manager.__enter__()
        except Exception as exc:
            call.record_error(repr(exc), _ms(start))
            raise
        return self.adapter.defer_response(raw, call, start, False)

    async def __aenter__(self):
        """Check off-loop, then open the SDK's asynchronous raw context."""
        call, blocked = await asyncio.to_thread(self._prepare, True)
        if blocked is not None:
            return blocked
        call.begin_model_call()
        start = time.perf_counter()
        try:
            self.manager = self.original(*self.args, **call.request.kwargs)
            raw = await self.manager.__aenter__()
        except BaseException as exc:
            await asyncio.shield(
                asyncio.to_thread(
                    call._write,
                    ResponseFields(),
                    _ms(start),
                    "cancelled" if isinstance(exc, asyncio.CancelledError) else "error",
                    str(exc),
                )
            )
            raise
        return self.adapter.defer_response(raw, call, start, True)


class _BlockedRaw:
    """Response-shaped empty result for a pre-check that prevented HTTP entirely."""

    def __init__(self, message):
        """No native HTTP response or provider request was created."""
        self.message = message
        self.headers = {}
        self.status_code = 200
        self.http_response = SimpleNamespace(headers=self.headers, status_code=200)

    def parse(self, **kwargs):
        """Return the native empty Message or an explicitly requested dictionary."""
        return self.message.model_dump() if kwargs.get("to") is dict else self.message

    def close(self):
        """There is no HTTP resource to close."""
        return _CompletedClose()

    def json(self):
        """Return the blocked message's native JSON representation."""
        return self.message.model_dump()


class _AsyncBlockedRaw(_BlockedRaw):
    """Asynchronous raw blocked response with the same empty native message."""

    async def parse(self, **kwargs):
        """Preserve asynchronous raw-parser calling conventions."""
        return super().parse(**kwargs)

    async def json(self):
        """No response body needs reading."""
        return super().json()

"""The OpenAI Responses API adapter (``client.responses.create``).

The second surface on the same client, with its own shapes: Prompts sit in
``instructions`` (a string) and in ``input`` (a string, or a list of items
whose ``content`` is a string or a list of ``input_text`` blocks); usage is
counted as ``input_tokens`` / ``output_tokens``; the reply is spread over
``output`` items; and a stream is a sequence of typed events rather than
chunks of one shape.

It is also the one surface that names its own predecessor:
``previous_response_id`` says which response this call continues, and
``conversation_hint`` hands that to ``core`` so chained calls group into one
conversation with no user code.

Input items are dicts shaped like chat messages, so the Prompt substitution,
current-turn and rewrite helpers are the chat adapter's. Only ``create`` is
instrumented — ``responses.stream()`` and ``responses.parse()`` are separate
SDK methods and pass through untracked.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from .base import ProviderAdapter, Request, ResponseFields, StreamAbsorber, Target, TrackedPrompt
from .openai_wrapper import (
    _apply_rewrite,
    _extract_input_text,
    _joined_text,
    _process_messages,
    _reported_cost,
    _resolve_text,
)

# The kwargs that carry text. Everything else is a request parameter.
_PAYLOAD_KWARGS = ("input", "instructions")

# The events that end a stream; each carries the final Response object.
_TERMINAL_EVENTS = ("response.completed", "response.incomplete", "response.failed")


class OpenAIResponsesAdapter(ProviderAdapter):
    """``client.responses.create`` — sync and async clients alike."""

    provider = "openai-responses"

    def locate(self, client: Any) -> Target | None:
        """``client.responses.create`` is the surface; anything without it is
        not this provider."""
        responses = getattr(client, "responses", None)
        if responses is not None and callable(getattr(responses, "create", None)):
            return Target(responses, "create")
        return None

    def accepts(self, kwargs: dict[str, Any]) -> bool:
        """A Responses call is spelled with ``input`` / ``instructions``."""
        return any(key in kwargs for key in _PAYLOAD_KWARGS)

    def parse_request(self, kwargs: dict[str, Any]) -> Request:
        """Substitute the Prompts in ``instructions`` and ``input``; the
        newest non-system input item (or a string input) is the current turn."""
        tracked: list[TrackedPrompt] = []
        new_kwargs = dict(kwargs)

        # instructions: one string, the system prompt of this surface.
        resolved = _resolve_text(kwargs.get("instructions"))
        if resolved is not None:
            new_kwargs["instructions"], found = resolved
            tracked.extend(found)

        # input: a bare string (itself possibly a Prompt), or a list of items.
        value = kwargs.get("input")
        resolved = _resolve_text(value)
        if resolved is not None:
            new_kwargs["input"], found = resolved
            tracked.extend(found)
        elif isinstance(value, (list, tuple)):
            found, new_kwargs["input"] = _process_messages(value)
            tracked.extend(found)
        return _request(new_kwargs, tracked)

    def apply_rewrite(self, request: Request, text: str) -> Request:
        """Put the rewrite where the current turn is: the string input, or
        the newest non-system item of a list. No current turn, no rewrite."""
        if request.input_text is None:
            return request
        value = request.kwargs.get("input")
        rewritten = text if isinstance(value, str) else _apply_rewrite(value, text)
        return _request({**request.kwargs, "input": rewritten}, request.tracked)

    def conversation_hint(self, request: Request) -> str | None:
        """``previous_response_id``: the response this call continues."""
        previous = request.kwargs.get("previous_response_id")
        return previous if isinstance(previous, str) and previous else None

    def read_response(self, response: Any) -> ResponseFields:
        """Reply text from the ``output`` message items, counts from ``usage``
        (``input_tokens`` / ``output_tokens`` on this surface)."""
        usage = getattr(response, "usage", None)
        return ResponseFields(
            model=getattr(response, "model", None),
            response_id=getattr(response, "id", None),
            output_text=_extract_output_text(response),
            prompt_tokens=getattr(usage, "input_tokens", None),
            completion_tokens=getattr(usage, "output_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
            cost_usd=_reported_cost(usage),
        )

    def stream_absorber(self) -> StreamAbsorber:
        """A fresh event absorber for one streamed response."""
        return _ResponsesStreamAbsorber(self)

    def blocked_stub(self, request: Request, blocked: Any) -> Any:
        """A Response-shaped object with no output and the block noted on it,
        so ``response.output_text`` / ``response.output`` code degrades
        instead of crashing."""
        return SimpleNamespace(
            id=None,
            model=request.kwargs.get("model"),
            usage=None,
            output=[],
            output_text="",
            promptkeep_blocked=SimpleNamespace(check=blocked.name, message=blocked.message),
        )


class _ResponsesStreamAbsorber(StreamAbsorber):
    """Folds a Responses event stream: text from the ``output_text.delta``
    events, ids and usage from whichever events carry the Response object
    (``created`` first, a terminal event last — the only one with usage)."""

    def __init__(self, adapter: OpenAIResponsesAdapter) -> None:
        """Hold the adapter (the final Response is read the non-streaming way)."""
        self.adapter = adapter
        self.parts: list[str] = []
        self.final = ResponseFields()

    def absorb(self, event: Any) -> None:
        """Fold one event in: a text delta, or the Response riding on it."""
        if getattr(event, "type", None) == "response.output_text.delta":
            delta = getattr(event, "delta", None)
            if isinstance(delta, str) and delta:
                self.parts.append(delta)
        response = getattr(event, "response", None)
        if response is not None:
            self.final = self.adapter.read_response(response)

    def summary(self) -> ResponseFields:
        """The last Response seen, with the streamed deltas as its text — they
        are what the caller actually received, and all there is when a stream
        breaks before its terminal event."""
        text = "".join(self.parts) or self.final.output_text
        return ResponseFields(
            model=self.final.model,
            response_id=self.final.response_id,
            output_text=text,
            prompt_tokens=self.final.prompt_tokens,
            completion_tokens=self.final.completion_tokens,
            total_tokens=self.final.total_tokens,
            cost_usd=self.final.cost_usd,
        )


# --- request and response shapes ------------------------------------------------


def _request(kwargs: dict[str, Any], tracked: list[TrackedPrompt]) -> Request:
    """The Request for already-substituted kwargs — shared by parse_request and
    apply_rewrite, so a rewrite re-derives every text field the same way.

    ``payload`` (what a check reads as ``ctx.messages``) is the call laid out
    as a chat-style message list — instructions as the system message, a
    string input as one user message — so a check written against one OpenAI
    surface reads the other.
    """
    instructions = kwargs.get("instructions")
    value = kwargs.get("input")
    items = [{"role": "user", "content": value}] if isinstance(value, str) else value
    system = [{"role": "system", "content": instructions}] if isinstance(instructions, str) else []
    payload = system + list(items) if isinstance(items, (list, tuple)) else system
    return Request(
        kwargs=kwargs,
        tracked=tracked,
        payload=payload,
        input_text=_extract_input_text(items),
        joined_text=_joined_text(payload),
        request_params={k: v for k, v in kwargs.items() if k not in _PAYLOAD_KWARGS},
    )


def _extract_output_text(response: Any) -> str | None:
    """The reply's text: every ``output_text`` block of every ``message`` item
    in ``output``, joined — what the SDK's ``response.output_text`` computes,
    which is the fallback when there is no ``output`` list to walk. None for
    an empty reply (a pure tool call, an error)."""
    output = getattr(response, "output", None)
    if not isinstance(output, (list, tuple)):
        text = getattr(response, "output_text", None)
        return text if isinstance(text, str) and text else None
    parts = []
    for item in output:
        if getattr(item, "type", None) != "message":
            continue
        for block in getattr(item, "content", None) or ():
            text = getattr(block, "text", None)
            if getattr(block, "type", None) == "output_text" and isinstance(text, str):
                parts.append(text)
    return "".join(parts) or None

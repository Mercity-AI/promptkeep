"""The provider adapter interface: what an integration has to know about one
SDK so the shared orchestration in ``core`` can track its calls.

Everything provider-agnostic — conversations, checks, run recording, the
stream proxies — lives once, in ``core.py``. An adapter answers a small set
of questions about one SDK's shapes: where the call method lives on a client,
where Prompt objects hide in a request, which text is the current turn, how to
read a response, how to fold a stream. It never sees the database or a check.

Adapters are registered in ``integrations/__init__.py``; ``wrap()`` asks each
one to ``locate`` the client it was given and instruments every surface that
matches, so one client can carry several tracked surfaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ..prompts import Prompt

# (prompt, variables used, rendered text) — everything a run row needs from
# the request side.
TrackedPrompt = tuple[Prompt, dict[str, Any], str]


@dataclass(frozen=True)
class Target:
    """Where a provider's call method lives: ``getattr(owner, attribute)``.
    The adapter's replacement is installed with ``setattr`` on the same owner."""

    owner: Any
    attribute: str


@dataclass(frozen=True)
class Request:
    """One outgoing call, as an adapter parsed it.

    kwargs: the call's keyword arguments with every Prompt / RenderedText
      replaced by a plain string — exactly what the provider will receive.
    tracked: one entry per Prompt found, in order of appearance.
    payload: the message payload as the provider sees it (for OpenAI chat,
      the messages list) — what a check receives as ``ctx.messages``.
    input_text: the current turn's text (the newest non-system message), or
      None when there is no such message. Stored as the turn's ``input_text``
      and read by checks as ``ctx.last_text``.
    joined_text: all outgoing text as one string — what a pre-check scans.
    request_params: kwargs minus the message payload, recorded on the run.
    """

    kwargs: dict[str, Any]
    tracked: list[TrackedPrompt]
    payload: Any
    input_text: str | None
    joined_text: str
    request_params: dict[str, Any]


@dataclass(frozen=True)
class ResponseFields:
    """What a run row records from a response. Every field is optional: an
    error has no response, a stream may never report usage."""

    model: str | None = None
    response_id: str | None = None
    output_text: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class StreamAbsorber(ABC):
    """Folds a stream's chunks into one ResponseFields. One instance per stream;
    ``core`` feeds it every chunk and asks for the summary when the stream ends."""

    @abstractmethod
    def absorb(self, chunk: Any) -> None:
        """Fold one chunk in: ids, usage, and any text delta."""

    @abstractmethod
    def summary(self) -> ResponseFields:
        """Everything gathered so far, as a response would have reported it."""


class ProviderAdapter(ABC):
    """One provider SDK's shapes. Subclass, set ``provider``, implement the
    questions below; register the instance in ``integrations/__init__.py``.

    Adapters must be defensive by construction — treat every attribute on a
    request or response as possibly missing — because ``core`` calls
    ``read_response`` and the absorber inside the user's request path. It
    shields them anyway (a raising adapter loses telemetry, never a call), but
    a warning per call is not a good look.
    """

    provider: str = ""

    @abstractmethod
    def locate(self, client: Any) -> Target | None:
        """The call method to instrument on this client, or None if the client
        does not expose this provider's surface."""

    @abstractmethod
    def parse_request(self, kwargs: dict[str, Any]) -> Request:
        """Find the Prompts in a call's kwargs, substitute their rendered text,
        and pick out the current turn. Must not mutate ``kwargs`` or anything
        inside it. Called after promptkeep's own kwargs have been stripped."""

    @abstractmethod
    def apply_rewrite(self, request: Request, text: str) -> Request:
        """A pre-check replaced the current turn with ``text``: return a new
        Request whose kwargs/payload carry it and whose input_text/joined_text
        reflect it. When the request has no current turn (a trailing system
        message), return it unchanged."""

    @abstractmethod
    def read_response(self, response: Any) -> ResponseFields:
        """The run-row fields of a (non-streaming) response."""

    @abstractmethod
    def stream_absorber(self) -> StreamAbsorber:
        """A fresh absorber for one streamed call."""

    @abstractmethod
    def blocked_stub(self, request: Request, blocked: Any) -> Any:
        """A response-shaped object for ``on_block="return"``: enough of the
        provider's response shape that existing code degrades instead of
        crashing. ``blocked`` is the CheckResult that gated the call."""

    def is_streaming(self, request: Request) -> bool:
        """Whether this call returns a stream. ``stream=True`` is the spelling
        every supported SDK uses; override if a provider differs."""
        return bool(request.kwargs.get("stream"))

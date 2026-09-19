"""The OpenAI ``chat.completions`` adapter.

All the shape knowledge for one provider surface: where ``create`` lives,
that Prompts sit in ``messages[*].content`` (a string or a list of text
blocks), that the newest non-system message is the current turn, and how a
response and its streamed chunks report text, ids, usage and — on endpoints
that say so, like OpenRouter — what the call cost. Nothing here records
anything — ``core.py`` does the tracking through the adapter
interface in ``base.py``.

We never monkey-patch the ``openai`` module: only the client object the user
passed to ``wrap()`` gets its ``chat.completions.create`` replaced. Message
dicts are copied, never mutated, and the API receives plain strings —
byte-for-byte what an unwrapped client would send with ``prompt.text``.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

from ..prompts import Prompt, RenderedText
from .base import ProviderAdapter, Request, ResponseFields, StreamAbsorber, Target, TrackedPrompt

# The system-level roles: never the current turn, never stored as input_text.
# "developer" is the newer-model spelling of the same thing.
_SYSTEM_ROLES = ("system", "developer")


class OpenAIChatAdapter(ProviderAdapter):
    """``client.chat.completions.create`` — sync and async clients alike."""

    provider = "openai"

    def locate(self, client: Any) -> Target | None:
        """``client.chat.completions.create`` is the surface; anything without
        it is not this provider."""
        completions = getattr(getattr(client, "chat", None), "completions", None)
        if completions is not None and callable(getattr(completions, "create", None)):
            return Target(completions, "create")
        return None

    def parse_request(self, kwargs: dict[str, Any]) -> Request:
        """Substitute the Prompts in ``messages``; the newest non-system
        message is the current turn."""
        tracked, messages = _process_messages(kwargs.get("messages"))
        new_kwargs = dict(kwargs)
        if "messages" in kwargs:
            new_kwargs["messages"] = messages
        return Request(
            kwargs=new_kwargs,
            tracked=tracked,
            payload=messages,
            input_text=_extract_input_text(messages),
            joined_text=_joined_text(messages),
            request_params={k: v for k, v in new_kwargs.items() if k != "messages"},
        )

    def apply_rewrite(self, request: Request, text: str) -> Request:
        """Put the rewrite on the newest non-system message and re-derive the
        current-turn and joined texts from the result."""
        messages = _apply_rewrite(request.payload, text)
        return replace(
            request,
            kwargs={**request.kwargs, "messages": messages},
            payload=messages,
            input_text=_extract_input_text(messages),
            joined_text=_joined_text(messages),
        )

    def read_response(self, response: Any) -> ResponseFields:
        """Reply text from ``choices[0].message.content``, counts from ``usage``."""
        usage = getattr(response, "usage", None)
        return ResponseFields(
            model=getattr(response, "model", None),
            response_id=getattr(response, "id", None),
            output_text=_extract_output_text(response),
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
            cost_usd=_reported_cost(usage),
        )

    def stream_absorber(self) -> StreamAbsorber:
        """A fresh chunk absorber for one streamed completion."""
        return _ChatStreamAbsorber()

    def blocked_stub(self, request: Request, blocked: Any) -> Any:
        """A chat-completion-shaped object with no choices and the block noted
        on it, so ``response.choices`` code degrades instead of crashing."""
        return SimpleNamespace(
            id=None,
            model=request.kwargs.get("model"),
            usage=None,
            choices=[],
            promptkeep_blocked=SimpleNamespace(check=blocked.name, message=blocked.message),
        )


class _ChatStreamAbsorber(StreamAbsorber):
    """Accumulates chat-completion chunks: ids and usage as they appear, and
    every ``choices[0].delta.content`` piece as the reply text."""

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.model: str | None = None
        self.response_id: str | None = None
        self.usage: Any = None

    def absorb(self, chunk: Any) -> None:
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

    def summary(self) -> ResponseFields:
        return ResponseFields(
            model=self.model,
            response_id=self.response_id,
            output_text="".join(self.parts) or None,
            prompt_tokens=getattr(self.usage, "prompt_tokens", None),
            completion_tokens=getattr(self.usage, "completion_tokens", None),
            total_tokens=getattr(self.usage, "total_tokens", None),
            cost_usd=_reported_cost(self.usage),
        )


# --- usage ---------------------------------------------------------------------


def _reported_cost(usage: Any) -> float | None:
    """The cost the endpoint itself reported for a call, in US dollars.

    OpenAI's own API reports none. OpenRouter — the same chat shape — puts
    ``cost`` on every usage block (in credits, which are dollars), including
    a stream's final chunk; the OpenAI SDK keeps fields it doesn't model, so
    it reads like any other attribute. Anything that isn't a plain number is
    treated as "not reported" rather than guessed at.
    """
    cost = getattr(usage, "cost", None)
    if isinstance(cost, bool) or not isinstance(cost, (int, float)):
        return None
    return float(cost)


# --- message processing --------------------------------------------------------


def _resolve_text(value) -> tuple[str, list[TrackedPrompt]] | None:
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
    tracked: list[TrackedPrompt] = []
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


def _extract_input_text(messages) -> str | None:
    """The newest message's text content — what's actually new at this turn.

    Earlier turns (including the model's own prior reply) already live in
    the conversation as earlier rows; the system prompt, if any, is captured
    by the tracked Prompt's own version lineage instead of being repeated
    here. Multimodal content blocks are flattened to their text parts.
    """
    if not messages or not isinstance(messages, (list, tuple)):
        return None
    last = messages[-1]
    if not isinstance(last, dict) or last.get("role") in _SYSTEM_ROLES:
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


def _joined_text(messages) -> str:
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
    if not isinstance(last, dict) or last.get("role") in _SYSTEM_ROLES:
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


def _extract_output_text(response) -> str | None:
    """The assistant's text reply, or None (errors, empty, non-string)."""
    choices = getattr(response, "choices", None)
    if choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
    return None

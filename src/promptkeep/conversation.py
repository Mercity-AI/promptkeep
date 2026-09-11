"""Group tracked calls into a conversation (session).

The context manager below is the primary API — `contextvars`-based, so it's
safe across threads and async tasks, and it never touches the database
itself: nothing is written until a wrapped client call actually happens
inside the block. The wrapper (`integrations/openai_wrapper.py`) reads
`current()` to resolve which conversation, if any, is active, then does the
create-or-reuse write against storage at record time.

An explicit `promptkeep_conversation="..."` kwarg on individual calls is the
escape hatch for call sites that can't wrap a block (queue workers,
callbacks) — see the wrapper's `_resolve_conversation`. Both paths write the
same `conversation_id` column; this module only owns the ambient case.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from typing import Any

_current: contextvars.ContextVar[_Active | None] = contextvars.ContextVar(
    "promptkeep_conversation", default=None
)


@dataclass(frozen=True)
class _Active:
    """The ambient conversation: what `conversation()` stashes in the contextvar."""

    external_id: str
    title: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


class conversation:
    """Group every tracked call inside this block into one conversation.

        with promptkeep.conversation("user-42-session-9", user_id=42):
            client.chat.completions.create(...)   # turn 0
            client.chat.completions.create(...)   # turn 1

    Also works as `async with` for async call sites. Keyword arguments
    beyond `title` become the conversation's metadata (recorded once, the
    first time this external_id is seen).
    """

    def __init__(self, external_id: str, title: str | None = None, **metadata: Any):
        if not isinstance(external_id, str) or not external_id.strip():
            raise ValueError("conversation() requires a non-empty external_id")
        self._active = _Active(external_id, title, metadata)
        self._token: contextvars.Token | None = None

    @property
    def external_id(self) -> str:
        """The id passed in — the same one you'll look history up by."""
        return self._active.external_id

    def __enter__(self) -> conversation:
        self._token = _current.set(self._active)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        _current.reset(self._token)
        return False

    async def __aenter__(self) -> conversation:
        return self.__enter__()

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return self.__exit__(exc_type, exc, tb)


def current() -> _Active | None:
    """The ambient conversation set by an enclosing `with conversation(...)`,
    or None outside any block."""
    return _current.get()

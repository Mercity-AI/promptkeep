"""``promptkeep.call()`` / ``acall()``: the explicit-result shape.

The wrapper's default is to hand back the provider's native response with a
``RunHandle`` attached as ``response.promptkeep`` — existing code keeps
working untouched. ``call()`` is the alternative for new code: the same rows
are written, but the reply text and the verdict come back on one settled
``CallResult`` instead of being read off the response. Provider-agnostic:
the client's adapter finds the surface to call and reads the reply.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .base import ProviderAdapter, Target
from .core import is_instrumented
from .registry import locate

logger = logging.getLogger("promptkeep")


@dataclass(frozen=True)
class CallResult:
    """What ``call()`` returns — the explicit shape, no attribute-poking.

    text is the model's reply; verification is the aggregate verdict; run_key
    and checks come from the same RunHandle the attach path exposes. response
    is the untouched provider object, still available if you need it.
    """

    text: str | None
    verification: str
    run_key: str | None
    checks: list
    response: Any


def call(client: Any, **kwargs: Any) -> CallResult:
    """Make a tracked, checked call and get a result object directly.

        result = promptkeep.call(client, model="gpt-5.5", messages=[...])
        result.text          # the reply
        result.verification  # "ok" | "warn" | "failed" | "pending"

    Same rows as the attach path — just a nicer shape for new code. Blocked
    calls raise PromptBlocked (or, under on_block="return", come back with
    verification="failed" and text=None). Non-streaming only: passing
    stream=True raises (use the wrapped client's create(stream=True) instead).
    """
    adapter, target = _entry(client, kwargs)
    response = getattr(target.owner, target.attribute)(**kwargs)
    return _result(adapter, response)


async def acall(client: Any, **kwargs: Any) -> CallResult:
    """Async twin of call(), for async clients. Non-streaming only."""
    adapter, target = _entry(client, kwargs)
    response = await getattr(target.owner, target.attribute)(**kwargs)
    return _result(adapter, response)


def _entry(client: Any, kwargs: dict[str, Any]) -> tuple[ProviderAdapter, Target]:
    """Validate the call and find the surface it goes through.

    A stream can't produce a settled result — the reply and verdicts don't
    exist until it drains — so that is refused loudly. An unwrapped client
    still works but records nothing and runs no checks, which would make
    ``verification`` a permanent "ok"; warn rather than pretend.
    """
    if kwargs.get("stream"):
        raise ValueError(
            "promptkeep.call()/acall() are non-streaming. Drop stream=True, or call the "
            "wrapped client's create(stream=True) directly and read response.promptkeep "
            "once the stream finishes."
        )
    located = locate(client)
    if located is None:
        raise TypeError(f"promptkeep.call(): no supported provider surface on {client!r}")
    adapter, target = located
    if not is_instrumented(target):
        logger.warning(
            "promptkeep.call(): client is not wrapped, so no tracking or checks ran "
            "(verification will always be 'ok'). Pass promptkeep.wrap(client)."
        )
    return adapter, target


def _result(adapter: ProviderAdapter, response: Any) -> CallResult:
    """Build a CallResult from a (possibly promptkeep-annotated) response."""
    try:
        text = adapter.read_response(response).output_text
    except Exception:
        logger.warning("promptkeep: %s adapter failed to read response", adapter.provider)
        text = None
    handle = getattr(response, "promptkeep", None)
    if handle is not None:
        return CallResult(text, handle.verification, handle.run_key, handle.checks, response)
    return CallResult(text, "ok", None, [], response)

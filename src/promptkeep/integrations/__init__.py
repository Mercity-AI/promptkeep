"""Provider integrations. ``wrap()`` is the public entry point; the adapter
registry below is what it consults.

Each adapter (``base.ProviderAdapter``) knows one SDK's shapes. ``wrap()``
asks every registered adapter to ``locate`` its surface on the client and
instruments each one it finds with the shared interceptor in ``core``.
"""

from __future__ import annotations

from typing import Any, List, Optional

from .base import ProviderAdapter, Request, ResponseFields, StreamAbsorber, Target
from .core import instrument, is_instrumented
from .openai_wrapper import OpenAIChatAdapter

# Registration order is also search order. Built-ins first; third parties
# append with register_adapter().
_ADAPTERS: List[ProviderAdapter] = [OpenAIChatAdapter()]


def adapters() -> List[ProviderAdapter]:
    """The registered adapters, in search order (a copy)."""
    return list(_ADAPTERS)


def register_adapter(adapter: ProviderAdapter) -> None:
    """Add a provider adapter so ``wrap()`` recognizes its clients. Idempotent
    per adapter class: registering a second instance of the same class
    replaces the first."""
    if not isinstance(adapter, ProviderAdapter):
        raise TypeError(
            f"register_adapter() expects a ProviderAdapter, got {type(adapter).__name__}"
        )
    _ADAPTERS[:] = [a for a in _ADAPTERS if type(a) is not type(adapter)]
    _ADAPTERS.append(adapter)


def wrap(target):
    """Wrap a provider client class or instance so Prompt objects work as
    message content and every call is recorded as a run.

        from openai import OpenAI
        from promptkeep import wrap

        OpenAI = wrap(OpenAI)          # wrap the class...
        client = OpenAI(api_key=...)   # ...then use it exactly as before

        # or wrap a live client:
        client = wrap(OpenAI(api_key=...))

    Every surface a registered adapter recognizes on the client is
    instrumented; wrapping twice is a no-op. Raises TypeError when no adapter
    recognizes the object at all.
    """
    if isinstance(target, type):
        return _wrap_class(target)
    _instrument_or_raise(target)
    return target


def is_wrapped(client: Any) -> Optional[bool]:
    """Whether a client's provider surfaces carry promptkeep's tracking:
    True/False when at least one adapter recognizes the client, None when
    none does (nothing to be wrapped or unwrapped)."""
    targets = [t for t in (a.locate(client) for a in _ADAPTERS) if t is not None]
    if not targets:
        return None
    return any(is_instrumented(t) for t in targets)


def _instrument_or_raise(client: Any) -> None:
    if instrument(client, _ADAPTERS) == 0:
        known = ", ".join(a.provider for a in _ADAPTERS)
        raise TypeError(
            f"wrap() found no supported provider surface on {client!r} "
            f"(registered adapters: {known})"
        )


def _wrap_class(cls):
    """Subclass the client class so every instance self-instruments on init."""

    class WrappedClient(cls):
        """The user's client class plus tracking; behaves identically otherwise."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            _instrument_or_raise(self)

    # Keep the wrapper indistinguishable in reprs, logs, and debuggers.
    WrappedClient.__name__ = cls.__name__
    WrappedClient.__qualname__ = cls.__qualname__
    WrappedClient.__doc__ = cls.__doc__
    return WrappedClient


__all__ = [
    "wrap",
    "is_wrapped",
    "adapters",
    "register_adapter",
    "ProviderAdapter",
    "Request",
    "ResponseFields",
    "StreamAbsorber",
    "Target",
    "OpenAIChatAdapter",
]

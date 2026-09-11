"""The adapter registry and ``wrap()``, the public entry point.

Each adapter (``base.ProviderAdapter``) knows one SDK's shapes. ``wrap()``
asks every registered adapter to ``locate`` its surface on the client it
was given and installs the shared interceptor from ``core`` on each one it
finds — so one client can carry several tracked surfaces, and a third party
can teach ``wrap()`` a new client shape with ``register_adapter()``.
"""

from __future__ import annotations

from typing import Any

from .base import ProviderAdapter, Target
from .core import instrument, is_instrumented
from .openai_wrapper import OpenAIChatAdapter

# Registration order is also search order. Built-ins first; third parties
# append with register_adapter().
_ADAPTERS: list[ProviderAdapter] = [OpenAIChatAdapter()]


def adapters() -> list[ProviderAdapter]:
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


def locate(client: Any) -> tuple[ProviderAdapter, Target] | None:
    """The first registered adapter that recognizes ``client``, with the
    surface it found — or None when no adapter does."""
    for adapter in _ADAPTERS:
        target = adapter.locate(client)
        if target is not None:
            return adapter, target
    return None


def wrap(target: Any) -> Any:
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


def is_wrapped(client: Any) -> bool | None:
    """Whether a client's provider surfaces carry promptkeep's tracking:
    True/False when at least one adapter recognizes the client, None when
    none does (nothing to be wrapped or unwrapped)."""
    targets = [t for t in (a.locate(client) for a in _ADAPTERS) if t is not None]
    if not targets:
        return None
    return any(is_instrumented(t) for t in targets)


def _instrument_or_raise(client: Any) -> None:
    """Instrument every recognized surface; an unrecognized object is a usage error."""
    if instrument(client, _ADAPTERS) == 0:
        known = ", ".join(a.provider for a in _ADAPTERS)
        raise TypeError(
            f"wrap() found no supported provider surface on {client!r} "
            f"(registered adapters: {known})"
        )


def _wrap_class(cls: type) -> type:
    """Subclass the client class so every instance self-instruments on init."""

    class WrappedClient(cls):  # type: ignore[misc,valid-type]
        """The user's client class plus tracking; behaves identically otherwise."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Construct the real client, then instrument it."""
            super().__init__(*args, **kwargs)
            _instrument_or_raise(self)

    # Keep the wrapper indistinguishable in reprs, logs, and debuggers.
    WrappedClient.__name__ = cls.__name__
    WrappedClient.__qualname__ = cls.__qualname__
    WrappedClient.__doc__ = cls.__doc__
    return WrappedClient

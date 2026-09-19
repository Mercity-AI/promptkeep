"""Provider integrations: ``wrap()``, the adapter interface and registry, and
the explicit ``call()`` shape. See ``base`` for what an adapter is, ``core``
for the shared interceptor, ``registry`` for how clients are recognized."""

from __future__ import annotations

from .base import ProviderAdapter, Request, ResponseFields, StreamAbsorber, Target
from .call import CallResult, acall, call
from .openai_responses import OpenAIResponsesAdapter
from .openai_wrapper import OpenAIChatAdapter
from .registry import adapters, is_wrapped, register_adapter, wrap

__all__ = [
    "wrap",
    "call",
    "acall",
    "CallResult",
    "is_wrapped",
    "adapters",
    "register_adapter",
    "ProviderAdapter",
    "Request",
    "ResponseFields",
    "StreamAbsorber",
    "Target",
    "OpenAIChatAdapter",
    "OpenAIResponsesAdapter",
]

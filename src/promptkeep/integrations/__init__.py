"""Provider integrations. `wrap()` is the public entry point."""

from __future__ import annotations


def wrap(target):
    """Wrap an OpenAI client class or instance so Prompt objects work as
    message content and every call is recorded as a run.

        from openai import OpenAI
        from promptkeep import wrap

        OpenAI = wrap(OpenAI)          # wrap the class...
        client = OpenAI(api_key=...)   # ...then use it exactly as before

        # or wrap a live client:
        client = wrap(OpenAI(api_key=...))
    """
    from .openai_wrapper import wrap_openai_class, wrap_openai_instance

    if isinstance(target, type):
        return wrap_openai_class(target)
    chat = getattr(target, "chat", None)
    completions = getattr(chat, "completions", None)
    if completions is not None and hasattr(completions, "create"):
        return wrap_openai_instance(target)
    raise TypeError(
        "wrap() expects an OpenAI client class or instance"
        f" (an object with .chat.completions.create); got {target!r}"
    )


__all__ = ["wrap"]

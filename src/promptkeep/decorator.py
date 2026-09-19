"""@prompt decorator for computed prompts.

The decorated function returns the *raw template* (placeholders intact); the
decorator turns each call into a Prompt object, using the call's arguments as
the variables dict. Both raw and rendered live on the returned Prompt, so
lineage and run tracking work exactly like literal prompts. ``async def``
template functions are supported: awaiting the decorated call yields the Prompt.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import textwrap
from collections.abc import Callable
from typing import Any

from .prompts import Prompt


def _function_source_hash(fn: Callable) -> str | None:
    """Hash the function's source so history can tell code changes apart from
    same-code output changes. None when source isn't available (e.g. REPL)."""
    try:
        source = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError):
        return None
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def prompt(name: str, strict: bool | None = None, exact_match: bool = False):
    """Turn a template-building function into a Prompt factory.

        @prompt(name="REVIEW_SYSTEM")
        def review_sys_prompt(var1="some value"):
            return "You are a reviewer. Focus on {var1}."

        p = review_sys_prompt(var1="security")   # -> Prompt
        p.raw    # the template the function returned
        p.text   # rendered with the call's arguments

    The version identity is the returned template text (content-hash dedup,
    normalized so placeholder renames don't create versions; exact_match=True
    opts into raw-text identity). A hash of the function's source is stored
    alongside each version so history can tell "code changed" apart from
    "same code, different output".
    """
    # Catch the bare-decorator mistake (@prompt without parentheses) early.
    if callable(name):
        raise TypeError(
            '@prompt requires a name: use @prompt(name="MY_PROMPT") — '
            "the name is the prompt's stable identity"
        )
    if not isinstance(name, str) or not name.strip():
        raise ValueError("@prompt requires a non-empty name string")

    def decorate(fn: Callable[..., Any]):
        """Wrap fn so each call yields a Prompt built from its return value.

        An ``async def`` template function gets an async wrapper: awaiting it
        yields the Prompt. Everything else — variable capture, validation,
        lineage — is identical to the sync case.
        """
        # Computed once at decoration time; identical for every call.
        fn_hash = _function_source_hash(fn)
        signature = inspect.signature(fn)

        def package(template: Any, variables: dict) -> Prompt:
            """Validate fn's return value and wrap it as a Prompt."""
            if not isinstance(template, str):
                raise TypeError(
                    f"@prompt function {fn.__name__!r} must return a template string,"
                    f" got {type(template).__name__}"
                )
            return Prompt(
                template,
                variables,
                name=name,
                strict=strict,
                exact_match=exact_match,
                source="decorator",
                fn_source_hash=fn_hash,
            )

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Prompt:
                """Await fn, then package its returned template into a Prompt."""
                variables = _call_variables(signature, args, kwargs)
                return package(await fn(*args, **kwargs), variables)

            async_wrapper.prompt_name = name  # type: ignore[attr-defined]
            return async_wrapper

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Prompt:
            """Call fn, then package its returned template into a Prompt."""
            variables = _call_variables(signature, args, kwargs)
            return package(fn(*args, **kwargs), variables)

        wrapper.prompt_name = name  # type: ignore[attr-defined]
        return wrapper

    return decorate


def _call_variables(signature: inspect.Signature, args: tuple, kwargs: dict) -> dict:
    """The full call (defaults applied) as the variables dict: **kwargs entries
    become top-level variables, *args is recorded as a list under its name."""
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    variables: dict = {}
    for param_name, value in bound.arguments.items():
        kind = signature.parameters[param_name].kind
        if kind is inspect.Parameter.VAR_KEYWORD:
            variables.update(value)
        elif kind is inspect.Parameter.VAR_POSITIONAL:
            variables[param_name] = list(value)
        else:
            variables[param_name] = value
    return variables

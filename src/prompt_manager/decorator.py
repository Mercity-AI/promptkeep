"""@prompt decorator for computed prompts.

The decorated function returns the *raw template* (placeholders intact); the
decorator turns each call into a Prompt object, using the call's arguments as
the variables dict. Both raw and rendered live on the returned Prompt, so
lineage and run tracking work exactly like literal prompts.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import textwrap
from typing import Any, Callable, Optional

from .prompts import Prompt


def _function_source_hash(fn: Callable) -> Optional[str]:
    try:
        source = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError):
        return None
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def prompt(name: str, strict: Optional[bool] = None):
    """Turn a template-building function into a Prompt factory.

        @prompt(name="REVIEW_SYSTEM")
        def review_sys_prompt(var1="some value"):
            return "You are a reviewer. Focus on {var1}."

        p = review_sys_prompt(var1="security")   # -> Prompt
        p.raw    # the template the function returned
        p.text   # rendered with the call's arguments

    The version identity is the returned template text (content-hash dedup);
    a hash of the function's source is stored alongside each version so
    history can tell "code changed" apart from "same code, different output".
    """
    if callable(name):
        raise TypeError(
            '@prompt requires a name: use @prompt(name="MY_PROMPT") — '
            "the name is the prompt's stable identity"
        )
    if not isinstance(name, str) or not name.strip():
        raise ValueError("@prompt requires a non-empty name string")

    def decorate(fn: Callable[..., str]):
        fn_hash = _function_source_hash(fn)
        signature = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Prompt:
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
            template = fn(*args, **kwargs)
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
                source="decorator",
                fn_source_hash=fn_hash,
            )

        wrapper.prompt_name = name  # type: ignore[attr-defined]
        return wrapper

    return decorate

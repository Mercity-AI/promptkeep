"""The Prompt class and the provenance-carrying RenderedText string."""

from __future__ import annotations

from typing import Any, Dict, Optional, Set

from .rendering import extract_placeholders, render

_UNSET = object()


class RenderedText(str):
    """A plain string that remembers which Prompt (and variables) produced it.

    Behaves exactly like `str` everywhere (JSON, f-strings, SDKs). The wrapped
    OpenAI client uses the hidden provenance to track runs even when you pass
    `prompt.text` instead of the Prompt object itself.
    """

    _pm_prompt: Optional["Prompt"]
    _pm_variables: Dict[str, Any]

    def __new__(cls, value: str, prompt: Optional["Prompt"] = None, variables=None):
        self = super().__new__(cls, value)
        self._pm_prompt = prompt
        self._pm_variables = dict(variables or {})
        return self

    @property
    def prompt(self) -> Optional["Prompt"]:
        return self._pm_prompt

    @property
    def variables(self) -> Dict[str, Any]:
        return dict(self._pm_variables)


class Prompt:
    """A named, versioned prompt template.

    `name` is the stable identity; the template text is the versioned content.
    Instances are immutable — `.format(**vars)` returns a new Prompt with
    updated variables (same name, same template, same version).

        p = Prompt("xyz, {var1}", variables={"var1": "some value"}, name="REVIEW_SYSTEM")
        p.text   # rendered string (RenderedText)
        p.raw    # raw template with placeholders intact
    """

    __slots__ = (
        "_name",
        "_template",
        "_variables",
        "_strict",
        "_source",
        "_fn_source_hash",
        "_registration",
        "_frozen",
    )

    def __init__(
        self,
        text: str,
        variables: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        strict: Optional[bool] = None,
        source: str = "literal",
        fn_source_hash: Optional[str] = None,
    ):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Prompt text must be a non-empty string")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                "Prompt requires a name (its stable identity), e.g. "
                "Prompt('...', name='REVIEW_SYSTEM')"
            )
        if variables is None:
            variables = {}
        if not isinstance(variables, dict):
            raise TypeError(f"variables must be a dict, got {type(variables).__name__}")
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_template", text)
        object.__setattr__(self, "_variables", dict(variables))
        object.__setattr__(self, "_strict", strict)
        object.__setattr__(self, "_source", source)
        object.__setattr__(self, "_fn_source_hash", fn_source_hash)
        object.__setattr__(self, "_registration", _UNSET)
        object.__setattr__(self, "_frozen", True)

    # --- immutability -----------------------------------------------------

    def __setattr__(self, key, value):
        raise AttributeError(
            "Prompt objects are immutable; use .format(**variables) to derive a new one"
        )

    def __delattr__(self, key):
        raise AttributeError("Prompt objects are immutable")

    # --- identity & content -----------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def raw(self) -> str:
        """The raw template text, placeholders intact."""
        return self._template

    @property
    def variables(self) -> Dict[str, Any]:
        return dict(self._variables)

    @property
    def placeholders(self) -> Set[str]:
        return extract_placeholders(self._template)

    @property
    def source(self) -> str:
        return self._source

    @property
    def fn_source_hash(self) -> Optional[str]:
        return self._fn_source_hash

    # --- rendering ----------------------------------------------------------

    def render(self, **overrides: Any) -> RenderedText:
        """Render with the stored variables, optionally overridden per-call."""
        merged = {**self._variables, **overrides}
        self._ensure_registered()
        value = render(self._template, merged, strict=self._effective_strict())
        return RenderedText(value, prompt=self, variables=merged)

    @property
    def text(self) -> RenderedText:
        """The rendered prompt — a real string, safe to pass anywhere."""
        return self.render()

    def format(self, **overrides: Any) -> "Prompt":
        """Return a new Prompt with updated variables (same name/template/version)."""
        return Prompt(
            self._template,
            {**self._variables, **overrides},
            name=self._name,
            strict=self._strict,
            source=self._source,
            fn_source_hash=self._fn_source_hash,
        )

    def _effective_strict(self) -> bool:
        if self._strict is not None:
            return self._strict
        from .config import get_settings

        return get_settings().strict

    # --- versioning ---------------------------------------------------------

    def _ensure_registered(self):
        """Lazily record this template as a version in the DB (once per object).

        Returns (version_id, version_number) or None when tracking is disabled
        or the write failed. Never raises.
        """
        registration = self._registration
        if registration is _UNSET:
            from . import storage

            registration = storage.register_version(
                self._name, self._template, self._source, self._fn_source_hash
            )
            object.__setattr__(self, "_registration", registration)
        return registration

    @property
    def version(self) -> Optional[int]:
        """This template's version number under its name (None if tracking is off)."""
        registration = self._ensure_registered()
        return registration[1] if registration else None

    # --- dunders ------------------------------------------------------------

    def __str__(self) -> str:
        return str(self.text)

    def __repr__(self) -> str:
        template = self._template if len(self._template) <= 50 else self._template[:47] + "..."
        return f"Prompt(name={self._name!r}, raw={template!r})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Prompt):
            return NotImplemented
        return (
            self._name == other._name
            and self._template == other._template
            and self._variables == other._variables
        )

    def __hash__(self) -> int:
        items = tuple(sorted((k, repr(v)) for k, v in self._variables.items()))
        return hash((self._name, self._template, items))

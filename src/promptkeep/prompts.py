"""The Prompt class (a named, versioned template) and RenderedText (a plain
string that remembers which Prompt and variables produced it).

These two types are the heart of the library: Prompt carries identity and
lineage; RenderedText lets rendered strings flow through any SDK while still
being traceable back to their source for run tracking.
"""

from __future__ import annotations

from typing import Any

from . import storage
from .config import get_settings
from .rendering import extract_placeholders, render, template_hash

# "Registration not attempted yet". Once attempted, the slot holds
# (database path, result) — the result being None when tracking was disabled or
# the write failed. The path is part of the memo because a version id means
# nothing in another file: a Prompt defined at import time can outlive a
# configure(db_path=...), and a stale id would file its runs under whatever
# version happens to own that id over there (or fail the foreign key and lose
# them).
_UNSET = object()


class RenderedText(str):
    """A plain string that remembers which Prompt (and variables) produced it.

    Behaves exactly like `str` everywhere (JSON, f-strings, SDKs). The wrapped
    OpenAI client uses the hidden provenance to track runs even when you pass
    `prompt.text` instead of the Prompt object itself.
    """

    _pm_prompt: Prompt | None
    _pm_variables: dict[str, Any]

    def __new__(cls, value: str, prompt: Prompt | None = None, variables=None):
        """Build the string value and attach its provenance attributes."""
        self = super().__new__(cls, value)
        self._pm_prompt = prompt
        self._pm_variables = dict(variables or {})
        return self

    @property
    def prompt(self) -> Prompt | None:
        """The Prompt this string was rendered from (None if constructed bare)."""
        return self._pm_prompt

    @property
    def variables(self) -> dict[str, Any]:
        """A copy of the variables used for this particular rendering."""
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
        "_exact_match",
        "_source",
        "_fn_source_hash",
        "_pre",
        "_post",
        "_registration",
        "_frozen",
    )

    def __init__(
        self,
        text: str,
        variables: dict[str, Any] | None = None,
        name: str | None = None,
        strict: bool | None = None,
        exact_match: bool = False,
        source: str = "literal",
        fn_source_hash: str | None = None,
        pre: list | None = None,
        post: list | None = None,
    ):
        """Validate inputs and freeze the instance (source/fn_source_hash are
        internal, set by the @prompt decorator).

        exact_match=True opts out of normalized version matching: the raw
        template text (including placeholder names) becomes the version
        identity, so renaming {var1} -> {x} DOES create a new version.
        """
        # Validate the two identity-critical inputs up front.
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

        # __setattr__ is blocked after construction, so set fields via object.
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_template", text)
        object.__setattr__(self, "_variables", dict(variables))
        object.__setattr__(self, "_strict", strict)
        object.__setattr__(self, "_exact_match", bool(exact_match))
        object.__setattr__(self, "_source", source)
        object.__setattr__(self, "_fn_source_hash", fn_source_hash)
        # Checks attached at prompt scope. Tuples so the frozen prompt can't
        # have its check list mutated out from under a registered version.
        object.__setattr__(self, "_pre", tuple(pre or ()))
        object.__setattr__(self, "_post", tuple(post or ()))
        object.__setattr__(self, "_registration", _UNSET)
        object.__setattr__(self, "_frozen", True)

    # --- immutability -----------------------------------------------------

    def __setattr__(self, key, value):
        """Block mutation: a Prompt must never drift from its registered version."""
        raise AttributeError(
            "Prompt objects are immutable; use .format(**variables) to derive a new one"
        )

    def __delattr__(self, key):
        """Block attribute deletion for the same reason as __setattr__."""
        raise AttributeError("Prompt objects are immutable")

    # --- identity & content -----------------------------------------------

    @property
    def name(self) -> str:
        """The prompt's stable identity — versions are tracked under this name."""
        return self._name

    @property
    def raw(self) -> str:
        """The raw template text, placeholders intact."""
        return self._template

    @property
    def variables(self) -> dict[str, Any]:
        """A copy of the stored variables (mutating it cannot affect the Prompt)."""
        return dict(self._variables)

    @property
    def placeholders(self) -> set[str]:
        """Variable names the template references, e.g. {'var1', 'topic'}."""
        return extract_placeholders(self._template)

    @property
    def source(self) -> str:
        """How the prompt was defined: 'literal' (class) or 'decorator'."""
        return self._source

    @property
    def fn_source_hash(self) -> str | None:
        """Hash of the @prompt function's source code (None for literal prompts)."""
        return self._fn_source_hash

    @property
    def exact_match(self) -> bool:
        """True when version identity includes placeholder names (opt-in)."""
        return self._exact_match

    @property
    def pre(self) -> tuple:
        """Pre-checks (gates) attached at this prompt's scope."""
        return self._pre

    @property
    def post(self) -> tuple:
        """Post-checks (audits) attached at this prompt's scope."""
        return self._post

    # --- rendering ----------------------------------------------------------

    def render(self, **overrides: Any) -> RenderedText:
        """Render with the stored variables, optionally overridden per-call."""
        merged = {**self._variables, **overrides}
        # First use of the template registers its version (lazy, write-once).
        self._ensure_registered()
        value = render(self._template, merged, strict=self._effective_strict())
        return RenderedText(value, prompt=self, variables=merged)

    @property
    def text(self) -> RenderedText:
        """The rendered prompt — a real string, safe to pass anywhere."""
        return self.render()

    def format(self, **overrides: Any) -> Prompt:
        """Return a new Prompt with updated variables (same name/template/version)."""
        derived = Prompt(
            self._template,
            {**self._variables, **overrides},
            name=self._name,
            strict=self._strict,
            exact_match=self._exact_match,
            source=self._source,
            fn_source_hash=self._fn_source_hash,
            pre=self._pre,
            post=self._post,
        )
        # Same name and template means the same version: a registration already
        # made carries over. (A failed one doesn't — the derived prompt retries.)
        memo = self._registration
        if memo is not _UNSET and memo[1] is not None:
            object.__setattr__(derived, "_registration", memo)
        return derived

    def _effective_strict(self) -> bool:
        """Per-prompt strict flag if set, otherwise the global configured default."""
        if self._strict is not None:
            return self._strict
        return get_settings().strict

    # --- versioning ---------------------------------------------------------

    def _ensure_registered(self):
        """Lazily record this template as a version in the DB (once per object).

        Returns (version_id, version_number) or None when tracking is disabled
        or the write failed. Never raises.
        """
        # The memo is good only for the database it was made against.
        database = str(get_settings().db_path)
        memo = self._registration
        if memo is not _UNSET and memo[0] == database:
            return memo[1]

        registration = storage.register_version(
            self._name,
            self._template,
            self._source,
            self._fn_source_hash,
            exact_match=self._exact_match,
        )
        object.__setattr__(self, "_registration", (database, registration))
        return registration

    @property
    def version(self) -> int | None:
        """This template's version number under its name (None if tracking is off)."""
        registration = self._ensure_registered()
        return registration[1] if registration else None

    @classmethod
    def variants(cls, name: str) -> list[Prompt]:
        """Every stored version of ``name`` as a usable Prompt, oldest first.

            v4, v5 = Prompt.variants("REVIEW_SYSTEM")[-2:]
            chosen = random.choice([v4, v5]).format(focus="security")   # your own A/B
            client.chat.completions.create(..., messages=[{"role": "system", "content": chosen}])

        The code only ever holds the *current* template; this is how an older
        one comes back — to re-run a session against it, to compare it with
        its successor, or to split traffic between two. Each variant already
        knows its version (``.version`` costs no database round-trip) and
        records its runs under it. A version stores a template, never
        variables — those are run data — so variants come back with none:
        ``.format(**variables)`` them. Checks aren't stored either.

        An explicit read: it opens the database (so not at import time) and
        raises if the database is broken. Empty for an unknown name, or when
        tracking is disabled.
        """
        database = str(get_settings().db_path)
        return [cls._stored(name, row, database) for row in storage.version_rows(name)]

    @classmethod
    def load(
        cls,
        name: str,
        version: int | None = None,
        *,
        strict: bool | None = None,
        pre: list | None = None,
        post: list | None = None,
    ) -> Prompt:
        """One stored version of ``name`` as a Prompt — ``version``, or the
        latest (the highest-numbered) when it is omitted.

            review = Prompt.load("REVIEW_SYSTEM", version=4)   # pinned
            review = Prompt.load("REVIEW_SYSTEM")              # whatever is newest
            review.format(focus="security").text

        This is the prompt registry: the template comes from the database
        instead of a literal in the code, so the prompt in play can change
        without a deploy — register a new version anywhere that writes to the
        same file (a script, a notebook, another service) and the next load
        picks it up. Like ``variants()``, the result already knows its version
        and records its runs under it; it carries no variables. Checks and
        strictness aren't stored with a version, so they are passed here.

        Latest means highest-numbered, not most recently used: if the code
        went back to an older template, that template kept its old number, and
        the newest version is still the one after it.

        An explicit read — do it at startup or per request, not at import
        time. Raises ValueError when the prompt has no stored versions (an
        unknown name, or tracking disabled) or no such version.
        """
        rows = storage.version_rows(name)
        if not rows:
            disabled = "" if get_settings().enabled else " (tracking is disabled)"
            raise ValueError(f"no stored versions of prompt {name!r}{disabled}")

        # Pick the row: the one asked for, else the newest.
        if version is None:
            row = rows[-1]
        else:
            row = next((r for r in rows if r["version"] == version), None)
            if row is None:
                raise ValueError(f"prompt {name!r} has no version {version}")
        database = str(get_settings().db_path)
        return cls._stored(name, row, database, strict=strict, pre=pre, post=post)

    @classmethod
    def _stored(
        cls,
        name: str,
        row: dict[str, Any],
        database: str,
        strict: bool | None = None,
        pre: list | None = None,
        post: list | None = None,
    ) -> Prompt:
        """A stored version row rebuilt as a Prompt already bound to it.

        It must reproduce the identity the version was hashed under, or using
        it would mint a new version: one registered with exact_match=True
        hashed its raw text, which shows as a stored hash that differs from
        the normalized one. The registration memo is seeded from the row —
        it came from the lineage, so no lookup is needed.
        """
        exact = template_hash(row["template"]) != row["template_hash"]
        stored = cls(
            row["template"],
            name=name,
            strict=strict,
            exact_match=exact,
            source=row["source"],
            fn_source_hash=row["fn_source_hash"],
            pre=pre,
            post=post,
        )
        object.__setattr__(stored, "_registration", (database, (row["id"], row["version"])))
        return stored

    # --- dunders ------------------------------------------------------------

    def __str__(self) -> str:
        """Render on str() so accidental f-string usage still produces the text."""
        return str(self.text)

    def __repr__(self) -> str:
        """Short debugging form; avoids touching the DB (no version lookup)."""
        template = self._template if len(self._template) <= 50 else self._template[:47] + "..."
        return f"Prompt(name={self._name!r}, raw={template!r})"

    def __eq__(self, other) -> bool:
        """Prompts are equal when name, template, and variables all match."""
        if not isinstance(other, Prompt):
            return NotImplemented
        return (
            self._name == other._name
            and self._template == other._template
            and self._variables == other._variables
        )

    def __hash__(self) -> int:
        """Hash consistent with __eq__; variable values go through repr() so
        unhashable values (lists, dicts) still work."""
        items = tuple(sorted((k, repr(v)) for k, v in self._variables.items()))
        return hash((self._name, self._template, items))

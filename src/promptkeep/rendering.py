"""Template rendering with `{placeholder}` syntax.

Lenient by default: placeholders without a matching variable are left intact
in the output (so JSON snippets and code braces inside prompts survive), and
templates that can't be parsed at all are returned unchanged with a warning.
Strict mode raises instead.
"""

from __future__ import annotations

import logging
from string import Formatter
from typing import Iterable, Mapping, Optional, Set

logger = logging.getLogger("promptkeep")

# One shared Formatter gives us stdlib-compatible parsing of {field:spec!conv}.
_formatter = Formatter()


class MissingVariableError(KeyError):
    """Raised in strict mode when a placeholder has no matching variable."""

    def __init__(self, missing: Iterable[str]):
        self.missing = sorted(set(missing))
        super().__init__(
            "Missing variables for placeholders: " + ", ".join(repr(m) for m in self.missing)
        )

    def __str__(self) -> str:
        """Return the plain message (KeyError repr-quotes its args otherwise)."""
        return self.args[0]


class TemplateParseError(ValueError):
    """Raised in strict mode when the template is not valid format syntax."""


def _base_name(field_name: str) -> str:
    """Extract the variable name from a field: `user.name` / `user[0]` -> `user`."""
    return field_name.split(".")[0].split("[")[0]


def _rebuild_placeholder(field_name: str, conversion: Optional[str], spec: Optional[str]) -> str:
    """Reassemble a parsed placeholder into its original `{field!conv:spec}` text."""
    out = "{" + field_name
    if conversion:
        out += "!" + conversion
    if spec:
        out += ":" + spec
    return out + "}"


def normalize_template(template: str) -> str:
    """Canonical form used for version matching: placeholder *names* are
    replaced by positional tokens ({v0}, {v1}, ... in order of first
    appearance) so renaming a variable doesn't change a prompt's identity.

    Everything that affects the prompt structurally is preserved: static
    text, placeholder positions, repetition patterns ({a}..{a} vs {a}..{b}),
    attribute/index paths, conversions, and format specs. Unparseable
    templates normalize to themselves.
    """
    try:
        parsed = list(_formatter.parse(template))
    except ValueError:
        return template
    mapping: dict = {}
    out = []
    for literal, field_name, spec, conversion in parsed:
        # parse() unescapes {{ }}; re-escape so literal braces stay literal.
        out.append(literal.replace("{", "{{").replace("}", "}}"))
        if field_name is None:
            continue
        base = _base_name(field_name)
        if base and not base.isdigit():
            # Same variable -> same token everywhere; keep any .attr/[idx] tail.
            if base not in mapping:
                mapping[base] = f"v{len(mapping)}"
            canonical = mapping[base] + field_name[len(base) :]
        else:
            # Positional/empty fields are not named variables; leave untouched.
            canonical = field_name
        out.append(_rebuild_placeholder(canonical, conversion, spec))
    return "".join(out)


def extract_placeholders(template: str) -> Set[str]:
    """Return the set of variable names referenced by the template.

    `{user[name]}` and `{user.name}` both report `user`. Positional
    placeholders (`{}` / `{0}`) are not supported and are ignored here.
    """
    names: Set[str] = set()
    try:
        for _literal, field_name, _spec, _conv in _formatter.parse(template):
            if field_name:
                base = _base_name(field_name)
                if base and not base.isdigit():
                    names.add(base)
    except ValueError:
        # Unparseable template: report no placeholders rather than blow up.
        return set()
    return names


def render(template: str, variables: Optional[Mapping] = None, strict: bool = False) -> str:
    """Substitute variables into the template.

    Lenient (default): unknown placeholders stay as literal `{name}` text and
    unparseable templates are returned as-is. Strict: raises
    MissingVariableError / TemplateParseError instead.
    """
    variables = variables or {}

    # Parse up front so a syntax error surfaces before any output is built.
    try:
        parsed = list(_formatter.parse(template))
    except ValueError as exc:
        if strict:
            raise TemplateParseError(f"Invalid template syntax: {exc}") from exc
        logger.warning(
            "promptkeep: template could not be parsed (%s); returning it unrendered", exc
        )
        return template

    # Walk the parsed segments: emit literal text as-is, substitute known
    # variables, and keep unknown placeholders literal (or collect for strict).
    out = []
    missing = []
    for literal, field_name, spec, conversion in parsed:
        out.append(literal)
        if field_name is None:
            continue
        base = _base_name(field_name)
        if base and not base.isdigit() and base in variables:
            try:
                # Full stdlib semantics: attribute/index access, !r/!s, :specs.
                value, _ = _formatter.get_field(field_name, (), variables)
                if conversion:
                    value = _formatter.convert_field(value, conversion)
                out.append(format(value, spec or ""))
                continue
            except Exception:
                if strict:
                    raise
                # Lookup or format-spec failed (e.g. `{user[missing]}` or a
                # JSON-ish spec); keep the placeholder literal instead.
        missing.append(field_name)
        out.append(_rebuild_placeholder(field_name, conversion, spec))

    if strict and missing:
        raise MissingVariableError(missing)
    return "".join(out)

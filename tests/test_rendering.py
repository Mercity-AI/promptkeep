"""Tests for the template rendering engine: substitution, leniency, strictness."""

import pytest

from promptkeep.rendering import (
    MissingVariableError,
    TemplateParseError,
    extract_placeholders,
    normalize_template,
    render,
)


class TestRender:
    """render(): substitution behavior in lenient and strict modes."""

    def test_basic_substitution(self):
        """A single {var} is replaced by its value."""
        assert render("Hello {name}!", {"name": "world"}) == "Hello world!"

    def test_multiple_and_repeated_placeholders(self):
        """Each occurrence is substituted, including repeats of the same name."""
        out = render("{a} and {b} and {a}", {"a": "1", "b": "2"})
        assert out == "1 and 2 and 1"

    def test_missing_variable_stays_literal(self):
        """Lenient default: unknown placeholders survive as literal text."""
        assert render("Hello {name}!", {}) == "Hello {name}!"

    def test_extra_variables_ignored(self):
        """Variables without a matching placeholder are simply unused."""
        assert render("Hello {name}!", {"name": "x", "unused": "y"}) == "Hello x!"

    def test_non_string_values(self):
        """Non-string values are formatted with standard str() semantics."""
        assert render("{n} items, pi={pi}", {"n": 3, "pi": 3.14}) == "3 items, pi=3.14"

    def test_json_in_template_survives(self):
        """JSON examples inside a prompt must pass through untouched."""
        template = 'Reply with JSON like {"score": 5} for {topic}.'
        out = render(template, {"topic": "cats"})
        assert out == 'Reply with JSON like {"score": 5} for cats.'

    def test_double_brace_escaping(self):
        """Standard {{...}} escaping produces literal braces."""
        assert render("literal {{braces}} here {v}", {"v": "x"}) == "literal {braces} here x"

    def test_format_spec(self):
        """Format specs like :03d apply to substituted values."""
        assert render("{n:03d}", {"n": 7}) == "007"

    def test_conversion(self):
        """!r / !s conversions apply to substituted values."""
        assert render("{v!r}", {"v": "hi"}) == "'hi'"

    def test_index_access(self):
        """Placeholders can index into dict/list variables: {user[name]}."""
        assert render("{user[name]}", {"user": {"name": "ada"}}) == "ada"

    def test_failed_index_access_stays_literal_when_lenient(self):
        """A failed lookup inside a present variable keeps the placeholder."""
        assert render("{user[missing]}", {"user": {}}) == "{user[missing]}"

    def test_unparseable_template_returned_as_is_when_lenient(self):
        """Templates with broken brace syntax come back unrendered, not raising."""
        broken = "closing brace only } here"
        assert render(broken, {"a": 1}) == broken

    def test_strict_raises_on_missing(self):
        """Strict mode raises and names every unresolved placeholder."""
        with pytest.raises(MissingVariableError) as excinfo:
            render("{a} {b}", {"a": 1}, strict=True)
        assert "b" in str(excinfo.value)

    def test_strict_raises_on_unparseable(self):
        """Strict mode surfaces template syntax errors."""
        with pytest.raises(TemplateParseError):
            render("bad } brace", {}, strict=True)

    def test_strict_ok_when_all_present(self):
        """Strict mode is silent when every placeholder resolves."""
        assert render("{a}", {"a": 1}, strict=True) == "1"


class TestNormalizeTemplate:
    """normalize_template(): the canonical form behind version matching."""

    def test_renamed_variable_same_form(self):
        """Only static text matters — {var1} and {x} normalize identically."""
        assert normalize_template("xyz, {var1}") == normalize_template("xyz, {x}")

    def test_different_static_text_differs(self):
        """Changing the words around a placeholder changes the form."""
        assert normalize_template("xyz, {a}") != normalize_template("abc, {a}")

    def test_repetition_pattern_preserved(self):
        """{a}..{a} (one value twice) is not the same prompt as {a}..{b}."""
        assert normalize_template("{a} then {a}") == normalize_template("{x} then {x}")
        assert normalize_template("{a} then {a}") != normalize_template("{a} then {b}")

    def test_order_of_first_appearance(self):
        """Tokens are assigned by first appearance, so swapped roles still match."""
        assert normalize_template("{a} vs {b}") == normalize_template("{b} vs {a}")

    def test_format_spec_and_conversion_preserved(self):
        """Specs/conversions are structure: renaming matches, respeccing doesn't."""
        assert normalize_template("{x:03d}") == normalize_template("{y:03d}")
        assert normalize_template("{x:03d}") != normalize_template("{x:04d}")
        assert normalize_template("{x!r}") != normalize_template("{x}")

    def test_attribute_and_index_tails_preserved(self):
        """{user.name} matches {u.name} but not {user.age}."""
        assert normalize_template("{user.name}") == normalize_template("{u.name}")
        assert normalize_template("{user.name}") != normalize_template("{user.age}")

    def test_escaped_braces_are_static_text(self):
        """{{literal}} content is text, not a variable — it must not normalize."""
        assert normalize_template("{{a}}") != normalize_template("{{b}}")
        assert normalize_template("{{a}} {x}") == normalize_template("{{a}} {y}")

    def test_idempotent(self):
        """Normalizing an already-normalized template is a no-op."""
        once = normalize_template("Hi {name}, meet {name} and {other}.")
        assert normalize_template(once) == once

    def test_unparseable_normalizes_to_itself(self):
        """Broken syntax falls back to the raw text."""
        assert normalize_template("bad } brace") == "bad } brace"


class TestExtractPlaceholders:
    """extract_placeholders(): reporting which variables a template uses."""

    def test_simple(self):
        """Plain placeholders are reported by name."""
        assert extract_placeholders("Hello {name}, {age}") == {"name", "age"}

    def test_deduplicates(self):
        """Repeated placeholders appear once in the result set."""
        assert extract_placeholders("{a} {a} {b}") == {"a", "b"}

    def test_ignores_escaped_braces(self):
        """{{literal}} escapes are not variables."""
        assert extract_placeholders("{{literal}} {real}") == {"real"}

    def test_reports_base_name_for_attribute_and_index(self):
        """{user.name} and {items[0]} report their base variable names."""
        assert extract_placeholders("{user.name} {items[0]}") == {"user", "items"}

    def test_no_placeholders(self):
        """Plain text has no placeholders."""
        assert extract_placeholders("plain text") == set()

    def test_unparseable_returns_empty(self):
        """Broken syntax yields an empty set rather than raising."""
        assert extract_placeholders("bad } brace") == set()

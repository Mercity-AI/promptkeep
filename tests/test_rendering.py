import pytest

from prompt_manager.rendering import (
    MissingVariableError,
    TemplateParseError,
    extract_placeholders,
    render,
)


class TestRender:
    def test_basic_substitution(self):
        assert render("Hello {name}!", {"name": "world"}) == "Hello world!"

    def test_multiple_and_repeated_placeholders(self):
        out = render("{a} and {b} and {a}", {"a": "1", "b": "2"})
        assert out == "1 and 2 and 1"

    def test_missing_variable_stays_literal(self):
        assert render("Hello {name}!", {}) == "Hello {name}!"

    def test_extra_variables_ignored(self):
        assert render("Hello {name}!", {"name": "x", "unused": "y"}) == "Hello x!"

    def test_non_string_values(self):
        assert render("{n} items, pi={pi}", {"n": 3, "pi": 3.14}) == "3 items, pi=3.14"

    def test_json_in_template_survives(self):
        template = 'Reply with JSON like {"score": 5} for {topic}.'
        out = render(template, {"topic": "cats"})
        assert out == 'Reply with JSON like {"score": 5} for cats.'

    def test_double_brace_escaping(self):
        assert render("literal {{braces}} here {v}", {"v": "x"}) == "literal {braces} here x"

    def test_format_spec(self):
        assert render("{n:03d}", {"n": 7}) == "007"

    def test_conversion(self):
        assert render("{v!r}", {"v": "hi"}) == "'hi'"

    def test_index_access(self):
        assert render("{user[name]}", {"user": {"name": "ada"}}) == "ada"

    def test_failed_index_access_stays_literal_when_lenient(self):
        assert render("{user[missing]}", {"user": {}}) == "{user[missing]}"

    def test_unparseable_template_returned_as_is_when_lenient(self):
        broken = "closing brace only } here"
        assert render(broken, {"a": 1}) == broken

    def test_strict_raises_on_missing(self):
        with pytest.raises(MissingVariableError) as excinfo:
            render("{a} {b}", {"a": 1}, strict=True)
        assert "b" in str(excinfo.value)

    def test_strict_raises_on_unparseable(self):
        with pytest.raises(TemplateParseError):
            render("bad } brace", {}, strict=True)

    def test_strict_ok_when_all_present(self):
        assert render("{a}", {"a": 1}, strict=True) == "1"


class TestExtractPlaceholders:
    def test_simple(self):
        assert extract_placeholders("Hello {name}, {age}") == {"name", "age"}

    def test_deduplicates(self):
        assert extract_placeholders("{a} {a} {b}") == {"a", "b"}

    def test_ignores_escaped_braces(self):
        assert extract_placeholders("{{literal}} {real}") == {"real"}

    def test_reports_base_name_for_attribute_and_index(self):
        assert extract_placeholders("{user.name} {items[0]}") == {"user", "items"}

    def test_no_placeholders(self):
        assert extract_placeholders("plain text") == set()

    def test_unparseable_returns_empty(self):
        assert extract_placeholders("bad } brace") == set()

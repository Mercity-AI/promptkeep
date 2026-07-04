"""Tests for the Prompt class: construction, rendering, immutability, versioning."""

import json

import pytest

import promptkeep
from promptkeep import Prompt, RenderedText


class TestConstruction:
    """Constructor validation and the documented signature."""

    def test_requires_name(self):
        """A Prompt without a name has no identity and must be rejected."""
        with pytest.raises(ValueError, match="name"):
            Prompt("some text")

    def test_requires_nonempty_text(self):
        """Empty or whitespace-only templates are rejected."""
        with pytest.raises(ValueError):
            Prompt("", name="X")
        with pytest.raises(ValueError):
            Prompt("   ", name="X")

    def test_variables_must_be_dict(self):
        """Variables must be a dict, not any other container."""
        with pytest.raises(TypeError):
            Prompt("hi {a}", variables=["a"], name="X")

    def test_matches_planned_signature(self):
        """The keyword form from the original design sketch works verbatim."""
        p = Prompt(text="xyz, {var1}", variables={"var1": "some value"}, name="REVIEW_SYSTEM")
        assert p.name == "REVIEW_SYSTEM"
        assert p.text == "xyz, some value"


class TestRenderingBehavior:
    """.text / .raw / .render() and the provenance-carrying result string."""

    def test_text_is_rendered(self):
        """.text substitutes the stored variables."""
        p = Prompt("Hello {who}", {"who": "world"}, name="GREET")
        assert p.text == "Hello world"

    def test_raw_keeps_placeholders(self):
        """.raw is the untouched template."""
        p = Prompt("Hello {who}", {"who": "world"}, name="GREET")
        assert p.raw == "Hello {who}"

    def test_text_is_a_real_string(self):
        """.text passes every duck-type check a plain str would."""
        p = Prompt("Hello {who}", {"who": "world"}, name="GREET")
        assert isinstance(p.text, str)
        assert json.dumps({"content": p.text}) == '{"content": "Hello world"}'
        assert p.text + "!" == "Hello world!"

    def test_text_carries_provenance(self):
        """.text remembers its Prompt and variables for downstream tracking."""
        p = Prompt("Hello {who}", {"who": "world"}, name="GREET")
        rendered = p.text
        assert isinstance(rendered, RenderedText)
        assert rendered.prompt is p
        assert rendered.variables == {"who": "world"}

    def test_str_returns_rendered(self):
        """str(prompt) and f-strings produce the rendered text."""
        p = Prompt("Hello {who}", {"who": "world"}, name="GREET")
        assert str(p) == "Hello world"
        assert f"{p}" == "Hello world"

    def test_render_with_overrides(self):
        """render(**overrides) is one-shot; the stored variables stay put."""
        p = Prompt("Hello {who}", {"who": "world"}, name="GREET")
        assert p.render(who="there") == "Hello there"
        assert p.text == "Hello world"  # original untouched

    def test_missing_variable_lenient_by_default(self):
        """Unfilled placeholders stay literal unless strict mode is on."""
        p = Prompt("Hello {who}", name="GREET")
        assert p.text == "Hello {who}"

    def test_per_prompt_strict(self):
        """strict=True on one Prompt raises on missing variables."""
        p = Prompt("Hello {who}", name="GREET", strict=True)
        with pytest.raises(promptkeep.MissingVariableError):
            _ = p.text

    def test_global_strict_via_configure(self):
        """configure(strict=True) flips the default for all prompts."""
        promptkeep.configure(strict=True)
        p = Prompt("Hello {who}", name="GREET")
        with pytest.raises(promptkeep.MissingVariableError):
            _ = p.text

    def test_placeholders(self):
        """.placeholders lists the template's variables, ignoring escapes."""
        p = Prompt("{a} and {b} and {{not_one}}", name="X")
        assert p.placeholders == {"a", "b"}


class TestImmutability:
    """Prompts are frozen; derivation happens through .format()."""

    def test_setattr_raises(self):
        """Both public and private attributes are locked after construction."""
        p = Prompt("hi", name="X")
        with pytest.raises(AttributeError):
            p.name = "other"
        with pytest.raises(AttributeError):
            p._template = "sneaky"

    def test_format_returns_new_prompt(self):
        """.format() derives a sibling with new variables, same identity."""
        p1 = Prompt("Hello {who}", {"who": "world"}, name="GREET")
        p2 = p1.format(who="there")
        assert p2 is not p1
        assert p2.text == "Hello there"
        assert p1.text == "Hello world"
        assert p2.name == p1.name
        assert p2.raw == p1.raw

    def test_variables_property_returns_copy(self):
        """Mutating the .variables snapshot cannot reach into the Prompt."""
        p = Prompt("Hello {who}", {"who": "world"}, name="GREET")
        p.variables["who"] = "mutated"
        assert p.text == "Hello world"


class TestEquality:
    """__eq__ / __hash__ semantics."""

    def test_equal_prompts(self):
        """Same name + template + variables means equal and same hash."""
        a = Prompt("hi {x}", {"x": 1}, name="X")
        b = Prompt("hi {x}", {"x": 1}, name="X")
        assert a == b
        assert hash(a) == hash(b)

    def test_different_variables_not_equal(self):
        """Different variables break equality even with the same template."""
        a = Prompt("hi {x}", {"x": 1}, name="X")
        b = Prompt("hi {x}", {"x": 2}, name="X")
        assert a != b

    def test_not_equal_to_string(self):
        """A Prompt never equals a bare string (identity matters)."""
        assert Prompt("hi", name="X") != "hi"


class TestVersioning:
    """Version numbers as observed from the Prompt object."""

    def test_version_assigned_on_use(self):
        """First use of a name registers version 1."""
        p = Prompt("hi {x}", name="VTEST")
        assert p.version == 1

    def test_same_text_same_version(self):
        """Identical text dedupes to one version regardless of variables."""
        p1 = Prompt("hi {x}", name="VTEST")
        p2 = Prompt("hi {x}", {"x": "different variables"}, name="VTEST")
        assert p1.version == p2.version == 1

    def test_changed_text_bumps_version(self):
        """Editing the template under the same name creates the next version."""
        p1 = Prompt("hi {x}", name="VTEST")
        p2 = Prompt("hi {x}!!", name="VTEST")
        assert p1.version == 1
        assert p2.version == 2

    def test_format_preserves_version(self):
        """.format() changes variables only, so the version is unchanged."""
        p1 = Prompt("hi {x}", {"x": 1}, name="VTEST")
        assert p1.version == 1
        assert p1.format(x=2).version == 1

    def test_disabled_tracking_gives_none(self, tmp_path):
        """With tracking off, rendering works and version is None."""
        promptkeep.configure(enabled=False)
        p = Prompt("hi {x}", name="VTEST")
        assert p.version is None
        assert p.text == "hi {x}"  # rendering still works

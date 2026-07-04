import json

import pytest

import prompt_manager
from prompt_manager import Prompt, RenderedText


class TestConstruction:
    def test_requires_name(self):
        with pytest.raises(ValueError, match="name"):
            Prompt("some text")

    def test_requires_nonempty_text(self):
        with pytest.raises(ValueError):
            Prompt("", name="X")
        with pytest.raises(ValueError):
            Prompt("   ", name="X")

    def test_variables_must_be_dict(self):
        with pytest.raises(TypeError):
            Prompt("hi {a}", variables=["a"], name="X")

    def test_matches_planned_signature(self):
        p = Prompt(text="xyz, {var1}", variables={"var1": "some value"}, name="REVIEW_SYSTEM")
        assert p.name == "REVIEW_SYSTEM"
        assert p.text == "xyz, some value"


class TestRenderingBehavior:
    def test_text_is_rendered(self):
        p = Prompt("Hello {who}", {"who": "world"}, name="GREET")
        assert p.text == "Hello world"

    def test_raw_keeps_placeholders(self):
        p = Prompt("Hello {who}", {"who": "world"}, name="GREET")
        assert p.raw == "Hello {who}"

    def test_text_is_a_real_string(self):
        p = Prompt("Hello {who}", {"who": "world"}, name="GREET")
        assert isinstance(p.text, str)
        assert json.dumps({"content": p.text}) == '{"content": "Hello world"}'
        assert p.text + "!" == "Hello world!"

    def test_text_carries_provenance(self):
        p = Prompt("Hello {who}", {"who": "world"}, name="GREET")
        rendered = p.text
        assert isinstance(rendered, RenderedText)
        assert rendered.prompt is p
        assert rendered.variables == {"who": "world"}

    def test_str_returns_rendered(self):
        p = Prompt("Hello {who}", {"who": "world"}, name="GREET")
        assert str(p) == "Hello world"
        assert f"{p}" == "Hello world"

    def test_render_with_overrides(self):
        p = Prompt("Hello {who}", {"who": "world"}, name="GREET")
        assert p.render(who="there") == "Hello there"
        assert p.text == "Hello world"  # original untouched

    def test_missing_variable_lenient_by_default(self):
        p = Prompt("Hello {who}", name="GREET")
        assert p.text == "Hello {who}"

    def test_per_prompt_strict(self):
        p = Prompt("Hello {who}", name="GREET", strict=True)
        with pytest.raises(prompt_manager.MissingVariableError):
            _ = p.text

    def test_global_strict_via_configure(self):
        prompt_manager.configure(strict=True)
        p = Prompt("Hello {who}", name="GREET")
        with pytest.raises(prompt_manager.MissingVariableError):
            _ = p.text

    def test_placeholders(self):
        p = Prompt("{a} and {b} and {{not_one}}", name="X")
        assert p.placeholders == {"a", "b"}


class TestImmutability:
    def test_setattr_raises(self):
        p = Prompt("hi", name="X")
        with pytest.raises(AttributeError):
            p.name = "other"
        with pytest.raises(AttributeError):
            p._template = "sneaky"

    def test_format_returns_new_prompt(self):
        p1 = Prompt("Hello {who}", {"who": "world"}, name="GREET")
        p2 = p1.format(who="there")
        assert p2 is not p1
        assert p2.text == "Hello there"
        assert p1.text == "Hello world"
        assert p2.name == p1.name
        assert p2.raw == p1.raw

    def test_variables_property_returns_copy(self):
        p = Prompt("Hello {who}", {"who": "world"}, name="GREET")
        p.variables["who"] = "mutated"
        assert p.text == "Hello world"


class TestEquality:
    def test_equal_prompts(self):
        a = Prompt("hi {x}", {"x": 1}, name="X")
        b = Prompt("hi {x}", {"x": 1}, name="X")
        assert a == b
        assert hash(a) == hash(b)

    def test_different_variables_not_equal(self):
        a = Prompt("hi {x}", {"x": 1}, name="X")
        b = Prompt("hi {x}", {"x": 2}, name="X")
        assert a != b

    def test_not_equal_to_string(self):
        assert Prompt("hi", name="X") != "hi"


class TestVersioning:
    def test_version_assigned_on_use(self):
        p = Prompt("hi {x}", name="VTEST")
        assert p.version == 1

    def test_same_text_same_version(self):
        p1 = Prompt("hi {x}", name="VTEST")
        p2 = Prompt("hi {x}", {"x": "different variables"}, name="VTEST")
        assert p1.version == p2.version == 1

    def test_changed_text_bumps_version(self):
        p1 = Prompt("hi {x}", name="VTEST")
        p2 = Prompt("hi {x}!!", name="VTEST")
        assert p1.version == 1
        assert p2.version == 2

    def test_format_preserves_version(self):
        p1 = Prompt("hi {x}", {"x": 1}, name="VTEST")
        assert p1.version == 1
        assert p1.format(x=2).version == 1

    def test_disabled_tracking_gives_none(self, tmp_path):
        prompt_manager.configure(enabled=False)
        p = Prompt("hi {x}", name="VTEST")
        assert p.version is None
        assert p.text == "hi {x}"  # rendering still works

"""Tests for the @prompt decorator: computed prompts and their versioning."""

import pytest

from promptkeep import Prompt, history, prompt


class TestDecorator:
    """Call mechanics: arguments become variables, return value becomes template."""

    def test_returns_prompt_object(self):
        """The pseudocode form from the design sketch produces a full Prompt."""

        @prompt(name="REVIEW_SYSTEM")
        def review_sys_prompt(var1="some value"):
            return "xyz, {var1}"

        p = review_sys_prompt()
        assert isinstance(p, Prompt)
        assert p.name == "REVIEW_SYSTEM"
        assert p.raw == "xyz, {var1}"
        assert p.text == "xyz, some value"

    def test_call_kwargs_become_variables(self):
        """Explicit call kwargs are recorded and used for rendering."""

        @prompt(name="DECO")
        def make(var1="default", tone="neutral"):
            return "say {var1} in a {tone} tone"

        p = make(var1="hello", tone="warm")
        assert p.text == "say hello in a warm tone"
        assert p.variables == {"var1": "hello", "tone": "warm"}

    def test_defaults_are_applied(self):
        """Unpassed parameters fall back to their declared defaults."""

        @prompt(name="DECO")
        def make(var1="default"):
            return "value: {var1}"

        assert make().text == "value: default"

    def test_positional_args_work(self):
        """Positional arguments bind to their parameter names as variables."""

        @prompt(name="DECO")
        def make(var1):
            return "value: {var1}"

        assert make("positional").text == "value: positional"

    def test_var_keyword_flattened(self):
        """**kwargs entries land as individual top-level variables."""

        @prompt(name="DECO")
        def make(**kwargs):
            return "a={a} b={b}"

        p = make(a=1, b=2)
        assert p.variables == {"a": 1, "b": 2}
        assert p.text == "a=1 b=2"

    def test_computation_in_body(self):
        """Real logic in the body works; {{...}} escapes survive the f-string."""

        @prompt(name="DECO")
        def make(n=3):
            bullets = "\n".join(f"- example {i}" for i in range(n))
            return f"Examples:\n{bullets}\nNow answer about {{topic}}."

        p = make(n=2)
        assert "- example 0\n- example 1" in p.raw
        assert "{topic}" in p.raw  # placeholder survives the f-string
        assert p.render(topic="cats").endswith("about cats.")

    def test_non_string_return_raises(self):
        """A decorated function must return a template string."""

        @prompt(name="DECO")
        def make():
            return 42

        with pytest.raises(TypeError, match="must return a template string"):
            make()

    def test_requires_name(self):
        """Bare @prompt (no name) is a usage error caught at decoration time."""
        with pytest.raises(TypeError, match="requires a name"):

            @prompt
            def make():
                return "hi"

    def test_wraps_preserves_function_identity(self):
        """functools.wraps keeps the function introspectable after decoration."""

        @prompt(name="DECO")
        def my_special_prompt():
            """docs live here"""
            return "hi"

        assert my_special_prompt.__name__ == "my_special_prompt"
        assert my_special_prompt.__doc__ == "docs live here"
        assert my_special_prompt.prompt_name == "DECO"


class TestDecoratorVersioning:
    """Lineage semantics when the template is computed at call time."""

    def test_stable_template_single_version(self):
        """Varying only the variables never creates new versions."""

        @prompt(name="DECOV")
        def make(var1="x"):
            return "fixed template with {var1}"

        make(var1="a").version
        make(var1="b").version
        assert len(history.versions("DECOV")) == 1

    def test_computed_template_change_creates_version(self):
        """Genuinely different computed text is a new version; repeats dedup."""

        @prompt(name="DECOV")
        def make(n=1):
            return "examples: " + ", ".join(str(i) for i in range(n))

        assert make(n=1).version == 1
        assert make(n=2).version == 2
        assert make(n=1).version == 1  # dedup back to v1

    def test_fn_source_hash_recorded(self):
        """Decorator versions carry a hash of the function's source code."""

        @prompt(name="DECOV")
        def make():
            return "hi"

        make().version
        (version,) = history.versions("DECOV")
        assert version.source == "decorator"
        assert version.fn_source_hash is not None
        assert len(version.fn_source_hash) == 64  # sha256 hex

    def test_literal_prompts_have_no_fn_hash(self):
        """Class-constructed prompts are marked 'literal' with no fn hash."""
        Prompt("hi", name="LIT").version
        (version,) = history.versions("LIT")
        assert version.source == "literal"
        assert version.fn_source_hash is None

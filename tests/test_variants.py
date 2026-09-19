"""Prompt.variants(): every stored version of a prompt, back as Prompt objects."""

import promptkeep
from promptkeep import Prompt, history, storage, wrap
from tests.fakes import FakeClient


def register(*templates, name="VAR_SYS", **kwargs):
    """Register each template as a version of ``name``, in order."""
    for template in templates:
        Prompt(template, name=name, **kwargs).version


class TestVariants:
    def test_every_version_oldest_first(self):
        register("Review {what}.", "Review {what} briefly.", "Review {what} in depth.")
        variants = Prompt.variants("VAR_SYS")
        assert [v.version for v in variants] == [1, 2, 3]
        assert [v.raw for v in variants] == [
            "Review {what}.",
            "Review {what} briefly.",
            "Review {what} in depth.",
        ]
        assert all(isinstance(v, Prompt) and v.name == "VAR_SYS" for v in variants)

    def test_unknown_name_is_empty(self):
        assert Prompt.variants("NEVER_DEFINED") == []

    def test_disabled_tracking_is_empty(self):
        register("Review {what}.")
        promptkeep.configure(enabled=False)
        assert Prompt.variants("VAR_SYS") == []

    def test_variants_carry_no_variables_until_formatted(self):
        Prompt("Review {what}.", {"what": "code"}, name="VAR_SYS").version
        (variant,) = Prompt.variants("VAR_SYS")
        assert variant.variables == {}
        assert str(variant.format(what="the diff").text) == "Review the diff."

    def test_loading_never_creates_a_version(self):
        register("Review {what}.", "Review {what} briefly.")
        for variant in Prompt.variants("VAR_SYS"):
            variant.text
            variant.format(what="x").version
        assert len(history.versions("VAR_SYS")) == 2

    def test_a_variant_knows_its_version_without_a_lookup(self, monkeypatch):
        register("Review {what}.", "Review {what} briefly.")
        old, _new = Prompt.variants("VAR_SYS")

        def no_lookups(*args, **kwargs):
            raise AssertionError("a variant must not re-register")

        monkeypatch.setattr(storage, "register_version", no_lookups)
        assert old.version == 1
        assert old.format(what="x").version == 1  # .format() keeps the registration

    def test_an_exact_match_version_round_trips(self):
        """{var1} -> {x} is a new version only under exact_match; a variant
        rebuilt without it would dedupe into v1 and lose its identity."""
        register("Review {var1}.", "Review {x}.", exact_match=True)
        first, second = Prompt.variants("VAR_SYS")
        assert (first.exact_match, second.exact_match) == (True, True)
        assert second.format(x="code").version == 2
        assert len(history.versions("VAR_SYS")) == 2

    def test_a_decorator_version_keeps_its_source(self):
        Prompt("Built {x}.", name="VAR_FN", source="decorator", fn_source_hash="abc123").version
        (variant,) = Prompt.variants("VAR_FN")
        assert (variant.source, variant.fn_source_hash) == ("decorator", "abc123")

    def test_runs_made_with_an_old_variant_are_filed_under_it(self):
        register("Review {what}.", "Review {what} briefly.")
        old = Prompt.variants("VAR_SYS")[0].format(what="code")
        wrap(FakeClient()).chat.completions.create(
            model="m", messages=[{"role": "system", "content": old}]
        )
        (run,) = history.runs("VAR_SYS")
        assert run.version == 1
        assert run.rendered_text == "Review code."

"""Prompt.variants() and Prompt.load(): stored versions of a prompt, back as Prompt objects."""

import pytest

import promptkeep
from promptkeep import (
    MissingVariableError,
    Prompt,
    PromptBlocked,
    Verdict,
    check,
    history,
    storage,
    wrap,
)
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


class TestLoad:
    """Prompt.load(): one stored version — pinned, or the latest."""

    def test_latest_by_default(self):
        register("Review {what}.", "Review {what} briefly.")
        loaded = Prompt.load("VAR_SYS")
        assert (loaded.version, loaded.raw) == (2, "Review {what} briefly.")

    def test_a_pinned_version(self):
        register("Review {what}.", "Review {what} briefly.")
        assert Prompt.load("VAR_SYS", version=1).raw == "Review {what}."

    def test_latest_is_the_highest_number_not_the_last_used(self):
        register("Review {what}.", "Review {what} briefly.", "Review {what}.")
        assert Prompt.load("VAR_SYS").version == 2

    def test_a_version_registered_elsewhere_is_picked_up_by_the_next_load(self):
        register("Review {what}.")
        assert Prompt.load("VAR_SYS").version == 1
        register("Review {what}, then summarise.")  # another process, a notebook, ...
        assert Prompt.load("VAR_SYS").raw == "Review {what}, then summarise."

    def test_an_unknown_name_or_version_raises(self):
        register("Review {what}.")
        with pytest.raises(ValueError, match="no stored versions of prompt 'NOPE'"):
            Prompt.load("NOPE")
        with pytest.raises(ValueError, match="has no version 7"):
            Prompt.load("VAR_SYS", version=7)

    def test_disabled_tracking_says_so(self):
        register("Review {what}.")
        promptkeep.configure(enabled=False)
        with pytest.raises(ValueError, match="tracking is disabled"):
            Prompt.load("VAR_SYS")

    def test_loading_and_using_never_creates_a_version(self, monkeypatch):
        register("Review {what}.", "Review {what} briefly.")
        loaded = Prompt.load("VAR_SYS", version=1)

        def no_lookups(*args, **kwargs):
            raise AssertionError("a loaded prompt must not re-register")

        monkeypatch.setattr(storage, "register_version", no_lookups)
        wrap(FakeClient()).chat.completions.create(
            model="m", messages=[{"role": "system", "content": loaded.format(what="code")}]
        )
        (run,) = history.runs("VAR_SYS")
        assert (run.version, run.rendered_text) == (1, "Review code.")

    def test_checks_and_strictness_are_given_at_load(self):
        register("Review {what}.")

        @check.pre(name="never")
        def never(ctx):
            return Verdict.block("not today")

        loaded = Prompt.load("VAR_SYS", strict=True, pre=[never])
        with pytest.raises(MissingVariableError):
            loaded.text
        with pytest.raises(PromptBlocked):
            wrap(FakeClient()).chat.completions.create(
                model="m", messages=[{"role": "system", "content": loaded.format(what="x")}]
            )

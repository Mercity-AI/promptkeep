"""Tests for the history API: versions listing, diffing, run queries."""

import pytest

from promptkeep import Prompt, history, tracking


def _run(prompt, **overrides):
    """Record a minimal run for a prompt, with per-test field overrides."""
    defaults = dict(provider="openai", model="gpt-test", output_text="out")
    defaults.update(overrides)
    tracking.record_prompt_run(prompt, prompt.variables, str(prompt.text), **defaults)


class TestVersions:
    """history.versions(): the lineage view."""

    def test_empty_for_unknown_prompt(self):
        """Unknown names yield an empty list, not an error."""
        assert history.versions("NOPE") == []

    def test_fields(self):
        """Every VersionInfo field is populated from the stored row."""
        Prompt("hello {x}", name="H").version
        (v,) = history.versions("H")
        assert v.version == 1
        assert v.template == "hello {x}"
        assert len(v.template_hash) == 64
        assert v.source == "literal"
        assert v.created_at  # ISO timestamp present


class TestDiff:
    """history.diff(): comparing two versions of a template."""

    def test_unified_diff_between_versions(self):
        """The diff shows removed/added lines with version-labeled headers."""
        Prompt("You are a reviewer.\nFocus on style.", name="D").version
        Prompt("You are a reviewer.\nFocus on correctness.", name="D").version
        d = history.diff("D", 1, 2)
        assert "-Focus on style." in d
        assert "+Focus on correctness." in d
        assert "D v1" in d and "D v2" in d

    def test_unknown_version_raises(self):
        """Asking for a nonexistent version is a caller error."""
        Prompt("hi", name="D").version
        with pytest.raises(ValueError, match="no version 9"):
            history.diff("D", 1, 9)


class TestRuns:
    """history.runs(): querying recorded executions."""

    def test_newest_first_and_limit(self):
        """Results come newest-first and respect the limit."""
        p = Prompt("hi {x}", name="R")
        for i in range(5):
            _run(p, output_text=f"out {i}")
        recorded = history.runs("R", limit=3)
        assert len(recorded) == 3
        assert recorded[0].output_text == "out 4"  # newest first

    def test_filter_by_version(self):
        """version= narrows runs to one template revision."""
        p1 = Prompt("hi v1", name="R")
        p2 = Prompt("hi v2", name="R")
        _run(p1)
        _run(p2)
        _run(p2)
        assert len(history.runs("R")) == 3
        assert len(history.runs("R", version=1)) == 1
        assert len(history.runs("R", version=2)) == 2

    def test_empty_for_unknown_prompt(self):
        """Unknown names yield an empty list, not an error."""
        assert history.runs("NOPE") == []

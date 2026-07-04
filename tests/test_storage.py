"""Tests for the storage layer: lineage semantics, concurrency, run recording."""

import threading

import promptkeep
from promptkeep import Prompt, history
from promptkeep import config as pm_config
from promptkeep import storage, tracking


class TestLineage:
    """Version registration: sequencing, dedup, isolation, concurrency."""

    def test_versions_are_sequential(self):
        """Each distinct template under one name gets the next version number."""
        for i in range(3):
            Prompt(f"template revision {i}", name="SEQ").version
        recorded = history.versions("SEQ")
        assert [v.version for v in recorded] == [1, 2, 3]
        assert recorded[0].template == "template revision 0"
        assert recorded[2].template == "template revision 2"

    def test_dedup_by_content_hash(self):
        """Registering identical text twice creates exactly one version."""
        Prompt("same text", name="DEDUP").version
        Prompt("same text", name="DEDUP").version
        assert len(history.versions("DEDUP")) == 1

    def test_dedup_survives_cache_reset(self, isolated_db):
        """Dedup relies on the DB, not the in-process cache."""
        assert Prompt("same text", name="DEDUP").version == 1
        # Simulate a new process: drop in-memory caches, keep the same DB file.
        storage.reset_caches()
        promptkeep.configure(db_path=isolated_db)
        assert Prompt("same text", name="DEDUP").version == 1
        assert Prompt("new text", name="DEDUP").version == 2

    def test_renamed_variable_is_same_version(self):
        """Renaming a placeholder ({var1} -> {x}) must not create a version."""
        assert Prompt("review this: {var1}", name="NORM").version == 1
        assert Prompt("review this: {x}", name="NORM").version == 1
        assert len(history.versions("NORM")) == 1
        # But changing the static text around it is a real new version.
        assert Prompt("review that: {x}", name="NORM").version == 2

    def test_repetition_pattern_is_a_real_difference(self):
        """Same static text, different variable structure -> distinct versions."""
        assert Prompt("{a} and then {a}", name="NORMREP").version == 1
        assert Prompt("{a} and then {b}", name="NORMREP").version == 2
        # Renaming either still dedups to its structural twin.
        assert Prompt("{z} and then {z}", name="NORMREP").version == 1

    def test_names_are_independent_lineages(self):
        """Two prompts sharing text but not name version independently."""
        assert Prompt("shared text", name="A").version == 1
        assert Prompt("shared text", name="B").version == 1
        assert Prompt("other text", name="A").version == 2
        assert len(history.versions("B")) == 1

    def test_disabled_creates_no_db_file(self, tmp_path):
        """Disabled tracking must do zero filesystem I/O."""
        pm_config.reset()
        promptkeep.configure(db_path=tmp_path / "nope.db", enabled=False)
        p = Prompt("hi {x}", {"x": 1}, name="X")
        assert p.text == "hi 1"
        assert p.version is None
        assert not (tmp_path / "nope.db").exists()

    def test_concurrent_registration_from_threads(self):
        """Eight threads registering distinct texts must produce versions 1..8
        with no errors — exercises the IMMEDIATE-transaction retry path."""
        errors = []

        def register(i):
            """Register one distinct template from a worker thread."""
            try:
                version = Prompt(f"threaded text {i}", name="THREADED").version
                assert version is not None
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=register, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        recorded = history.versions("THREADED")
        assert len(recorded) == 8
        assert sorted(v.version for v in recorded) == list(range(1, 9))


class TestRunRecording:
    """record_prompt_run -> runs table round-trips."""

    def test_record_and_read_back(self):
        """Every field written to a run row survives the read back intact."""
        p = Prompt("hi {x}", {"x": 1}, name="RUNS")
        tracking.record_prompt_run(
            p,
            {"x": 1},
            "hi 1",
            provider="openai",
            model="gpt-test",
            request_params={"temperature": 0.2},
            response_id="resp_42",
            output_text="hello back",
            prompt_tokens=11,
            completion_tokens=7,
            total_tokens=18,
            latency_ms=123,
        )
        (run,) = history.runs("RUNS")
        assert run.prompt_name == "RUNS"
        assert run.version == 1
        assert run.variables == {"x": 1}
        assert run.rendered_text == "hi 1"
        assert run.model == "gpt-test"
        assert run.request_params == {"temperature": 0.2}
        assert run.response_id == "resp_42"
        assert run.output_text == "hello back"
        assert run.total_tokens == 18
        assert run.latency_ms == 123
        assert run.status == "ok"
        assert run.error is None

    def test_non_json_variables_do_not_crash(self):
        """Unserializable variable values degrade to repr(), not exceptions."""
        p = Prompt("hi {x}", name="RUNS")
        tracking.record_prompt_run(
            p, {"x": object()}, "hi ...", provider="openai", model="gpt-test"
        )
        (run,) = history.runs("RUNS")
        assert "object" in run.variables["x"]  # stored via repr fallback

    def test_recording_failure_is_swallowed(self, monkeypatch):
        """A DB explosion during recording must never propagate to the caller."""
        p = Prompt("hi {x}", name="RUNS")

        def boom(**kwargs):
            """Stand-in for a storage layer that is completely broken."""
            raise RuntimeError("db exploded")

        monkeypatch.setattr(storage, "record_run", boom)
        # must not raise
        tracking.record_prompt_run(p, {}, "hi", provider="openai")

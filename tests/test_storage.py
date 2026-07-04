import threading

import prompt_manager
from prompt_manager import Prompt, history
from prompt_manager import config as pm_config
from prompt_manager import storage, tracking


class TestLineage:
    def test_versions_are_sequential(self):
        for i in range(3):
            Prompt(f"template revision {i}", name="SEQ").version
        recorded = history.versions("SEQ")
        assert [v.version for v in recorded] == [1, 2, 3]
        assert recorded[0].template == "template revision 0"
        assert recorded[2].template == "template revision 2"

    def test_dedup_by_content_hash(self):
        Prompt("same text", name="DEDUP").version
        Prompt("same text", name="DEDUP").version
        assert len(history.versions("DEDUP")) == 1

    def test_dedup_survives_cache_reset(self, isolated_db):
        assert Prompt("same text", name="DEDUP").version == 1
        # Simulate a new process: drop in-memory caches, keep the same DB file.
        storage.reset_caches()
        prompt_manager.configure(db_path=isolated_db)
        assert Prompt("same text", name="DEDUP").version == 1
        assert Prompt("new text", name="DEDUP").version == 2

    def test_names_are_independent_lineages(self):
        assert Prompt("shared text", name="A").version == 1
        assert Prompt("shared text", name="B").version == 1
        assert Prompt("other text", name="A").version == 2
        assert len(history.versions("B")) == 1

    def test_disabled_creates_no_db_file(self, tmp_path):
        pm_config.reset()
        prompt_manager.configure(db_path=tmp_path / "nope.db", enabled=False)
        p = Prompt("hi {x}", {"x": 1}, name="X")
        assert p.text == "hi 1"
        assert p.version is None
        assert not (tmp_path / "nope.db").exists()

    def test_concurrent_registration_from_threads(self):
        errors = []

        def register(i):
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
    def test_record_and_read_back(self):
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
        p = Prompt("hi {x}", name="RUNS")
        tracking.record_prompt_run(
            p, {"x": object()}, "hi ...", provider="openai", model="gpt-test"
        )
        (run,) = history.runs("RUNS")
        assert "object" in run.variables["x"]  # stored via repr fallback

    def test_recording_failure_is_swallowed(self, monkeypatch):
        p = Prompt("hi {x}", name="RUNS")

        def boom(**kwargs):
            raise RuntimeError("db exploded")

        monkeypatch.setattr(storage, "record_run", boom)
        # must not raise
        tracking.record_prompt_run(p, {}, "hi", provider="openai")

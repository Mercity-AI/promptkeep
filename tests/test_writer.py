"""Tests for the background writer: queuing, flush, drops, mode dispatch.

The autouse fixture pins write_mode="sync"; each test here opts into the
mode it exercises.
"""

import threading

import promptkeep
from promptkeep import Prompt, history, storage, tracking, writer


def _record(name="BG", text="hi"):
    """One run through the normal tracking path."""
    p = Prompt(text + " {x}", {"x": 1}, name=name)
    tracking.record_prompt_run(p, {"x": 1}, text + " 1", provider="openai", model="m")


class TestBackgroundMode:
    def test_rows_land_after_flush(self):
        """Queued rows are all on disk once flush() returns True."""
        promptkeep.configure(write_mode="background")
        for _ in range(5):
            _record()
        assert promptkeep.flush(timeout=5) is True
        assert len(history.runs("BG")) == 5

    def test_version_registration_stays_synchronous(self):
        """Prompt.version is a read-back value — background mode must not
        defer it."""
        promptkeep.configure(write_mode="background")
        assert Prompt("sync version {x}", name="BGV").version == 1

    def test_turn_indexes_stay_sequential_with_queued_rows(self):
        """The in-process counter, not the (stale) DB MAX, numbers turns:
        rapid turns recorded before anything is written must not collide."""
        promptkeep.configure(write_mode="background")
        cid = storage.get_or_create_conversation("bg-sess")
        for i in range(4):
            storage.record_run(provider="openai", conversation_id=cid, input_text=f"q{i}")
        assert promptkeep.flush(timeout=5) is True
        turns = [t.turn_index for t in history.conversation("bg-sess").turns]
        assert turns == [0, 1, 2, 3]

    def test_flush_without_any_writes_returns_immediately(self):
        assert promptkeep.flush(timeout=1) is True

    def test_worker_survives_a_broken_batch(self, monkeypatch):
        """A write failure loses that batch but the next one still lands."""
        promptkeep.configure(write_mode="background")
        original = storage.write_batch
        calls = {"n": 0}

        def flaky(rows):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("disk exploded")
            original(rows)

        monkeypatch.setattr(storage, "write_batch", flaky)
        _record(name="FLAKY")
        promptkeep.flush(timeout=5)
        _record(name="FLAKY")
        promptkeep.flush(timeout=5)
        assert calls["n"] >= 2
        assert len(history.runs("FLAKY")) == 1  # first batch lost, second landed


class TestOverflow:
    def test_full_queue_drops_and_counts(self, monkeypatch):
        """With the worker blocked, overflowing the queue drops rows and
        counts them instead of growing or raising."""
        promptkeep.configure(write_mode="background", queue_size=100)
        gate = threading.Event()
        original = storage.write_batch

        def blocked(rows):
            gate.wait(timeout=10)
            original(rows)

        monkeypatch.setattr(storage, "write_batch", blocked)
        before = writer.dropped_count()
        # Overfill: the worker can take at most one batch off the queue
        # while blocked, so >> queue_size submissions must drop some.
        for i in range(400):
            storage.record_run(provider="openai", version_id=None, status="ok")
        assert writer.dropped_count() > before
        gate.set()
        promptkeep.flush(timeout=10)


class TestOffMode:
    def test_off_drops_runs_but_keeps_versions(self):
        promptkeep.configure(write_mode="off")
        p = Prompt("off mode {x}", {"x": 1}, name="OFF")
        assert p.version == 1  # lineage still works
        tracking.record_prompt_run(p, {"x": 1}, "off mode 1", provider="openai")
        promptkeep.flush(timeout=2)
        assert history.runs("OFF") == []


class TestSyncMode:
    def test_sync_rows_exist_immediately(self):
        """The conftest default: no flush needed before reading back."""
        _record(name="SYNCED")
        assert len(history.runs("SYNCED")) == 1

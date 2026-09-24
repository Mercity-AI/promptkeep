"""Retention: configure(retention_days=...) and the sweep that enforces it."""

from datetime import UTC, datetime, timedelta

import pytest

import promptkeep
from promptkeep import Prompt, config, history, storage, tracking
from promptkeep.models import ConversationRecord, RunRecord


def _days_ago(days):
    """An ISO timestamp ``days`` in the past, in the format rows are stored in."""
    return (datetime.now(UTC) - timedelta(days=days)).isoformat()


def _backdate_run(run_key, days):
    """Pretend a run was recorded ``days`` ago."""
    RunRecord.update(created_at=_days_ago(days)).where(RunRecord.run_key == run_key).execute()


def _backdate_conversation(external_id, days):
    """Pretend a conversation was last active ``days`` ago."""
    ConversationRecord.update(updated_at=_days_ago(days)).where(
        ConversationRecord.external_id == external_id
    ).execute()


def _keys():
    """The run keys on file, oldest first."""
    return [run.run_key for run in reversed(history.all_runs(limit=None))]


class TestSetting:
    def test_off_by_default(self):
        assert config.get_settings().retention_days is None

    def test_configure_takes_a_positive_number_of_days(self):
        promptkeep.configure(retention_days=90)
        assert config.get_settings().retention_days == 90.0

    @pytest.mark.parametrize("bad", [0, -1, float("inf"), float("nan"), True, "90"])
    def test_anything_else_is_refused(self, bad):
        with pytest.raises(ValueError, match="retention_days"):
            promptkeep.configure(retention_days=bad)

    def test_the_environment_variable(self, monkeypatch):
        monkeypatch.setenv("PROMPTKEEP_RETENTION_DAYS", "30")
        assert config.get_settings().retention_days == 30.0

    @pytest.mark.parametrize("junk", ["", "soon", "0", "-5", "inf"])
    def test_a_bad_environment_value_keeps_everything(self, monkeypatch, junk):
        monkeypatch.setenv("PROMPTKEEP_RETENTION_DAYS", junk)
        assert config.get_settings().retention_days is None

    def test_configure_wins_over_the_environment(self, monkeypatch):
        monkeypatch.setenv("PROMPTKEEP_RETENTION_DAYS", "30")
        promptkeep.configure(retention_days=7)
        assert config.get_settings().retention_days == 7.0


class TestPrune:
    """storage.prune(): what goes, what stays."""

    def test_old_runs_go_with_their_labels_and_recent_ones_stay(self):
        old = storage.record_run(
            provider="openai",
            checks=[{"name": "gate", "phase": "pre", "status": "ok", "score": None}],
        )
        promptkeep.feedback(old, score=1.0, label="thumbs_up")
        recent = storage.record_run(provider="openai")
        _backdate_run(old, 91)

        assert storage.prune(90) == 1
        assert _keys() == [recent]
        assert history.checks(old) == []

    def test_prompts_and_versions_are_never_pruned(self):
        prompt = Prompt("Review {what}.", {"what": "code"}, name="KEEP_ME")
        key = tracking.record_prompt_run(prompt, {"what": "code"}, "Review code.", provider="x")
        _backdate_run(key, 400)
        storage.prune(1)
        assert history.runs("KEEP_ME") == []
        assert [v.version for v in history.versions("KEEP_ME")] == [1]

    def test_a_conversation_goes_whole_once_its_last_turn_is_old(self):
        cid = storage.get_or_create_conversation("stale")
        storage.record_run(provider="openai", conversation_id=cid, input_text="q0")
        storage.record_run(provider="openai", conversation_id=cid, input_text="q1")
        _backdate_conversation("stale", 100)
        assert storage.prune(90) == 2
        assert history.list_conversations() == []
        assert _keys() == []

    def test_a_conversation_still_in_use_keeps_its_old_turns(self):
        cid = storage.get_or_create_conversation("long-running")
        opening = storage.record_run(provider="openai", conversation_id=cid, input_text="q0")
        storage.record_run(provider="openai", conversation_id=cid, input_text="q1")
        _backdate_run(opening, 200)  # the session started long ago, but is active now
        assert storage.prune(90) == 0
        assert len(history.conversation("long-running").turns) == 2

    def test_a_run_whose_conversation_is_gone_goes_by_its_own_age(self):
        database = storage.get_db()
        database.execute_sql("PRAGMA foreign_keys = OFF")  # a file migrated without the FK
        try:
            orphan = storage.record_run(provider="openai", conversation_id=999, turn_index=0)
        finally:
            database.execute_sql("PRAGMA foreign_keys = ON")
        _backdate_run(orphan, 91)
        assert storage.prune(90) == 1

    def test_large_sweeps_are_chunked(self, monkeypatch):
        monkeypatch.setattr(storage, "_PRUNE_CHUNK", 2)
        for number in range(5):
            cid = storage.get_or_create_conversation(f"c{number}")
            storage.record_run(provider="openai", conversation_id=cid)
            _backdate_conversation(f"c{number}", 100)
            _backdate_run(storage.record_run(provider="openai"), 100)
        assert storage.prune(90) == 10
        assert _keys() == [] and history.list_conversations() == []

    def test_a_pruned_conversation_can_start_again_under_its_id(self):
        cid = storage.get_or_create_conversation("resumed")
        storage.record_run(provider="openai", conversation_id=cid, input_text="long ago")
        _backdate_conversation("resumed", 100)
        storage.prune(90)

        # The caches forgot it: the next turn opens a fresh conversation.
        cid = storage.get_or_create_conversation("resumed")
        storage.record_run(provider="openai", conversation_id=cid, input_text="today")
        (turn,) = history.conversation("resumed").turns
        assert (turn.turn_index, turn.input_text) == (0, "today")

    def test_a_turn_whose_conversation_vanished_is_kept_outside_it(self, caplog):
        """Another process pruned the conversation this one has cached: the
        turn is recorded unattached rather than lost, and the next turn
        starts the conversation afresh."""
        cid = storage.get_or_create_conversation("pruned-elsewhere")
        storage.record_run(provider="openai", conversation_id=cid)
        RunRecord.delete().execute()
        ConversationRecord.delete().execute()

        key = storage.record_run(provider="openai", conversation_id=cid, input_text="late")
        (run,) = history.all_runs()
        assert (run.run_key, run.conversation_id, run.turn_index) == (key, None, None)
        assert "was pruned" in caplog.text

        again = storage.get_or_create_conversation("pruned-elsewhere")
        storage.record_run(provider="openai", conversation_id=again, input_text="next")
        (turn,) = history.conversation("pruned-elsewhere").turns
        assert (turn.turn_index, turn.input_text) == (0, "next")


class TestSweep:
    """The sweep configure(retention_days=...) schedules on the write path."""

    def _count_sweeps(self, monkeypatch):
        """Replace prune with a counter; returns the list of its calls."""
        calls = []
        monkeypatch.setattr(storage, "prune", lambda days: calls.append(days) or 0)
        return calls

    def test_no_retention_no_sweep(self, monkeypatch):
        calls = self._count_sweeps(monkeypatch)
        storage.record_run(provider="openai")
        assert calls == []

    def test_the_first_run_sweeps_and_the_rest_of_the_hour_does_not(self, monkeypatch):
        promptkeep.configure(retention_days=30)
        calls = self._count_sweeps(monkeypatch)
        for _ in range(3):
            storage.record_run(provider="openai")
        assert calls == [30.0]

        # An hour on, the next run sweeps again.
        monkeypatch.setattr(storage, "_PRUNE_INTERVAL", 0.0)
        storage._next_prune.clear()
        storage.record_run(provider="openai")
        assert calls == [30.0, 30.0]

    def test_a_sweep_actually_deletes(self):
        old = storage.record_run(provider="openai")
        _backdate_run(old, 40)
        promptkeep.configure(retention_days=30)
        fresh = storage.record_run(provider="openai")
        assert _keys() == [fresh]

    def test_in_background_mode_the_writer_sweeps(self):
        old = storage.record_run(provider="openai")
        _backdate_run(old, 40)
        promptkeep.configure(retention_days=30, write_mode="background")
        fresh = storage.record_run(provider="openai")
        assert promptkeep.flush(timeout=5)
        assert _keys() == [fresh]

    def test_a_failing_sweep_costs_neither_the_run_nor_the_batch(self, monkeypatch, caplog):
        def broken(days):
            raise RuntimeError("database is locked")

        monkeypatch.setattr(storage, "prune", broken)
        promptkeep.configure(retention_days=30)
        key = storage.record_run(provider="openai")
        assert _keys() == [key]
        assert "retention sweep failed" in caplog.text

        storage._next_prune.clear()
        promptkeep.configure(write_mode="background")
        queued = storage.record_run(provider="openai")
        assert promptkeep.flush(timeout=5)
        assert _keys() == [key, queued]
        assert "failed to persist" not in caplog.text

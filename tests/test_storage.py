"""Tests for the storage layer: lineage semantics, concurrency, run recording."""

import sqlite3
import threading

import peewee as pw
import pytest

import promptkeep
from promptkeep import Prompt, history, migrations, storage, tracking
from promptkeep import config as pm_config


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

    def test_exact_match_opts_out_of_normalization(self):
        """exact_match=True makes placeholder names part of the identity."""
        assert Prompt("review this: {var1}", name="EXACT", exact_match=True).version == 1
        # Rename -> a genuinely new version under exact matching.
        assert Prompt("review this: {x}", name="EXACT", exact_match=True).version == 2
        # And re-registering an existing spelling still dedups to its row.
        assert Prompt("review this: {var1}", name="EXACT", exact_match=True).version == 1
        assert len(history.versions("EXACT")) == 2

    def test_exact_match_survives_format(self):
        """.format() derivatives keep the parent's matching mode."""
        p = Prompt("hi {a}", {"a": 1}, name="EXACTFMT", exact_match=True)
        assert p.format(a=2).exact_match is True
        assert p.format(a=2).version == p.version

    def test_names_are_independent_lineages(self):
        """Two prompts sharing text but not name version independently."""
        assert Prompt("shared text", name="A").version == 1
        assert Prompt("shared text", name="B").version == 1
        assert Prompt("other text", name="A").version == 2
        assert len(history.versions("B")) == 1

    def test_disabled_creates_no_db_file(self, tmp_path):
        """Disabled tracking must do zero filesystem I/O."""
        pm_config.reset()
        storage.reset_caches()
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


class TestSchemaMigration:
    """Upgrading a real pre-conversations (schema v2) database file."""

    def _build_v2_database(self, path):
        """Hand-build the exact table shapes schema v2 left behind, with one
        legacy row in each table — promptkeep hasn't touched this file yet
        (configure() is lazy), so this simulates a genuine existing DB."""
        raw = pw.SqliteDatabase(str(path), pragmas={"journal_mode": "wal", "foreign_keys": 1})
        raw.connect()
        raw.execute_sql(
            "CREATE TABLE prompts (id INTEGER PRIMARY KEY, name TEXT UNIQUE, created_at TEXT)"
        )
        raw.execute_sql(
            "CREATE TABLE prompt_versions (id INTEGER PRIMARY KEY, prompt_id INTEGER, "
            "version INTEGER, template TEXT, template_hash TEXT, source TEXT, "
            "fn_source_hash TEXT, created_at TEXT)"
        )
        raw.execute_sql(
            "CREATE TABLE runs (id INTEGER PRIMARY KEY, version_id INTEGER NOT NULL, "
            "variables TEXT, rendered_text TEXT NOT NULL, provider TEXT, model TEXT, "
            "request_params TEXT, response_id TEXT, output_text TEXT, prompt_tokens INTEGER, "
            "completion_tokens INTEGER, total_tokens INTEGER, latency_ms INTEGER, status TEXT, "
            "error TEXT, created_at TEXT)"
        )
        raw.execute_sql(
            "INSERT INTO prompts (id, name, created_at) VALUES (1, 'OLD', '2020-01-01')"
        )
        raw.execute_sql(
            "INSERT INTO prompt_versions (id, prompt_id, version, template, template_hash, "
            "source, created_at) VALUES (1, 1, 1, 'hi {v0}', 'abc', 'literal', '2020-01-01')"
        )
        raw.execute_sql(
            "INSERT INTO runs (id, version_id, rendered_text, provider, status, created_at) "
            "VALUES (1, 1, 'hi there', 'openai', 'ok', '2020-01-01')"
        )
        raw.execute_sql("PRAGMA user_version = 2")
        raw.close()

    def test_v2_database_gains_conversations_and_keeps_its_data(self, isolated_db):
        """The v3 step adds the conversations table and the new runs columns,
        drops NOT NULL on version_id/rendered_text, and the pre-existing row
        survives the table rebuild intact."""
        self._build_v2_database(isolated_db)

        # First real DB touch: triggers _migrate() against the pre-built file.
        assert history.runs("OLD")[0].rendered_text == "hi there"

        conn = sqlite3.connect(str(isolated_db))
        assert conn.execute("PRAGMA user_version").fetchone()[0] == migrations.SCHEMA_VERSION
        columns = {row[1] for row in conn.execute("PRAGMA table_info(runs)")}
        assert {"conversation_id", "turn_index", "input_text"} <= columns
        conn.close()

        # version_id/rendered_text are now nullable: a conversation-only turn
        # (no wrapped Prompt) must be legal on a migrated database too.
        cid = storage.get_or_create_conversation("post-migration")
        storage.record_run(provider="openai", conversation_id=cid, input_text="q", output_text="a")
        assert history.conversation("post-migration").turns[0].input_text == "q"

    def test_reopening_an_already_migrated_database_does_not_crash(self, isolated_db):
        """Regression test: the first draft of _migrate() never persisted
        PRAGMA user_version after the v3 step, so every reopen mistook an
        already-migrated file for a stale v2 one and re-ran add_column,
        crashing with a 'duplicate column' error."""
        self._build_v2_database(isolated_db)
        history.runs("OLD")  # first open: migrates

        storage.reset_caches()
        promptkeep.configure(db_path=isolated_db, enabled=True, strict=False)
        # Must not raise.
        assert history.runs("OLD")[0].rendered_text == "hi there"

    def test_v2_all_the_way_to_latest(self, isolated_db):
        """A v2 file migrates through every later step in one pass: the checks
        table appears, the legacy run gets a run_key, and it still reads back."""
        self._build_v2_database(isolated_db)
        (legacy,) = history.runs("OLD")
        assert legacy.rendered_text == "hi there"
        assert legacy.run_key  # backfilled by the v5 step
        assert legacy.original_input_text is None

        conn = sqlite3.connect(str(isolated_db))
        assert conn.execute("PRAGMA user_version").fetchone()[0] == migrations.SCHEMA_VERSION
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "checks" in tables
        run_cols = {r[1] for r in conn.execute("PRAGMA table_info(runs)")}
        assert {"run_key", "original_input_text"} <= run_cols
        check_cols = {r[1] for r in conn.execute("PRAGMA table_info(checks)")}
        assert "rewritten" in check_cols
        conn.close()

    def _build_v4_database(self, path):
        """The shape the checks branch first shipped (v4): a checks table
        without `rewritten`, runs without `run_key`, one verdict on file."""
        self._build_v2_database(path)
        raw = pw.SqliteDatabase(str(path), pragmas={"journal_mode": "wal", "foreign_keys": 1})
        raw.connect()
        raw.execute_sql(
            "CREATE TABLE conversations (id INTEGER PRIMARY KEY, external_id TEXT UNIQUE, "
            "title TEXT, metadata TEXT, created_at TEXT, updated_at TEXT)"
        )
        raw.execute_sql("ALTER TABLE runs ADD COLUMN conversation_id INTEGER")
        raw.execute_sql("ALTER TABLE runs ADD COLUMN turn_index INTEGER")
        raw.execute_sql("ALTER TABLE runs ADD COLUMN input_text TEXT")
        raw.execute_sql(
            "CREATE TABLE checks (id INTEGER PRIMARY KEY, run_id INTEGER NOT NULL, "
            "name TEXT, phase TEXT, status TEXT, score REAL, message TEXT, "
            "latency_ms INTEGER, created_at TEXT)"
        )
        raw.execute_sql(
            "INSERT INTO checks (run_id, name, phase, status, created_at) "
            "VALUES (1, 'legacy_gate', 'pre', 'ok', '2020-01-01')"
        )
        raw.execute_sql("PRAGMA user_version = 4")
        raw.close()

    def test_v4_database_gains_run_keys_and_the_rewrite_columns(self, isolated_db):
        """A v4 file (checks table already present, in its first shape) gets
        the v5 additions without tripping on the existing checks table, and
        its legacy verdict is reachable through the run's new key."""
        self._build_v4_database(isolated_db)
        (legacy,) = history.runs("OLD")
        assert legacy.run_key

        conn = sqlite3.connect(str(isolated_db))
        assert conn.execute("PRAGMA user_version").fetchone()[0] == migrations.SCHEMA_VERSION
        check_cols = {r[1] for r in conn.execute("PRAGMA table_info(checks)")}
        assert "rewritten" in check_cols
        # run_key is unique from here on.
        index_sql = " ".join(
            r[0] or "" for r in conn.execute("SELECT sql FROM sqlite_master WHERE type='index'")
        )
        assert "UNIQUE" in index_sql and "run_key" in index_sql
        # v6: the cost column arrives empty — old runs never recorded a cost.
        run_cols = {r[1] for r in conn.execute("PRAGMA table_info(runs)")}
        assert "cost_usd" in run_cols
        assert legacy.cost_usd is None
        # v7: chained calls look their predecessor up by response id.
        assert "response_id" in index_sql
        conn.close()

        (chk,) = history.checks(legacy.run_key)
        assert chk.name == "legacy_gate" and chk.rewritten is None

    def test_interrupted_step_rolls_back_whole(self, isolated_db, monkeypatch):
        """A step and its user_version bump commit together. If the v3 step
        dies partway (here: after adding one column), the transaction rolls it
        back entirely — no half-added columns, version stays 2 — and a clean
        retry migrates all the way rather than hitting 'duplicate column'."""
        import playhouse.migrate as pm

        self._build_v2_database(isolated_db)
        real_migrate = pm.migrate

        def die_after_first_op(*ops):
            real_migrate(ops[0])  # add conversation_id, then crash mid-step
            raise RuntimeError("interrupted migration")

        monkeypatch.setattr(pm, "migrate", die_after_first_op)
        with pytest.raises(Exception):
            history.runs("OLD")  # first touch triggers the (failing) migration

        # Nothing partial survived: still at v2, no leaked column.
        conn = sqlite3.connect(str(isolated_db))
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
        cols = {r[1] for r in conn.execute("PRAGMA table_info(runs)")}
        assert "conversation_id" not in cols
        conn.close()

        # A clean retry completes the migration.
        monkeypatch.undo()
        storage.reset_caches()
        promptkeep.configure(db_path=isolated_db, enabled=True, strict=False)
        assert history.runs("OLD")[0].rendered_text == "hi there"
        conn = sqlite3.connect(str(isolated_db))
        assert conn.execute("PRAGMA user_version").fetchone()[0] == migrations.SCHEMA_VERSION
        conn.close()

        # The checks table is usable: a run with bundled verdicts round-trips.
        run_key = storage.record_run(
            provider="openai",
            model="m",
            checks=[
                {
                    "name": "g",
                    "phase": "pre",
                    "status": "ok",
                    "score": None,
                    "message": None,
                    "latency_ms": 1,
                }
            ],
        )
        assert [c.name for c in history.checks(run_key)] == ["g"]


class TestLateVerdicts:
    """record_check / the writer's check items: verdicts that arrive after
    their run was recorded."""

    def _row(self, name="late"):
        return {
            "name": name,
            "phase": "post",
            "status": "ok",
            "score": 0.9,
            "message": None,
            "latency_ms": 3,
            "rewritten": None,
        }

    def test_late_verdict_attaches_to_its_run_by_key(self):
        run_key = storage.record_run(provider="openai", model="m")
        storage.record_check(run_key, self._row())
        (chk,) = history.checks(run_key)
        assert chk.name == "late" and chk.score == 0.9

    def test_verdict_for_a_never_persisted_run_is_skipped_not_raised(self, caplog):
        """The run may have been evicted on queue overflow. The verdict is
        dropped with a warning; nothing raises, and a batch carrying it still
        lands its other items."""
        with caplog.at_level("WARNING", logger="promptkeep"):
            storage.record_check("no-such-run", self._row())
        assert any("never persisted" in r.message for r in caplog.records)

        good = storage.record_run(provider="openai", model="m")
        storage.write_batch(
            [
                {"_kind": "check", "run_key": "no-such-run", "created_at": "t", **self._row()},
                {"_kind": "check", "run_key": good, "created_at": "t", **self._row("kept")},
            ]
        )
        assert [c.name for c in history.checks(good)] == ["kept"]

    def test_verdict_without_a_run_key_is_a_no_op(self):
        storage.record_check(None, self._row())  # tracking was off: nothing to attach to
        assert history.all_runs() == []


class TestChainConversation:
    """storage.chain_conversation: the conversation a call continuing a given
    response belongs in (the wrapper-level behavior is in test_responses.py)."""

    def _run(self, response_id, **fields):
        return storage.record_run(provider="openai-responses", response_id=response_id, **fields)

    def test_unknown_response_is_none(self):
        assert storage.chain_conversation("resp_never_seen") is None

    def test_first_link_starts_a_conversation_and_adopts_the_run(self):
        self._run("resp_1", output_text="a1")
        cid = storage.chain_conversation("resp_1")
        assert cid is not None
        assert storage.reserve_turn_index(cid) == 1  # turn 0 is the adopted run
        (turn,) = history.conversation("response:resp_1").turns
        assert (turn.turn_index, turn.output_text) == (0, "a1")

    def test_asking_twice_is_the_same_conversation_and_adopts_once(self):
        self._run("resp_1")
        assert storage.chain_conversation("resp_1") == storage.chain_conversation("resp_1")
        assert len(history.conversation("response:resp_1").turns) == 1

    def test_adoption_never_moves_a_run_already_in_a_conversation(self):
        mine = storage.get_or_create_conversation("mine")
        self._run("resp_1", conversation_id=mine)
        storage._adopt_run({"response_id": "resp_1", "conversation": mine + 1})
        assert len(history.conversation("mine").turns) == 1

    def test_the_in_process_index_is_bounded(self, monkeypatch):
        monkeypatch.setattr(storage, "_RESPONSE_INDEX_SIZE", 3)
        for number in range(10):
            self._run(f"resp_{number}")
        assert len(storage._response_index) == 3
        # An aged-out response is still found — on disk.
        assert storage.chain_conversation("resp_0") is not None

    def test_write_mode_off_leaves_no_conversation_behind(self):
        self._run("resp_1")
        promptkeep.configure(write_mode="off")
        assert storage.chain_conversation("resp_1") is None
        promptkeep.configure(write_mode="sync")
        assert history.list_conversations() == []

    def test_disabled_tracking_chains_nothing(self):
        promptkeep.configure(enabled=False)
        assert storage.chain_conversation("resp_1") is None

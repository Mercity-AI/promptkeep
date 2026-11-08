"""Storage, migration, history and retention coverage for agent traces."""

import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

import promptkeep
from promptkeep import Prompt, controls, history, migrations, storage, tracking, wrap
from promptkeep.models import (
    ConversationRecord,
    RunRecord,
    SpanRecord,
    new_run_key,
)
from tests.fakes import FakeClient, make_response


def _now(days=0):
    """An ISO timestamp relative to now, in the format storage uses."""
    return (datetime.now(UTC) - timedelta(days=days)).isoformat()


def _root(*, started_at=None, ended_at=None, status="ok", sampled=True, name="root"):
    """Record one complete root span and return its trace key."""
    key = new_run_key()
    storage.record_span(
        span_key=key,
        trace_key=key,
        parent_span_key=None,
        kind="agent",
        name=name,
        started_at=started_at or _now(),
        ended_at=ended_at,
        duration_ms=1,
        status=status,
        sampled=sampled,
    )
    return key


def _model(trace_key, parent_span_key, *, status="ok", started_at=None):
    """Record one model span under a trace and return its key."""
    key = new_run_key()
    storage.record_span(
        span_key=key,
        trace_key=trace_key,
        parent_span_key=parent_span_key,
        kind="llm",
        name="model",
        started_at=started_at or _now(),
        ended_at=_now(),
        duration_ms=1,
        status=status,
        sampled=True,
    )
    return key


def _run(span_key, **fields):
    """Record a run linked to a model span."""
    defaults = dict(provider="openai", model="m", output_text="out", span_key=span_key)
    defaults.update(fields)
    return storage.record_run(**defaults)


class TestSchema:
    def test_fresh_database_has_spans_and_the_run_link(self):
        database = storage.get_db()
        tables = {
            row[0]
            for row in database.execute_sql(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert "spans" in tables
        run_columns = {row[1] for row in database.execute_sql("PRAGMA table_info(runs)").fetchall()}
        assert "span_key" in run_columns
        assert migrations.SCHEMA_VERSION == 12

    def test_historical_runs_keep_a_null_span_link(self):
        key = storage.record_run(provider="openai")
        assert RunRecord.get(RunRecord.run_key == key).span_key is None

    def test_v8_migrates_to_v9_without_losing_runs(self, isolated_db):
        prompt = Prompt("v8 {x}", {"x": 1}, name="V8")
        key = tracking.record_prompt_run(
            prompt, {"x": 1}, "v8 1", provider="openai", output_text="kept"
        )
        assert key is not None

        storage.reset_caches()
        conn = sqlite3.connect(str(isolated_db))
        conn.execute("DROP TABLE spans")
        for index_row in conn.execute("PRAGMA index_list(runs)").fetchall():
            index_name = index_row[1]
            columns = [row[2] for row in conn.execute(f"PRAGMA index_info({index_name})")]
            if "span_key" in columns:
                conn.execute(f"DROP INDEX {index_name}")
        conn.execute("ALTER TABLE runs DROP COLUMN span_key")
        conn.execute("PRAGMA user_version = 8")
        conn.commit()
        conn.close()

        assert history.runs("V8")[0].output_text == "kept"
        conn = sqlite3.connect(str(isolated_db))
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 12
        columns = {row[1] for row in conn.execute("PRAGMA table_info(runs)")}
        assert "span_key" in columns
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'spans'"
        ).fetchone()
        conn.close()
        assert RunRecord.get(RunRecord.run_key == key).span_key is None

    def test_v2_legacy_path_reaches_the_span_schema(self, isolated_db):
        # The oldest supported fixture already runs every later step; assert
        # the new table and run column specifically, not just the version.
        from tests.test_storage import TestSchemaMigration

        TestSchemaMigration()._build_v2_database(isolated_db)
        assert history.runs("OLD")[0].rendered_text == "hi there"
        conn = sqlite3.connect(str(isolated_db))
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'spans'"
        ).fetchone()
        assert "span_key" in {row[1] for row in conn.execute("PRAGMA table_info(runs)")}
        conn.close()


class TestSpanUpsert:
    def test_final_row_restores_a_missing_start(self):
        trace_key = new_run_key()
        storage.record_span(
            span_key=trace_key,
            trace_key=trace_key,
            parent_span_key=None,
            kind="agent",
            name="restored",
            started_at=_now(days=1),
            ended_at=_now(),
            duration_ms=5,
            status="ok",
            sampled=True,
        )
        trace = history.trace(trace_key)
        assert trace.root.status == "ok"
        assert trace.root.name == "restored"

    def test_terminal_state_cannot_regress_to_running(self):
        trace_key = new_run_key()
        storage.record_span(
            span_key=trace_key,
            trace_key=trace_key,
            parent_span_key=None,
            kind="agent",
            name="terminal",
            started_at=_now(),
            ended_at=_now(),
            duration_ms=2,
            status="ok",
            sampled=True,
        )
        storage.record_span(
            span_key=trace_key,
            trace_key=trace_key,
            parent_span_key=None,
            kind="agent",
            name="terminal",
            started_at=_now(),
            status="running",
            sampled=True,
        )
        row = SpanRecord.get(SpanRecord.span_key == trace_key)
        assert row.status == "ok"
        assert row.ended_at is not None

    def test_explicit_null_and_omitted_payloads_are_distinct(self):
        trace_key = _root()
        omitted = new_run_key()
        explicit = new_run_key()
        storage.record_span(
            span_key=omitted,
            trace_key=trace_key,
            parent_span_key=trace_key,
            kind="tool",
            name="omitted",
            started_at=_now(),
            ended_at=_now(),
            status="ok",
        )
        storage.record_span(
            span_key=explicit,
            trace_key=trace_key,
            parent_span_key=trace_key,
            kind="tool",
            name="explicit",
            started_at=_now(),
            ended_at=_now(),
            status="ok",
            input_json="null",
            output_json="null",
        )
        by_name = {span.name: span for span in history.trace(trace_key).spans}
        assert by_name["omitted"].input is history._UNSET
        assert by_name["omitted"].output is history._UNSET
        assert by_name["omitted"].has_input is False
        assert by_name["omitted"].has_output is False
        assert by_name["explicit"].input is None
        assert by_name["explicit"].output is None
        assert by_name["explicit"].has_input is True
        assert by_name["explicit"].has_output is True

    def test_a_final_row_never_clears_a_stored_input(self):
        trace_key = _root()
        tool_key = new_run_key()
        storage.record_span(
            span_key=tool_key,
            trace_key=trace_key,
            parent_span_key=trace_key,
            kind="tool",
            name="once",
            started_at=_now(),
            status="running",
            input_json='"payload"',
        )
        storage.record_span(
            span_key=tool_key,
            trace_key=trace_key,
            parent_span_key=trace_key,
            kind="tool",
            name="once",
            started_at=_now(),
            ended_at=_now(),
            duration_ms=1,
            status="ok",
            input_json=None,
        )
        row = SpanRecord.get(SpanRecord.span_key == tool_key)
        assert row.status == "ok"
        assert row.input_json == '"payload"'

    def test_batch_writes_begin_immediately(self, monkeypatch):
        # _insert_span/_insert_check read before writing; a deferred batch
        # transaction would fail its read→write lock upgrade when another
        # connection commits in between, dropping the whole batch.
        database = storage.get_db().obj
        original = database.execute_sql
        statements = []

        def capture(sql, *args, **kwargs):
            statements.append(sql)
            return original(sql, *args, **kwargs)

        monkeypatch.setattr(database, "execute_sql", capture)
        storage.write_batch(
            [
                {
                    "_kind": "span",
                    "span_key": new_run_key(),
                    "trace_key": new_run_key(),
                    "parent_span_key": None,
                    "kind": "agent",
                    "name": "batch",
                    "started_at": _now(),
                    "ended_at": _now(),
                    "duration_ms": 1,
                    "status": "ok",
                    "input_json": None,
                    "output_json": None,
                    "error": None,
                    "sampled": True,
                }
            ]
        )
        assert any("BEGIN IMMEDIATE" in statement.upper() for statement in statements)


class TestTraceReads:
    def test_list_traces_is_newest_first_and_bounded(self):
        first = _root(started_at=_now(days=3), name="first")
        second = _root(started_at=_now(days=2), name="second")
        third = _root(started_at=_now(days=1), name="third")
        listed = history.list_traces(limit=2)
        assert [span.span_key for span in listed] == [third, second]
        assert history.list_traces(limit=0) == []
        assert history.list_traces(limit=-1) == []
        assert first not in [span.span_key for span in listed]

    def test_list_traces_does_not_load_root_payloads(self):
        with promptkeep.agent("listed", input={"large": "input"}) as task:
            task.output({"large": "output"})
        listed = history.list_traces()[0]
        assert listed.span_key == task.span_key
        assert listed.input is history._UNSET
        assert listed.output is history._UNSET
        loaded = history.trace(task.trace_key).root
        assert loaded.input == {"large": "input"}
        assert loaded.output == {"large": "output"}

    def test_list_traces_excludes_nested_agents(self):
        root = _root()
        nested = new_run_key()
        storage.record_span(
            span_key=nested,
            trace_key=root,
            parent_span_key=root,
            kind="agent",
            name="nested",
            started_at=_now(),
            ended_at=_now(),
            status="ok",
        )
        assert [span.span_key for span in history.list_traces()] == [root]

    def test_unknown_trace_and_disabled_reads_raise_or_return_empty(self):
        with pytest.raises(ValueError, match="no trace found"):
            history.trace("missing")
        promptkeep.configure(enabled=False)
        assert history.list_traces() == []
        with pytest.raises(ValueError, match="no trace found"):
            history.trace("missing")

    def test_missing_root_and_parent_are_readable(self):
        trace_key = new_run_key()
        child = new_run_key()
        storage.record_span(
            span_key=child,
            trace_key=trace_key,
            parent_span_key="never-written",
            kind="tool",
            name="orphan",
            started_at=_now(),
            ended_at=_now(),
            status="ok",
        )
        trace = history.trace(trace_key)
        assert trace.root is None
        assert trace.status is None
        assert trace.sampled is None
        (span,) = trace.spans
        assert span.parent_span_key == "never-written"
        assert span.span_key == child

    def test_missing_linked_run_is_marked_unavailable(self):
        root = _root()
        model = _model(root, root)
        trace = history.trace(root)
        llm = next(span for span in trace.spans if span.span_key == model)
        assert llm.runs == []
        assert llm.run_unavailable is True
        assert llm.primary_run is None
        assert trace.total_tokens == 0
        assert trace.total_cost is None
        assert trace.cost_complete is False

    def test_spans_are_ordered_by_start_with_stable_ties(self):
        root = _root(started_at="2020-01-01T00:00:00+00:00")
        later = new_run_key()
        earlier = new_run_key()
        storage.record_span(
            span_key=later,
            trace_key=root,
            parent_span_key=root,
            kind="tool",
            name="later",
            started_at="2020-01-01T00:00:02+00:00",
            ended_at="2020-01-01T00:00:03+00:00",
            status="ok",
        )
        storage.record_span(
            span_key=earlier,
            trace_key=root,
            parent_span_key=root,
            kind="tool",
            name="earlier",
            started_at="2020-01-01T00:00:01+00:00",
            ended_at="2020-01-01T00:00:02+00:00",
            status="ok",
        )
        names = [span.name for span in history.trace(root).spans]
        assert names == ["root", "earlier", "later"]

    def test_malformed_parent_cycle_does_not_hang(self):
        trace_key = new_run_key()
        first, second = new_run_key(), new_run_key()
        storage.record_span(
            span_key=first,
            trace_key=trace_key,
            parent_span_key=second,
            kind="tool",
            name="first",
            started_at=_now(),
            ended_at=_now(),
            status="ok",
        )
        storage.record_span(
            span_key=second,
            trace_key=trace_key,
            parent_span_key=first,
            kind="tool",
            name="second",
            started_at=_now(),
            ended_at=_now(),
            status="ok",
        )
        assert len(history.trace(trace_key).spans) == 2

    def test_primary_run_metrics_are_counted_once(self):
        root = _root()
        model = _model(root, root)
        _run(model, total_tokens=10, cost_usd=0.0)
        _run(model, total_tokens=99, cost_usd=None)
        trace = history.trace(root)
        assert trace.model_calls == 1
        assert trace.total_tokens == 10
        assert trace.total_cost == 0.0
        # The second run is the same physical call, not a second cost source.
        assert trace.cost_complete is True

    def test_unknown_cost_on_a_later_model_call_marks_incomplete(self):
        root = _root()
        first = _model(root, root)
        second = _model(root, root)
        _run(first, total_tokens=10, cost_usd=0.0)
        _run(second, total_tokens=5, cost_usd=None)
        trace = history.trace(root)
        assert trace.model_calls == 2
        assert trace.total_tokens == 15
        assert trace.total_cost == 0.0
        assert trace.cost_complete is False

    def test_known_zero_cost_is_complete(self):
        root = _root()
        model = _model(root, root)
        _run(model, total_tokens=3, cost_usd=0.0)
        trace = history.trace(root)
        assert trace.total_cost == 0.0
        assert trace.cost_complete is True

    def test_root_error_is_status_not_a_child_error_count(self):
        root = _root(status="error")
        child = new_run_key()
        storage.record_span(
            span_key=child,
            trace_key=root,
            parent_span_key=root,
            kind="tool",
            name="failed-tool",
            started_at=_now(),
            ended_at=_now(),
            status="error",
        )
        trace = history.trace(root)
        assert trace.status == "error"
        assert trace.error_count == 1  # the tool, not the root as well

    def test_no_model_calls_have_unknown_cost(self):
        trace = history.trace(_root())
        assert trace.model_calls == 0
        assert trace.total_tokens == 0
        assert trace.total_cost is None
        assert trace.cost_complete is False

    def test_duration_is_observed_wall_interval_not_a_sum(self):
        root = _root(
            started_at="2020-01-01T00:00:00+00:00",
            ended_at="2020-01-01T00:00:10+00:00",
        )
        model = _model(root, root, started_at="2020-01-01T00:00:02+00:00")
        SpanRecord.update(ended_at="2020-01-01T00:00:04+00:00").where(
            SpanRecord.span_key == model
        ).execute()
        assert history.trace(root).duration == 10.0

    def test_root_output_is_not_silently_replaced_by_model_text(self):
        root = _root()
        model = _model(root, root)
        _run(model, output_text="model text")
        trace = history.trace(root)
        assert trace.root.output is history._UNSET


class TestRetention:
    def test_whole_trace_expires_with_its_runs_checks_and_payloads(self):
        prompt = Prompt("retain {x}", {"x": 1}, name="RETAIN")
        client = wrap(FakeClient(response=make_response(content="answer")))
        with promptkeep.agent("old", input={"secret": "payload"}) as task:
            client.chat.completions.create(
                model="m",
                messages=[
                    {"role": "developer", "content": prompt},
                    {"role": "user", "content": "hi"},
                ],
            )
        (run,) = history.runs("RETAIN")
        storage.record_check(
            run.run_key,
            {"name": "audit", "phase": "post", "status": "ok", "message": "fine"},
        )
        SpanRecord.update(started_at=_now(days=100), ended_at=_now(days=100)).where(
            SpanRecord.trace_key == task.trace_key
        ).execute()
        RunRecord.update(created_at=_now(days=100)).where(
            RunRecord.run_key == run.run_key
        ).execute()

        assert storage.prune(90) == 1
        assert SpanRecord.select().where(SpanRecord.trace_key == task.trace_key).count() == 0
        assert history.runs("RETAIN") == []
        assert history.checks(run.run_key) == []
        assert [version.version for version in history.versions("RETAIN")] == [1]

    def test_a_recent_trace_keeps_its_older_linked_run(self):
        root = _root(started_at=_now(days=1))
        model = _model(root, root, started_at=_now(days=1))
        key = _run(model, total_tokens=5)
        RunRecord.update(created_at=_now(days=100)).where(RunRecord.run_key == key).execute()
        assert storage.prune(90) == 0
        assert history.runs("RETAIN_KEEP") == []  # no Prompt, but the row survived
        assert RunRecord.get(RunRecord.run_key == key).run_key == key

    def test_old_abandoned_running_root_expires(self):
        trace_key = new_run_key()
        storage.record_span(
            span_key=trace_key,
            trace_key=trace_key,
            parent_span_key=None,
            kind="agent",
            name="abandoned",
            started_at=_now(days=100),
            status="running",
            sampled=True,
        )
        assert storage.prune(90) == 0
        assert history.list_traces() == []

    def test_conversation_pruning_keeps_spans_readable_without_the_run(self):
        client = wrap(FakeClient())
        with promptkeep.conversation("session"):
            with promptkeep.agent("conversation") as task:
                client.chat.completions.create(
                    model="m", messages=[{"role": "user", "content": "hi"}]
                )
        ConversationRecord.update(updated_at=_now(days=100)).where(
            ConversationRecord.external_id == "session"
        ).execute()
        assert storage.prune(90) == 1
        trace = history.trace(task.trace_key)
        assert trace.root.status == "ok"
        (llm,) = [span for span in trace.spans if span.kind == "llm"]
        assert llm.run_unavailable is True
        assert llm.runs == []
        assert history.list_conversations() == []

    def test_prompt_and_version_history_survive_trace_expiry(self):
        prompt = Prompt("survive {x}", {"x": 1}, name="SURVIVE")
        root = _root(started_at=_now(days=100), ended_at=_now(days=100))
        model = _model(root, root, started_at=_now(days=100))
        SpanRecord.update(started_at=_now(days=100), ended_at=_now(days=100)).where(
            SpanRecord.span_key == model
        ).execute()
        key = tracking.record_prompt_run(
            prompt, {"x": 1}, "survive 1", provider="openai", span_key=model
        )
        RunRecord.update(created_at=_now(days=100)).where(RunRecord.run_key == key).execute()
        storage.prune(90)
        assert history.runs("SURVIVE") == []
        assert [version.version for version in history.versions("SURVIVE")] == [1]


class TestSamplingPolicy:
    def test_cohort_is_passed_through_but_problems_still_keep(self):
        promptkeep.configure(sample_rate=0.5)
        settings = promptkeep.get_settings()
        assert controls.keep_run(settings, {"status": "ok"}, None, cohort=False) is False
        assert controls.keep_run(settings, {"status": "error"}, None, cohort=False) is True
        warned = {"status": "ok", "_checks": [{"status": "warn"}]}
        assert controls.keep_run(settings, warned, None, cohort=False) is True

    def test_redaction_happens_before_a_span_reaches_the_queue(self, monkeypatch):
        from promptkeep import writer

        promptkeep.configure(
            write_mode="background", redact=lambda text: text.replace("secret", "[X]")
        )
        seen = []
        monkeypatch.setattr(writer, "submit", lambda item: seen.append(item))
        with promptkeep.agent("secret", input="secret in") as task:
            task.output("secret out")
        span_rows = [item for item in seen if item.get("_kind") == "span"]
        assert span_rows
        assert any(item["name"] == "secret" for item in span_rows)  # names stay structural
        assert any(item.get("input_json") == '"[X] in"' for item in span_rows)
        assert any(item.get("output_json") == '"[X] out"' for item in span_rows)
        assert all("secret" not in (item.get("input_json") or "") for item in span_rows)
        assert all("secret" not in (item.get("output_json") or "") for item in span_rows)

    def test_span_redaction_failure_never_stores_original(self):
        promptkeep.configure(redact=lambda text: (_ for _ in ()).throw(RuntimeError("boom")))
        with promptkeep.agent("secret") as task:
            task.output("secret")
        assert history.list_traces() == []
        assert SpanRecord.select().count() == 0

    def test_span_name_is_kept_and_error_is_redacted(self):
        promptkeep.configure(redact=lambda text: text.replace("secret", "[X]"))
        with pytest.raises(RuntimeError):
            with promptkeep.agent("secret agent") as task:
                raise RuntimeError("secret error")
        trace = history.trace(task.trace_key)
        assert trace.root.name == "secret agent"
        assert "[X] error" in trace.root.error


def test_trace_reads_are_batched(monkeypatch):
    """Reading many spans does not issue one query per span."""
    trace_key = _root()
    for _ in range(25):
        _run(_model(trace_key, trace_key))
    database = storage.get_db().obj
    original = database.execute_sql
    queries = []

    def count(sql, *args, **kwargs):
        queries.append(sql)
        return original(sql, *args, **kwargs)

    monkeypatch.setattr(database, "execute_sql", count)
    result = history.trace(trace_key)
    assert len(result.spans) == 26
    assert len(queries) <= 4

"""Native imports preserve source facts and replace snapshots atomically."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import asdict

import pytest
import zstandard

import promptkeep
from promptkeep import cli, history, storage
from promptkeep.importers import import_codex


def record(kind, payload, second=0):
    """One native rollout envelope with an original UTC timestamp."""
    return dict(type=kind, timestamp=f"2026-10-01T12:00:{second:02d}Z", payload=payload)


def session():
    """A turn with one failed tool, mirrored usage and explicit completion."""
    usage = dict(input_tokens=10, cached_input_tokens=3, output_tokens=2, total_tokens=12)
    return [
        record("session_meta", dict(id="session-1", cli_version="fixture", cwd="/private/secret")),
        record("event_msg", dict(type="task_started", turn_id="turn-1"), 1),
        record(
            "response_item",
            dict(
                type="message",
                role="user",
                content=[dict(type="input_text", text="secret question")],
            ),
            2,
        ),
        record(
            "response_item",
            dict(
                type="function_call",
                call_id="call-1",
                name="exec",
                arguments='{"command":"secret"}',
            ),
            3,
        ),
        record(
            "response_item",
            dict(
                type="function_call_output",
                call_id="call-1",
                output="Process exited with code 1\nOutput:\nsecret failure",
            ),
            5,
        ),
        record(
            "token_usage_record",
            dict(
                response_id="r-1",
                turn_id="turn-1",
                usage=usage,
                thread_token_usage=usage,
                turn_token_usage=usage,
            ),
            6,
        ),
        record("event_msg", dict(type="token_count", info=dict(total_token_usage=usage)), 6),
        record(
            "event_msg",
            dict(
                type="task_complete",
                turn_id="turn-1",
                started_at="2026-10-01T12:00:01Z",
                completed_at="2026-10-01T12:00:07Z",
                last_agent_message="finished",
            ),
            7,
        ),
    ]


def write(tmp_path, events=None, name="rollout.jsonl"):
    """Write a synthetic session; no private user traces enter fixtures."""
    path = tmp_path / name
    path.write_text(
        "".join(json.dumps(e) + "\n" for e in (session() if events is None else events))
    )
    return path


def test_native_repeat_and_compressed_import(tmp_path):
    path = write(tmp_path)
    report = import_codex(path)
    assert report.outcome == "created"
    assert (report.agent_spans, report.tool_spans, report.model_spans) == (2, 1, 1)
    trace = history.trace(report.trace_key)
    tool = next(s for s in trace.spans if s.kind == "tool")
    assert tool.status == "error"
    assert tool.duration_ms == 2000
    assert tool.started_at == "2026-10-01T12:00:03+00:00"
    assert trace.root.status == "ok"
    assert import_codex(path).outcome == "unchanged"
    compressed = tmp_path / "renamed.jsonl.zst"
    compressed.write_bytes(zstandard.ZstdCompressor().compress(path.read_bytes()))
    assert import_codex(compressed).outcome == "unchanged"
    summary = history.imported_session(report.trace_key)
    assert summary.summary["total_usage"]["total_tokens"] == 12
    assert len(summary.usage) == 3  # response, turn and thread; never summed together
    assert trace.total_tokens == 12  # identified response; timing remains unknown


def test_replacement_and_invalid_snapshot_preserve_previous(tmp_path):
    path = write(tmp_path)
    first = import_codex(path)
    events = session()
    events[-1]["payload"]["last_agent_message"] = "replacement"
    write(tmp_path, events)
    assert import_codex(path).outcome == "replaced"
    assert len(history.trace(first.trace_key).spans) == 4
    path.write_text(path.read_text() + "{broken\n")
    with pytest.raises(ValueError, match="invalid_json"):
        import_codex(path, strict=True)
    assert history.imported_session(first.trace_key).summary["malformed_records"] == 0
    tolerant = import_codex(path)
    assert tolerant.malformed_records == 1


def test_redaction_and_large_payload(tmp_path):
    promptkeep.configure(redact=lambda s: s.replace("secret", "[X]"))
    report = import_codex(write(tmp_path))
    trace = history.trace(report.trace_key)
    assert "secret" not in json.dumps(asdict(history.imported_session(report.trace_key)))
    tool = next(s for s in trace.spans if s.kind == "tool")
    assert "[X]" in tool.input
    events = session()
    events[4]["payload"]["output"] = "界" * 40000
    report = import_codex(write(tmp_path, events))
    tool = next(s for s in history.trace(report.trace_key).spans if s.kind == "tool")
    assert "promptkeep truncated" in tool.output


def test_redaction_failure_keeps_previous_import(tmp_path):
    path = write(tmp_path)
    report = import_codex(path)
    promptkeep.configure(redact=lambda _: (_ for _ in ()).throw(RuntimeError("hook failed")))
    with pytest.raises(RuntimeError):
        import_codex(path)
    assert len(history.trace(report.trace_key).spans) == 4


def test_explicit_destination_is_independent(tmp_path, isolated_db):
    target = tmp_path / "other.db"
    report = import_codex(write(tmp_path), db_path=target)
    assert promptkeep.get_settings().db_path == isolated_db
    with closing(sqlite3.connect(target)) as db, db:
        assert db.execute("SELECT COUNT(*) FROM spans").fetchone()[0] == 4
    assert not history.list_traces()
    assert report.outcome == "created"


def test_two_importers_produce_one_snapshot(tmp_path, isolated_db):
    path = write(tmp_path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: import_codex(path).outcome, range(2)))
    assert sorted(outcomes) == ["created", "unchanged"]
    assert len(history.list_traces()) == 1


def test_conflicting_call_identity_is_fatal(tmp_path):
    events = session()
    events.insert(
        4,
        record(
            "response_item",
            dict(type="function_call", call_id="call-1", name="different", arguments="changed"),
            4,
        ),
    )
    with pytest.raises(ValueError, match="conflicting_operation"):
        import_codex(write(tmp_path, events))
    assert not history.list_traces()


def test_import_cli_and_inspection(tmp_path, capsys):
    path = write(tmp_path)
    cli.main(["import", "codex", str(path), "--json"])
    report = json.loads(capsys.readouterr().out)
    cli.main(["traces"])
    assert report["trace_key"] in capsys.readouterr().out
    cli.main(["trace", report["trace_key"], "--json"])
    result = json.loads(capsys.readouterr().out)
    assert len(result["spans"]) == 4
    assert result["imported_session"]["summary"]["total_usage"]["total_tokens"] == 12


def test_expiry_removes_import_owned_records(tmp_path):
    report = import_codex(write(tmp_path))
    storage.prune(1)
    assert history.list_traces() == []
    with pytest.raises(ValueError):
        history.imported_session(report.trace_key)
    assert storage.get_db().execute_sql("SELECT COUNT(*) FROM imported_usage").fetchone()[0] == 0


def test_unknown_records_and_off_mode(tmp_path):
    path = write(tmp_path, session() + [record("future_record", {})])
    assert import_codex(path, strict=True).ignored_records >= 1
    promptkeep.configure(write_mode="off")
    with pytest.raises(ValueError, match="requires enabled"):
        import_codex(path)


def test_parallel_tools_partial_calls_and_opaque_records(tmp_path):
    events = session()
    events.insert(
        4,
        record(
            "response_item",
            dict(type="custom_tool_call", call_id="call-2", name="search", input="query"),
            4,
        ),
    )
    events.insert(
        5,
        record(
            "response_item",
            dict(
                type="custom_tool_call_output",
                call_id="call-2",
                output={"is_error": True, "content": "failed"},
            ),
            6,
        ),
    )
    events.insert(
        6,
        record(
            "response_item",
            dict(type="function_call", call_id="unfinished", name="unfinished", arguments="{}"),
            6,
        ),
    )
    events.append(
        record(
            "response_item",
            dict(type="function_call_output", call_id="orphan", output="unmatched"),
            8,
        )
    )
    events.append(
        record("response_item", dict(type="reasoning", encrypted_content="never-store"), 8)
    )
    report = import_codex(write(tmp_path, events))
    tools = [s for s in history.trace(report.trace_key).spans if s.kind == "tool"]
    assert len(tools) == 4
    assert {s.name: s.status for s in tools}["unfinished"] == "incomplete"
    assert sum(s.status == "error" for s in tools) == 2
    orphan = next(s for s in tools if s.name == "codex.tool")
    assert orphan.status == "incomplete"
    assert "never-store" not in json.dumps(asdict(history.imported_session(report.trace_key)))


def test_completed_items_are_deduplicated_and_explicitly_timed(tmp_path):
    events = session()
    events.insert(
        5,
        record(
            "event_msg",
            dict(
                type="item_completed",
                turn_id="turn-1",
                started_at_ms=1790856003000,
                completed_at_ms=1790856005000,
                item=dict(
                    type="CommandExecution", id="call-1", aggregated_output="completed", exit_code=1
                ),
            ),
            5,
        ),
    )
    events.insert(
        3,
        record(
            "event_msg",
            dict(
                type="item_completed",
                turn_id="turn-1",
                item=dict(
                    type="UserMessage",
                    id="user-1",
                    content=[dict(type="text", text="secret question")],
                ),
            ),
            2,
        ),
    )
    report = import_codex(write(tmp_path, events))
    trace = history.trace(report.trace_key)
    assert report.agent_spans == 2 and report.tool_spans == 1
    tool = next(s for s in trace.spans if s.kind == "tool")
    assert tool.metadata["timing_quality"] == "explicit"
    assert tool.duration_ms == 2000


def test_missing_metadata_corrupt_compression_and_bad_shapes(tmp_path):
    with pytest.raises(ValueError, match="missing_session_metadata"):
        import_codex(write(tmp_path, []))
    invalid = tmp_path / "bad.jsonl.zst"
    invalid.write_bytes(b"not zstd")
    with pytest.raises(zstandard.ZstdError):
        import_codex(invalid)
    events = session() + [
        dict(type="response_item", payload="bad"),
        record("turn_context", {"turn_id": []}),
    ]
    report = import_codex(write(tmp_path, events))
    assert report.malformed_records >= 1
    with pytest.raises(ValueError):
        import_codex(write(tmp_path, events), strict=True)


def test_usage_legacy_snapshots_are_not_summed(tmp_path):
    events = [e for e in session() if e["type"] != "token_usage_record"]
    events.append(
        record(
            "event_msg",
            dict(
                type="token_count",
                info=dict(
                    total_token_usage={"input_tokens": 20, "output_tokens": 4, "total_tokens": 24}
                ),
            ),
            8,
        )
    )
    report = import_codex(write(tmp_path, events))
    imported = history.imported_session(report.trace_key)
    assert imported.summary["total_usage"]["total_tokens"] == 24
    assert len(imported.usage) == 1


def test_import_destination_is_pinned_across_configuration_change(tmp_path, isolated_db):
    other = tmp_path / "changed.db"

    def redact(text):
        promptkeep.configure(db_path=other)
        return text

    promptkeep.configure(redact=redact)
    report = import_codex(write(tmp_path))
    with closing(sqlite3.connect(isolated_db)) as db, db:
        assert (
            db.execute("SELECT trace_key FROM imported_sessions").fetchone()[0] == report.trace_key
        )
    assert not other.exists()


def test_failed_destination_write_rolls_back_replacement(tmp_path, isolated_db):
    path = write(tmp_path)
    report = import_codex(path)
    # A real database rejection after deletion must roll back the whole replacement.
    with closing(sqlite3.connect(isolated_db)) as db, db:
        db.execute(
            "CREATE TRIGGER reject_import BEFORE INSERT ON spans "
            "BEGIN SELECT RAISE(ABORT, 'test failure'); END"
        )
    events = session()
    events[-1]["payload"]["last_agent_message"] = "different snapshot"
    with pytest.raises(Exception, match="test failure"):
        import_codex(write(tmp_path, events))
    trace = history.trace(report.trace_key)
    assert next(s for s in trace.spans if s.name == "codex.turn").output == "finished"


def test_large_import_does_not_use_background_queue(tmp_path):
    promptkeep.configure(write_mode="background", queue_size=1)
    events = session()[:3]
    for number in range(1500):
        events.append(
            record(
                "response_item",
                dict(type="function_call", call_id=f"c{number}", name="tool", arguments="{}"),
                3,
            )
        )
        events.append(
            record(
                "response_item",
                dict(type="function_call_output", call_id=f"c{number}", output="ok"),
                4,
            )
        )
    report = import_codex(write(tmp_path, events))
    assert report.tool_spans == 1500
    assert len(history.trace(report.trace_key).spans) == 1502


def test_legacy_turn_boundaries_do_not_merge_user_turns(tmp_path):
    events = [session()[0]]
    for index in range(2):
        events.extend(
            [
                record("event_msg", dict(type="task_started"), index * 4 + 1),
                record(
                    "response_item",
                    dict(
                        type="message",
                        role="user",
                        content=[dict(type="input_text", text=f"question {index}")],
                    ),
                    index * 4 + 2,
                ),
                record(
                    "event_msg",
                    dict(type="task_complete", last_agent_message=f"answer {index}"),
                    index * 4 + 3,
                ),
            ]
        )
    report = import_codex(write(tmp_path, events))
    turns = [s for s in history.trace(report.trace_key).spans if s.name == "codex.turn"]
    assert len(turns) == 2
    assert {s.input for s in turns} == {"question 0", "question 1"}
    assert {s.output for s in turns} == {"answer 0", "answer 1"}


def test_v9_upgrade_preserves_existing_history(tmp_path, isolated_db):
    prompt = promptkeep.Prompt("old prompt", name="OLD")
    storage.record_run(
        version_id=prompt._ensure_registered()[0], provider="old", output_text="kept"
    )
    storage.reset_caches()
    with closing(sqlite3.connect(isolated_db)) as db, db:
        db.execute("DROP TABLE imported_usage")
        db.execute("DROP TABLE imported_sessions")
        db.execute("ALTER TABLE runs DROP COLUMN usage_details")
        db.execute("ALTER TABLE spans DROP COLUMN metadata_json")
        db.execute("PRAGMA user_version=9")
    report = import_codex(write(tmp_path))
    assert history.runs("OLD")[0].output_text == "kept"
    assert history.imported_session(report.trace_key).source == "codex"
    assert storage.get_db().execute_sql("PRAGMA user_version").fetchone()[0] == 12


def test_same_usage_and_call_ids_in_different_sessions_are_distinct(tmp_path):
    first = import_codex(write(tmp_path))
    events = session()
    events[0]["payload"]["id"] = "session-2"
    second = import_codex(write(tmp_path, events))
    assert first.trace_key != second.trace_key
    assert len(history.list_traces()) == 2
    assert history.imported_session(first.trace_key).summary["total_usage"]["total_tokens"] == 12
    assert history.imported_session(second.trace_key).summary["total_usage"]["total_tokens"] == 12


def test_archive_reports_invalid_observed_intervals(tmp_path):
    events = session()
    events[4]["timestamp"] = "2026-10-01T12:00:01Z"  # Tool completes before its recorded start.
    report = import_codex(write(tmp_path, events))
    tool = next(s for s in history.trace(report.trace_key).spans if s.kind == "tool")
    assert tool.duration_ms is None
    assert any(d.code == "negative_interval" for d in report.diagnostics)


def test_terminal_json_envelope_and_absent_timestamps(tmp_path):
    events = session()
    events[4]["payload"]["output"] = json.dumps({"output": "failed", "metadata": {"exit_code": 2}})
    events[2].pop("timestamp")  # The turn already has an observed start.
    report = import_codex(write(tmp_path, events))
    trace = history.trace(report.trace_key)
    assert next(s for s in trace.spans if s.kind == "tool").status == "error"
    assert next(s for s in trace.spans if s.name == "codex.turn").input == "secret question"

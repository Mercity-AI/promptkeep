"""Source-derived regression fixtures and third-party adapter contract checks."""

import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import pytest

from promptkeep import cli, history, storage
from promptkeep.events import key
from promptkeep.importers import (
    AgentTrace,
    ClaudeCodeTrace,
    CodexTrace,
    OpenCodeTrace,
    TraceEvent,
    import_codex,
    import_trace,
    register_agent_trace,
)
from promptkeep.watch import Watcher
from tests.test_codex_import import write
from tests.test_remaining import claude_records, settle

FIXTURES = Path(__file__).parent / "fixtures" / "agent_traces"


@pytest.mark.parametrize("variant", ["legacy", "modern"])
def test_opencode_export_formats(tmp_path, variant):
    """Both upstream formats preserve two models, one failure and visible text only."""
    report = import_trace(FIXTURES / f"opencode-{variant}.json")
    trace = history.trace(report.trace_key)
    assert (report.agent_spans, report.model_spans, report.tool_spans) == (2, 2, 1)
    assert {r.model for s in trace.spans for r in s.runs} == {"model-a", "model-b"}
    assert trace.total_tokens == 41
    assert trace.total_cost is None
    assert trace.root.links == [key("opencode", "opencode-parent", "opencode-parent")]
    tool = next(s for s in trace.spans if s.kind == "tool")
    assert tool.status == "error" and tool.duration_ms == 1000
    assert next(s for s in trace.spans if s.span_key == tool.parent_span_key).kind == "llm"
    assert all(
        s.metadata["cost_provenance"] == "harness_reported" for s in trace.spans if s.kind == "llm"
    )
    assert "PRIVATE REASONING" not in json.dumps(asdict(trace), default=str)
    assert "/excluded/private" not in json.dumps(asdict(trace), default=str)
    assert import_trace(FIXTURES / f"opencode-{variant}.json").outcome == "unchanged"


def test_export_watch_rejects_partial_replacement(tmp_path):
    """Only complete validated snapshots advance history and checkpoint state."""
    path = tmp_path / "export.json"
    shutil.copyfile(FIXTURES / "opencode-modern.json", path)
    with Watcher(path, interval=0.02) as watcher:
        settle(lambda: bool(history.list_traces()))
        root = history.list_traces()[0].span_key
        old = history.imported_session(root).summary
        path.write_text('{"info":')
        time.sleep(0.08)
        assert history.imported_session(root).summary == old
        replacement = json.loads((FIXTURES / "opencode-modern.json").read_text())
        replacement["messages"] = replacement["messages"][:2]
        path.write_text(json.dumps(replacement))
        settle(lambda: history.trace(root).model_calls == 1)
    assert watcher.failed_count == 0


class TinyTrace(AgentTrace):
    """A third-party source can import and watch using only public lifecycle helpers."""

    source = "contract-tiny"
    revision = 1

    @classmethod
    def detect(cls, record):
        """Identify only this fixture's explicit format marker."""
        return isinstance(record, dict) and record.get("format") == "contract-tiny"

    def consume(self, record, line):
        """Interpret a source record without storage/SQL or built-in parser imports."""
        self.session = record["session"]
        event = TraceEvent(
            self.source,
            self.session,
            self.session,
            "tiny",
            kind="agent",
            status="ok",
            started_at=record["at"],
            ended_at=record["at"],
            input=record["text"],
        )
        self.trace, self.first, self.last = event.trace_key, record["at"], record["at"]
        self.stage_event(event)


register_agent_trace(TinyTrace)


def test_third_party_registration_archive_watch_and_checkpoint(tmp_path):
    """Dispatch needs no edits when an independently registered source is added."""
    assert (
        CodexTrace.__bases__
        == ClaudeCodeTrace.__bases__
        == OpenCodeTrace.__bases__
        == (AgentTrace,)
    )
    value = dict(
        format="contract-tiny", session="tiny", at="2026-10-01T00:00:00+00:00", text="hello"
    )
    path = write(tmp_path, [value])
    report = import_trace(path)
    assert history.trace(report.trace_key).root.input == "hello"
    with Watcher(path, interval=0.02) as watcher:
        settle(lambda: history.trace(report.trace_key).root.status == "running")
    assert watcher.failed_count == 0
    with Watcher(path, interval=0.02) as watcher:
        with path.open("a") as stream:
            stream.write(json.dumps(value | {"text": "updated"}) + "\n")
        settle(lambda: history.trace(report.trace_key).root.input == "updated")
    assert watcher.failed_count == 0


def test_auto_cli_mixed_directory_and_unknown_content(tmp_path, capsys):
    """Each watched file selects its own source from content."""
    codex = write(tmp_path, name="codex.jsonl")
    write(tmp_path, claude_records(), name="claude.jsonl")
    shutil.copyfile(FIXTURES / "opencode-modern.json", tmp_path / "export.json")
    cli.main(["import", str(codex), "--json"])
    assert json.loads(capsys.readouterr().out)["model_spans"] == 1
    with Watcher(tmp_path, interval=0.02) as watcher:
        settle(lambda: len(history.list_traces()) == 3)
    assert watcher.failed_count == 0
    path = write(tmp_path, [dict(secret="never echo this")])
    with pytest.raises(ValueError, match="unknown trace format") as error:
        import_trace(path)
    assert "secret" not in str(error.value)


def test_revision_rebuild_reconciles_obsolete_owned_spans(tmp_path, isolated_db):
    """A corrected parser replaces only its owned snapshot on checkpoint upgrade."""
    path = write(tmp_path)
    report = import_codex(path)
    with Watcher(path, source="codex", interval=0.02):
        settle(lambda: history.trace(report.trace_key).root.status == "running")
    with storage.connection_context():
        db = storage.get_db()
        db.execute_sql("UPDATE watch_checkpoints SET parser_revision=0")
        db.execute_sql(
            "INSERT INTO spans(span_key,trace_key,kind,name,status) VALUES (?,?,?,?,?)",
            ("obsolete", report.trace_key, "agent", "wrong-turn", "ok"),
        )
    with Watcher(path, interval=0.02):
        settle(lambda: all(s.name != "wrong-turn" for s in history.trace(report.trace_key).spans))
    assert import_trace(path).model_spans == 1


@pytest.mark.parametrize(
    "name", ["codex-seconds.jsonl", "opencode-modern.json", "opencode-legacy.json"]
)
def test_archive_and_watch_complete_source_agree(tmp_path, name):
    """Delivery modes preserve the same facts for a complete original source."""
    path = tmp_path / name
    shutil.copyfile(FIXTURES / name, path)
    report = import_trace(path)

    def facts():
        """Compare source facts while excluding physical database row IDs."""
        return sorted(
            (
                span.span_key,
                span.kind,
                span.name,
                span.started_at,
                span.ended_at,
                span.duration_ms,
                span.status,
                span.primary_run.model if span.primary_run else None,
                span.primary_run.total_tokens if span.primary_run else None,
            )
            for span in history.trace(report.trace_key).spans
        )

    expected = facts()
    with Watcher(path, interval=0.02) as watcher:
        if path.suffix == ".jsonl":
            settle(lambda: history.trace(report.trace_key).root.status == "running")
        else:
            settle(lambda: bool(watcher._snapshots))
    assert watcher.failed_count == 0
    assert facts() == expected

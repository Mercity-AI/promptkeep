"""Regression checks for native timestamp, usage and turn classification facts."""

from pathlib import Path

from promptkeep import history
from promptkeep.importers import import_claude_code, import_codex, import_trace
from tests.test_codex_import import session, write
from tests.test_remaining import claude_records

FIXTURES = Path(__file__).parent / "fixtures" / "agent_traces"


def test_codex_seconds_duration_switches_repeats_and_resets():
    """Reviewed counts: one turn, three observations, 42 tokens across two epochs."""
    report = import_trace(FIXTURES / "codex-seconds.jsonl")
    trace = history.trace(report.trace_key)
    turn = next(s for s in trace.spans if s.name == "codex.turn")
    assert turn.started_at == "2026-10-01T12:00:01+00:00"
    assert turn.ended_at == "2026-10-01T12:01:09+00:00"
    assert turn.duration_ms == 68000
    models = sorted((s for s in trace.spans if s.kind == "llm"), key=lambda s: s.observed_at)
    assert [s.primary_run.model for s in models] == ["first-model", "second-model", "second-model"]
    assert [s.primary_run.total_tokens for s in models] == [12, 24, 6]
    assert all(
        s.primary_run.response_id is None and s.started_at is None and s.duration_ms is None
        for s in models
    )
    summary = history.imported_session(report.trace_key).summary
    assert summary["total_usage"]["total_tokens"] == 42
    assert summary["response_usage"]["total_tokens"] == 42
    assert summary["unattributed_usage"] == {}
    assert any(d.code == "usage_accounting_reset" for d in report.diagnostics)


def test_duration_disagreement_preserves_explicit_facts(tmp_path):
    """An invalid source interval never silently changes its recorded duration."""
    events = session()
    events[-1]["payload"].update(started_at=1790856007, completed_at=1790856001, duration_ms=68000)
    report = import_codex(write(tmp_path, events))
    turn = next(s for s in history.trace(report.trace_key).spans if s.name == "codex.turn")
    assert turn.duration_ms == 68000
    assert turn.ended_at < turn.started_at
    assert any(d.code == "negative_interval" for d in report.diagnostics)
    events[-1]["payload"]["completed_at"] = 1790856010
    report = import_codex(write(tmp_path, events))
    assert any(d.code == "duration_timestamp_disagreement" for d in report.diagnostics)


def test_claude_classifies_envelopes_and_completes_refusals(tmp_path):
    """Injected records do not become prompts; quoted marker text remains a prompt."""
    events = claude_records()
    template = events[0]
    extra = []
    for index, (content, flags) in enumerate(
        [
            ("<local-command-stdout>done</local-command-stdout>", {}),
            ("context", {"isMeta": True}),
            ("summary", {"isCompactSummary": True}),
        ]
    ):
        extra.append(
            {
                **template,
                "uuid": f"injected-{index}",
                "message": {"role": "user", "content": content},
                **flags,
            }
        )
    quoted = {
        **template,
        "uuid": "quoted",
        "parentUuid": "b",
        "message": {"role": "user", "content": "Explain [Request interrupted by user] please"},
    }
    refusal = {
        **events[-1],
        "uuid": "refusal",
        "parentUuid": "quoted",
        "message": dict(
            id="refusal-response",
            model="claude",
            content=[dict(type="refusal", refusal="declined")],
            stop_reason="refusal",
            usage=dict(input_tokens=1, output_tokens=1),
        ),
    }
    report = import_claude_code(
        write(tmp_path, events[:1] + extra + events[1:] + [quoted, refusal])
    )
    trace = history.trace(report.trace_key)
    assert len([s for s in trace.spans if s.name == "claude-code.turn"]) == 2
    assert len([s for s in trace.spans if s.kind == "operation"]) == 3
    model = next(
        s for s in trace.spans if s.primary_run and s.primary_run.response_id == "refusal-response"
    )
    assert model.metadata["refusal"] is True
    assert trace.root.status == "ok"

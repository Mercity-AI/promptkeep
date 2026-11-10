"""External integration end-to-end contracts, using real SDKs without paid calls."""

import asyncio
import json
import time
import uuid
from types import SimpleNamespace

import pytest

import promptkeep
from promptkeep import history, storage
from promptkeep.events import Event, key
from promptkeep.importers import import_claude_code, import_codex, import_dataset
from promptkeep.ingestion import record
from promptkeep.watch import Watcher
from tests.test_anthropic import client, kwargs
from tests.test_codex_import import record as codex_record
from tests.test_codex_import import session, write


@pytest.fixture(autouse=True)
def retire_watchers(monkeypatch):
    """An assertion failure must not leave a background watcher touching another test."""
    started = []
    original = Watcher.start

    def start(self):
        """Track explicitly started workers for deterministic teardown."""
        started.append(self)
        return original(self)

    monkeypatch.setattr(Watcher, "start", start)
    yield
    for watcher in started:
        watcher.stop()
    for watcher in started:
        assert watcher.join(6)


def settle(condition, timeout=5):
    """Wait for meaningful observable state, not an assumed worker sleep."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if condition():
            return
        time.sleep(0.02)
    raise AssertionError("worker did not reach expected state")


def test_idempotent_events_unknown_time_links_and_redaction():
    promptkeep.configure(redact=lambda text: text.replace("secret", "[X]"))
    root = Event(
        "fixture", "trace", "trace", "root", kind="agent", started_at="2026-11-08T00:00:00+00:00"
    )
    model = Event(
        "fixture",
        "trace",
        "response",
        "model",
        kind="llm",
        parent_id="trace",
        output="secret",
        response_id="r",
        usage={"input_tokens": 5, "output_tokens": 2},
        observed_at="2026-11-08T00:00:01+00:00",
        timing_quality="unknown",
        links=("future",),
    )
    record(model)
    record(root)
    record(model)
    trace = history.trace(root.trace_key)
    assert trace.model_calls == 1 and trace.total_tokens == 7
    span = next(s for s in trace.spans if s.kind == "llm")
    assert span.started_at is None and span.duration_ms is None
    assert span.links == [key("fixture", "trace", "future")]
    assert "secret" not in str(span.runs)


def claude_records():
    """Native assistant messages reuse response identities across streamed blocks."""

    def event(kind, id, parent, message, second):
        return dict(
            type=kind,
            uuid=id,
            parentUuid=parent,
            sessionId="claude-session",
            timestamp=f"2026-11-08T00:00:0{second}Z",
            message=message,
        )

    return [
        event("user", "u", None, dict(role="user", content="question"), 0),
        event(
            "assistant",
            "a",
            "u",
            dict(
                id="response-1",
                model="claude",
                role="assistant",
                content=[dict(type="tool_use", id="tool-1", name="search", input={"query": "q"})],
                usage={"input_tokens": 3, "output_tokens": 1},
                stop_reason="tool_use",
            ),
            1,
        ),
        event(
            "user",
            "result",
            "a",
            dict(
                role="user",
                content=[
                    dict(type="tool_result", tool_use_id="tool-1", content="failed", is_error=True)
                ],
            ),
            2,
        ),
        event(
            "assistant",
            "b",
            "result",
            dict(
                id="response-2",
                model="claude",
                role="assistant",
                content=[dict(type="text", text="answer")],
                usage={"input_tokens": 5, "output_tokens": 2, "cache_read_input_tokens": 7},
                stop_reason="end_turn",
            ),
            3,
        ),
    ]


def test_claude_native_and_dataset_share_identity(tmp_path):
    path = write(tmp_path, claude_records())
    report = import_claude_code(path)
    trace = history.trace(report.trace_key)
    assert report.model_spans == 2 and report.tool_spans == 1
    assert trace.total_tokens == 18
    tool = next(s for s in trace.spans if s.kind == "tool")
    assert tool.status == "error"
    assert next(s for s in trace.spans if s.span_key == tool.parent_span_key).kind == "llm"
    result = import_dataset(path, dataset="ultralazr/claude-code-traces")
    assert result.sessions[0].outcome == "unchanged"
    assert len(history.list_traces()) == 1


def test_codex_dataset_orders_records_without_duplicate_sessions(tmp_path):
    rows = [
        dict(**value, session_id="session-1", record_index=index, schema_version=1)
        for index, value in enumerate(session())
    ]
    path = write(tmp_path, list(reversed(rows)))
    report = import_dataset(path, dataset="build-small-hackathon/hackathon-advisor-codex-traces")
    assert not report.failures and len(report.sessions) == 1
    native = import_codex(write(tmp_path))
    assert native.trace_key == report.sessions[0].trace_key
    assert len(history.list_traces()) == 1


def test_native_agents_processor_and_provider_enrichment():
    from agents.tracing.span_data import GenerationSpanData

    from promptkeep.integrations.openai_agents import PromptKeepTracingProcessor
    from tests.fakes import FakeClient

    processor = PromptKeepTracingProcessor()
    trace = SimpleNamespace(trace_id="trace-native", name="workflow")
    span = SimpleNamespace(
        trace_id=trace.trace_id,
        span_id="model-native",
        parent_id=None,
        started_at="2026-11-08T00:00:00+00:00",
        ended_at=None,
        error=None,
        span_data=GenerationSpanData(model="test", usage={"input_tokens": 3, "output_tokens": 2}),
    )
    processor.on_trace_start(trace)
    processor.on_span_start(span)
    c = promptkeep.wrap(FakeClient())
    c.chat.completions.create(model="test", messages=[dict(role="user", content="hi")])
    span.ended_at = "2026-11-08T00:00:01+00:00"
    processor.on_span_end(span)
    processor.on_trace_end(trace)
    result = history.trace(key("openai-agents", trace.trace_id, trace.trace_id))
    assert result.model_calls == 1
    assert len(history.all_runs()) == 1
    processor.shutdown()


def test_langchain_real_runnable_and_model():
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from langchain_core.runnables import RunnableLambda

    from promptkeep.integrations.langchain import PromptKeepCallbackHandler

    handler = PromptKeepCallbackHandler()
    runnable = RunnableLambda(lambda value: value) | FakeListChatModel(responses=["answer"])
    assert runnable.invoke("question", config={"callbacks": [handler]}).content == "answer"
    roots = history.list_traces()
    assert len(roots) == 1
    assert history.trace(roots[0].span_key).model_calls == 1
    handler.shutdown()


def test_setup_is_idempotent_and_restores_owned_methods():
    from tests.fakes import FakeClient

    c = FakeClient()
    original = c.chat.completions.create
    installation = promptkeep.setup(clients=[c], integrations=["langchain"])
    assert promptkeep.setup(clients=[c], integrations=["langchain"]) is installation
    assert installation.langchain_callbacks
    assert c.chat.completions.create is not original
    assert installation.shutdown(timeout=5)
    assert c.chat.completions.create == original
    assert installation.shutdown()


def test_otel_ingestion_and_export_do_not_loop():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult

    from promptkeep.integrations.opentelemetry import OTLPExport, PromptKeepSpanExporter

    class Collector:
        def __init__(self):
            self.spans = []

        def export(self, spans):
            self.spans.extend(spans)
            return SpanExportResult.SUCCESS

        def shutdown(self):
            pass

    collector = Collector()
    outbound = OTLPExport("http://unused", exporter=collector)
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(PromptKeepSpanExporter()))
    with provider.get_tracer("test").start_as_current_span(
        "model",
        attributes={
            "gen_ai.operation.name": "chat",
            "gen_ai.usage.input_tokens": 3,
            "gen_ai.usage.output_tokens": 2,
        },
    ):
        pass
    assert history.trace(history.list_traces()[0].span_key).total_tokens == 5
    with promptkeep.agent("manual"):
        pass
    assert outbound.flush(5)
    assert len(collector.spans) == 1
    assert collector.spans[0].attributes["promptkeep.origin"] == "promptkeep"
    provider.shutdown()
    assert outbound.shutdown(5)


def test_watch_restarts_and_buffers_incomplete_lines(tmp_path):
    path = write(tmp_path, session()[:3])
    watcher = Watcher("codex", path, interval=0.03).start()
    settle(lambda: bool(history.list_traces()))
    with path.open("ab") as file:
        raw = json.dumps(session()[3]).encode()
        file.write(raw[: len(raw) // 2])
    time.sleep(0.08)
    assert not any(s.kind == "tool" for s in history.trace(history.list_traces()[0].span_key).spans)
    with path.open("ab") as file:
        file.write(raw[len(raw) // 2 :] + b"\n")
    settle(
        lambda: any(
            s.kind == "tool" for s in history.trace(history.list_traces()[0].span_key).spans
        )
    )
    watcher.stop()
    assert watcher.join(5)
    assert watcher.failed_count == 0
    resumed = Watcher("codex", path, interval=0.03).start()
    with path.open("a") as file:
        for event in session()[4:]:
            file.write(json.dumps(event) + "\n")
    settle(lambda: history.trace(history.list_traces()[0].span_key).total_tokens == 12)
    resumed.stop()
    assert resumed.join(5)
    assert resumed.failed_count == 0
    assert len(history.list_traces()) == 1
    assert import_codex(path).model_spans == 1


@pytest.mark.parametrize("async_", [False, True])
@pytest.mark.parametrize("surface", ["raw", "raw_close", "raw_stream", "parse"])
def test_anthropic_raw_and_parse(async_, surface):
    c = client(async_)

    async def run():
        if surface == "parse":
            result = await c.messages.parse(**kwargs())
            assert result.content[0].text == "hello"
        elif surface == "raw_stream":
            async with c.messages.with_streaming_response.create(**kwargs()) as raw:
                assert (await raw.parse()).content[0].text == "hello"
        else:
            raw = await c.messages.with_raw_response.create(**kwargs())
            if surface == "raw":
                assert (await raw.parse()).content[0].text == "hello"
            await raw.close()
        await c.close()

    if async_:
        asyncio.run(run())
    else:
        if surface == "parse":
            assert c.messages.parse(**kwargs()).content[0].text == "hello"
        elif surface == "raw_stream":
            with c.messages.with_streaming_response.create(**kwargs()) as raw:
                assert raw.parse().content[0].text == "hello"
        else:
            raw = c.messages.with_raw_response.create(**kwargs())
            if surface == "raw":
                assert raw.parse().content[0].text == "hello"
            raw.close()
        c.close()
    rows = history.runs("SYSTEM")
    assert len(rows) == 1
    assert rows[0].status == ("incomplete" if surface == "raw_close" else "ok")


def test_late_checks_and_framework_spans_keep_original_database(tmp_path, isolated_db):
    import threading

    from promptkeep.integrations.langchain import PromptKeepCallbackHandler
    from tests.fakes import FakeClient

    release = threading.Event()

    @promptkeep.check.post(name="late")
    def late(ctx):
        release.wait(5)
        return promptkeep.Verdict.ok()

    promptkeep.configure(post=[late])
    c = promptkeep.wrap(FakeClient())
    handler = PromptKeepCallbackHandler()
    run_id = uuid.uuid4()
    handler.on_chain_start({}, {}, run_id=run_id, name="workflow")
    c.chat.completions.create(model="test", messages=[dict(role="user", content="hi")])
    other = tmp_path / "other.db"
    promptkeep.configure(db_path=other)
    handler.on_chain_end({}, run_id=run_id)
    release.set()
    assert promptkeep.flush(5)
    assert history.list_traces() == []
    promptkeep.configure(db_path=isolated_db)
    trace = history.trace(key("langchain", str(run_id), str(run_id)))
    assert trace.root.status == "ok" and trace.model_calls == 1
    assert len(history.checks(history.all_runs()[0].run_key)) == 1
    handler.shutdown()


@pytest.mark.parametrize("async_", [False, True])
@pytest.mark.parametrize("raw_stream", [False, True])
def test_anthropic_blocked_raw_keeps_response_protocol(async_, raw_stream):
    @promptkeep.check.pre(name="stop")
    def stop(ctx):
        return promptkeep.Verdict.block("stop")

    promptkeep.configure(pre=[stop], on_block="return")
    c = client(async_)

    async def run():
        if raw_stream:
            async with c.messages.with_streaming_response.create(**kwargs()) as raw:
                assert (await raw.parse()).content == []
        else:
            raw = await c.messages.with_raw_response.create(**kwargs())
            assert (await raw.parse()).content == []
            await raw.close()
        await c.close()

    if async_:
        asyncio.run(run())
    else:
        if raw_stream:
            with c.messages.with_streaming_response.create(**kwargs()) as raw:
                assert raw.parse().content == []
        else:
            raw = c.messages.with_raw_response.create(**kwargs())
            assert raw.parse().content == []
            raw.close()
        c.close()
    assert history.runs("SYSTEM")[0].status == "blocked"


@pytest.mark.parametrize("async_", [False, True])
def test_langgraph_parallel_branches_preserve_one_trace(async_):
    import operator
    from typing import Annotated

    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from langgraph.graph import END, START, StateGraph
    from typing_extensions import TypedDict

    from promptkeep.integrations.langchain import PromptKeepCallbackHandler

    class State(TypedDict):
        answers: Annotated[list[str], operator.add]

    def node(answer):
        model = FakeListChatModel(responses=[answer])

        def sync(state):
            return {"answers": [model.invoke("question").content]}

        async def async_node(state):
            return {"answers": [(await model.ainvoke("question")).content]}

        return async_node if async_ else sync

    graph = StateGraph(State)
    for name in ("a", "b"):
        graph.add_node(name, node(name))
        graph.add_edge(START, name)
        graph.add_edge(name, END)
    graph = graph.compile()
    handler = PromptKeepCallbackHandler()
    config = {"callbacks": [handler]}
    result = (
        asyncio.run(graph.ainvoke({"answers": []}, config))
        if async_
        else graph.invoke({"answers": []}, config)
    )
    assert sorted(result["answers"]) == ["a", "b"]
    roots = history.list_traces()
    assert len(roots) == 1
    trace = history.trace(roots[0].span_key)
    assert trace.model_calls == 2 and trace.root.status == "ok"
    assert all(span.status != "running" for span in trace.spans)
    handler.shutdown()


def test_langgraph_interrupt_and_resume_keep_explicit_runs():
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command, interrupt
    from typing_extensions import TypedDict

    from promptkeep.integrations.langchain import PromptKeepCallbackHandler

    class State(TypedDict):
        answer: str

    def pause(state):
        return {"answer": interrupt("continue?")}

    builder = StateGraph(State)
    builder.add_node("pause", pause)
    builder.add_edge(START, "pause")
    builder.add_edge("pause", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    handler = PromptKeepCallbackHandler()
    config = {"configurable": {"thread_id": "test"}, "callbacks": [handler]}
    first = graph.invoke({"answer": ""}, config)
    assert first["__interrupt__"]
    assert graph.invoke(Command(resume="yes"), config)["answer"] == "yes"
    roots = history.list_traces()
    assert len(roots) == 2
    assert {root.status for root in roots} == {"incomplete", "ok"}
    handler.shutdown()


def test_setup_failure_restores_client_and_preserves_external_changes(monkeypatch):
    import promptkeep.integrations.opentelemetry as otel
    from tests.fakes import FakeClient

    c = FakeClient()
    original = c.chat.completions.create

    def fail(*args, **kwargs):
        raise RuntimeError("fixture exporter failure")

    monkeypatch.setattr(otel, "OTLPExport", fail)
    with pytest.raises(RuntimeError):
        promptkeep.setup(clients=[c], otlp_endpoint="http://localhost:1")
    assert c.chat.completions.create == original
    installation = promptkeep.setup(clients=[c])

    def external(**kwargs):
        """An unrelated library takes ownership after setup."""
        return None

    c.chat.completions.create = external
    installation.shutdown(5)
    assert c.chat.completions.create is external


def test_watch_rotation_and_failed_commit_recovery(tmp_path, monkeypatch):
    path = write(tmp_path, session())
    watcher = Watcher("codex", path, interval=0.02).start()
    settle(lambda: bool(history.list_traces()))
    trace_key = history.list_traces()[0].span_key
    settle(lambda: history.trace(trace_key).model_calls == 1)
    replacement = write(tmp_path, session()[:3], name="replacement.jsonl")
    replacement.replace(path)
    time.sleep(0.1)
    original = storage.persist_watch
    failed = []

    def flaky(*args, **kwargs):
        if not failed:
            failed.append(True)
            raise RuntimeError("fixture transaction failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(storage, "persist_watch", flaky)
    with path.open("a") as file:
        file.write(
            json.dumps(
                codex_record(
                    "response_item",
                    dict(type="function_call", call_id="new-tool", name="second", arguments="{}"),
                    8,
                )
            )
            + "\n"
        )
    settle(lambda: sum(s.kind == "tool" for s in history.trace(trace_key).spans) == 2)
    watcher.stop()
    assert watcher.join(5)
    assert failed and watcher.failed_count == 1
    assert len(history.list_traces()) == 1


def test_claude_watching_and_text_parts_are_idempotent(tmp_path):
    events = claude_records()
    second = json.loads(json.dumps(events[-1]))
    second["uuid"] = "text-part-two"
    second["message"]["content"][0]["text"] = " continued"
    events.append(second)
    path = write(tmp_path, events)
    report = import_claude_code(path)
    model = next(
        s
        for s in history.trace(report.trace_key).spans
        if s.primary_run and s.primary_run.response_id == "response-2"
    )
    assert model.primary_run.output_text == "answer continued"
    turn = next(s for s in history.trace(report.trace_key).spans if s.name == "claude-code.turn")
    assert turn.output == "answer continued"
    with Watcher("claude-code", path, interval=0.02) as watcher:
        settle(lambda: history.trace(report.trace_key).root.status == "running")
        time.sleep(0.08)
    assert watcher.failed_count == 0
    assert history.trace(report.trace_key).model_calls == 2


def test_otlp_timeout_and_loop_suppression():
    import threading

    from opentelemetry.sdk.trace.export import SpanExportResult

    from promptkeep.integrations.opentelemetry import OTLPExport

    gate = threading.Event()
    entered = threading.Event()

    class Slow:
        def export(self, spans):
            entered.set()
            gate.wait(5)
            return SpanExportResult.SUCCESS

        def shutdown(self):
            pass

    outbound = OTLPExport("http://unused", exporter=Slow())
    try:
        with promptkeep.agent("slow-export"):
            pass
        assert entered.wait(2)
        assert not outbound.flush(0.01)
        assert not outbound.shutdown(0.01)
    finally:
        gate.set()
        assert outbound.shutdown(5)


def test_otlp_explicit_replay_uses_redacted_content_and_skips_inbound_spans():
    from opentelemetry.sdk.trace.export import SpanExportResult

    from promptkeep.integrations.opentelemetry import OTLPExport

    class Collector:
        def __init__(self):
            self.spans = []

        def export(self, spans):
            self.spans.extend(spans)
            return SpanExportResult.SUCCESS

        def shutdown(self):
            pass

    promptkeep.configure(redact=lambda text: text.replace("secret", "[X]"))
    with promptkeep.agent("historical-root", input="secret prompt"):
        with promptkeep.tool("historical-tool", input="secret tool"):
            pass
    root = history.list_traces()[0]
    collector = Collector()
    outbound = OTLPExport("http://unused", exporter=collector, include_content=True)
    try:
        assert outbound.replay([root.trace_key], timeout=5) == 2
        assert len(collector.spans) == 2
        root_span = next(span for span in collector.spans if span.parent is None)
        tool_span = next(span for span in collector.spans if span.parent is not None)
        assert root_span.attributes["promptkeep.input_json"] == '"[X] prompt"'
        assert tool_span.parent.span_id == root_span.context.span_id

        inbound = Event(
            "opentelemetry",
            "native-trace",
            "native-span",
            "already-ingested",
            kind="agent",
            started_at="2026-11-08T00:00:00+00:00",
            ended_at="2026-11-08T00:00:01+00:00",
        )
        record(inbound)
        imported_trace_key = inbound.trace_key
        assert outbound.replay([imported_trace_key], timeout=5) == 0
        assert len(collector.spans) == 2
        with pytest.raises(ValueError, match="no trace found"):
            outbound.replay(["missing-trace"], timeout=5)
    finally:
        assert outbound.shutdown(5)
    with pytest.raises(RuntimeError, match="shutdown"):
        outbound.replay([root.trace_key], timeout=1)


def test_real_otlp_http_collector_receives_only_redacted_content():
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

    from promptkeep.integrations.opentelemetry import OTLPExport

    received = []

    class Collector(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            message = ExportTraceServiceRequest()
            message.ParseFromString(body)
            received.append(message)
            self.send_response(200)
            self.send_header("Content-Type", "application/x-protobuf")
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Collector)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}/v1/traces"
    promptkeep.configure(redact=lambda text: text.replace("secret", "[X]"))
    outbound = OTLPExport(endpoint, include_content=True)
    try:
        with promptkeep.agent("local-export", input="secret"):
            pass
        assert outbound.flush(5)
        assert received and "secret" not in str(received[0])
        assert "[X]" in str(received[0])
    finally:
        assert outbound.shutdown(5)
        server.shutdown()
        server.server_close()
        thread.join(5)


def test_native_agents_hooks_preserve_existing_processors():
    import agents.tracing as sdk
    from agents.tracing import custom_span, trace

    before = sdk.get_trace_provider()._multi_processor._processors
    installation = promptkeep.setup(integrations=["openai_agents"])
    try:
        with trace("native-events"):
            with custom_span("step", data={"key": "value"}):
                pass
        roots = history.list_traces()
        assert len(roots) == 1
        assert len(history.trace(roots[0].span_key).spans) == 2
        after = sdk.get_trace_provider()._multi_processor._processors
        assert all(processor in after for processor in before)
    finally:
        assert installation.shutdown(5)


def test_claude_subagent_does_not_replace_parent_session(tmp_path):
    parent = import_claude_code(write(tmp_path, claude_records(), name="parent.jsonl"))
    records = claude_records()
    for event in records:
        event.update(agentId="child-agent", isSidechain=True)
    child = import_claude_code(write(tmp_path, records, name="child.jsonl"))
    assert child.session_id != parent.session_id
    assert len(history.list_traces()) == 2
    trace = history.trace(child.trace_key)
    assert trace.root.links == [parent.trace_key]
    assert history.trace(parent.trace_key).model_calls == 2
    with Watcher("claude-code", tmp_path / "child.jsonl", interval=0.02):
        settle(lambda: history.trace(child.trace_key).root.status == "running")
    assert history.trace(parent.trace_key).model_calls == 2


def test_committed_observers_do_not_receive_rolled_back_events():
    seen = []
    unsubscribe = storage.observe_spans(seen.append)
    try:
        with storage.connection_context():
            storage.get_db()
            with pytest.raises(RuntimeError):
                with storage.notification_transaction(storage.db_proxy):
                    record(Event("rollback", "trace", "trace", "root", kind="agent"))
                    raise RuntimeError("rollback")
        assert seen == []
        assert history.list_traces() == []
    finally:
        unsubscribe()


def test_real_v10_not_null_timestamps_migrate_without_losing_spans(isolated_db):
    import sqlite3
    from contextlib import closing

    with promptkeep.agent("legacy") as original:
        pass
    storage.reset_caches()
    with closing(sqlite3.connect(isolated_db)) as db, db:
        columns = [
            row[1]
            for row in db.execute("PRAGMA table_info(spans)")
            if row[1] not in ("observed_at", "content_parts_json")
        ]
        # Rebuild the actual previous shape, including its non-null execution start.
        db.execute("ALTER TABLE spans RENAME TO spans_current")
        definitions = []
        for row in db.execute("PRAGMA table_info(spans_current)"):
            name, type_ = row[1], row[2]
            if name not in columns:
                continue
            definitions.append(
                '"'
                + name
                + '" '
                + type_
                + (
                    " PRIMARY KEY"
                    if row[5]
                    else " NOT NULL"
                    if name == "started_at" or row[3]
                    else ""
                )
            )
        db.execute("CREATE TABLE spans (" + ",".join(definitions) + ")")
        names = ",".join(columns)
        db.execute(f"INSERT INTO spans ({names}) SELECT {names} FROM spans_current")
        db.execute("DROP TABLE spans_current")
        db.execute("CREATE UNIQUE INDEX legacy_span_key ON spans(span_key)")
        for table in ("span_links", "external_identities", "watch_checkpoints"):
            db.execute(f"DROP TABLE {table}")
        db.execute("PRAGMA user_version=10")
    assert history.trace(original.trace_key).root.name == "legacy"
    record(
        Event(
            "migration",
            "root",
            "response",
            "model",
            kind="llm",
            observed_at="2026-11-08T00:00:00+00:00",
        )
    )
    with storage.connection_context():
        assert storage.get_db().execute_sql("PRAGMA user_version").fetchone()[0] == 12
        assert (
            storage.db_proxy.execute_sql(
                "SELECT started_at FROM spans WHERE kind='llm'"
            ).fetchone()[0]
            is None
        )

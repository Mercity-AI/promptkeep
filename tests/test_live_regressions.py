"""Durable prepared inputs, sampling ancestry, warning limits and owned connections."""

import logging
import queue
import threading
from pathlib import Path

import pytest

import promptkeep
from promptkeep import agents, history, storage, writer
from promptkeep.importers import TraceEvent
from promptkeep.ingestion import record
from promptkeep.integrations._bridge import Bridge
from tests.fakes import FakeClient, make_response, make_tool_call


def test_evicted_start_reuses_redacted_input(monkeypatch):
    """A real bounded queue evicts the start while its final update restores input."""
    pending = queue.Queue(maxsize=1)
    monkeypatch.setattr(writer, "_ensure_started", lambda: pending)
    seen = []

    def redact(text):
        """A non-idempotent hook proves preparation occurs once per supplied value."""
        seen.append(text)
        return text.replace("secret", "safe").replace("safe", "[safe]")

    promptkeep.configure(write_mode="background", redact=redact)
    with promptkeep.agent("task", input={"value": "secret"}) as task:
        pass
    assert writer.dropped_count() == 1
    terminal = pending.get_nowait()
    pending.task_done()
    storage.write_batch([terminal])
    assert history.trace(task.trace_key).root.input == {"value": "[safe]"}
    assert len(seen) == 1


def test_omitted_input_and_explicit_null_are_distinct():
    """JSON null is a recorded fact; omitted input stays unavailable."""
    with promptkeep.agent("omitted") as omitted:
        pass
    with promptkeep.agent("null", input=None) as null:
        pass
    assert not history.trace(omitted.trace_key).root.has_input
    assert history.trace(null.trace_key).root.has_input
    assert history.trace(null.trace_key).root.input is None


def test_retained_tool_recovers_sampled_model_ancestor():
    """A known sampled-out parent keeps its identity without its discarded usage."""
    promptkeep.configure(sample_rate=0)
    client = promptkeep.wrap(
        FakeClient(response=make_response(tool_calls=[make_tool_call(call_id="call-structural")]))
    )
    with promptkeep.agent("task") as root:
        client.chat.completions.create(
            model="model", messages=[dict(role="user", content="secret")]
        )
        with promptkeep.tool("shell", call_id="call-structural") as tool:
            tool.fail("exit 1")
    trace = history.trace(root.trace_key)
    retained = next(s for s in trace.spans if s.kind == "tool")
    parent = next(s for s in trace.spans if s.span_key == retained.parent_span_key)
    assert parent.kind == "llm" and parent.sampled is False
    assert parent.metadata["sampled_out"] is True
    assert parent.runs == [] and not parent.has_input and not parent.has_output
    assert trace.total_tokens == 0 and trace.model_calls == 1


def test_outside_warning_is_shared_across_500_names(monkeypatch, caplog):
    """One initial warning summarizes the next 499 suppressed calls after a minute."""
    monkeypatch.setattr(agents, "_warning_last", None)
    monkeypatch.setattr(agents, "_warning_suppressed", 0)
    caplog.set_level(logging.WARNING, logger="promptkeep")
    for index in range(500):
        with promptkeep.tool(f"tool-{index}") as tool:
            tool.output("harmless")
    assert len(caplog.records) == 1
    monkeypatch.setattr(agents, "_warning_last", agents._warning_last - 61)
    with promptkeep.tool("next"):
        pass
    assert len(caplog.records) == 2
    assert "499 suppressed" in caplog.records[-1].message
    promptkeep.configure(enabled=False)
    monkeypatch.setattr(agents, "_warning_last", None)
    with promptkeep.tool("disabled"):
        pass
    assert len(caplog.records) == 2


def test_nested_lazy_connection_owner_and_exception_cleanup():
    """An inner storage operation cannot retire the outer trace's lazy connection."""
    with pytest.raises(RuntimeError):
        with promptkeep.agent("task"):
            database = storage.db_proxy.obj
            with storage.connection_context():
                assert storage.get_db().obj is database
            assert not database.is_closed()
            raise RuntimeError("application failure")
    assert database.is_closed()


def test_500_short_lived_threads_close_owned_connections(monkeypatch):
    """Connection counters and supported OS descriptor inventories detect leaks."""
    counts = dict(open=0, close=0)
    lock = threading.Lock()
    connect, close = storage._Database._connect, storage._Database.close
    errors = []
    fd_path = Path("/dev/fd")
    baseline = len(list(fd_path.iterdir())) if fd_path.exists() else None

    def opened(database):
        """Count each physical open on its owning thread."""
        result = connect(database)
        with lock:
            counts["open"] += 1
        return result

    def closed(database):
        """Count only open-to-closed transitions."""
        was_open = not database.is_closed()
        result = close(database)
        if was_open:
            with lock:
                counts["close"] += 1
        return result

    def work():
        """A short-lived caller uses nested scopes and leaves no connection behind."""
        try:
            with promptkeep.agent("thread"):
                with promptkeep.tool("nested"):
                    pass
            assert storage.db_proxy.obj.is_closed()
        except BaseException as exc:
            errors.append(exc)

    monkeypatch.setattr(storage._Database, "_connect", opened)
    monkeypatch.setattr(storage._Database, "close", closed)
    for _ in range(25):
        threads = [threading.Thread(target=work) for _ in range(20)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=20)
            assert not thread.is_alive()
    assert not errors
    assert counts == dict(open=500, close=500)
    if baseline is not None:
        assert len(list(fd_path.iterdir())) <= baseline + 3


def test_framework_input_is_prepared_once_and_event_is_immutable():
    """Framework starts and ends share prepared input through the common normalizer."""
    seen = []

    def redact(text):
        """Track input preparations without relying on idempotent redaction."""
        seen.append(text)
        return text.replace("secret", "safe")

    promptkeep.configure(redact=redact)
    original = {"nested": ["secret"]}
    event = TraceEvent(
        "contract", "trace", "trace", "root", kind="agent", status="running", input=original
    )
    original["nested"].append("changed")
    with pytest.raises(TypeError):
        event.usage["input_tokens"] = 1
    bridge = Bridge()
    bridge.start(event)
    bridge.end(TraceEvent("contract", "trace", "trace", "root", kind="agent"))
    assert history.trace(event.trace_key).root.input == {"nested": ["safe"]}
    assert len(seen) == 1
    null = TraceEvent("contract", "null", "null", "null", kind="agent", input_present=True)
    assert record(null)
    assert history.trace(null.trace_key).root.has_input


def test_framework_sampling_keeps_observed_structural_ancestors():
    """A retained callback tool can reference a sampled-out callback model."""
    promptkeep.configure(sample_rate=0)
    root = TraceEvent("callback-sampling", "root", "root", "task", kind="agent", status="running")
    model = TraceEvent(
        "callback-sampling",
        "root",
        "model",
        "model",
        kind="llm",
        parent_id="root",
        input="discard",
        usage={"total_tokens": 99},
    )
    tool = TraceEvent(
        "callback-sampling",
        "root",
        "tool",
        "shell",
        kind="tool",
        parent_id="model",
        status="error",
        error="exit 1",
    )
    assert record(root) and record(model) and record(tool)
    trace = history.trace(root.trace_key)
    assert trace.model_calls == 1 and trace.total_tokens == 0
    parent = next(s for s in trace.spans if s.kind == "llm")
    assert parent.sampled is False and parent.runs == [] and not parent.has_input
    assert trace.root.metadata["sampled_out"] is True


def test_each_physical_call_shares_sampling_across_prompt_versions(monkeypatch):
    """Two tracked prompts in one call cannot independently sample their run rows."""
    from promptkeep import controls

    promptkeep.configure(sample_rate=0.5)
    draws = iter((0.0, 1.0))
    monkeypatch.setattr(controls.random, "random", lambda: next(draws))
    system = promptkeep.Prompt("system", name="SYSTEM")
    user = promptkeep.Prompt("question", name="USER")
    client = promptkeep.wrap(FakeClient())
    client.chat.completions.create(
        model="fixture",
        messages=[dict(role="system", content=system.text), dict(role="user", content=user.text)],
    )
    assert len(history.runs("SYSTEM")) == len(history.runs("USER")) == 1
    assert next(draws) == 1.0

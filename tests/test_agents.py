"""Focused tests for explicit agent/tool scopes and model-call association.

The fixtures are the same fakes the wrapper tests use: no network, no live
provider. Each test starts from the autouse fresh database.
"""

import asyncio
import logging
import os
import time
from datetime import datetime

import pytest

import promptkeep
from promptkeep import Prompt, PromptBlocked, Verdict, agents, check, controls, history, wrap
from tests.fakes import (
    FakeAsyncClient,
    FakeClient,
    FakeResponsesClient,
    make_chunk,
    make_response,
    make_responses_response,
    make_tool_call,
    make_tool_call_delta,
    make_tool_call_item,
    make_usage,
)


def _client(**kwargs):
    """A wrapped sync fake client."""
    return wrap(FakeClient(**kwargs))


def _spans(trace_key, kind=None):
    """The trace's spans, optionally narrowed to one kind."""
    spans = history.trace(trace_key).spans
    return [span for span in spans if kind is None or span.kind == kind]


class TestScopeLifecycle:
    def test_root_tool_nested_agent_and_payloads(self):
        with promptkeep.agent("root", input={"q": "why"}) as root:
            with promptkeep.tool("search", input={"query": "why"}) as search:
                search.output({"hits": 2})
            with promptkeep.agent("child", input=[1]) as child:
                with promptkeep.tool("inner") as inner:
                    inner.output("done")
                child.output(None)
            root.output({"answer": 42})

        trace = history.trace(root.trace_key)
        by_key = {span.span_key: span for span in trace.spans}
        assert trace.root is by_key[root.span_key]
        assert trace.status == "ok"
        assert trace.sampled is True
        assert by_key[search.span_key].kind == "tool"
        assert by_key[search.span_key].parent_span_key == root.span_key
        assert by_key[search.span_key].input == {"query": "why"}
        assert by_key[search.span_key].output == {"hits": 2}
        assert by_key[child.span_key].kind == "agent"
        assert by_key[child.span_key].parent_span_key == root.span_key
        assert by_key[child.span_key].trace_key == root.trace_key
        assert by_key[inner.span_key].parent_span_key == child.span_key
        assert by_key[child.span_key].output is None
        assert by_key[root.span_key].input == {"q": "why"}
        assert by_key[root.span_key].output == {"answer": 42}

    def test_output_omitted_vs_explicit_null_and_last_value_wins(self):
        with promptkeep.agent("omitted") as omitted:
            pass
        with promptkeep.agent("null") as null:
            null.output(None)
        with promptkeep.agent("last") as last:
            last.output("first")
            last.output("last")

        omitted_root = history.trace(omitted.trace_key).root
        assert omitted_root.output is history._UNSET
        assert omitted_root.has_output is False
        null_root = history.trace(null.trace_key).root
        assert null_root.output is None
        assert null_root.has_output is True
        assert history.trace(last.trace_key).root.output == "last"

    def test_fail_marks_a_scope_error_without_raising(self):
        with promptkeep.agent("tool-failure") as task:
            with promptkeep.tool("flaky") as tool:
                tool.output("partial")
                tool.fail("exit code 1")

        trace = history.trace(task.trace_key)
        assert trace.status == "ok"
        assert trace.error_count == 1
        (tool_span,) = _spans(task.trace_key, "tool")
        assert tool_span.status == "error"
        assert tool_span.error == "exit code 1"
        assert tool_span.output == "partial"

    def test_input_and_output_are_snapshotted(self):
        source = {"question": "before"}
        result = {"answer": 1}
        with promptkeep.agent("snapshot", input=source) as scope:
            source["question"] = "after"
            scope.output(result)
            result["answer"] = 2

        root = history.trace(scope.trace_key).root
        assert root.input == {"question": "before"}
        assert root.output == {"answer": 1}

    def test_exception_identity_is_preserved_and_context_restored(self):
        error = RuntimeError("boom")
        with pytest.raises(RuntimeError) as caught:
            with promptkeep.agent("fails") as scope:
                raise error
        assert caught.value is error
        assert agents.current() is None

        trace = history.trace(scope.trace_key)
        assert trace.status == "error"
        assert "boom" in trace.root.error

        with promptkeep.agent("after") as after:
            pass
        assert history.trace(after.trace_key).root.parent_span_key is None

    def test_tool_outside_agent_warns_and_records_nothing(self, caplog):
        with caplog.at_level(logging.WARNING, logger="promptkeep"):
            with promptkeep.tool("orphan") as orphan:
                orphan.output("ignored")
        assert orphan.trace_key is None
        assert history.list_traces() == []
        assert "outside an agent" in caplog.text

    def test_reentry_and_closed_output_are_usage_errors(self):
        scope = promptkeep.agent("once")
        with scope:
            with pytest.raises(RuntimeError, match="re-entered"):
                with scope:
                    pass
        with pytest.raises(RuntimeError, match="re-entered"):
            with scope:
                pass
        with pytest.raises(RuntimeError, match="open"):
            scope.output("late")
        with pytest.raises(RuntimeError, match="open"):
            scope.fail("late")

    @pytest.mark.parametrize("name", ["", "   ", None, 1])
    def test_names_must_be_non_empty_strings(self, name):
        with pytest.raises(ValueError, match="non-empty string name"):
            promptkeep.agent(name)
        with pytest.raises(ValueError, match="non-empty string name"):
            promptkeep.tool(name)

    def test_disabled_and_off_scopes_are_harmless(self):
        promptkeep.configure(enabled=False)
        with promptkeep.agent("disabled") as disabled:
            with promptkeep.tool("tool") as tool:
                tool.output("ignored")
        assert disabled.span_key and tool.span_key
        assert disabled.trace_key == disabled.span_key
        assert history.list_traces() == []

        promptkeep.configure(enabled=True, write_mode="off")
        with promptkeep.agent("off") as off:
            off.output("ignored")
        assert history.list_traces() == []

    def test_suppressed_scopes_do_not_persist(self):
        with promptkeep.suppress():
            with promptkeep.agent("judge") as judge:
                with promptkeep.tool("lookup") as lookup:
                    lookup.output("ignored")
        assert judge.span_key and lookup.span_key
        assert history.list_traces() == []

    def test_suppressed_child_failure_does_not_promote_sampled_out_trace(self):
        promptkeep.configure(sample_rate=0.0)
        with promptkeep.agent("sampled-out"):
            with promptkeep.suppress():
                with pytest.raises(RuntimeError, match="suppressed failure"):
                    with promptkeep.tool("hidden"):
                        raise RuntimeError("suppressed failure")
            with promptkeep.tool("ordinary") as ordinary:
                ordinary.output("ok")

        assert history.list_traces() == []
        assert history.all_runs() == []

    def test_async_cancellation_is_cancelled_not_error(self):
        async def go():
            with pytest.raises(asyncio.CancelledError):
                async with promptkeep.agent("cancelled") as task:
                    raise asyncio.CancelledError()
            return task.trace_key

        trace = history.trace(asyncio.run(go()))
        assert trace.status == "cancelled"
        assert trace.root.error == "cancelled"

    def test_async_tasks_do_not_share_context(self):
        async def make(name):
            async with promptkeep.agent(name) as task:
                await asyncio.sleep(0)
                return task.trace_key

        async def go():
            return await asyncio.gather(make("one"), make("two"))

        keys = asyncio.run(go())
        traces = [history.trace(key) for key in keys]
        assert sorted(trace.root.name for trace in traces) == ["one", "two"]
        assert all(trace.root.parent_span_key is None for trace in traces)

    def test_database_change_mid_scope_detaches_telemetry(self, tmp_path):
        first = promptkeep.get_settings().db_path
        with promptkeep.agent("changing") as task:
            promptkeep.configure(db_path=tmp_path / "other.db")
            with promptkeep.tool("child") as child:
                child.output("ignored")
            task.output("still runs")

        # The root start was in the first database; the final and child were
        # skipped rather than written into the unrelated file.
        promptkeep.configure(db_path=first)
        trace = history.trace(task.trace_key)
        assert [span.kind for span in trace.spans] == ["agent"]
        assert trace.root.status == "running"

        promptkeep.configure(db_path=tmp_path / "other.db")
        assert history.list_traces() == []

    def test_a_frame_inherited_across_a_process_identity_change_is_ignored(self, monkeypatch):
        real_pid = os.getpid()
        client = _client()
        with promptkeep.agent("parent") as task:
            monkeypatch.setattr(agents.os, "getpid", lambda: real_pid + 1)
            assert agents.current() is None
            client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])

        # The root start was written before the identity changed; the final
        # row and the model call were not attached to the old trace.
        trace = history.trace(task.trace_key)
        assert [span.kind for span in trace.spans] == ["agent"]
        assert trace.root.status == "running"


class TestModelRecording:
    def test_plain_call_inside_agent_records_one_model_span(self):
        client = _client()
        with promptkeep.agent("plain") as task:
            response = client.chat.completions.create(
                model="m", messages=[{"role": "user", "content": "hi"}]
            )

        trace = history.trace(task.trace_key)
        (llm,) = _spans(task.trace_key, "llm")
        assert response.promptkeep.span_key == llm.span_key
        assert llm.parent_span_key == task.span_key
        assert llm.name == "gpt-test"  # the model the provider reported
        assert len(llm.runs) == 1
        assert llm.runs[0].prompt_name is None
        assert llm.runs[0].output_text == "hello!"
        assert trace.total_tokens == 15
        assert trace.model_calls == 1

    def test_model_span_interval_covers_the_provider_call_only(self):
        @check.pre(name="slow")
        def slow(ctx):
            time.sleep(0.05)
            return Verdict.ok()

        prompt = Prompt("system", name="SLOW_CHECK", pre=[slow])
        client = _client()
        with promptkeep.agent("timing") as task:
            client.chat.completions.create(
                model="m",
                messages=[
                    {"role": "developer", "content": prompt},
                    {"role": "user", "content": "hi"},
                ],
            )

        (llm,) = _spans(task.trace_key, "llm")
        assert llm.ended_at is not None and llm.duration_ms is not None
        start = datetime.fromisoformat(llm.started_at)
        end = datetime.fromisoformat(llm.ended_at)
        measured = (end - start).total_seconds() * 1000
        assert abs(measured - llm.duration_ms) <= 1
        assert llm.duration_ms < 50  # the pre-check's sleep is not in the span

    def test_a_trace_inside_a_conversation_keeps_one_existing_turn(self):
        client = _client()
        with promptkeep.conversation("trace-session"):
            with promptkeep.agent("conversation-task") as task:
                with promptkeep.tool("lookup") as tool:
                    tool.output("ok")
                client.chat.completions.create(
                    model="m", messages=[{"role": "user", "content": "hi"}]
                )

        (llm,) = _spans(task.trace_key, "llm")
        assert llm.runs[0].conversation_id == "trace-session"
        assert llm.runs[0].turn_index == 0
        conversation = history.conversation("trace-session")
        assert [turn.run_key for turn in conversation.turns] == [llm.runs[0].run_key]

    def test_call_outside_agent_keeps_existing_behavior(self):
        client = _client()
        response = client.chat.completions.create(
            model="m", messages=[{"role": "user", "content": "hi"}]
        )
        assert not hasattr(response, "promptkeep")
        assert history.all_runs() == []
        assert history.list_traces() == []

    def test_model_inside_tool_is_child_of_tool(self):
        client = _client()
        with promptkeep.agent("root") as task:
            with promptkeep.tool("search") as search:
                client.chat.completions.create(
                    model="m", messages=[{"role": "user", "content": "hi"}]
                )
        (llm,) = _spans(task.trace_key, "llm")
        assert llm.parent_span_key == search.span_key

    def test_early_stream_close_records_what_arrived(self):
        chunks = [make_chunk(content="Hel"), make_chunk(content="lo")]
        client = _client(stream_chunks=chunks)
        with promptkeep.agent("early-close") as task:
            with client.chat.completions.create(
                model="m", stream=True, messages=[{"role": "user", "content": "hi"}]
            ) as stream:
                assert next(iter(stream)).choices[0].delta.content == "Hel"
        (llm,) = _spans(task.trace_key, "llm")
        assert llm.status == "ok"
        assert llm.runs[0].output_text == "Hel"

    def test_stream_error_records_a_failed_model_span(self):
        class BrokenStream:
            def __iter__(self):
                yield make_chunk(content="Hi")
                raise ConnectionError("stream dropped")

        client = FakeClient()
        client.chat.completions.create = lambda **kwargs: BrokenStream()
        wrapped = wrap(client)
        with promptkeep.agent("stream-error") as task:
            with pytest.raises(ConnectionError, match="stream dropped"):
                list(
                    wrapped.chat.completions.create(
                        model="m", stream=True, messages=[{"role": "user", "content": "hi"}]
                    )
                )
        (llm,) = _spans(task.trace_key, "llm")
        assert llm.status == "error"
        assert "stream dropped" in llm.error
        assert llm.runs[0].status == "error"
        assert llm.runs[0].output_text == "Hi"

    def test_a_stream_can_finish_after_the_root_has_closed(self):
        chunks = [make_chunk(content="late")]
        client = _client(stream_chunks=chunks)
        with promptkeep.agent("late-child") as task:
            stream = client.chat.completions.create(
                model="m", stream=True, messages=[{"role": "user", "content": "hi"}]
            )
        # The root has closed; the captured frame still names its trace.
        list(stream)
        trace = history.trace(task.trace_key)
        (llm,) = _spans(task.trace_key, "llm")
        assert trace.root.status == "ok"
        assert llm.parent_span_key == task.span_key
        assert llm.runs[0].output_text == "late"
        assert trace.duration >= llm.duration_seconds

    def test_adapter_read_failure_loses_metadata_not_the_call_or_span(self, monkeypatch):
        from promptkeep.integrations import OpenAIChatAdapter

        def broken(self, response):
            raise RuntimeError("cannot read response")

        monkeypatch.setattr(OpenAIChatAdapter, "read_response", broken)
        client = _client()
        with promptkeep.agent("adapter-failure") as task:
            response = client.chat.completions.create(
                model="m", messages=[{"role": "user", "content": "hi"}]
            )
        assert response.choices[0].message.content == "hello!"
        (llm,) = _spans(task.trace_key, "llm")
        assert llm.status == "ok"
        assert llm.runs[0].output_text is None

    def test_run_handle_span_key_is_none_outside_a_trace(self):
        prompt = Prompt("system", name="NO_TRACE")
        client = _client()
        response = client.chat.completions.create(
            model="m",
            messages=[
                {"role": "developer", "content": prompt},
                {"role": "user", "content": "hi"},
            ],
        )
        assert response.promptkeep.span_key is None

    def test_sync_stream_records_when_it_ends(self):
        chunks = [
            make_chunk(content="Hel"),
            make_chunk(content="lo"),
            make_chunk(usage=make_usage(prompt_tokens=8, completion_tokens=2)),
        ]
        client = _client(stream_chunks=chunks)
        with promptkeep.agent("stream") as task:
            stream = client.chat.completions.create(
                model="m", stream=True, messages=[{"role": "user", "content": "hi"}]
            )
            assert stream.promptkeep.span_key
            assert list(stream)

        (llm,) = _spans(task.trace_key, "llm")
        assert llm.status == "ok"
        assert llm.runs[0].output_text == "Hello"
        assert llm.runs[0].total_tokens == 10
        assert llm.duration_ms is not None

    def test_async_non_streaming_records(self):
        client = wrap(FakeAsyncClient())

        async def go():
            async with promptkeep.agent("async") as task:
                response = await client.chat.completions.create(
                    model="m", messages=[{"role": "user", "content": "hi"}]
                )
                return task.trace_key, response

        key, response = asyncio.run(go())
        (llm,) = _spans(key, "llm")
        assert response.promptkeep.span_key == llm.span_key
        assert llm.runs[0].output_text == "hello!"

    def test_async_streaming_records(self):
        chunks = [
            make_chunk(content="Hi "),
            make_chunk(content="there"),
            make_chunk(usage=make_usage(prompt_tokens=4, completion_tokens=2)),
        ]
        client = wrap(FakeAsyncClient(stream_chunks=chunks))

        async def go():
            async with promptkeep.agent("async-stream") as task:
                stream = await client.chat.completions.create(
                    model="m", stream=True, messages=[{"role": "user", "content": "hi"}]
                )
                received = [chunk async for chunk in stream]
                return task.trace_key, received

        key, received = asyncio.run(go())
        assert len(received) == 3
        (llm,) = _spans(key, "llm")
        assert llm.runs[0].output_text == "Hi there"
        assert llm.runs[0].total_tokens == 6

    def test_provider_error_is_a_caught_child_failure(self):
        client = _client(error=RuntimeError("rate limited"))
        with promptkeep.agent("error") as task:
            with pytest.raises(RuntimeError, match="rate limited"):
                client.chat.completions.create(
                    model="m", messages=[{"role": "user", "content": "hi"}]
                )

        trace = history.trace(task.trace_key)
        (llm,) = _spans(task.trace_key, "llm")
        assert trace.status == "ok"  # the application caught it
        assert trace.error_count == 1
        assert llm.status == "error"
        assert "rate limited" in llm.error
        assert llm.runs[0].status == "error"

    def test_blocked_call_records_a_blocked_model_span(self):
        @check.pre(name="gate")
        def gate(ctx):
            return Verdict.block("not allowed")

        prompt = Prompt("system", name="BLOCKED", pre=[gate])
        client = _client()
        with promptkeep.agent("blocked") as task:
            with pytest.raises(PromptBlocked):
                client.chat.completions.create(
                    model="m",
                    messages=[
                        {"role": "developer", "content": prompt},
                        {"role": "user", "content": "hi"},
                    ],
                )

        trace = history.trace(task.trace_key)
        (llm,) = _spans(task.trace_key, "llm")
        assert trace.status == "ok"
        assert trace.error_count == 1
        assert llm.status == "blocked"
        assert "gate" in llm.error
        assert llm.runs[0].status == "blocked"
        assert [c.name for c in llm.checks] == ["gate"]

    def test_two_prompts_share_one_model_span_without_multiplying_metrics(self):
        first = Prompt("first {x}", {"x": 1}, name="FIRST")
        second = Prompt("second {x}", {"x": 2}, name="SECOND")
        client = _client(response=make_response(cost=0.25))
        with promptkeep.agent("multi") as task:
            response = client.chat.completions.create(
                model="m",
                messages=[
                    {"role": "developer", "content": first},
                    {"role": "user", "content": second},
                ],
            )

        trace = history.trace(task.trace_key)
        (llm,) = _spans(task.trace_key, "llm")
        assert response.promptkeep.span_key == llm.span_key
        assert {run.prompt_name for run in llm.runs} == {"FIRST", "SECOND"}
        assert len(llm.runs) == 2
        assert trace.model_calls == 1
        assert trace.total_tokens == 15  # the response's usage, not 2 x 15
        assert trace.total_cost == 0.25
        assert trace.cost_complete is True
        assert trace.versions_used == {"FIRST": [1], "SECOND": [1]}
        assert len(history.runs("FIRST")) == 1
        assert len(history.runs("SECOND")) == 1

    def test_checks_attach_to_the_primary_run_on_the_model_span(self):
        @check.post(name="audit", mode="blocking")
        def audit(ctx):
            return Verdict.ok()

        prompt = Prompt("system", name="CHECKED", post=[audit])
        client = _client()
        with promptkeep.agent("checked") as task:
            client.chat.completions.create(
                model="m",
                messages=[
                    {"role": "developer", "content": prompt},
                    {"role": "user", "content": "hi"},
                ],
            )
        (llm,) = _spans(task.trace_key, "llm")
        assert [verdict.name for verdict in llm.checks] == ["audit"]
        assert llm.checks[0].status == "ok"

    def test_explicit_conversation_override_can_drop_a_run_but_keep_the_span(self):
        # Conversation ids 1 and 2 have different stable cohorts at 0.6:
        # id 1 is kept, id 2 is sampled out.
        promptkeep.configure(sample_rate=0.6)
        client = _client()
        with promptkeep.conversation("kept-session"):
            with promptkeep.agent("kept-trace") as task:
                client.chat.completions.create(
                    model="m",
                    messages=[{"role": "user", "content": "hi"}],
                    promptkeep_conversation="dropped-session",
                )

        trace = history.trace(task.trace_key)
        assert trace.sampled is True
        (llm,) = _spans(task.trace_key, "llm")
        assert llm.sampled is False
        assert llm.runs == []
        assert llm.run_unavailable is True
        assert trace.model_calls == 1
        assert trace.total_tokens == 0

    def test_suppressed_judge_call_is_not_a_trace_span(self):
        client = _client()
        with promptkeep.agent("judge-root") as task:
            with promptkeep.suppress():
                client.chat.completions.create(
                    model="m", messages=[{"role": "user", "content": "judge this"}]
                )
        trace = history.trace(task.trace_key)
        assert [span.kind for span in trace.spans] == ["agent"]
        assert history.all_runs() == []

    def test_background_mode_flush_makes_the_trace_readable(self):
        promptkeep.configure(write_mode="background")
        client = _client()
        with promptkeep.agent("background") as task:
            with promptkeep.tool("tool") as tool:
                tool.output("ok")
            client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])
        assert promptkeep.flush(timeout=5) is True
        trace = history.trace(task.trace_key)
        assert {span.kind for span in trace.spans} == {"agent", "tool", "llm"}
        assert trace.status == "ok"


class TestToolCallLinking:
    def test_a_tool_call_id_becomes_a_child_of_its_model_span(self):
        response = make_response(content=None, tool_calls=[make_tool_call(call_id="call_1")])
        client = _client(response=response)
        with promptkeep.agent("linked") as task:
            client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])
            with promptkeep.tool("search", call_id="call_1") as search:
                search.output("found")

        (llm,) = _spans(task.trace_key, "llm")
        (tool_span,) = _spans(task.trace_key, "tool")
        assert tool_span.parent_span_key == llm.span_key

    def test_a_streamed_tool_call_links_too(self):
        chunks = [make_chunk(tool_calls=[make_tool_call_delta(call_id="call_s")])]
        client = _client(stream_chunks=chunks)
        with promptkeep.agent("stream-linked") as task:
            stream = client.chat.completions.create(
                model="m", stream=True, messages=[{"role": "user", "content": "hi"}]
            )
            list(stream)
            with promptkeep.tool("search", call_id="call_s"):
                pass

        (llm,) = _spans(task.trace_key, "llm")
        (tool_span,) = _spans(task.trace_key, "tool")
        assert tool_span.parent_span_key == llm.span_key

    def test_a_responses_function_call_links_too(self):
        response = make_responses_response(
            text=None, tool_calls=[make_tool_call_item(call_id="call_r")]
        )
        client = wrap(FakeResponsesClient(responses=[response]))
        with promptkeep.agent("responses-linked") as task:
            client.responses.create(model="m", input="hi")
            with promptkeep.tool("search", call_id="call_r"):
                pass

        (llm,) = _spans(task.trace_key, "llm")
        (tool_span,) = _spans(task.trace_key, "tool")
        assert tool_span.parent_span_key == llm.span_key

    def test_an_unknown_call_id_keeps_the_ambient_parent(self):
        with promptkeep.agent("unlinked") as task:
            with promptkeep.tool("search", call_id="call_missing"):
                pass

        (tool_span,) = _spans(task.trace_key, "tool")
        assert tool_span.parent_span_key == task.span_key

    def test_a_call_id_from_another_trace_keeps_the_ambient_parent(self):
        response = make_response(content=None, tool_calls=[make_tool_call(call_id="call_x")])
        client = _client(response=response)
        with promptkeep.agent("first"):
            client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])
        with promptkeep.agent("second") as second:
            with promptkeep.tool("search", call_id="call_x"):
                pass

        (tool_span,) = _spans(second.trace_key, "tool")
        assert tool_span.parent_span_key == second.span_key

    def test_a_non_string_call_id_is_a_usage_error(self):
        with pytest.raises(ValueError, match="call_id"):
            promptkeep.tool("search", call_id=1)


class TestSamplingAndPayloads:
    def test_sampled_out_success_is_not_stored(self):
        promptkeep.configure(sample_rate=0.0)
        client = _client()
        with promptkeep.agent("dropped") as task:
            with promptkeep.tool("tool") as tool:
                tool.output("ok")
            client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])
        assert history.list_traces() == []
        assert history.all_runs() == []
        with pytest.raises(ValueError, match="no trace found"):
            history.trace(task.trace_key)

    def test_trace_cohort_governs_a_model_run_outside_a_conversation(self, monkeypatch):
        # The root draws once; the model run must inherit that draw rather
        # than flipping a second, independent coin at write time.
        promptkeep.configure(sample_rate=0.5)
        draws = iter([0.0, 1.0])
        monkeypatch.setattr(controls.random, "random", lambda: next(draws))
        client = _client()
        with promptkeep.agent("kept") as task:
            client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])

        trace = history.trace(task.trace_key)
        assert trace.sampled is True
        (llm,) = _spans(task.trace_key, "llm")
        assert len(llm.runs) == 1
        assert trace.model_calls == 1

    def test_sampled_out_problem_produces_a_partial_problem_trace(self):
        promptkeep.configure(sample_rate=0.0)
        client = _client(error=RuntimeError("boom"))
        with promptkeep.agent("problem") as task:
            with promptkeep.tool("ok-tool") as ok_tool:
                ok_tool.output("fine")
            with pytest.raises(RuntimeError):
                client.chat.completions.create(
                    model="m", messages=[{"role": "user", "content": "hi"}]
                )

        trace = history.trace(task.trace_key)
        assert trace.sampled is False
        assert trace.status == "ok"
        assert trace.error_count == 1
        assert {span.kind for span in trace.spans} == {"agent", "llm"}
        (llm,) = _spans(task.trace_key, "llm")
        assert llm.sampled is False
        assert llm.runs[0].status == "error"

    def test_sampled_out_caught_tool_error_is_kept(self):
        promptkeep.configure(sample_rate=0.0)
        with promptkeep.agent("tool-error") as task:
            with pytest.raises(ValueError):
                with promptkeep.tool("fails") as tool:
                    tool.output("partial")
                    raise ValueError("tool broke")

        trace = history.trace(task.trace_key)
        assert trace.sampled is False
        assert trace.status == "ok"
        assert trace.error_count == 1
        assert {span.kind for span in trace.spans} == {"agent", "tool"}
        (tool_span,) = _spans(task.trace_key, "tool")
        assert tool_span.status == "error"
        assert tool_span.output == "partial"

    def test_redaction_covers_payloads_and_errors_but_not_names(self):
        promptkeep.configure(redact=lambda text: text.replace("secret", "[X]"))
        with pytest.raises(RuntimeError, match="secret"):
            with promptkeep.agent("secret agent", input={"key": "secret"}) as task:
                with promptkeep.tool("secret tool", input="secret in") as tool:
                    tool.output("secret out")
                    raise RuntimeError("secret error")

        trace = history.trace(task.trace_key)
        assert trace.root.name == "secret agent"
        assert trace.root.input == {"key": "[X]"}
        assert "[X] error" in trace.root.error
        (tool_span,) = _spans(task.trace_key, "tool")
        assert tool_span.name == "secret tool"
        assert tool_span.input == "[X] in"
        assert tool_span.output == "[X] out"

    def test_redaction_failure_drops_the_span(self):
        promptkeep.configure(redact=lambda text: (_ for _ in ()).throw(RuntimeError("boom")))
        with promptkeep.agent("dropped", input="secret") as task:
            task.output("secret")
        assert history.list_traces() == []
        assert task.trace_key

    def test_unparsable_redacted_json_is_kept_as_raw_text(self):
        promptkeep.configure(redact=lambda text: text.replace('"', "'"))
        with promptkeep.agent("raw", input={"key": "value"}) as task:
            pass
        root = history.trace(task.trace_key).root
        assert isinstance(root.input, str)
        assert "'key'" in root.input

    def test_large_payload_is_truncated_visibly(self):
        with promptkeep.agent("large", input="x" * (70 * 1024)) as task:
            pass
        root = history.trace(task.trace_key).root
        assert isinstance(root.input, str)
        assert "promptkeep truncated" in root.input
        assert len(root.input.encode("utf-8")) < 70 * 1024

    def test_serialization_failure_loses_only_the_payload(self):
        class BadRepr:
            def __repr__(self):
                raise RuntimeError("no repr")

        with promptkeep.agent("bad", input=BadRepr()) as task:
            pass
        root = history.trace(task.trace_key).root
        assert root.input is history._UNSET
        assert root.has_input is False
        assert root.status == "ok"

    def test_terminal_row_reuses_the_prepared_input(self, monkeypatch):
        from promptkeep import writer

        promptkeep.configure(write_mode="background")
        seen = []
        monkeypatch.setattr(writer, "submit", lambda item: seen.append(item))
        with promptkeep.agent("once", input={"big": "x" * 100}) as task:
            task.output("done")

        span_items = [item for item in seen if item.get("_kind") == "span"]
        carried = [item for item in span_items if item.get("input_json") is not None]
        assert len(carried) == 2
        assert carried[0]["status"] == "running"
        assert carried[0]["input_json"] is carried[1]["input_json"]


def test_inherited_scope_is_ignored_after_process_change(monkeypatch, caplog):
    """An inherited frame cannot attach child work to the parent's trace, and
    a tool entered under the changed identity warns instead of raising."""
    with promptkeep.agent("parent") as parent:
        pid = os.getpid()
        with monkeypatch.context() as patch:
            patch.setattr(agents.os, "getpid", lambda: pid + 1)
            assert agents.current() is None
            with caplog.at_level(logging.WARNING, logger="promptkeep"):
                with promptkeep.tool("child") as child:
                    child.output("ignored")
            assert child.trace_key is None
            assert "outside an agent" in caplog.text
        assert agents.current().span_key == parent.span_key

    # Only the parent's own rows were written: the child never attached.
    assert [span.kind for span in history.trace(parent.trace_key).spans] == ["agent"]

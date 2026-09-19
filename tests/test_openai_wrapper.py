"""Tests for wrap(): substitution, run tracking, streaming, async — all against
the fake OpenAI-shaped clients in tests/fakes.py (no network)."""

import asyncio

import pytest

from promptkeep import Prompt, history, storage, wrap
from tests.fakes import (
    FakeAsyncClient,
    FakeClient,
    make_chunk,
    make_response,
)


def make_prompt(name="WRAP_SYS", focus="brevity"):
    """A small tracked prompt to embed in request messages."""
    return Prompt("You are helpful. Focus on {focus}.", {"focus": focus}, name=name)


class TestWrapDispatch:
    """wrap() accepting classes, instances, and rejecting everything else."""

    def test_wrap_class_keeps_name_and_behavior(self):
        """Wrapping a class preserves its name, constructor, and isinstance."""
        Wrapped = wrap(FakeClient)
        assert Wrapped.__name__ == "FakeClient"
        client = Wrapped(api_key="sk-test")
        assert isinstance(client, FakeClient)
        assert client.api_key == "sk-test"

    def test_wrap_instance_returns_same_object(self):
        """Wrapping an instance instruments it in place."""
        client = FakeClient()
        assert wrap(client) is client

    def test_wrap_rejects_non_clients(self):
        """Objects no adapter recognizes are a usage error, and the message
        names what would have been recognized."""
        with pytest.raises(TypeError, match="no supported provider surface.*openai"):
            wrap(42)

    def test_double_wrap_is_idempotent(self):
        """Wrapping twice must not double-substitute or double-record."""
        client = wrap(wrap(FakeClient()))
        client.chat.completions.create(
            model="gpt-test", messages=[{"role": "user", "content": make_prompt()}]
        )
        assert len(client.calls) == 1
        assert len(history.runs("WRAP_SYS")) == 1


class TestSubstitution:
    """Prompt objects in messages become plain strings on the wire."""

    def test_prompt_object_becomes_plain_string(self):
        """The API receives rendered str content, other messages untouched."""
        client = wrap(FakeClient())
        p = make_prompt()
        client.chat.completions.create(
            model="gpt-test",
            messages=[
                {"role": "developer", "content": p},
                {"role": "user", "content": "How do I check isinstance?"},
            ],
        )
        (call,) = client.calls
        sent = call["messages"][0]["content"]
        assert type(sent) is str  # plain str, not Prompt/RenderedText
        assert sent == "You are helpful. Focus on brevity."
        assert call["messages"][1]["content"] == "How do I check isinstance?"

    def test_rendered_text_also_substituted_and_tracked(self):
        """Passing prompt.text (not the Prompt) still tracks via provenance."""
        client = wrap(FakeClient())
        p = make_prompt(focus="detail")
        client.chat.completions.create(
            model="gpt-test", messages=[{"role": "developer", "content": p.text}]
        )
        (call,) = client.calls
        assert type(call["messages"][0]["content"]) is str
        (run,) = history.runs("WRAP_SYS")
        assert run.variables == {"focus": "detail"}

    def test_original_message_dicts_not_mutated(self):
        """The caller's message list must come back exactly as they built it."""
        client = wrap(FakeClient())
        p = make_prompt()
        messages = [{"role": "developer", "content": p}]
        client.chat.completions.create(model="gpt-test", messages=messages)
        assert messages[0]["content"] is p  # user's list untouched

    def test_content_block_lists(self):
        """Prompts inside multi-part content blocks are substituted too."""
        client = wrap(FakeClient())
        p = make_prompt()
        client.chat.completions.create(
            model="gpt-test",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": p},
                        {"type": "image_url", "image_url": {"url": "http://x/y.png"}},
                    ],
                },
            ],
        )
        (call,) = client.calls
        block = call["messages"][0]["content"][0]
        assert type(block["text"]) is str
        assert block["text"] == "You are helpful. Focus on brevity."
        assert len(history.runs("WRAP_SYS")) == 1

    def test_plain_string_messages_pass_through_untracked(self):
        """Ordinary string content is neither altered nor recorded."""
        client = wrap(FakeClient())
        client.chat.completions.create(
            model="gpt-test", messages=[{"role": "user", "content": "just a string"}]
        )
        (call,) = client.calls
        assert call["messages"][0]["content"] == "just a string"
        assert history.runs("WRAP_SYS") == []


class TestRunTracking:
    """Run rows written by the interceptor for non-streaming calls."""

    def test_run_recorded_with_response_metadata(self):
        """Model, usage, output, params, and latency all land on the run row."""
        response = make_response(
            content="the answer",
            model="gpt-5.5",
            response_id="resp_99",
            prompt_tokens=20,
            completion_tokens=10,
        )
        client = wrap(FakeClient(response=response))
        client.chat.completions.create(
            model="gpt-5.5",
            temperature=0.3,
            messages=[{"role": "developer", "content": make_prompt()}],
        )
        (run,) = history.runs("WRAP_SYS")
        assert run.status == "ok"
        assert run.model == "gpt-5.5"
        assert run.response_id == "resp_99"
        assert run.output_text == "the answer"
        assert run.prompt_tokens == 20
        assert run.completion_tokens == 10
        assert run.total_tokens == 30
        assert run.latency_ms is not None
        assert run.rendered_text == "You are helpful. Focus on brevity."
        assert run.variables == {"focus": "brevity"}
        assert run.request_params["temperature"] == 0.3
        assert "messages" not in run.request_params

    def test_two_prompts_two_runs(self):
        """Each tracked prompt in one call gets its own run row."""
        client = wrap(FakeClient())
        client.chat.completions.create(
            model="gpt-test",
            messages=[
                {"role": "developer", "content": make_prompt(name="WRAP_SYS")},
                {"role": "user", "content": make_prompt(name="WRAP_USER", focus="speed")},
            ],
        )
        assert len(history.runs("WRAP_SYS")) == 1
        assert len(history.runs("WRAP_USER")) == 1

    def test_api_error_recorded_and_reraised(self):
        """API failures produce an error run and re-raise unchanged."""
        client = wrap(FakeClient(error=RuntimeError("rate limited")))
        with pytest.raises(RuntimeError, match="rate limited"):
            client.chat.completions.create(
                model="gpt-test", messages=[{"role": "developer", "content": make_prompt()}]
            )
        (run,) = history.runs("WRAP_SYS")
        assert run.status == "error"
        assert "rate limited" in run.error
        assert run.output_text is None

    def test_tracking_failure_never_breaks_the_call(self, monkeypatch):
        """A dead DB loses telemetry but the completion still returns."""

        def boom(**kwargs):
            """Stand-in for a storage layer that is completely broken."""
            raise RuntimeError("db exploded")

        monkeypatch.setattr(storage, "record_run", boom)
        client = wrap(FakeClient())
        response = client.chat.completions.create(
            model="gpt-test", messages=[{"role": "developer", "content": make_prompt()}]
        )
        assert response.choices[0].message.content == "hello!"


class TestStreaming:
    """The stream proxy: pass-through chunks, deferred run recording."""

    def _chunks(self):
        """Two content deltas plus a final usage-bearing chunk."""
        from types import SimpleNamespace

        return [
            make_chunk(content="Hel"),
            make_chunk(content="lo!"),
            make_chunk(
                usage=SimpleNamespace(prompt_tokens=8, completion_tokens=2, total_tokens=10)
            ),
        ]

    def test_chunks_pass_through_and_run_recorded_at_end(self):
        """Chunks arrive untouched; the run is written only after exhaustion."""
        client = wrap(FakeClient(stream_chunks=self._chunks()))
        stream = client.chat.completions.create(
            model="gpt-test",
            stream=True,
            messages=[{"role": "developer", "content": make_prompt()}],
        )
        assert history.runs("WRAP_SYS") == []  # nothing recorded yet
        received = list(stream)
        assert len(received) == 3
        (run,) = history.runs("WRAP_SYS")
        assert run.output_text == "Hello!"
        assert run.total_tokens == 10
        assert run.status == "ok"

    def test_context_manager_records_on_exit(self):
        """`with ... as stream:` records when the context closes."""
        client = wrap(FakeClient(stream_chunks=self._chunks()))
        with client.chat.completions.create(
            model="gpt-test",
            stream=True,
            messages=[{"role": "developer", "content": make_prompt()}],
        ) as stream:
            for _chunk in stream:
                pass
        (run,) = history.runs("WRAP_SYS")
        assert run.output_text == "Hello!"

    def test_stream_without_prompts_is_not_proxied(self):
        """Untracked streams are returned raw — zero proxy overhead."""
        from tests.fakes import FakeStream

        client = wrap(FakeClient(stream_chunks=self._chunks()))
        stream = client.chat.completions.create(
            model="gpt-test",
            stream=True,
            messages=[{"role": "user", "content": "plain"}],
        )
        assert isinstance(stream, FakeStream)


class TestAsync:
    """The async interceptor and async stream proxy."""

    def test_async_create_tracked(self):
        """Awaiting a wrapped async create substitutes and records."""
        client = wrap(FakeAsyncClient())

        async def go():
            """Drive one async completion call."""
            return await client.chat.completions.create(
                model="gpt-test",
                messages=[{"role": "developer", "content": make_prompt()}],
            )

        response = asyncio.run(go())
        assert response.choices[0].message.content == "hello!"
        (call,) = client.calls
        assert type(call["messages"][0]["content"]) is str
        (run,) = history.runs("WRAP_SYS")
        assert run.status == "ok"

    def test_async_streaming(self):
        """`async for` streams pass through and record on exhaustion."""
        from types import SimpleNamespace

        chunks = [
            make_chunk(content="Hi "),
            make_chunk(content="there"),
            make_chunk(usage=SimpleNamespace(prompt_tokens=4, completion_tokens=2, total_tokens=6)),
        ]
        client = wrap(FakeAsyncClient(stream_chunks=chunks))

        async def go():
            """Consume the async stream fully and collect its chunks."""
            stream = await client.chat.completions.create(
                model="gpt-test",
                stream=True,
                messages=[{"role": "developer", "content": make_prompt()}],
            )
            return [chunk async for chunk in stream]

        received = asyncio.run(go())
        assert len(received) == 3
        (run,) = history.runs("WRAP_SYS")
        assert run.output_text == "Hi there"
        assert run.total_tokens == 6

    def test_async_error_recorded(self):
        """Async API failures record an error run and re-raise."""
        client = wrap(FakeAsyncClient(error=RuntimeError("boom")))

        async def go():
            """Drive one failing async completion call."""
            await client.chat.completions.create(
                model="gpt-test",
                messages=[{"role": "developer", "content": make_prompt()}],
            )

        with pytest.raises(RuntimeError, match="boom"):
            asyncio.run(go())
        (run,) = history.runs("WRAP_SYS")
        assert run.status == "error"


class TestDecoratedAsyncMethods:
    """The real AsyncOpenAI hides ``chat.completions.create`` behind a plain-def
    decorator, so it doesn't look like a coroutine function. Treated as sync,
    every async chat run was recorded the moment the coroutine was *created*:
    no output, no usage, zero latency. With the decorator's ``__wrapped__``
    trail the method is recognized up front; without one, the awaitable it
    returns gives it away."""

    @pytest.fixture(params=[True, False], ids=["wraps-trail", "no-trail"])
    def make_client(self, request):
        from tests.fakes import FakeDecoratedAsyncCompletions

        def build(**kwargs):
            client = FakeAsyncClient()
            client.chat.completions = FakeDecoratedAsyncCompletions(
                leave_trail=request.param, **kwargs
            )
            return wrap(client)

        return build

    def ask(self, client, **extra):
        return client.chat.completions.create(
            model="gpt-test", messages=[{"role": "developer", "content": make_prompt()}], **extra
        )

    def test_the_fake_really_hides_its_coroutine_function(self):
        import inspect

        from tests.fakes import FakeDecoratedAsyncCompletions

        assert not inspect.iscoroutinefunction(FakeDecoratedAsyncCompletions().create)

    def test_the_run_is_recorded_after_the_await_with_the_response(self, make_client):
        client = make_client(response=make_response(content="awaited answer", cost=0.01))
        response = asyncio.run(self.ask(client))
        assert response.choices[0].message.content == "awaited answer"
        (run,) = history.runs("WRAP_SYS")
        assert run.output_text == "awaited answer"
        assert run.total_tokens == 15
        assert run.cost_usd == 0.01
        assert response.promptkeep.run_key == run.run_key

    def test_provider_error_is_recorded_once_and_reraised(self, make_client):
        client = make_client(error=RuntimeError("boom"))
        with pytest.raises(RuntimeError, match="boom"):
            asyncio.run(self.ask(client))
        (run,) = history.runs("WRAP_SYS")
        assert run.status == "error"

    def test_streaming(self, make_client):
        client = make_client(stream_chunks=[make_chunk(content="Hi "), make_chunk(content="there")])

        async def go():
            return [chunk async for chunk in await self.ask(client, stream=True)]

        assert len(asyncio.run(go())) == 2
        (run,) = history.runs("WRAP_SYS")
        assert run.output_text == "Hi there"

    def test_a_post_check_audits_the_response_not_the_coroutine(self, make_client):
        from promptkeep import check
        from promptkeep.checks import Verdict

        seen = {}

        @check.post(mode="blocking")
        def look(ctx):
            seen["output"] = ctx.output_text
            return Verdict.ok()

        client = make_client(response=make_response(content="audited"))
        response = asyncio.run(self.ask(client, promptkeep_post=[look]))
        assert seen["output"] == "audited"
        assert response.promptkeep.verification == "ok"

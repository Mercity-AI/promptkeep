import asyncio

import pytest

from prompt_manager import Prompt, history, wrap
from prompt_manager import storage
from tests.fakes import (
    FakeAsyncClient,
    FakeClient,
    make_chunk,
    make_response,
)


def make_prompt(name="WRAP_SYS", focus="brevity"):
    return Prompt("You are helpful. Focus on {focus}.", {"focus": focus}, name=name)


class TestWrapDispatch:
    def test_wrap_class_keeps_name_and_behavior(self):
        Wrapped = wrap(FakeClient)
        assert Wrapped.__name__ == "FakeClient"
        client = Wrapped(api_key="sk-test")
        assert isinstance(client, FakeClient)
        assert client.api_key == "sk-test"

    def test_wrap_instance_returns_same_object(self):
        client = FakeClient()
        assert wrap(client) is client

    def test_wrap_rejects_non_clients(self):
        with pytest.raises(TypeError, match="OpenAI client"):
            wrap(42)

    def test_double_wrap_is_idempotent(self):
        client = wrap(wrap(FakeClient()))
        client.chat.completions.create(
            model="gpt-test", messages=[{"role": "user", "content": make_prompt()}]
        )
        assert len(client.calls) == 1
        assert len(history.runs("WRAP_SYS")) == 1


class TestSubstitution:
    def test_prompt_object_becomes_plain_string(self):
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
        client = wrap(FakeClient())
        p = make_prompt()
        messages = [{"role": "developer", "content": p}]
        client.chat.completions.create(model="gpt-test", messages=messages)
        assert messages[0]["content"] is p  # user's list untouched

    def test_content_block_lists(self):
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
        client = wrap(FakeClient())
        client.chat.completions.create(
            model="gpt-test", messages=[{"role": "user", "content": "just a string"}]
        )
        (call,) = client.calls
        assert call["messages"][0]["content"] == "just a string"
        assert history.runs("WRAP_SYS") == []


class TestRunTracking:
    def test_run_recorded_with_response_metadata(self):
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
        def boom(**kwargs):
            raise RuntimeError("db exploded")

        monkeypatch.setattr(storage, "record_run", boom)
        client = wrap(FakeClient())
        response = client.chat.completions.create(
            model="gpt-test", messages=[{"role": "developer", "content": make_prompt()}]
        )
        assert response.choices[0].message.content == "hello!"


class TestStreaming:
    def _chunks(self):
        from types import SimpleNamespace

        return [
            make_chunk(content="Hel"),
            make_chunk(content="lo!"),
            make_chunk(
                usage=SimpleNamespace(prompt_tokens=8, completion_tokens=2, total_tokens=10)
            ),
        ]

    def test_chunks_pass_through_and_run_recorded_at_end(self):
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
        from tests.fakes import FakeStream

        client = wrap(FakeClient(stream_chunks=self._chunks()))
        stream = client.chat.completions.create(
            model="gpt-test",
            stream=True,
            messages=[{"role": "user", "content": "plain"}],
        )
        assert isinstance(stream, FakeStream)


class TestAsync:
    def test_async_create_tracked(self):
        client = wrap(FakeAsyncClient())

        async def go():
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
        from types import SimpleNamespace

        chunks = [
            make_chunk(content="Hi "),
            make_chunk(content="there"),
            make_chunk(usage=SimpleNamespace(prompt_tokens=4, completion_tokens=2, total_tokens=6)),
        ]
        client = wrap(FakeAsyncClient(stream_chunks=chunks))

        async def go():
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
        client = wrap(FakeAsyncClient(error=RuntimeError("boom")))

        async def go():
            await client.chat.completions.create(
                model="gpt-test",
                messages=[{"role": "developer", "content": make_prompt()}],
            )

        with pytest.raises(RuntimeError, match="boom"):
            asyncio.run(go())
        (run,) = history.runs("WRAP_SYS")
        assert run.status == "error"

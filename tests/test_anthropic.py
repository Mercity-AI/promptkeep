"""Real Anthropic SDK contract over an offline HTTP transport."""

import asyncio
import json

import anthropic
import httpx2
import pytest

import promptkeep
from promptkeep import Prompt, history


def message():
    """A complete native message with cache-aware usage."""
    return dict(
        id="msg_1",
        type="message",
        role="assistant",
        model="claude-test",
        content=[dict(type="text", text="hello")],
        stop_reason="end_turn",
        stop_sequence=None,
        usage=dict(
            input_tokens=3,
            output_tokens=2,
            cache_read_input_tokens=5,
            cache_creation_input_tokens=7,
        ),
    )


def events():
    """The same message as raw SSE, with cumulative output-token updates."""
    start = message()
    start.update(content=[], stop_reason=None)
    start["usage"]["output_tokens"] = 0
    return [
        dict(type="message_start", message=start),
        dict(type="content_block_start", index=0, content_block=dict(type="text", text="")),
        dict(type="content_block_delta", index=0, delta=dict(type="text_delta", text="hello")),
        dict(type="content_block_stop", index=0),
        dict(
            type="message_delta",
            delta=dict(stop_reason="end_turn", stop_sequence=None),
            usage=dict(output_tokens=2),
        ),
        dict(type="message_stop"),
    ]


def client(async_=False):
    """Use the real SDK serializer/parser, never the network."""

    def handle(request):
        body = json.loads(request.content)
        assert isinstance(body["system"], str)
        if body.get("stream"):
            data = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events())
            return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=data)
        return httpx2.Response(200, json=message())

    transport = httpx2.MockTransport(handle)
    if async_:
        return promptkeep.wrap(
            anthropic.AsyncAnthropic(
                api_key="test", max_retries=0, http_client=httpx2.AsyncClient(transport=transport)
            )
        )
    return promptkeep.wrap(
        anthropic.Anthropic(
            api_key="test", max_retries=0, http_client=httpx2.Client(transport=transport)
        )
    )


def kwargs():
    """A tracked system prompt and one plain user message."""
    return dict(
        model="claude-test",
        max_tokens=20,
        system=Prompt("Be kind", name="SYSTEM"),
        messages=[dict(role="user", content="hi")],
    )


def assert_run():
    """Usage is persisted without multiplying cache counters or snapshots."""
    rows = history.runs("SYSTEM")
    assert len(rows) == 1
    row = rows[0]
    assert row.output_text == "hello"
    assert row.prompt_tokens == 15
    assert row.completion_tokens == 2
    assert row.total_tokens == 17
    assert row.usage_details["input_tokens"] == 3


@pytest.mark.parametrize("surface", ["create", "stream", "helper", "final"])
def test_real_sync(surface):
    c = client()
    promptkeep.wrap(c)
    with promptkeep.agent("root") as root:
        if surface == "create":
            assert c.messages.create(**kwargs()).content[0].text == "hello"
        elif surface == "stream":
            stream = c.messages.create(**kwargs(), stream=True)
            list(stream)
            stream.close()
        else:
            with c.messages.stream(**kwargs()) as stream:
                if surface == "helper":
                    assert "".join(stream.text_stream) == "hello"
                else:
                    assert stream.get_final_message().content[0].text == "hello"
    assert_run()
    assert history.trace(root.trace_key).model_calls == 1
    c.close()


@pytest.mark.parametrize("surface", ["create", "stream", "helper", "final"])
def test_real_async(surface):
    async def run():
        c = client(True)
        async with promptkeep.agent("root") as root:
            if surface == "create":
                assert (await c.messages.create(**kwargs())).content[0].text == "hello"
            elif surface == "stream":
                stream = await c.messages.create(**kwargs(), stream=True)
                async for _ in stream:
                    pass
                await stream.close()
            else:
                async with c.messages.stream(**kwargs()) as stream:
                    if surface == "helper":
                        assert "".join([s async for s in stream.text_stream]) == "hello"
                    else:
                        assert (await stream.get_final_message()).content[0].text == "hello"
        await c.close()
        return root.trace_key

    key = asyncio.run(run())
    assert_run()
    assert history.trace(key).model_calls == 1


def test_explicit_close_records_partial_output_once():
    c = client()
    stream = c.messages.create(**kwargs(), stream=True)
    iterator = iter(stream)
    for _ in range(3):
        next(iterator)
    stream.close()
    stream.close()
    assert len(history.runs("SYSTEM")) == 1
    assert history.runs("SYSTEM")[0].output_text == "hello"
    c.close()


@pytest.mark.parametrize("async_", [False, True])
@pytest.mark.parametrize("helper", [False, True])
def test_blocked_streams_never_call_provider(async_, helper):
    @promptkeep.check.pre(name="gate")
    def gate(ctx):
        return promptkeep.Verdict.block("blocked")

    promptkeep.configure(pre=[gate], on_block="return")
    c = client(async_)
    if async_:

        async def run():
            if helper:
                async with c.messages.stream(**kwargs()) as stream:
                    assert (await stream.get_final_message()).content == []
            else:
                stream = await c.messages.create(**kwargs(), stream=True)
                assert len([e async for e in stream]) == 3
                await stream.close()
            await c.close()

        asyncio.run(run())
    elif helper:
        with c.messages.stream(**kwargs()) as stream:
            assert stream.get_final_message().content == []
        c.close()
    else:
        with c.messages.create(**kwargs(), stream=True) as stream:
            assert len(list(stream)) == 3
        c.close()
    assert history.runs("SYSTEM")[0].status == "blocked"


def test_helper_failure_preserves_exception_and_partial_output():
    c = client()
    failure = RuntimeError("application failed")
    with pytest.raises(RuntimeError) as caught:
        with c.messages.stream(**kwargs()) as stream:
            assert next(iter(stream.text_stream)) == "hello"
            raise failure
    assert caught.value is failure
    assert history.runs("SYSTEM")[0].status == "error"
    c.close()


def test_async_helper_cancellation_records_once():
    async def run():
        c = client(True)
        with pytest.raises(asyncio.CancelledError):
            async with c.messages.stream(**kwargs()) as stream:
                assert await anext(stream.text_stream) == "hello"
                raise asyncio.CancelledError()
        await c.close()

    asyncio.run(run())
    assert len(history.runs("SYSTEM")) == 1
    assert history.runs("SYSTEM")[0].status == "cancelled"


def test_structured_request_generators_rewrite_and_tool_links():
    from types import SimpleNamespace

    from promptkeep.integrations import AnthropicMessagesAdapter

    adapter = AnthropicMessagesAdapter()
    image = dict(type="image", source=dict(type="base64", data="opaque"))
    messages = [
        dict(
            role="user",
            content=[
                image,
                dict(type="text", text="old"),
                dict(type="tool_result", tool_use_id="call", content="tool text"),
            ],
        )
    ]
    request = adapter.parse_request(
        dict(
            system=[dict(type="text", text=Prompt("system", name="STRUCTURED"))],
            messages=(m for m in messages),
            model="claude-test",
        )
    )
    assert "tool text" in request.joined_text
    rewritten = adapter.apply_rewrite(request, "new")
    assert rewritten.kwargs["messages"][0]["content"][0] == image
    assert messages[0]["content"][1]["text"] == "old"
    assert rewritten.kwargs["messages"][0]["content"][1]["text"] == "new"
    fields = adapter.read_response(
        SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking="private"),
                SimpleNamespace(type="tool_use", id="call"),
            ]
        )
    )
    assert fields.output_text is None
    assert fields.tool_call_ids == ("call",)


def test_pre_rewrite_and_post_check_use_same_physical_call():
    seen = []

    @promptkeep.check.pre(name="rewrite")
    def rewrite(ctx):
        return promptkeep.Verdict.rewrite("sanitized")

    @promptkeep.check.post(name="audit", mode="blocking")
    def audit(ctx):
        seen.append((ctx.last_text, ctx.output_text))
        return promptkeep.Verdict.ok()

    promptkeep.configure(pre=[rewrite], post=[audit])
    c = client()
    with c.messages.stream(**kwargs()) as stream:
        stream.get_final_message()
    assert seen == [("sanitized", "hello")]
    assert len(history.runs("SYSTEM")) == 1
    c.close()


def test_lazy_manager_and_class_wrapping():
    c = promptkeep.wrap(anthropic.Anthropic)(api_key="test")
    manager = c.messages.stream(**kwargs())
    assert history.all_runs() == []
    assert history.list_prompts() == []
    del manager
    c.close()


def test_provider_error_is_re_raised_and_recorded():
    def fail(request):
        return httpx2.Response(
            500,
            json={"type": "error", "error": {"type": "api_error", "message": "fixture failure"}},
        )

    c = promptkeep.wrap(
        anthropic.Anthropic(
            api_key="test",
            max_retries=0,
            http_client=httpx2.Client(transport=httpx2.MockTransport(fail)),
        )
    )
    with pytest.raises(anthropic.InternalServerError, match="fixture failure"):
        with c.messages.stream(**kwargs()) as stream:
            stream.get_final_message()
    assert history.runs("SYSTEM")[0].status == "error"
    c.close()


def test_async_cancel_before_response_records_cancelled():
    async def fail(request):
        raise asyncio.CancelledError()

    async def run():
        c = promptkeep.wrap(
            anthropic.AsyncAnthropic(
                api_key="test",
                max_retries=0,
                http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(fail)),
            )
        )
        with pytest.raises(asyncio.CancelledError):
            await c.messages.create(**kwargs())
        await c.close()

    asyncio.run(run())
    assert history.runs("SYSTEM")[0].status == "cancelled"

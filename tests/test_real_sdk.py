"""The real ``openai`` SDK, driven end to end through promptkeep — with an
httpx ``MockTransport`` standing in for the network.

The fakes in tests/fakes.py are what *we think* the SDK looks like. This file
is the SDK itself: its client classes, its method decorators, its response
parsing, its SSE stream decoding — only the HTTP layer is canned. It exists
because a hand-rolled fake can't catch a wrong assumption about the thing it
imitates: the real ``AsyncOpenAI.chat.completions.create`` hides its ``async
def`` behind a sync decorator, which a fake ``async def create`` never
modelled, and every async chat run was recorded empty until this file ran.

Skipped when ``openai`` isn't installed (it is in the dev group, not a
dependency of the package).
"""

import asyncio
import json

import httpx
import pytest

import promptkeep
from promptkeep import Prompt, check, configure, history, wrap
from promptkeep.checks import Verdict

openai = pytest.importorskip("openai")


# --- canned HTTP bodies, as the APIs return them --------------------------------------


def chat_body(text="Lima"):
    """A chat completion from an OpenRouter-style endpoint (``usage.cost`` included)."""
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "openai/gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": text},
            }
        ],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 3,
            "total_tokens": 14,
            "cost": 0.0000123,
            "cost_details": {"upstream_inference_cost": None},
        },
    }


def chat_stream():
    """The same reply as SSE chunks; usage (and cost) ride on the last one."""
    chunks = [
        {
            "id": "chatcmpl-s",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "openai/gpt-4o-mini",
            "choices": [{"index": 0, "delta": {"content": part}, "finish_reason": None}],
        }
        for part in ("Li", "ma")
    ]
    chunks.append(
        {
            "id": "chatcmpl-s",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "openai/gpt-4o-mini",
            "choices": [],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7, "cost": 0.5},
        }
    )
    return "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"


def response_body(response_id, text, previous=None):
    """A Responses API body."""
    return {
        "id": response_id,
        "object": "response",
        "created_at": 1,
        "model": "gpt-5.5",
        "status": "completed",
        "previous_response_id": previous,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "output": [
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "usage": {
            "input_tokens": 9,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 2,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 11,
        },
    }


def response_stream(response_id, parts, previous=None):
    """The same, as the typed event stream ``stream=True`` produces."""
    full = response_body(response_id, "".join(parts), previous)
    events = [
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": {**full, "status": "in_progress", "output": [], "usage": None},
        }
    ]
    for number, part in enumerate(parts, start=1):
        events.append(
            {
                "type": "response.output_text.delta",
                "sequence_number": number,
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "delta": part,
                "logprobs": [],
            }
        )
    events.append(
        {"type": "response.completed", "sequence_number": len(parts) + 1, "response": full}
    )
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


class Endpoint:
    """The mock server: answers both surfaces, numbers its responses, and
    keeps every request body so a test can see what actually went out."""

    def __init__(self):
        self.requests = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        sse = {"content-type": "text/event-stream"}
        if request.url.path.endswith("/chat/completions"):
            if body.get("stream"):
                return httpx.Response(200, text=chat_stream(), headers=sse)
            return httpx.Response(200, json=chat_body())
        response_id = f"resp_{sum('input' in r or 'instructions' in r for r in self.requests)}"
        previous = body.get("previous_response_id")
        if body.get("stream"):
            text = response_stream(response_id, ["Santi", "ago"], previous)
            return httpx.Response(200, text=text, headers=sse)
        return httpx.Response(
            200, json=response_body(response_id, f"answer {response_id}", previous)
        )


@pytest.fixture
def endpoint():
    return Endpoint()


@pytest.fixture
def client(endpoint):
    """A wrapped, real ``openai.OpenAI``."""
    transport = httpx.MockTransport(endpoint)
    return wrap(
        openai.OpenAI(
            api_key="sk-test",
            base_url="http://mock/v1",
            http_client=httpx.Client(transport=transport),
        )
    )


@pytest.fixture
def async_client(endpoint):
    """A wrapped, real ``openai.AsyncOpenAI``."""
    transport = httpx.MockTransport(endpoint)
    return wrap(
        openai.AsyncOpenAI(
            api_key="sk-test",
            base_url="http://mock/v1",
            http_client=httpx.AsyncClient(transport=transport),
        )
    )


SYSTEM = Prompt("Answer in {n} words.", {"n": 3}, name="REAL_SDK")
MESSAGES = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Capital of Peru?"}]


class TestChatCompletions:
    def test_a_call_records_text_usage_and_reported_cost(self, client, endpoint):
        response = client.chat.completions.create(model="m", messages=MESSAGES)
        assert response.choices[0].message.content == "Lima"
        (sent,) = endpoint.requests
        assert sent["messages"][0] == {"role": "system", "content": "Answer in 3 words."}
        (run,) = history.runs("REAL_SDK")
        assert (run.output_text, run.total_tokens, run.response_id) == ("Lima", 14, "chatcmpl-1")
        assert run.cost_usd == pytest.approx(0.0000123)

    def test_the_handle_rides_on_the_pydantic_response_without_leaking_into_it(self, client):
        response = client.chat.completions.create(model="m", messages=MESSAGES)
        assert response.promptkeep.run_key == history.runs("REAL_SDK")[0].run_key
        assert "promptkeep" not in response.model_dump()
        assert "promptkeep" not in response.model_dump_json()

    def test_promptkeep_kwargs_never_reach_the_wire(self, client, endpoint):
        @check.pre()
        def fine(ctx):
            return Verdict.ok()

        client.chat.completions.create(
            model="m", messages=MESSAGES, promptkeep_pre=[fine], promptkeep_conversation="sess"
        )
        assert not any("promptkeep" in key for key in endpoint.requests[0])

    def test_a_stream(self, client):
        stream = client.chat.completions.create(model="m", messages=MESSAGES, stream=True)
        text = "".join(c.choices[0].delta.content or "" for c in stream if c.choices)
        assert text == "Lima"
        (run,) = history.runs("REAL_SDK")
        assert (run.output_text, run.total_tokens, run.cost_usd) == ("Lima", 7, 0.5)

    def test_a_stream_used_as_a_context_manager(self, client):
        with client.chat.completions.create(model="m", messages=MESSAGES, stream=True) as stream:
            for _chunk in stream:
                pass
        assert len(history.runs("REAL_SDK")) == 1

    def test_the_async_client(self, async_client):
        """The regression: ``AsyncOpenAI.chat.completions.create`` is an async
        def behind a sync decorator, and does not look like a coroutine function."""
        import inspect

        assert not inspect.iscoroutinefunction(
            openai.AsyncOpenAI(api_key="x").chat.completions.create
        )

        async def go():
            return await async_client.chat.completions.create(model="m", messages=MESSAGES)

        response = asyncio.run(go())
        assert response.choices[0].message.content == "Lima"
        (run,) = history.runs("REAL_SDK")
        assert (run.output_text, run.total_tokens) == ("Lima", 14)
        assert run.cost_usd == pytest.approx(0.0000123)
        assert response.promptkeep.run_key == run.run_key

    def test_an_async_stream(self, async_client):
        async def go():
            stream = await async_client.chat.completions.create(
                model="m", messages=MESSAGES, stream=True
            )
            return [chunk async for chunk in stream]

        assert len(asyncio.run(go())) == 3
        (run,) = history.runs("REAL_SDK")
        assert (run.output_text, run.total_tokens) == ("Lima", 7)

    def test_a_provider_error_is_recorded_and_reraised(self, endpoint):
        def failing(request):
            return httpx.Response(429, json={"error": {"message": "slow down", "type": "rate"}})

        client = wrap(
            openai.OpenAI(
                api_key="sk-test",
                base_url="http://mock/v1",
                max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(failing)),
            )
        )
        with pytest.raises(openai.RateLimitError):
            client.chat.completions.create(model="m", messages=MESSAGES)
        (run,) = history.runs("REAL_SDK")
        assert run.status == "error" and "slow down" in run.error


class TestResponses:
    def test_a_chain_is_followed_into_one_conversation(self, client, endpoint):
        first = client.responses.create(model="gpt-5.5", instructions=SYSTEM, input="Peru?")
        second = client.responses.create(
            model="gpt-5.5", instructions=SYSTEM, input="Chile?", previous_response_id=first.id
        )
        events = client.responses.create(
            model="gpt-5.5",
            instructions=SYSTEM,
            input="Bolivia?",
            previous_response_id=second.id,
            stream=True,
        )
        assert [e.type for e in events][-1] == "response.completed"
        assert first.output_text == "answer resp_1"  # the SDK's own convenience still works
        assert endpoint.requests[0]["instructions"] == "Answer in 3 words."

        convo = history.conversation("response:resp_1")
        assert [(t.turn_index, t.input_text, t.output_text) for t in convo.turns] == [
            (0, "Peru?", "answer resp_1"),
            (1, "Chile?", "answer resp_2"),
            (2, "Bolivia?", "Santiago"),
        ]
        assert [t.total_tokens for t in convo.turns] == [11, 11, 11]

    def test_the_sdks_own_output_items_can_be_fed_back_as_input(self, client):
        """The agent-loop idiom: ``input = previous.output + [new message]`` puts
        pydantic objects, not dicts, into the list."""
        first = client.responses.create(model="gpt-5.5", instructions=SYSTEM, input="Peru?")
        client.responses.create(
            model="gpt-5.5",
            instructions=SYSTEM,
            input=list(first.output) + [{"role": "user", "content": "thanks"}],
            previous_response_id=first.id,
        )
        assert history.conversation("response:resp_1").turns[1].input_text == "thanks"

    def test_async_and_background_writes_together(self, async_client):
        configure(write_mode="background")

        async def go():
            first = await async_client.responses.create(
                model="gpt-5.5", instructions=SYSTEM, input="Peru?"
            )
            stream = await async_client.responses.create(
                model="gpt-5.5",
                instructions=SYSTEM,
                input="Chile?",
                previous_response_id=first.id,
                stream=True,
            )
            async for _event in stream:
                pass

        asyncio.run(go())
        assert promptkeep.flush(timeout=5)
        convo = history.conversation("response:resp_1")
        assert [t.output_text for t in convo.turns] == ["answer resp_1", "Santiago"]


class TestWrappingTheClass:
    def test_wrap_the_class_then_construct(self, endpoint):
        Wrapped = wrap(openai.OpenAI)
        client = Wrapped(
            api_key="sk-test",
            base_url="http://mock/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(endpoint)),
        )
        assert isinstance(client, openai.OpenAI)
        client.chat.completions.create(model="m", messages=MESSAGES)
        client.responses.create(model="gpt-5.5", instructions=SYSTEM, input="hi")
        assert {run.provider for run in history.runs("REAL_SDK")} == {"openai", "openai-responses"}

    def test_call_works_on_a_real_client(self, client):
        result = promptkeep.call(client, model="gpt-5.5", instructions=SYSTEM, input="hi")
        assert result.text == "answer resp_1"
        assert promptkeep.call(client, model="m", messages=MESSAGES).text == "Lima"

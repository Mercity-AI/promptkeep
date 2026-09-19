"""Contract tests every provider adapter must pass.

Each adapter contributes a Scenario: how to build a fake client, a request
carrying a Prompt and a user turn, a response, and a stream — in *its*
provider's shapes. The tests below then ask the same questions of every
adapter, so a new provider gets its whole contract checked by adding one
Scenario here. The shared orchestration is exercised end to end by
test_openai_wrapper.py / test_checks.py / test_conversation.py; this file is
about the seam between it and an adapter.
"""

from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from promptkeep import Prompt, RenderedText, integrations, wrap
from promptkeep.integrations import (
    OpenAIChatAdapter,
    OpenAIResponsesAdapter,
    ProviderAdapter,
    Request,
    ResponseFields,
    StreamAbsorber,
    Target,
    is_wrapped,
    register_adapter,
    registry,
)
from tests.fakes import (
    FakeAsyncClient,
    FakeAsyncResponsesClient,
    FakeClient,
    FakeResponsesClient,
    make_chunk,
    make_response,
    make_responses_events,
    make_responses_response,
)


@dataclass
class Scenario:
    """One provider's shapes, for the contract below."""

    adapter: ProviderAdapter
    make_client: Callable[[], Any]
    make_async_client: Callable[[], Any]
    request: Callable[[Any, str], dict]  # (system prompt, user text) -> call kwargs
    system_only: Callable[[Any], dict]  # (system prompt,) -> kwargs with no user turn
    texts: Callable[[dict], list[Any]]  # every content value in the kwargs' payload
    response: Callable[[str], Any]  # reply text -> provider response object
    chunks: Callable[[list[str]], list]  # text pieces -> stream chunks (usage on the last)


def _chat_scenario() -> Scenario:
    return Scenario(
        adapter=OpenAIChatAdapter(),
        make_client=FakeClient,
        make_async_client=FakeAsyncClient,
        request=lambda system, user: {
            "model": "gpt-test",
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
        system_only=lambda system: {
            "model": "gpt-test",
            "messages": [{"role": "developer", "content": system}],
        },
        texts=lambda kwargs: [m["content"] for m in kwargs["messages"]],
        response=lambda text: make_response(content=text, model="gpt-answered"),
        chunks=lambda parts: (
            [make_chunk(content=p) for p in parts]
            + [
                make_chunk(
                    usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3, total_tokens=10)
                )
            ]
        ),
    )


def _responses_scenario() -> Scenario:
    """The Responses API: the system prompt is ``instructions``, the turn an
    ``input`` item list (test_responses.py covers the string-input spelling)."""
    return Scenario(
        adapter=OpenAIResponsesAdapter(),
        make_client=FakeResponsesClient,
        make_async_client=FakeAsyncResponsesClient,
        request=lambda system, user: {
            "model": "gpt-test",
            "temperature": 0,
            "instructions": system,
            "input": [{"role": "user", "content": user}],
        },
        system_only=lambda system: {"model": "gpt-test", "instructions": system},
        texts=lambda kwargs: (
            [kwargs["instructions"]] + [item["content"] for item in kwargs.get("input", [])]
        ),
        response=lambda text: make_responses_response(text, model="gpt-answered"),
        chunks=lambda parts: make_responses_events(parts),
    )


SCENARIOS = [_chat_scenario(), _responses_scenario()]


@pytest.fixture(params=SCENARIOS, ids=lambda s: s.adapter.provider)
def scenario(request):
    return request.param


@pytest.fixture
def prompt():
    return Prompt("You review {what}.", {"what": "code"}, name="ADAPTER_SYS")


class TestLocate:
    def test_finds_the_surface_on_a_client(self, scenario):
        target = scenario.adapter.locate(scenario.make_client())
        assert isinstance(target, Target)
        assert callable(getattr(target.owner, target.attribute))

    def test_none_for_foreign_objects(self, scenario):
        assert scenario.adapter.locate(42) is None
        assert scenario.adapter.locate(object()) is None
        assert scenario.adapter.locate(SimpleNamespace(chat=None)) is None


class TestParseRequest:
    def test_substitutes_prompts_with_plain_strings_and_tracks_them(self, scenario, prompt):
        kwargs = scenario.request(prompt, "please review this")
        request = scenario.adapter.parse_request(kwargs)
        assert isinstance(request, Request)
        for text in scenario.texts(request.kwargs):
            assert type(text) is str  # exactly str: what the provider would get
        assert "You review code." in scenario.texts(request.kwargs)
        [(tracked_prompt, variables, rendered)] = request.tracked
        assert tracked_prompt is prompt
        assert variables == {"what": "code"}
        assert rendered == "You review code."

    def test_rendered_text_is_tracked_like_the_prompt(self, scenario, prompt):
        """prompt.text carries provenance; passing it must track the same run."""
        request = scenario.adapter.parse_request(scenario.request(prompt.text, "hi"))
        [(tracked_prompt, _vars, _rendered)] = request.tracked
        assert tracked_prompt is prompt
        assert all(type(t) is str for t in scenario.texts(request.kwargs))

    def test_bare_rendered_text_is_just_a_string(self, scenario):
        request = scenario.adapter.parse_request(scenario.request(RenderedText("plain"), "hi"))
        assert request.tracked == []

    def test_never_mutates_the_callers_kwargs(self, scenario, prompt):
        kwargs = scenario.request(prompt, "hi")
        before = [type(t) for t in scenario.texts(kwargs)]
        scenario.adapter.parse_request(kwargs)
        assert [type(t) for t in scenario.texts(kwargs)] == before
        assert isinstance(scenario.texts(kwargs)[0], Prompt)  # still the object we passed

    def test_current_turn_and_joined_text(self, scenario, prompt):
        request = scenario.adapter.parse_request(scenario.request(prompt, "please review this"))
        assert request.input_text == "please review this"
        assert "You review code." in request.joined_text
        assert "please review this" in request.joined_text

    def test_system_only_request_has_no_current_turn(self, scenario, prompt):
        request = scenario.adapter.parse_request(scenario.system_only(prompt))
        assert request.input_text is None
        assert request.tracked[0][0] is prompt

    def test_request_params_exclude_the_payload(self, scenario, prompt):
        request = scenario.adapter.parse_request(scenario.request(prompt, "hi"))
        assert request.request_params["model"] == "gpt-test"
        assert request.request_params["temperature"] == 0
        for value in request.request_params.values():
            assert not isinstance(value, (list, Prompt))  # no message payload leaked in

    def test_no_prompt_still_parses(self, scenario):
        request = scenario.adapter.parse_request(scenario.request("literal system", "hi"))
        assert request.tracked == []
        assert request.input_text == "hi"


class TestApplyRewrite:
    def test_replaces_the_current_turn_everywhere(self, scenario, prompt):
        adapter = scenario.adapter
        original = adapter.parse_request(scenario.request(prompt, "my card is 4111"))
        rewritten = adapter.apply_rewrite(original, "my card is [REDACTED]")
        assert rewritten.input_text == "my card is [REDACTED]"
        assert "my card is [REDACTED]" in scenario.texts(rewritten.kwargs)
        assert "my card is 4111" not in scenario.texts(rewritten.kwargs)
        assert "my card is [REDACTED]" in rewritten.joined_text
        assert "my card is 4111" not in rewritten.joined_text
        assert "You review code." in rewritten.joined_text  # the system prompt survives
        assert rewritten.tracked == original.tracked
        assert rewritten.request_params == original.request_params
        # The original Request is untouched.
        assert original.input_text == "my card is 4111"
        assert "my card is 4111" in scenario.texts(original.kwargs)

    def test_no_current_turn_means_no_rewrite(self, scenario, prompt):
        adapter = scenario.adapter
        original = adapter.parse_request(scenario.system_only(prompt))
        rewritten = adapter.apply_rewrite(original, "anything")
        assert rewritten.input_text is None
        assert scenario.texts(rewritten.kwargs) == scenario.texts(original.kwargs)


class TestReadResponse:
    def test_fields(self, scenario):
        fields = scenario.adapter.read_response(scenario.response("the answer"))
        assert isinstance(fields, ResponseFields)
        assert fields.output_text == "the answer"
        assert fields.model == "gpt-answered"
        assert fields.response_id
        assert fields.total_tokens == fields.prompt_tokens + fields.completion_tokens

    def test_unreadable_response_is_all_none_not_an_error(self, scenario):
        for junk in (None, object(), 42, SimpleNamespace(choices=[])):
            fields = scenario.adapter.read_response(junk)
            assert fields.output_text is None
            assert fields.total_tokens is None


class TestStreamAbsorber:
    def test_folds_chunks_into_a_summary(self, scenario):
        absorber = scenario.adapter.stream_absorber()
        assert isinstance(absorber, StreamAbsorber)
        for chunk in scenario.chunks(["Hel", "lo", " there"]):
            absorber.absorb(chunk)
        fields = absorber.summary()
        assert fields.output_text == "Hello there"
        assert fields.model
        assert fields.response_id
        assert (fields.prompt_tokens, fields.completion_tokens, fields.total_tokens) == (7, 3, 10)

    def test_empty_stream_summary(self, scenario):
        fields = scenario.adapter.stream_absorber().summary()
        assert fields.output_text is None
        assert fields.total_tokens is None

    def test_one_absorber_per_stream(self, scenario):
        assert scenario.adapter.stream_absorber() is not scenario.adapter.stream_absorber()


class TestBlockedStub:
    def test_carries_the_model_and_accepts_a_handle(self, scenario, prompt):
        request = scenario.adapter.parse_request(scenario.request(prompt, "hi"))
        blocked = SimpleNamespace(name="no_pii", message="email in prompt")
        stub = scenario.adapter.blocked_stub(request, blocked)
        assert getattr(stub, "model", None) == "gpt-test"
        stub.promptkeep = "handle"  # the orchestrator attaches a RunHandle here
        assert stub.promptkeep == "handle"


class TestStreamingFlag:
    def test_stream_kwarg(self, scenario, prompt):
        adapter = scenario.adapter
        assert adapter.is_streaming(adapter.parse_request(scenario.request(prompt, "hi"))) is False
        streaming = {**scenario.request(prompt, "hi"), "stream": True}
        assert adapter.is_streaming(adapter.parse_request(streaming)) is True


class TestWrapThroughTheRegistry:
    def test_wrap_instruments_every_located_surface(self, scenario):
        client = scenario.make_client()
        assert is_wrapped(client) is False
        assert wrap(client) is client
        assert is_wrapped(client) is True
        target = scenario.adapter.locate(client)
        assert getattr(target.owner, target.attribute).__wrapped__  # functools.wraps kept it

    def test_async_client_gets_the_async_interceptor(self, scenario):
        import inspect

        client = wrap(scenario.make_async_client())
        target = scenario.adapter.locate(client)
        assert inspect.iscoroutinefunction(getattr(target.owner, target.attribute))

    def test_unrecognized_object_is_none_not_false(self):
        assert is_wrapped(object()) is None


class TestRegistry:
    @pytest.fixture(autouse=True)
    def restore_registry(self):
        before = integrations.adapters()
        yield
        registry._ADAPTERS[:] = before

    def test_register_adapter_makes_wrap_recognize_a_new_client_shape(self, prompt):
        """A minimal third-party adapter: a client whose method is
        ``client.ask(prompt=..., question=...)``."""

        class Absorber(StreamAbsorber):
            def absorb(self, chunk):
                pass

            def summary(self):
                return ResponseFields()

        class AskAdapter(ProviderAdapter):
            provider = "askbot"

            def locate(self, client):
                return Target(client, "ask") if callable(getattr(client, "ask", None)) else None

            def parse_request(self, kwargs):
                prompt_value = kwargs.get("prompt")
                tracked, text = [], prompt_value
                if isinstance(prompt_value, Prompt):
                    rendered = prompt_value.text
                    tracked = [(prompt_value, rendered.variables, str(rendered))]
                    text = str(rendered)
                new_kwargs = {**kwargs, "prompt": text}
                return Request(
                    kwargs=new_kwargs,
                    tracked=tracked,
                    payload=text,
                    input_text=kwargs.get("question"),
                    joined_text=f"{text}\n{kwargs.get('question', '')}",
                    request_params={k: v for k, v in kwargs.items() if k not in ("prompt",)},
                )

            def apply_rewrite(self, request, text):
                return request

            def read_response(self, response):
                return ResponseFields(output_text=getattr(response, "answer", None))

            def stream_absorber(self):
                return Absorber()

            def blocked_stub(self, request, blocked):
                return SimpleNamespace(answer=None)

        class AskClient:
            def __init__(self):
                self.calls = []

            def ask(self, **kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(answer="42")

        with pytest.raises(TypeError, match="no supported provider surface"):
            wrap(AskClient())
        register_adapter(AskAdapter())
        client = wrap(AskClient())
        response = client.ask(prompt=prompt, question="meaning of life?")
        assert response.answer == "42"
        (call,) = client.calls
        assert call["prompt"] == "You review code."  # substituted
        from promptkeep import history

        (run,) = history.runs("ADAPTER_SYS")
        assert run.provider == "askbot"
        assert run.output_text == "42"
        assert run.request_params == {"question": "meaning of life?"}

    def test_register_adapter_rejects_non_adapters(self):
        with pytest.raises(TypeError, match="ProviderAdapter"):
            register_adapter(object())

    def test_registering_the_same_class_twice_replaces(self):
        first, second = OpenAIChatAdapter(), OpenAIChatAdapter()
        register_adapter(first)
        register_adapter(second)
        instances = [a for a in integrations.adapters() if isinstance(a, OpenAIChatAdapter)]
        assert instances == [second]

"""The OpenAI Responses API surface, end to end: request shapes, recording,
streaming, checks — and the part that is its own, conversations followed
automatically through ``previous_response_id``. The adapter's side of the
seam is covered by the contract in test_adapters.py; the last class here runs
it against the real SDK's objects when ``openai`` is installed."""

import asyncio
from types import SimpleNamespace

import pytest

import promptkeep
from promptkeep import Prompt, check, configure, conversation, history, storage, wrap
from promptkeep.checks import Verdict
from promptkeep.integrations import OpenAIResponsesAdapter, ProviderAdapter, registry
from tests.fakes import (
    FakeAsyncResponsesClient,
    FakeFullClient,
    FakeResponsesClient,
    make_responses_events,
    make_responses_response,
    make_responses_usage,
)


def make_prompt(name="RESP_SYS"):
    """A small tracked prompt to pass as ``instructions``."""
    return Prompt("You answer questions about {topic}.", {"topic": "rivers"}, name=name)


def replies(*texts):
    """One canned response per call: resp_1, resp_2, ... answering in order."""
    return [
        make_responses_response(text, response_id=f"resp_{number}")
        for number, text in enumerate(texts, start=1)
    ]


class TestRequestShapes:
    """Where Prompts can sit in a Responses call, and what is the current turn."""

    adapter = OpenAIResponsesAdapter()

    def test_string_input_is_the_current_turn(self):
        request = self.adapter.parse_request(
            {"model": "m", "instructions": make_prompt(), "input": "longest river?"}
        )
        assert request.kwargs["instructions"] == "You answer questions about rivers."
        assert type(request.kwargs["instructions"]) is str
        assert request.input_text == "longest river?"
        assert request.payload == [
            {"role": "system", "content": "You answer questions about rivers."},
            {"role": "user", "content": "longest river?"},
        ]
        assert [tracked[0].name for tracked in request.tracked] == ["RESP_SYS"]

    def test_a_prompt_can_be_the_input_itself(self):
        question = Prompt("Summarize {doc}.", {"doc": "the report"}, name="RESP_INPUT")
        request = self.adapter.parse_request({"model": "m", "input": question})
        assert request.kwargs["input"] == "Summarize the report."
        assert type(request.kwargs["input"]) is str
        assert request.input_text == "Summarize the report."
        assert [tracked[0].name for tracked in request.tracked] == ["RESP_INPUT"]

    def test_prompts_inside_input_items_and_input_text_blocks(self):
        system, block = make_prompt("RESP_ITEM"), make_prompt("RESP_BLOCK")
        items = [
            {"role": "developer", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": block},
                    {"type": "input_image", "image_url": "https://example.com/river.png"},
                ],
            },
        ]
        request = self.adapter.parse_request({"model": "m", "input": items})
        assert [tracked[0].name for tracked in request.tracked] == ["RESP_ITEM", "RESP_BLOCK"]
        sent = request.kwargs["input"]
        assert sent[0]["content"] == "You answer questions about rivers."
        assert sent[1]["content"][0]["text"] == "You answer questions about rivers."
        assert sent[1]["content"][1] == items[1]["content"][1]  # the image block is untouched
        assert isinstance(items[0]["content"], Prompt)  # the caller's list is not mutated

    def test_a_tool_output_item_is_not_a_turn(self):
        """An agent loop's follow-up carries a function_call_output, not a message."""
        items = [{"type": "function_call_output", "call_id": "call_1", "output": '{"km": 6650}'}]
        request = self.adapter.parse_request(
            {"model": "m", "input": items, "previous_response_id": "resp_1"}
        )
        assert request.input_text is None
        assert request.kwargs["input"] == items

    def test_request_params_keep_everything_but_the_text(self):
        request = self.adapter.parse_request(
            {
                "model": "m",
                "instructions": make_prompt(),
                "input": "hi",
                "previous_response_id": "resp_9",
                "store": True,
            }
        )
        assert request.request_params == {
            "model": "m",
            "previous_response_id": "resp_9",
            "store": True,
        }

    def test_the_hint_is_the_previous_response_id(self):
        parse = self.adapter.parse_request
        assert self.adapter.conversation_hint(parse({"input": "hi"})) is None
        assert self.adapter.conversation_hint(parse({"previous_response_id": None})) is None
        hinted = parse({"input": "hi", "previous_response_id": "resp_9"})
        assert self.adapter.conversation_hint(hinted) == "resp_9"

    def test_rewrite_of_a_string_input(self):
        original = self.adapter.parse_request({"instructions": make_prompt(), "input": "card 4111"})
        rewritten = self.adapter.apply_rewrite(original, "card [REDACTED]")
        assert rewritten.kwargs["input"] == "card [REDACTED]"
        assert rewritten.input_text == "card [REDACTED]"
        assert "4111" not in rewritten.joined_text
        assert "rivers" in rewritten.joined_text
        assert original.kwargs["input"] == "card 4111"


class TestRecording:
    def test_a_tracked_call_records_a_run(self):
        usage = make_responses_usage(input_tokens=20, output_tokens=10, cost=0.0007)
        response = make_responses_response(
            "The Nile.", model="gpt-5.5", response_id="resp_77", usage=usage
        )
        client = wrap(FakeResponsesClient(responses=[response]))
        returned = client.responses.create(
            model="gpt-5.5", temperature=0.2, instructions=make_prompt(), input="longest river?"
        )
        assert returned is response
        (sent,) = client.calls
        assert sent["instructions"] == "You answer questions about rivers."
        (run,) = history.runs("RESP_SYS")
        assert run.provider == "openai-responses"
        assert run.model == "gpt-5.5"
        assert run.response_id == "resp_77"
        assert run.input_text == "longest river?"
        assert run.output_text == "The Nile."
        assert (run.prompt_tokens, run.completion_tokens, run.total_tokens) == (20, 10, 30)
        assert run.cost_usd == pytest.approx(0.0007)
        assert run.request_params == {"model": "gpt-5.5", "temperature": 0.2}
        assert returned.promptkeep.run_key == run.run_key

    def test_an_untracked_call_records_nothing(self):
        client = wrap(FakeResponsesClient())
        client.responses.create(model="m", input="no prompt here")
        assert history.all_runs() == []

    def test_a_provider_error_is_recorded_and_reraised(self):
        client = wrap(FakeResponsesClient(error=RuntimeError("rate limited")))
        with pytest.raises(RuntimeError, match="rate limited"):
            client.responses.create(model="m", instructions=make_prompt(), input="hi")
        (run,) = history.runs("RESP_SYS")
        assert run.status == "error"
        assert "rate limited" in run.error

    def test_a_reply_with_no_text_is_none(self):
        """A pure tool call has output items but no message."""
        response = make_responses_response(response_id="resp_tool")
        response.output = [SimpleNamespace(type="function_call", name="lookup", arguments="{}")]
        client = wrap(FakeResponsesClient(responses=[response]))
        client.responses.create(model="m", instructions=make_prompt(), input="hi")
        (run,) = history.runs("RESP_SYS")
        assert run.output_text is None
        assert run.response_id == "resp_tool"

    def test_async_client(self):
        client = wrap(FakeAsyncResponsesClient(responses=replies("async answer")))

        async def go():
            return await client.responses.create(model="m", instructions=make_prompt(), input="hi")

        response = asyncio.run(go())
        (run,) = history.runs("RESP_SYS")
        assert run.output_text == "async answer"
        assert response.promptkeep.run_key == run.run_key

    def test_both_surfaces_of_one_client_are_instrumented(self):
        client = wrap(FakeFullClient())
        client.chat.completions.create(
            model="m", messages=[{"role": "system", "content": make_prompt("VIA_CHAT")}]
        )
        client.responses.create(model="m", instructions=make_prompt("VIA_RESPONSES"))
        assert history.runs("VIA_CHAT")[0].provider == "openai"
        assert history.runs("VIA_RESPONSES")[0].provider == "openai-responses"

    def test_call_routes_by_how_the_kwargs_are_spelled(self):
        client = wrap(FakeFullClient())
        via_responses = promptkeep.call(client, model="m", instructions=make_prompt(), input="hi")
        assert via_responses.text == "hello!"
        assert len(client.responses.calls) == 1 and client.calls == []
        promptkeep.call(client, model="m", messages=[{"role": "user", "content": "hi"}])
        assert len(client.calls) == 1

    def test_call_with_kwargs_no_surface_takes(self):
        with pytest.raises(TypeError, match="takes these arguments"):
            promptkeep.call(wrap(FakeFullClient()), model="m", prompt="neither spelling")


class TestStreaming:
    def events(self):
        usage = make_responses_usage(7, 3, cost=0.002)
        return make_responses_events(["The ", "Nile", "."], response_id="resp_s1", usage=usage)

    def test_a_stream_records_once_it_has_drained(self):
        client = wrap(FakeResponsesClient(stream_events=self.events()))
        stream = client.responses.create(
            model="m", instructions=make_prompt(), input="hi", stream=True
        )
        assert history.runs("RESP_SYS") == []  # nothing until the stream ends
        received = list(stream)
        assert [e.type for e in received][0] == "response.created"
        assert len(received) == 5
        (run,) = history.runs("RESP_SYS")
        assert run.output_text == "The Nile."
        assert run.response_id == "resp_s1"
        assert (run.prompt_tokens, run.completion_tokens, run.total_tokens) == (7, 3, 10)
        assert run.cost_usd == pytest.approx(0.002)
        assert stream.promptkeep.run_key == run.run_key

    def test_async_stream(self):
        client = wrap(FakeAsyncResponsesClient(stream_events=self.events()))

        async def go():
            stream = await client.responses.create(
                model="m", instructions=make_prompt(), input="hi", stream=True
            )
            return [event async for event in stream]

        assert len(asyncio.run(go())) == 5
        (run,) = history.runs("RESP_SYS")
        assert run.output_text == "The Nile."
        assert run.total_tokens == 10

    def test_a_stream_that_breaks_keeps_what_arrived(self):
        """No terminal event ever came: the deltas are all there is."""
        events = self.events()[:3]  # created, "The ", "Nile" — then the line drops

        class BreakingStream:
            def __iter__(self):
                yield from events
                raise ConnectionError("dropped")

        client = FakeResponsesClient()
        client.responses.create = lambda **kwargs: BreakingStream()
        stream = wrap(client).responses.create(
            model="m", instructions=make_prompt(), input="hi", stream=True
        )
        with pytest.raises(ConnectionError):
            list(stream)
        (run,) = history.runs("RESP_SYS")
        assert run.status == "error"
        assert run.output_text == "The Nile"
        assert run.response_id == "resp_s1"  # from the created event


class TestChaining:
    """previous_response_id groups calls into a conversation with no user code."""

    def chain(self, client, count=3):
        """A chain of ``count`` calls, each naming the one before it."""
        previous = None
        for number in range(1, count + 1):
            response = client.responses.create(
                model="m",
                instructions=make_prompt(),
                input=f"question {number}",
                **({"previous_response_id": previous} if previous else {}),
            )
            previous = response.id

    def assert_one_thread(self, count=3):
        """Every call of the chain is one ordered conversation."""
        (summary,) = history.list_conversations()
        assert summary.external_id == "response:resp_1"
        assert summary.turn_count == count
        convo = history.conversation("response:resp_1")
        assert [t.turn_index for t in convo.turns] == list(range(count))
        assert [t.input_text for t in convo.turns] == [f"question {n}" for n in range(1, count + 1)]
        return convo

    def test_a_chain_becomes_one_conversation(self):
        self.chain(wrap(FakeResponsesClient(responses=replies("a1", "a2", "a3"))))
        convo = self.assert_one_thread()
        assert convo.replay() == [
            {"role": "system", "content": "You answer questions about rivers."},
            {"role": "user", "content": "question 1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "question 2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "question 3"},
            {"role": "assistant", "content": "a3"},
        ]

    def test_the_first_call_alone_is_not_a_conversation(self):
        self.chain(wrap(FakeResponsesClient(responses=replies("a1"))), count=1)
        assert history.list_conversations() == []
        (run,) = history.runs("RESP_SYS")
        assert run.conversation_id is None

    def test_in_background_mode_the_predecessor_may_still_be_queued(self):
        configure(write_mode="background")
        self.chain(wrap(FakeResponsesClient(responses=replies("a1", "a2", "a3"))))
        assert promptkeep.flush(timeout=5)
        self.assert_one_thread()

    def test_a_chain_continues_across_processes(self):
        """A later process knows nothing in memory: the database has the link."""
        client = wrap(FakeResponsesClient(responses=replies("a1", "a2", "a3")))
        self.chain(client, count=2)
        storage.reset_caches()
        client.responses.create(
            model="m", instructions=make_prompt(), input="question 3", previous_response_id="resp_2"
        )
        self.assert_one_thread()

    def test_a_follow_up_with_no_prompt_is_still_a_turn(self):
        client = wrap(FakeResponsesClient(responses=replies("a1", "a2")))
        first = client.responses.create(model="m", instructions=make_prompt(), input="question 1")
        client.responses.create(model="m", input="question 2", previous_response_id=first.id)
        convo = history.conversation("response:resp_1")
        assert [(t.turn_index, t.prompt_name) for t in convo.turns] == [(0, "RESP_SYS"), (1, None)]

    def test_a_chain_promptkeep_never_tracked_is_not_followed(self):
        client = wrap(FakeResponsesClient(responses=replies("a1", "a2")))
        first = client.responses.create(model="m", input="no prompt")
        client.responses.create(model="m", input="still none", previous_response_id=first.id)
        assert history.all_runs() == []
        assert history.list_conversations() == []

    def test_an_explicit_conversation_wins_over_the_hint(self):
        client = wrap(FakeResponsesClient(responses=replies("a1", "a2")))
        first = client.responses.create(model="m", instructions=make_prompt(), input="question 1")
        with conversation("my-session"):
            client.responses.create(
                model="m",
                instructions=make_prompt(),
                input="question 2",
                previous_response_id=first.id,
            )
        assert [c.external_id for c in history.list_conversations()] == ["my-session"]
        assert len(history.conversation("my-session").turns) == 1

    def test_a_chain_begun_inside_a_conversation_stays_in_it(self):
        client = wrap(FakeResponsesClient(responses=replies("a1", "a2")))
        with conversation("my-session"):
            first = client.responses.create(
                model="m", instructions=make_prompt(), input="question 1"
            )
        client.responses.create(
            model="m", instructions=make_prompt(), input="question 2", previous_response_id=first.id
        )
        assert [c.external_id for c in history.list_conversations()] == ["my-session"]
        assert [t.turn_index for t in history.conversation("my-session").turns] == [0, 1]

    def test_a_first_call_with_two_prompts_is_adopted_as_one_turn(self):
        client = wrap(FakeResponsesClient(responses=replies("a1", "a2")))
        first = client.responses.create(
            model="m", instructions=make_prompt("RESP_A"), input=make_prompt("RESP_B")
        )
        client.responses.create(model="m", input="question 2", previous_response_id=first.id)
        turns = history.conversation("response:resp_1").turns
        assert [(t.turn_index, t.prompt_name) for t in turns] == [
            (0, "RESP_A"),
            (0, "RESP_B"),
            (1, None),
        ]

    def test_a_streamed_response_can_be_chained_onto(self):
        events = make_responses_events(["a1"], response_id="resp_1")
        client = wrap(FakeResponsesClient(responses=replies("unused", "a2"), stream_events=events))
        list(
            client.responses.create(
                model="m", instructions=make_prompt(), input="question 1", stream=True
            )
        )
        client.responses.create(model="m", input="question 2", previous_response_id="resp_1")
        convo = history.conversation("response:resp_1")
        assert [t.output_text for t in convo.turns] == ["a1", "a2"]

    def test_an_unknown_previous_response_is_just_an_ordinary_call(self):
        client = wrap(FakeResponsesClient(responses=replies("a1")))
        client.responses.create(
            model="m", instructions=make_prompt(), input="hi", previous_response_id="resp_gone"
        )
        (run,) = history.runs("RESP_SYS")
        assert run.conversation_id is None

    def test_each_link_names_its_predecessor_as_parent(self):
        self.chain(wrap(FakeResponsesClient(responses=replies("a1", "a2", "a3"))))
        turns = history.conversation("response:resp_1").turns
        assert [t.parent_run_key for t in turns] == [None, turns[0].run_key, turns[1].run_key]
        assert history.conversation("response:resp_1").forks == {}

    def test_continuing_an_older_response_is_a_branch(self):
        """previous_response_id is how the Responses API forks: pointing back
        past the latest reply starts a new branch from that one."""
        client = wrap(FakeResponsesClient(responses=replies("a1", "a2", "a3")))
        self.chain(client, count=2)
        client.responses.create(
            model="m", instructions=make_prompt(), input="question 3", previous_response_id="resp_1"
        )
        convo = history.conversation("response:resp_1")
        assert convo.forks == {2: 0}
        assert [leaf.input_text for leaf in convo.leaves] == ["question 2", "question 3"]
        users = [m["content"] for m in convo.replay() if m["role"] == "user"]
        assert users == ["question 1", "question 3"]

    def test_the_parent_is_found_while_the_predecessor_is_queued(self):
        configure(write_mode="background")
        self.chain(wrap(FakeResponsesClient(responses=replies("a1", "a2"))), count=2)
        assert promptkeep.flush(timeout=5)
        first, second = history.conversation("response:resp_1").turns
        assert second.parent_run_key == first.run_key

    def test_a_call_with_two_prompts_is_continued_from_its_primary_run(self):
        client = wrap(FakeResponsesClient(responses=replies("a1", "a2")))
        first = client.responses.create(
            model="m", instructions=make_prompt("RESP_A"), input=make_prompt("RESP_B")
        )
        client.responses.create(model="m", input="question 2", previous_response_id=first.id)
        turns = history.conversation("response:resp_1").turns
        assert turns[2].parent_run_key == first.promptkeep.run_key == turns[0].run_key

    def test_an_explicit_parent_wins_over_the_chain(self):
        client = wrap(FakeResponsesClient(responses=replies("a1", "a2", "a3")))
        self.chain(client, count=2)
        root = history.conversation("response:resp_1").turns[0].run_key
        client.responses.create(
            model="m", input="question 3", previous_response_id="resp_2", promptkeep_parent=root
        )
        assert history.conversation("response:resp_1").forks == {2: 0}

    def test_a_failing_hint_costs_the_grouping_not_the_call(self, caplog):
        class BrokenHint(OpenAIResponsesAdapter):
            def conversation_hint(self, request):
                raise RuntimeError("adapter bug")

        before = registry.adapters()
        try:
            registry.register_adapter(BrokenHint())
            registry._ADAPTERS[:] = [a for a in registry._ADAPTERS if type(a) is BrokenHint]
            client = wrap(FakeResponsesClient(responses=replies("a1")))
            with caplog.at_level("WARNING", logger="promptkeep"):
                response = client.responses.create(
                    model="m", instructions=make_prompt(), input="hi", previous_response_id="x"
                )
        finally:
            registry._ADAPTERS[:] = before
        assert response.id == "resp_1"
        assert "conversation hint" in caplog.text
        assert len(history.runs("RESP_SYS")) == 1

    def test_adapters_without_the_notion_hint_nothing(self):
        assert ProviderAdapter.conversation_hint(OpenAIResponsesAdapter(), None) is None


class TestChecks:
    def test_a_pre_check_rewrite_reaches_the_wire_and_the_row(self):
        @check.pre()
        def scrub(ctx):
            return Verdict.rewrite(ctx.last_text.replace("4111", "[card]"))

        client = wrap(FakeResponsesClient())
        client.responses.create(
            model="m", instructions=make_prompt(), input="my card is 4111", promptkeep_pre=[scrub]
        )
        (sent,) = client.calls
        assert sent["input"] == "my card is [card]"
        assert "promptkeep_pre" not in sent
        (run,) = history.runs("RESP_SYS")
        assert run.input_text == "my card is [card]"
        assert run.original_input_text == "my card is 4111"

    def test_a_check_reads_the_call_as_messages(self):
        seen = {}

        @check.pre()
        def look(ctx):
            seen["messages"] = ctx.messages
            seen["provider"] = ctx.provider
            return Verdict.ok()

        wrap(FakeResponsesClient()).responses.create(
            model="m", instructions=make_prompt(), input="hi", promptkeep_pre=[look]
        )
        assert [m["role"] for m in seen["messages"]] == ["system", "user"]
        assert seen["provider"] == "openai-responses"

    def test_a_blocked_call_returns_a_response_shaped_stub(self):
        @check.pre()
        def never(ctx):
            return Verdict.block("not today")

        configure(on_block="return")
        client = wrap(FakeResponsesClient())
        stub = client.responses.create(
            model="m", instructions=make_prompt(), input="hi", promptkeep_pre=[never]
        )
        assert client.calls == []  # the provider was never called
        assert stub.output_text == "" and stub.output == []
        assert stub.promptkeep.verification == "failed"
        assert history.runs("RESP_SYS")[0].status == "blocked"

    def test_a_post_check_sees_the_reply(self):
        @check.post(mode="blocking")
        def short(ctx):
            return Verdict.warn("long") if len(ctx.output_text) > 3 else Verdict.ok()

        client = wrap(FakeResponsesClient(responses=replies("a long reply")))
        response = client.responses.create(
            model="m", instructions=make_prompt(), input="hi", promptkeep_post=[short]
        )
        assert response.promptkeep.verification == "warn"


class TestAgainstTheRealSDK:
    """The fakes above are hand-built; these build the SDK's own objects, so a
    shape the adapter misreads shows up here. Skipped without ``openai``."""

    @pytest.fixture
    def types(self):
        return pytest.importorskip("openai.types.responses")

    def payload(self, **overrides):
        """A minimal Response body, as the API returns it."""
        body = {
            "id": "resp_real",
            "object": "response",
            "created_at": 1_700_000_000,
            "model": "gpt-5.5-2026",
            "status": "completed",
            "output": [
                {"id": "rs_1", "type": "reasoning", "summary": []},
                {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": "The Nile", "annotations": []},
                        {"type": "output_text", "text": ", probably.", "annotations": []},
                    ],
                },
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "usage": {
                "input_tokens": 12,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 4,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 16,
                "cost": 0.00031,
            },
        }
        return {**body, **overrides}

    def test_read_response(self, types):
        response = types.Response.model_validate(self.payload())
        fields = OpenAIResponsesAdapter().read_response(response)
        assert fields.output_text == "The Nile, probably." == response.output_text
        assert fields.model == "gpt-5.5-2026"
        assert fields.response_id == "resp_real"
        assert (fields.prompt_tokens, fields.completion_tokens, fields.total_tokens) == (12, 4, 16)
        assert fields.cost_usd == pytest.approx(0.00031)  # a field the SDK doesn't model

    def test_stream_events(self, types):
        created = types.ResponseCreatedEvent.model_validate(
            {
                "type": "response.created",
                "sequence_number": 0,
                "response": self.payload(status="in_progress", output=[], usage=None),
            }
        )
        deltas = [
            types.ResponseTextDeltaEvent.model_validate(
                {
                    "type": "response.output_text.delta",
                    "sequence_number": number,
                    "item_id": "msg_1",
                    "output_index": 1,
                    "content_index": 0,
                    "delta": text,
                    "logprobs": [],
                }
            )
            for number, text in enumerate(["The ", "Nile"], start=1)
        ]
        completed = types.ResponseCompletedEvent.model_validate(
            {"type": "response.completed", "sequence_number": 3, "response": self.payload()}
        )
        absorber = OpenAIResponsesAdapter().stream_absorber()
        for event in [created, *deltas, completed]:
            absorber.absorb(event)
        fields = absorber.summary()
        assert fields.output_text == "The Nile"
        assert fields.response_id == "resp_real"
        assert fields.total_tokens == 16

    def test_the_handle_attaches_to_a_real_response_object(self, types):
        response = types.Response.model_validate(self.payload())
        client = wrap(FakeResponsesClient(responses=[response]))
        returned = client.responses.create(model="m", instructions=make_prompt(), input="hi")
        assert returned is response
        (run,) = history.runs("RESP_SYS")
        assert returned.promptkeep.run_key == run.run_key
        assert run.output_text == "The Nile, probably."

"""Tests for conversation grouping: storage, the context-manager attach API,
and the history.conversation() read side."""

import asyncio

import pytest

import promptkeep
from promptkeep import Prompt, history, storage, wrap
from tests.fakes import FakeAsyncClient, FakeClient, make_chunk, make_response


class TestConversationStorage:
    """Conversation resolution and turn bookkeeping at the storage layer."""

    def test_get_or_create_is_idempotent(self):
        """Same external_id always resolves to the same row; metadata is
        only recorded on first creation, later calls don't overwrite it."""
        first = storage.get_or_create_conversation("sess-1", title="First")
        second = storage.get_or_create_conversation("sess-1", title="Ignored")
        assert first == second
        assert storage.fetch_conversation("sess-1")["title"] == "First"

    def test_unknown_conversation_is_none(self):
        """Reading a conversation that was never created returns None, not an error."""
        assert storage.fetch_conversation("nope") is None

    def test_turn_index_increments_per_conversation(self):
        """Each recorded run advances that conversation's next turn number."""
        cid = storage.get_or_create_conversation("sess-2")
        assert storage.next_turn_index(cid) == 0
        storage.record_run(provider="openai", conversation_id=cid, input_text="hi")
        assert storage.next_turn_index(cid) == 1
        storage.record_run(provider="openai", conversation_id=cid, input_text="again")
        assert storage.next_turn_index(cid) == 2

    def test_run_without_prompt_has_no_version(self):
        """A conversation turn with no wrapped Prompt still gets a row."""
        cid = storage.get_or_create_conversation("sess-3")
        storage.record_run(provider="openai", conversation_id=cid, input_text="q", output_text="a")
        (turn,) = history.conversation("sess-3").turns
        assert turn.version is None
        assert turn.prompt_name is None
        assert turn.input_text == "q"
        assert turn.output_text == "a"


class TestConversationHistory:
    """history.conversation(): the read side."""

    def test_unknown_external_id_raises(self):
        """Reading a never-recorded conversation is a usage error, not an empty result."""
        with pytest.raises(ValueError, match="sess-missing"):
            history.conversation("sess-missing")

    def test_turns_ordered_oldest_first(self):
        """Turns come back in turn_index order regardless of insertion order noise."""
        cid = storage.get_or_create_conversation("sess-order")
        for i in range(3):
            storage.record_run(
                provider="openai", conversation_id=cid, input_text=f"q{i}", output_text=f"a{i}"
            )
        convo = history.conversation("sess-order")
        assert [t.input_text for t in convo.turns] == ["q0", "q1", "q2"]
        assert [t.turn_index for t in convo.turns] == [0, 1, 2]


class TestConversationContextManager:
    """promptkeep.conversation(): the ambient attach API used with the wrapper."""

    def test_no_calls_inside_creates_no_row(self):
        """Entering the block does no DB work by itself — resolution is lazy,
        deferred to the first tracked call, matching version registration."""
        with promptkeep.conversation("sess-empty"):
            pass
        assert storage.fetch_conversation("sess-empty") is None

    def test_calls_inside_block_share_conversation_with_sequential_turns(self):
        """A Prompt-backed turn 0 and a plain follow-up turn 1 both land in
        the same conversation, in order, with only the follow-up's text
        stored as input (the system prompt is covered by version lineage)."""
        client = wrap(FakeClient(response=make_response(content="turn reply")))
        with promptkeep.conversation("sess-multi", user_id=42):
            client.chat.completions.create(
                model="gpt-test",
                messages=[
                    {"role": "developer", "content": Prompt("system prompt", name="CONVO_SYS")}
                ],
            )
            client.chat.completions.create(
                model="gpt-test", messages=[{"role": "user", "content": "follow up question"}]
            )
        convo = history.conversation("sess-multi")
        assert convo.metadata == {"user_id": 42}
        assert len(convo.turns) == 2
        assert convo.turns[0].turn_index == 0
        assert convo.turns[0].prompt_name == "CONVO_SYS"
        assert convo.turns[0].input_text is None
        assert convo.turns[1].turn_index == 1
        assert convo.turns[1].prompt_name is None
        assert convo.turns[1].input_text == "follow up question"
        assert convo.turns[1].output_text == "turn reply"

    def test_explicit_kwarg_attaches_without_a_block(self):
        """The per-call kwarg works with no `with` block at all, and never
        reaches the real API."""
        client = wrap(FakeClient())
        client.chat.completions.create(
            model="gpt-test",
            messages=[{"role": "user", "content": "hi"}],
            promptkeep_conversation="sess-explicit",
        )
        (call,) = client.calls
        assert "promptkeep_conversation" not in call
        convo = history.conversation("sess-explicit")
        assert len(convo.turns) == 1
        assert convo.turns[0].input_text == "hi"

    def test_explicit_kwarg_wins_over_ambient_block(self):
        """A per-call override takes precedence over the enclosing block."""
        client = wrap(FakeClient())
        with promptkeep.conversation("sess-ambient"):
            client.chat.completions.create(
                model="gpt-test",
                messages=[{"role": "user", "content": "hi"}],
                promptkeep_conversation="sess-override",
            )
        assert storage.fetch_conversation("sess-ambient") is None
        assert len(history.conversation("sess-override").turns) == 1

    def test_async_context_manager(self):
        """`async with promptkeep.conversation(...)` attaches async calls too."""
        client = wrap(FakeAsyncClient())

        async def go():
            """Make one tracked async call inside the async conversation block."""
            async with promptkeep.conversation("sess-async"):
                await client.chat.completions.create(
                    model="gpt-test", messages=[{"role": "user", "content": "async hi"}]
                )

        asyncio.run(go())
        convo = history.conversation("sess-async")
        assert convo.turns[0].input_text == "async hi"

    def test_streaming_call_inside_conversation_is_recorded(self):
        """A streamed call with no tracked Prompt still records its turn,
        once the stream is exhausted."""
        client = wrap(FakeClient(stream_chunks=[make_chunk(content="ok")]))
        with promptkeep.conversation("sess-stream"):
            stream = client.chat.completions.create(
                model="gpt-test",
                stream=True,
                messages=[{"role": "user", "content": "stream this"}],
            )
            assert storage.fetch_conversation("sess-stream") is not None  # reserved up front
            list(stream)
        convo = history.conversation("sess-stream")
        assert convo.turns[0].output_text == "ok"
        assert convo.turns[0].input_text == "stream this"

    def test_developer_role_is_never_stored_as_input(self):
        """The newer 'developer' role is the system-prompt equivalent and
        must be skipped the same way 'system' is."""
        client = wrap(FakeClient())
        with promptkeep.conversation("sess-dev-role"):
            client.chat.completions.create(
                model="gpt-test",
                messages=[{"role": "developer", "content": "act as a helpful assistant"}],
            )
        (turn,) = history.conversation("sess-dev-role").turns
        assert turn.input_text is None

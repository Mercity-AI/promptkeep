"""Tests for conversation grouping: storage, the context-manager attach API,
and the history.conversation() read side."""

import asyncio
from types import SimpleNamespace

import pytest

import promptkeep
from promptkeep import Prompt, history, storage, tracking, wrap
from tests.fakes import FakeAsyncClient, FakeClient, make_chunk, make_response


def _recorded(external_id):
    """Whether a conversation row exists under this id."""
    return any(c.external_id == external_id for c in history.list_conversations())


class TestConversationStorage:
    """Conversation resolution and turn bookkeeping at the storage layer."""

    def test_get_or_create_is_idempotent(self):
        """Same external_id always resolves to the same row; metadata is
        only recorded on first creation, later calls don't overwrite it."""
        first = storage.get_or_create_conversation("sess-1", title="First")
        second = storage.get_or_create_conversation("sess-1", title="Ignored")
        assert first == second
        assert history.conversation("sess-1").title == "First"

    def test_unknown_conversation_is_none(self):
        """Reading a conversation that was never created returns None, not an error."""
        assert not _recorded("nope")

    def test_turn_indexes_are_sequential(self):
        """reserve_turn_index claims a turn per call, and record_run without
        an explicit turn_index claims its own — no number is ever reused."""
        cid = storage.get_or_create_conversation("sess-2")
        assert storage.reserve_turn_index(cid) == 0  # claims 0
        storage.record_run(provider="openai", conversation_id=cid, input_text="hi")  # claims 1
        storage.record_run(provider="openai", conversation_id=cid, input_text="again")  # claims 2
        assert storage.reserve_turn_index(cid) == 3

    def test_turn_counter_reseeds_from_db_in_new_process(self, isolated_db):
        """A fresh process (caches dropped, same DB file) continues the
        sequence from what's on disk instead of restarting at 0."""
        cid = storage.get_or_create_conversation("sess-reseed")
        storage.record_run(provider="openai", conversation_id=cid, input_text="t0")
        storage.record_run(provider="openai", conversation_id=cid, input_text="t1")
        storage.reset_caches()
        promptkeep.configure(db_path=isolated_db)
        cid2 = storage.get_or_create_conversation("sess-reseed")
        assert cid2 == cid
        assert storage.reserve_turn_index(cid2) == 2

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
        assert not _recorded("sess-empty")

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
        assert not _recorded("sess-ambient")
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
            assert _recorded("sess-stream")  # reserved up front
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


class TestConversationReadModel:
    """The derived views on ConversationInfo: replay(), versions_used,
    total_tokens, duration — and the filtered listing."""

    def _turn(self, cid, prompt=None, **fields):
        """Record one turn; through a Prompt's lineage when one is given."""
        defaults = dict(provider="openai", model="gpt-test", conversation_id=cid)
        defaults.update(fields)
        if prompt is None:
            return storage.record_run(**defaults)
        return tracking.record_prompt_run(prompt, prompt.variables, str(prompt.text), **defaults)

    def test_replay_rebuilds_messages_in_order(self):
        """System prompt once, then user/assistant pairs turn by turn."""
        sys_prompt = Prompt("You are terse.", name="REPLAY_SYS")
        cid = storage.get_or_create_conversation("replay-basic")
        self._turn(cid, sys_prompt, input_text="hi", output_text="hello")
        self._turn(cid, sys_prompt, input_text="more?", output_text="no")
        assert history.conversation("replay-basic").replay() == [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "more?"},
            {"role": "assistant", "content": "no"},
        ]

    def test_replay_emits_system_prompt_again_when_it_changes(self):
        """A version switch mid-session shows up as a new system message at
        the turn it took effect — that's the whole point of version-aware
        conversations."""
        v1 = Prompt("Be brief.", name="REPLAY_SW")
        v2 = Prompt("Be brief and cite sources.", name="REPLAY_SW")
        cid = storage.get_or_create_conversation("replay-switch")
        self._turn(cid, v1, input_text="q1", output_text="a1")
        self._turn(cid, v2, input_text="q2", output_text="a2")
        messages = history.conversation("replay-switch").replay()
        assert [m["role"] for m in messages] == [
            "system",
            "user",
            "assistant",
            "system",
            "user",
            "assistant",
        ]
        assert messages[3]["content"] == "Be brief and cite sources."

    def test_replay_skips_turns_that_never_completed(self):
        """Errors and blocked calls gave the model nothing to build on."""
        cid = storage.get_or_create_conversation("replay-skip")
        self._turn(cid, input_text="q1", output_text="a1")
        self._turn(cid, input_text="q2", status="error", error="boom")
        self._turn(cid, input_text="q2", status="blocked", error="blocked by check 'pii'")
        self._turn(cid, input_text="q2 again", output_text="a2")
        assert history.conversation("replay-skip").replay() == [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2 again"},
            {"role": "assistant", "content": "a2"},
        ]

    def test_replay_uses_the_turn_as_sent_after_a_rewrite(self):
        """input_text is what went out; the original stays in the audit trail."""
        cid = storage.get_or_create_conversation("replay-rewrite")
        self._turn(
            cid,
            input_text="my card is [REDACTED]",
            original_input_text="my card is 4111",
            output_text="ok",
        )
        (user, _assistant) = history.conversation("replay-rewrite").replay()
        assert user == {"role": "user", "content": "my card is [REDACTED]"}

    def test_replay_does_not_duplicate_a_prompt_that_was_the_user_turn(self):
        """A tracked Prompt in the user message is the input, not a system prompt."""
        ask = Prompt("Translate {word} to French.", {"word": "cat"}, name="REPLAY_USER")
        cid = storage.get_or_create_conversation("replay-userprompt")
        self._turn(cid, ask, input_text=str(ask.text), output_text="chat")
        assert history.conversation("replay-userprompt").replay() == [
            {"role": "user", "content": "Translate cat to French."},
            {"role": "assistant", "content": "chat"},
        ]

    def test_replay_with_system_override_swaps_the_prompt(self):
        """system= is the re-run-against-a-new-version hook: it goes first and
        the stored system prompts are dropped. A Prompt object passes through
        untouched so a wrapped client can track it."""
        old = Prompt("Old instructions.", name="REPLAY_OVR")
        new = Prompt("New instructions.", name="REPLAY_OVR")
        cid = storage.get_or_create_conversation("replay-override")
        self._turn(cid, old, input_text="q", output_text="a")
        messages = history.conversation("replay-override").replay(system=new)
        assert messages[0] == {"role": "system", "content": new}
        assert messages[0]["content"] is new
        assert [m["role"] for m in messages] == ["system", "user", "assistant"]
        assert not any(m["content"] == "Old instructions." for m in messages)

    def test_replay_through_the_wrapper_round_trips(self):
        """End to end: what the wrapper recorded replays as the messages an app
        would have accumulated itself."""
        sys_prompt = Prompt("You are a bot.", name="REPLAY_E2E")
        client = wrap(FakeClient(response=make_response(content="reply")))
        with promptkeep.conversation("replay-e2e"):
            client.chat.completions.create(
                model="gpt-test",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": "first"},
                ],
            )
            client.chat.completions.create(
                model="gpt-test",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "reply"},
                    {"role": "user", "content": "second"},
                ],
            )
        assert history.conversation("replay-e2e").replay() == [
            {"role": "system", "content": "You are a bot."},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "reply"},
        ]

    def test_versions_used_lists_each_name_in_order_of_first_use(self):
        a1 = Prompt("a one", name="VU_A")
        a2 = Prompt("a two", name="VU_A")
        b1 = Prompt("b one", name="VU_B")
        assert (a1.version, a2.version) == (1, 2)  # registration is lazy: pin the order
        cid = storage.get_or_create_conversation("vu")
        self._turn(cid, a2, output_text="x")  # v2 first
        self._turn(cid, b1, output_text="x")
        self._turn(cid, a1, output_text="x")
        self._turn(cid, a2, output_text="x")  # repeat: not listed twice
        self._turn(cid, output_text="x")  # bare turn: no lineage
        assert history.conversation("vu").versions_used == {"VU_A": [2, 1], "VU_B": [1]}

    def test_total_tokens_sums_reported_usage(self):
        cid = storage.get_or_create_conversation("tokens")
        self._turn(cid, total_tokens=30, output_text="x")
        self._turn(cid, total_tokens=None, status="error")  # unknown counts as 0
        self._turn(cid, total_tokens=12, output_text="y")
        assert history.conversation("tokens").total_tokens == 42

    def test_duration_spans_first_turn_start_to_last_turn_end(self):
        """Timestamps are taken at record time (end of call), so the first
        turn's latency is counted back from its timestamp."""
        cid = storage.get_or_create_conversation("dur")
        self._turn(cid, latency_ms=1500, output_text="x")
        self._turn(cid, latency_ms=200, output_text="y")
        duration = history.conversation("dur").duration
        assert duration >= 1.5
        assert duration < 30  # two immediate records: no wall-clock gap to speak of

    def test_duration_is_computed_from_turn_fields(self):
        """Pin the arithmetic on hand-built turns, independent of the clock."""
        from dataclasses import replace

        cid = storage.get_or_create_conversation("dur-fixed")
        self._turn(cid, latency_ms=0, output_text="x")
        convo = history.conversation("dur-fixed")
        (turn,) = convo.turns
        turns = [
            replace(turn, created_at="2026-09-11T10:00:05+00:00", latency_ms=2000),
            replace(turn, created_at="2026-09-11T10:01:03+00:00", latency_ms=500),
        ]
        assert replace(convo, turns=turns).duration == 60.0
        assert replace(convo, turns=[]).duration == 0.0

    def test_list_conversations_filters_by_prompt_and_version(self):
        """prompt= keeps sessions the prompt drove; version= narrows to one
        version; the turn count stays the session's full length."""
        v1 = Prompt("one", name="LC")
        v2 = Prompt("two", name="LC")
        other = Prompt("other", name="LC_OTHER")
        assert (v1.version, v2.version) == (1, 2)
        a = storage.get_or_create_conversation("lc-a")
        b = storage.get_or_create_conversation("lc-b")
        c = storage.get_or_create_conversation("lc-c")
        self._turn(a, v1, output_text="x")
        self._turn(a, output_text="x")  # a bare turn — still counts as a turn
        self._turn(b, v2, output_text="x")
        self._turn(c, other, output_text="x")

        def ids(**kw):
            return sorted(s.external_id for s in history.list_conversations(**kw))

        assert ids() == ["lc-a", "lc-b", "lc-c"]
        assert ids(prompt="LC") == ["lc-a", "lc-b"]
        assert ids(prompt="LC", version=1) == ["lc-a"]
        assert ids(prompt="LC", version=2) == ["lc-b"]
        assert ids(prompt="LC", version=9) == []
        assert ids(prompt="NOPE") == []
        (summary,) = history.list_conversations(prompt="LC", version=1)
        assert summary.turn_count == 2

    def test_list_conversations_version_without_prompt_is_an_error(self):
        with pytest.raises(ValueError, match="requires prompt="):
            history.list_conversations(version=1)


class TestBranching:
    """Conversations as trees: parent_run_key, and the views that follow it."""

    def _turn(self, cid, text, parent=None, status="ok"):
        """Record one question/answer turn, optionally branching from ``parent``."""
        return storage.record_run(
            provider="openai",
            conversation_id=cid,
            input_text=text,
            output_text=f"re: {text}",
            status=status,
            parent_run_key=parent,
        )

    def _regenerated(self):
        """q0 -> q1, then q1 regenerated as q2 (both continue q0), then q3
        continuing the regeneration. Returns the run keys in turn order."""
        cid = storage.get_or_create_conversation("branchy")
        k0 = self._turn(cid, "q0")
        k1 = self._turn(cid, "q1")
        k2 = self._turn(cid, "q2", parent=k0)
        k3 = self._turn(cid, "q3")
        return [k0, k1, k2, k3]

    def test_a_linear_conversation_has_no_forks_and_one_leaf(self):
        cid = storage.get_or_create_conversation("straight")
        keys = [self._turn(cid, f"q{n}") for n in range(3)]
        convo = history.conversation("straight")
        assert convo.forks == {}
        assert [leaf.run_key for leaf in convo.leaves] == [keys[-1]]
        assert [t.run_key for t in convo.path(keys[-1])] == keys

    def test_the_parent_is_recorded_and_read_back(self):
        k0, _k1, k2, k3 = self._regenerated()
        turns = history.conversation("branchy").turns
        assert [t.parent_run_key for t in turns] == [None, None, k0, None]

    def test_forks_and_leaves_describe_the_tree(self):
        _k0, k1, _k2, k3 = self._regenerated()
        convo = history.conversation("branchy")
        assert convo.forks == {2: 0}
        assert [leaf.run_key for leaf in convo.leaves] == [k1, k3]

    def test_path_follows_parents_back_to_the_root(self):
        k0, k1, k2, k3 = self._regenerated()
        convo = history.conversation("branchy")
        assert [t.run_key for t in convo.path(k3)] == [k0, k2, k3]
        assert [t.run_key for t in convo.path(k1)] == [k0, k1]

    def test_replay_follows_the_latest_branch_by_default(self):
        self._regenerated()
        messages = history.conversation("branchy").replay()
        assert [m["content"] for m in messages if m["role"] == "user"] == ["q0", "q2", "q3"]

    def test_replay_upto_follows_the_chosen_branch(self):
        _k0, k1, _k2, _k3 = self._regenerated()
        messages = history.conversation("branchy").replay(upto=k1)
        assert [m["content"] for m in messages] == ["q0", "re: q0", "q1", "re: q1"]

    def test_a_failed_turn_stays_on_its_path_but_out_of_the_replay(self):
        cid = storage.get_or_create_conversation("with-error")
        k0 = self._turn(cid, "q0")
        k1 = self._turn(cid, "q1", status="error")
        k2 = self._turn(cid, "q2")
        convo = history.conversation("with-error")
        assert [t.run_key for t in convo.path(k2)] == [k0, k1, k2]
        assert [m["content"] for m in convo.replay() if m["role"] == "user"] == ["q0", "q2"]

    def test_a_parent_outside_the_conversation_starts_a_branch(self):
        outside = storage.record_run(provider="openai", output_text="planner said so")
        cid = storage.get_or_create_conversation("sub-agent")
        k0 = self._turn(cid, "q0")
        k1 = self._turn(cid, "q1", parent=outside)
        convo = history.conversation("sub-agent")
        assert convo.forks == {1: None}
        assert [t.run_key for t in convo.path(k1)] == [k1]
        assert [leaf.run_key for leaf in convo.leaves] == [k0, k1]

    def test_path_of_a_run_elsewhere_is_an_error(self):
        self._regenerated()
        with pytest.raises(ValueError, match="not in conversation"):
            history.conversation("branchy").path("no-such-run")

    def test_rows_of_one_call_stay_together_on_a_path(self):
        cid = storage.get_or_create_conversation("two-prompts")
        k0 = self._turn(cid, "q0")
        turn = storage.reserve_turn_index(cid)
        a = storage.record_run(provider="openai", conversation_id=cid, turn_index=turn)
        b = storage.record_run(provider="openai", conversation_id=cid, turn_index=turn)
        assert [t.run_key for t in history.conversation("two-prompts").path(b)] == [k0, a, b]

    def test_a_wrapped_call_branches_from_a_response_a_handle_or_a_key(self):
        client = wrap(FakeClient(response=make_response(content="reply")))
        ask = {"model": "gpt-test", "messages": [{"role": "user", "content": "q"}]}
        with promptkeep.conversation("regen"):
            # The fake hands back one response object for every call, so keep
            # the handle — later calls re-attach theirs to that same object.
            handle = client.chat.completions.create(**ask).promptkeep
            root = handle.run_key
            client.chat.completions.create(**ask)
            client.chat.completions.create(
                **ask, promptkeep_parent=SimpleNamespace(promptkeep=handle)
            )
            client.chat.completions.create(**ask, promptkeep_parent=handle)
            client.chat.completions.create(**ask, promptkeep_parent=root)
        assert all("promptkeep_parent" not in call for call in client.calls)
        turns = history.conversation("regen").turns
        assert [t.parent_run_key for t in turns] == [None, None, root, root, root]
        assert history.conversation("regen").forks == {2: 0, 3: 0, 4: 0}

    def test_a_malformed_parent_costs_the_link_not_the_call(self, caplog):
        client = wrap(FakeClient(response=make_response(content="reply")))
        response = client.chat.completions.create(
            model="gpt-test",
            messages=[{"role": "user", "content": "q"}],
            promptkeep_conversation="odd-parent",
            promptkeep_parent=42,
        )
        assert response.choices[0].message.content == "reply"
        (turn,) = history.conversation("odd-parent").turns
        assert turn.parent_run_key is None
        assert "promptkeep_parent" in caplog.text

    def test_the_untracked_passthrough_strips_the_parent_kwarg(self):
        client = wrap(FakeClient())
        with promptkeep.suppress():
            client.chat.completions.create(
                model="gpt-test", messages=[{"role": "user", "content": "q"}], promptkeep_parent="k"
            )
        assert "promptkeep_parent" not in client.calls[0]

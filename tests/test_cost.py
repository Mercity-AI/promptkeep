"""Cost tracking: the run's cost is whatever the provider reported on the
response's usage block (OpenRouter does, OpenAI's own API doesn't) — read by
the adapter, stored as ``runs.cost_usd``, never estimated."""

import pytest

from promptkeep import Prompt, conversation, history, storage, wrap
from promptkeep.integrations import OpenAIChatAdapter
from tests.fakes import FakeClient, make_chunk, make_response, make_usage


def make_prompt(name="COST_SYS"):
    """A small tracked prompt to embed in request messages."""
    return Prompt("You are helpful about {topic}.", {"topic": "billing"}, name=name)


def ask(client, **extra):
    """One tracked call through the wrapped client."""
    return client.chat.completions.create(
        model="openrouter/some-model",
        messages=[{"role": "system", "content": make_prompt()}, {"role": "user", "content": "hi"}],
        **extra,
    )


class TestReportedCost:
    def test_cost_on_the_usage_block_is_recorded(self):
        client = wrap(FakeClient(response=make_response(cost=0.000142)))
        ask(client)
        (run,) = history.runs("COST_SYS")
        assert run.cost_usd == pytest.approx(0.000142)

    def test_no_reported_cost_is_none_not_zero(self):
        """OpenAI's own API reports no cost: unknown must not read as free."""
        client = wrap(FakeClient(response=make_response()))
        ask(client)
        (run,) = history.runs("COST_SYS")
        assert run.cost_usd is None

    def test_a_zero_cost_is_kept_as_zero(self):
        """A free model really does cost 0 — distinct from "not reported"."""
        client = wrap(FakeClient(response=make_response(cost=0)))
        ask(client)
        (run,) = history.runs("COST_SYS")
        assert run.cost_usd == 0.0

    def test_stream_cost_comes_from_the_final_usage_chunk(self):
        chunks = [
            make_chunk(content="Hi "),
            make_chunk(content="there"),
            make_chunk(usage=make_usage(4, 2, cost=0.0031)),
        ]
        client = wrap(FakeClient(stream_chunks=chunks))
        list(ask(client, stream=True))
        (run,) = history.runs("COST_SYS")
        assert run.output_text == "Hi there"
        assert run.cost_usd == pytest.approx(0.0031)

    @pytest.mark.parametrize("junk", ["0.5", True, None, object(), [1]])
    def test_a_cost_that_is_not_a_number_is_ignored(self, junk):
        usage = make_usage()
        usage.cost = junk
        response = make_response()
        response.usage = usage
        assert OpenAIChatAdapter().read_response(response).cost_usd is None

    def test_a_failed_call_has_no_cost(self):
        client = wrap(FakeClient(error=RuntimeError("boom")))
        with pytest.raises(RuntimeError):
            ask(client)
        (run,) = history.runs("COST_SYS")
        assert run.status == "error"
        assert run.cost_usd is None


class TestConversationCost:
    def test_total_cost_sums_the_turns(self):
        client = wrap(FakeClient(response=make_response(cost=0.25)))
        with conversation("cost-thread"):
            ask(client)
            ask(client)
        assert history.conversation("cost-thread").total_cost == pytest.approx(0.5)

    def test_total_cost_is_none_when_nothing_was_reported(self):
        client = wrap(FakeClient())
        with conversation("no-cost-thread"):
            ask(client)
        assert history.conversation("no-cost-thread").total_cost is None

    def test_a_call_with_two_prompts_is_counted_once(self):
        """Two tracked Prompts in one call produce two rows repeating the same
        usage and cost; the conversation totals must not double them."""
        client = wrap(FakeClient(response=make_response(cost=0.1)))
        with conversation("two-prompt-thread"):
            client.chat.completions.create(
                model="m",
                messages=[
                    {"role": "system", "content": make_prompt("COST_A")},
                    {"role": "user", "content": make_prompt("COST_B")},
                ],
            )
        convo = history.conversation("two-prompt-thread")
        assert len(convo.turns) == 2
        assert convo.total_cost == pytest.approx(0.1)
        assert convo.total_tokens == 15


class TestDirectRecording:
    def test_record_run_takes_a_cost(self):
        version_id, _ = storage.register_version("COST_DIRECT", "template {x}")
        storage.record_run(version_id=version_id, provider="manual", cost_usd=1.5)
        (run,) = history.runs("COST_DIRECT")
        assert run.cost_usd == 1.5


class TestFormatCost:
    @pytest.mark.parametrize(
        ("value", "shown"),
        [
            (None, "—"),
            (0, "$0.00"),
            (0.95, "$0.95"),
            (12.4, "$12.40"),
            (0.000142, "$0.000142"),
            (0.0031, "$0.0031"),
            (0.00000004, "$0.00"),
        ],
    )
    def test_shows_as_many_decimals_as_it_takes(self, value, shown):
        assert history.format_cost(value) == shown

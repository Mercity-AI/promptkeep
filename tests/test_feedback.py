"""promptkeep.feedback(): a judgement attached to a run after the fact, by its
key — and the RunHandle every recorded call carries so that key is reachable."""

import asyncio

import pytest

import promptkeep
from promptkeep import Prompt, check, configure, conversation, history, storage, wrap
from promptkeep.checks import Verdict
from tests.fakes import FakeAsyncClient, FakeClient, make_chunk


def make_prompt(name="FB_SYS"):
    """A small tracked prompt to embed in request messages."""
    return Prompt("You answer questions about {topic}.", {"topic": "tax"}, name=name)


def ask(client, **extra):
    """One tracked, unchecked call through the wrapped client."""
    return client.chat.completions.create(
        model="gpt-test",
        messages=[{"role": "system", "content": make_prompt()}, {"role": "user", "content": "hi"}],
        **extra,
    )


class TestEveryRecordedCallHasAHandle:
    """feedback() needs a run_key; before this, only checked calls exposed one."""

    def test_an_unchecked_tracked_call_carries_its_run_key(self):
        response = ask(wrap(FakeClient()))
        (run,) = history.runs("FB_SYS")
        assert response.promptkeep.run_key == run.run_key
        assert response.promptkeep.prompt_version == 1
        assert response.promptkeep.checks == []
        assert response.promptkeep.verification == "ok"

    def test_a_bare_conversation_turn_carries_one_too(self):
        client = wrap(FakeClient())
        with conversation("fb-thread"):
            response = client.chat.completions.create(
                model="gpt-test", messages=[{"role": "user", "content": "hello"}]
            )
        (turn,) = history.conversation("fb-thread").turns
        assert response.promptkeep.run_key == turn.run_key
        assert response.promptkeep.prompt_version is None

    def test_a_call_that_records_nothing_is_left_untouched(self):
        response = wrap(FakeClient()).chat.completions.create(
            model="gpt-test", messages=[{"role": "user", "content": "untracked"}]
        )
        assert not hasattr(response, "promptkeep")

    def test_async_call(self):
        client = wrap(FakeAsyncClient())
        response = asyncio.run(ask(client))
        (run,) = history.runs("FB_SYS")
        assert response.promptkeep.run_key == run.run_key

    def test_a_stream_gets_its_key_once_it_has_drained(self):
        client = wrap(FakeClient(stream_chunks=[make_chunk(content="Hi")]))
        stream = ask(client, stream=True)
        assert stream.promptkeep.run_key is None  # nothing recorded yet
        list(stream)
        (run,) = history.runs("FB_SYS")
        assert stream.promptkeep.run_key == run.run_key

    def test_run_key_is_none_when_nothing_was_stored(self):
        configure(write_mode="off")
        response = ask(wrap(FakeClient()))
        assert response.promptkeep.run_key is None

    def test_the_key_is_valid_before_the_background_writer_lands_the_row(self):
        configure(write_mode="background")
        response = ask(wrap(FakeClient()))
        key = response.promptkeep.run_key
        assert key is not None
        promptkeep.feedback(key, score=1.0, label="thumbs_up")  # queued behind its run
        assert promptkeep.flush(timeout=5)
        (label,) = history.checks(key)
        assert (label.phase, label.name, label.score) == ("feedback", "thumbs_up", 1.0)


class TestFeedback:
    @pytest.fixture
    def run_key(self):
        return ask(wrap(FakeClient())).promptkeep.run_key

    def test_score_label_and_comment_are_stored(self, run_key):
        promptkeep.feedback(run_key, score=0.0, label="hallucination", comment="invented a law")
        (label,) = history.checks(run_key)
        assert label.phase == "feedback"
        assert label.name == "hallucination"
        assert label.score == 0.0
        assert label.message == "invented a law"
        assert label.rewritten is None

    def test_a_label_is_optional(self, run_key):
        promptkeep.feedback(run_key, score=0.5)
        (label,) = history.checks(run_key)
        assert label.name == "feedback"

    def test_a_run_collects_any_number_of_them(self, run_key):
        promptkeep.feedback(run_key, label="thumbs_up")
        promptkeep.feedback(run_key, score=4, comment="second reviewer")
        first, second = history.checks(run_key)
        assert (first.name, second.score) == ("thumbs_up", 4.0)

    def test_nothing_to_say_is_a_usage_error(self, run_key):
        with pytest.raises(ValueError, match="at least one"):
            promptkeep.feedback(run_key)

    @pytest.mark.parametrize("junk", ["1.0", True, [1]])
    def test_score_must_be_a_number(self, run_key, junk):
        with pytest.raises(TypeError, match="must be a number"):
            promptkeep.feedback(run_key, score=junk)

    def test_no_run_key_is_a_no_op(self):
        promptkeep.feedback(None, score=1.0)  # an unrecorded run's handle

    def test_an_unknown_run_is_skipped_with_a_warning_not_raised(self, caplog):
        with caplog.at_level("WARNING", logger="promptkeep"):
            promptkeep.feedback("no-such-run", score=1.0)
        assert "never persisted" in caplog.text

    def test_a_broken_database_costs_the_label_not_the_caller(self, run_key, monkeypatch):
        def explode(item):
            raise RuntimeError("disk on fire")

        monkeypatch.setattr(storage, "_insert_check", explode)
        promptkeep.feedback(run_key, score=1.0)
        assert history.checks(run_key) == []

    def test_the_comment_goes_through_the_redact_hook(self, run_key):
        configure(redact=lambda text: text.replace("alice@example.com", "[email]"))
        promptkeep.feedback(run_key, comment="alice@example.com said it was wrong")
        (label,) = history.checks(run_key)
        assert label.message == "[email] said it was wrong"


class TestFeedbackNextToChecks:
    def test_it_sits_beside_the_verdicts_without_moving_the_headline(self):
        @check.post(mode="blocking")
        def always_fine(ctx):
            return Verdict.ok()

        response = ask(wrap(FakeClient()), promptkeep_post=[always_fine])
        key = response.promptkeep.run_key
        promptkeep.feedback(key, score=0.0, label="thumbs_down")
        labels = history.checks(key)
        assert [c.phase for c in labels] == ["post", "feedback"]
        assert history.verdict("ok", labels) == "ok"

    def test_feedback_alone_is_not_a_check_result(self):
        key = ask(wrap(FakeClient())).promptkeep.run_key
        promptkeep.feedback(key, label="thumbs_up")
        assert history.verdict("ok", history.checks(key)) is None

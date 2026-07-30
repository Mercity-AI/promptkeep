"""Tests for checks: pre-gates, post-audits, verdicts, RunHandle, suppression.

Runs against the fake OpenAI clients (no network). Checks execute through
the real wrapper path.
"""

import asyncio
import time
from types import SimpleNamespace

import pytest

import promptkeep
from promptkeep import Prompt, PromptBlocked, Verdict, acall, call, check, history, storage, wrap
from tests.fakes import FakeAsyncClient, FakeClient, make_chunk, make_response


def _client(content="Paris is the capital of France."):
    return wrap(FakeClient(response=make_response(content=content)))


def _aclient(content="Paris is the capital of France."):
    return wrap(FakeAsyncClient(response=make_response(content=content)))


def _usage_chunk():
    return make_chunk(usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2, total_tokens=7))


class TestVerdict:
    def test_from_score_thresholds(self):
        assert Verdict.from_score(0.9, 0.5).status == "ok"
        assert Verdict.from_score(0.2, 0.5).status == "warn"
        assert Verdict.from_score(0.5, 0.5).status == "ok"  # boundary is inclusive

    def test_constructors(self):
        assert Verdict.ok().status == "ok"
        assert Verdict.warn("x").status == "warn"
        assert Verdict.block("x").status == "block"
        assert Verdict.rewrite("y").rewritten == "y"


class TestPreChecks:
    def test_block_raises_and_records_blocked_run(self):
        @check.pre(name="no_pii")
        def gate(ctx):
            return Verdict.block("email") if "@" in ctx.rendered else Verdict.ok()

        p = Prompt("sys", name="P", pre=[gate])
        client = _client()
        with pytest.raises(PromptBlocked, match="no_pii"):
            client.chat.completions.create(
                model="m",
                messages=[
                    {"role": "developer", "content": p},
                    {"role": "user", "content": "reach me at a@b.com"},
                ],
            )
        # The provider was never called.
        assert client.calls == []
        # A blocked run is still recorded, with the block verdict.
        (run,) = history.all_runs()
        assert run.status == "blocked"
        (chk,) = storage.fetch_checks(run.id)
        assert chk["name"] == "no_pii" and chk["status"] == "block"

    def test_block_return_mode_yields_stub(self):
        promptkeep.configure(on_block="return")

        @check.pre(name="gate")
        def gate(ctx):
            return Verdict.block("nope")

        p = Prompt("sys", name="P", pre=[gate])
        client = _client()
        resp = client.chat.completions.create(
            model="m", messages=[{"role": "developer", "content": p}]
        )
        assert client.calls == []
        assert resp.promptkeep_blocked.check == "gate"
        assert resp.promptkeep.verification == "failed"

    def test_ok_passes_through(self):
        @check.pre(name="always_ok")
        def gate(ctx):
            return Verdict.ok()

        p = Prompt("sys", name="P", pre=[gate])
        client = _client()
        resp = client.chat.completions.create(
            model="m", messages=[{"role": "developer", "content": p}]
        )
        assert len(client.calls) == 1
        assert resp.promptkeep.verification == "ok"

    def test_warn_is_recorded_but_continues(self):
        @check.pre(name="length")
        def gate(ctx):
            return Verdict.warn("long") if len(ctx.rendered) > 2 else Verdict.ok()

        p = Prompt("long system prompt", name="P", pre=[gate])
        client = _client()
        resp = client.chat.completions.create(
            model="m", messages=[{"role": "developer", "content": p}]
        )
        assert len(client.calls) == 1
        assert resp.promptkeep.verification == "warn"

    def test_rewrite_targets_only_the_last_message(self):
        @check.pre(name="redact")
        def gate(ctx):
            if "SECRET" in (ctx.last_text or ""):
                return Verdict.rewrite(ctx.last_text.replace("SECRET", "[redacted]"))
            return Verdict.ok()

        p = Prompt("system", name="P", pre=[gate])
        client = _client()
        client.chat.completions.create(
            model="m",
            messages=[
                {"role": "developer", "content": p},
                {"role": "user", "content": "my SECRET token"},
            ],
        )
        sent = client.calls[0]["messages"]
        assert sent[0]["content"] == "system"  # system prompt untouched
        assert sent[1]["content"] == "my [redacted] token"


class TestPostChecks:
    def test_blocking_post_grades_output(self):
        @check.post(name="grounded", mode="blocking")
        def grade(ctx):
            score = 0.9 if "Paris" in (ctx.output_text or "") else 0.1
            return Verdict.from_score(score, 0.5)

        p = Prompt("sys", name="P", post=[grade])
        resp = _client("Paris is the capital.").chat.completions.create(
            model="m", messages=[{"role": "developer", "content": p}]
        )
        (chk,) = [c for c in resp.promptkeep.checks if c.name == "grounded"]
        assert chk.status == "ok" and chk.score == 0.9

    def test_async_post_lands_after_wait(self):
        @check.post(name="slow", mode="async")
        def grade(ctx):
            time.sleep(0.03)
            return Verdict.from_score(0.8, 0.5)

        p = Prompt("sys", name="P", post=[grade])
        resp = _client().chat.completions.create(
            model="m", messages=[{"role": "developer", "content": p}]
        )
        assert resp.promptkeep.verification == "pending"
        resp.promptkeep.wait(timeout=5)
        assert resp.promptkeep.verification == "ok"
        names = [c["name"] for c in storage.fetch_checks(resp.promptkeep.run_id)]
        assert "slow" in names


class TestSafety:
    def test_crashing_check_never_breaks_the_call(self):
        @check.post(name="boom", mode="blocking")
        def grade(ctx):
            raise RuntimeError("check exploded")

        p = Prompt("sys", name="P", post=[grade])
        resp = _client().chat.completions.create(
            model="m", messages=[{"role": "developer", "content": p}]
        )
        # Call still returns; the check is recorded as an error.
        assert resp.choices[0].message.content == "Paris is the capital of France."
        (chk,) = [c for c in resp.promptkeep.checks if c.name == "boom"]
        assert chk.status == "error"

    def test_llm_judge_check_does_not_record_itself(self):
        """A check that calls an LLM must not create its own run row."""
        judge = _client("0.9")

        @check.post(name="judge", mode="blocking")
        def grade(ctx):
            r = judge.chat.completions.create(
                model="m", messages=[{"role": "user", "content": "grade"}]
            )
            return Verdict.from_score(float(r.choices[0].message.content), 0.5)

        p = Prompt("sys", name="P", post=[grade])
        client = _client()
        client.chat.completions.create(model="m", messages=[{"role": "developer", "content": p}])
        # Exactly one run: the outer call. The judge's call was suppressed.
        assert len(history.all_runs()) == 1

    def test_pre_timeout_fails_open_by_default(self):
        @check.pre(name="slow_gate", timeout=0.05)
        def gate(ctx):
            time.sleep(0.5)
            return Verdict.block("too late")

        p = Prompt("sys", name="P", pre=[gate])
        client = _client()
        # Fails open: the call proceeds despite the slow gate.
        resp = client.chat.completions.create(
            model="m", messages=[{"role": "developer", "content": p}]
        )
        assert len(client.calls) == 1
        (chk,) = [c for c in resp.promptkeep.checks if c.name == "slow_gate"]
        assert chk.status == "warn"

    def test_pre_timeout_can_fail_closed(self):
        @check.pre(name="strict_gate", timeout=0.05, on_timeout="closed")
        def gate(ctx):
            time.sleep(0.5)
            return Verdict.ok()

        p = Prompt("sys", name="P", pre=[gate])
        client = _client()
        with pytest.raises(PromptBlocked, match="strict_gate"):
            client.chat.completions.create(
                model="m", messages=[{"role": "developer", "content": p}]
            )
        assert client.calls == []


class TestScopes:
    def test_global_and_per_call_checks_apply(self):
        seen = []

        @check.pre(name="global_gate")
        def g(ctx):
            seen.append("global")
            return Verdict.ok()

        @check.pre(name="call_gate")
        def c(ctx):
            seen.append("call")
            return Verdict.ok()

        promptkeep.configure(pre=[g])
        client = _client()
        client.chat.completions.create(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            promptkeep_pre=[c],
        )
        assert "global" in seen and "call" in seen

    def test_check_kwargs_never_reach_the_provider(self):
        @check.pre(name="g")
        def g(ctx):
            return Verdict.ok()

        client = _client()
        client.chat.completions.create(
            model="m", messages=[{"role": "user", "content": "hi"}], promptkeep_pre=[g]
        )
        assert "promptkeep_pre" not in client.calls[0]


class TestPIIConversationFlow:
    """The realistic story: a global PII gate guards a multi-turn chat, blocks
    a pasted secret mid-conversation, and the retype goes through."""

    def test_secret_blocked_midconversation_then_retype_passes(self):
        import re

        secret_re = re.compile(r"sk-[A-Za-z0-9\-]{6,}")

        @check.pre(name="no_secrets")
        def gate(ctx):
            if ctx.last_text and secret_re.search(ctx.last_text):
                return Verdict.block("secret in message")
            return Verdict.ok()

        promptkeep.configure(pre=[gate])  # global: guards every turn
        client = _client()
        sys_p = Prompt("You are support.", name="SUPPORT")
        msgs = [{"role": "system", "content": sys_p}]

        with promptkeep.conversation("sess-pii", title="401 help"):
            # turn 0: normal, passes
            msgs.append({"role": "user", "content": "my deploy returns 401"})
            client.chat.completions.create(model="m", messages=msgs)
            msgs.append({"role": "assistant", "content": "likely an expired key"})

            # turn 1: user pastes a raw key -> blocked, never sent
            msgs.append({"role": "user", "content": "here: sk-live-9fA2Bq7Xk"})
            with pytest.raises(PromptBlocked, match="no_secrets"):
                client.chat.completions.create(model="m", messages=msgs)
            msgs.pop()  # app drops the leaked message

            # turn 2: retype without the secret -> passes
            msgs.append({"role": "user", "content": "it ends 7Xk, stopped today"})
            client.chat.completions.create(model="m", messages=msgs)

        convo = history.conversation("sess-pii")
        statuses = [t.status for t in convo.turns]
        # three recorded turns: ok, blocked, ok — in order.
        assert statuses == ["ok", "blocked", "ok"]
        # the blocked turn has the block verdict and no output.
        blocked = convo.turns[1]
        assert blocked.output_text is None
        (chk,) = storage.fetch_checks(blocked.id)
        assert chk["name"] == "no_secrets" and chk["status"] == "block"
        # the secret never reached the provider on any call.
        assert not any("sk-live" in str(c.get("messages")) for c in client.calls)


class TestAsyncChecks:
    """Checks on the AsyncOpenAI path."""

    def test_async_pre_and_post_run(self):
        @check.pre(name="gate")
        def gate(ctx):
            return Verdict.ok()

        @check.post(name="grounded", mode="blocking")
        def grade(ctx):
            return Verdict.from_score(0.9 if "Paris" in (ctx.output_text or "") else 0.1, 0.5)

        p = Prompt("sys", name="AP", pre=[gate], post=[grade])
        client = _aclient()

        async def go():
            return await client.chat.completions.create(
                model="m", messages=[{"role": "developer", "content": p}]
            )

        resp = asyncio.run(go())
        assert {c.name for c in resp.promptkeep.checks} == {"gate", "grounded"}
        assert resp.promptkeep.verification == "ok"

    def test_async_pre_block_raises(self):
        @check.pre(name="no_pii")
        def gate(ctx):
            return Verdict.block("email") if "@" in ctx.rendered else Verdict.ok()

        p = Prompt("sys", name="AP", pre=[gate])
        client = _aclient()

        async def go():
            await client.chat.completions.create(
                model="m",
                messages=[
                    {"role": "developer", "content": p},
                    {"role": "user", "content": "a@b.com"},
                ],
            )

        with pytest.raises(PromptBlocked, match="no_pii"):
            asyncio.run(go())
        assert client.calls == []

    def test_awaited_resolves_async_post(self):
        @check.post(name="slow", mode="async")
        def grade(ctx):
            time.sleep(0.03)
            return Verdict.from_score(0.8, 0.5)

        p = Prompt("sys", name="AP", post=[grade])
        client = _aclient()

        async def go():
            resp = await client.chat.completions.create(
                model="m", messages=[{"role": "developer", "content": p}]
            )
            assert resp.promptkeep.verification == "pending"
            await resp.promptkeep.awaited(timeout=5)
            return resp.promptkeep.verification

        assert asyncio.run(go()) == "ok"


class TestStreamingChecks:
    """Checks on streamed responses: pre before the stream, post after it drains."""

    def _chunks(self):
        return [make_chunk(content="Par"), make_chunk(content="is!"), _usage_chunk()]

    def test_sync_stream_pre_upfront_post_after_drain(self):
        @check.pre(name="gate")
        def gate(ctx):
            return Verdict.ok()

        @check.post(name="grounded", mode="blocking")
        def grade(ctx):
            return Verdict.from_score(0.9 if "Paris" in (ctx.output_text or "") else 0.1, 0.5)

        p = Prompt("sys", name="SP", pre=[gate], post=[grade])
        client = wrap(FakeClient(stream_chunks=self._chunks()))
        stream = client.chat.completions.create(
            model="m", stream=True, messages=[{"role": "developer", "content": p}]
        )
        # pre-check is known upfront; post-check not yet.
        assert [c.name for c in stream.promptkeep.checks] == ["gate"]
        list(stream)  # drain
        names = {c.name for c in stream.promptkeep.checks}
        assert names == {"gate", "grounded"}
        assert storage.fetch_all_runs()[0]["output_text"] == "Paris!"

    def test_sync_stream_pre_block_never_starts_stream(self):
        @check.pre(name="no_pii")
        def gate(ctx):
            return Verdict.block("email") if "@" in ctx.rendered else Verdict.ok()

        p = Prompt("sys", name="SP", pre=[gate])
        client = wrap(FakeClient(stream_chunks=self._chunks()))
        with pytest.raises(PromptBlocked, match="no_pii"):
            client.chat.completions.create(
                model="m",
                stream=True,
                messages=[
                    {"role": "developer", "content": p},
                    {"role": "user", "content": "a@b.com"},
                ],
            )
        assert client.calls == []

    def test_async_stream_checks(self):
        @check.post(name="grounded", mode="blocking")
        def grade(ctx):
            return Verdict.from_score(0.9 if "Paris" in (ctx.output_text or "") else 0.1, 0.5)

        p = Prompt("sys", name="SP", post=[grade])
        client = wrap(FakeAsyncClient(stream_chunks=self._chunks()))

        async def go():
            stream = await client.chat.completions.create(
                model="m", stream=True, messages=[{"role": "developer", "content": p}]
            )
            _ = [c async for c in stream]
            return stream

        stream = asyncio.run(go())
        assert {c.name for c in stream.promptkeep.checks} == {"grounded"}


class TestCallHelper:
    """promptkeep.call() / acall(): the explicit result shape."""

    def test_call_returns_text_and_verification(self):
        @check.post(name="grounded", mode="blocking")
        def grade(ctx):
            return Verdict.from_score(0.9 if "Paris" in (ctx.output_text or "") else 0.1, 0.5)

        p = Prompt("sys", name="CP", post=[grade])
        result = call(_client(), model="m", messages=[{"role": "developer", "content": p}])
        assert result.text == "Paris is the capital of France."
        assert result.verification == "ok"
        assert result.run_id is not None
        assert [c.name for c in result.checks] == ["grounded"]

    def test_call_on_block_return_gives_failed_result(self):
        promptkeep.configure(on_block="return")

        @check.pre(name="gate")
        def gate(ctx):
            return Verdict.block("nope")

        p = Prompt("sys", name="CP", pre=[gate])
        result = call(_client(), model="m", messages=[{"role": "developer", "content": p}])
        assert result.text is None
        assert result.verification == "failed"

    def test_acall_async(self):
        p = Prompt("sys", name="CP")
        client = _aclient()

        async def go():
            return await acall(client, model="m", messages=[{"role": "developer", "content": p}])

        result = asyncio.run(go())
        assert result.text == "Paris is the capital of France."

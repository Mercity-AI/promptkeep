"""Tests for the production controls: run sampling and the redaction hook.

Both act in storage.record_run / record_check — the one place every run row
and verdict passes through — so they cover the wrapper, the tracking helpers
and direct storage calls alike.
"""

import pytest

import promptkeep
from promptkeep import Prompt, Verdict, check, config, history, storage, tracking, wrap
from tests.fakes import FakeClient, make_response


def _ok_run(prompt, **fields):
    defaults = dict(provider="openai", model="m", output_text="fine")
    defaults.update(fields)
    return tracking.record_prompt_run(prompt, prompt.variables, str(prompt.text), **defaults)


class TestSampling:
    def test_rate_zero_keeps_only_problems(self):
        """0.0 means 'store only what went wrong': uneventful runs vanish,
        errors, blocked calls and non-ok verdicts all survive."""
        promptkeep.configure(sample_rate=0.0)
        p = Prompt("s {x}", {"x": 1}, name="SAMPLE")
        assert _ok_run(p) is None
        assert _ok_run(p, status="error", error="boom") is not None
        assert _ok_run(p, status="blocked", error="blocked by check 'pii'") is not None
        warned = {"name": "c", "phase": "post", "status": "warn", "message": "meh"}
        assert _ok_run(p, checks=[warned]) is not None
        passed = {"name": "c", "phase": "post", "status": "ok", "message": None}
        assert _ok_run(p, checks=[passed]) is None
        statuses = sorted(r.status for r in history.runs("SAMPLE"))
        assert statuses == ["blocked", "error", "ok"]  # the 'ok' one carries the warn

    def test_default_rate_keeps_everything(self):
        p = Prompt("s {x}", {"x": 1}, name="SAMPLE_ALL")
        for _ in range(5):
            assert _ok_run(p) is not None
        assert len(history.runs("SAMPLE_ALL")) == 5

    def test_version_registration_is_never_sampled(self):
        promptkeep.configure(sample_rate=0.0)
        assert Prompt("registered anyway", name="SAMPLE_VER").version == 1

    def test_decision_is_whole_conversation(self):
        """A session is kept whole or dropped whole — never a partial replay.
        The decision comes from the conversation's row id, so it's also the
        same in every process that shares the file."""
        promptkeep.configure(sample_rate=0.5)
        kept_counts = []
        for i in range(40):
            cid = storage.get_or_create_conversation(f"sampled-{i}")
            for turn in range(3):
                storage.record_run(
                    provider="openai", conversation_id=cid, input_text=f"q{turn}", output_text="a"
                )
            kept_counts.append(len(history.conversation(f"sampled-{i}").turns))
        assert set(kept_counts) <= {0, 3}
        # 40 independent coin flips: all landing the same way is a 2^-39 event.
        assert 0 in kept_counts and 3 in kept_counts

    def test_dropped_conversation_still_keeps_its_errors(self):
        """The always-keep rule wins over the per-session decision, and the
        surviving turn keeps its real position."""
        promptkeep.configure(sample_rate=0.0)
        cid = storage.get_or_create_conversation("sampled-err")
        storage.record_run(provider="openai", conversation_id=cid, input_text="q0", output_text="a")
        storage.record_run(
            provider="openai", conversation_id=cid, input_text="q1", status="error", error="x"
        )
        (turn,) = history.conversation("sampled-err").turns
        assert turn.status == "error"
        assert turn.turn_index == 1

    def test_sampled_out_checked_call_still_returns_its_verdicts(self):
        """The checks ran and ride on the response; only the row is absent."""
        promptkeep.configure(sample_rate=0.0)

        @check.post(name="grade", mode="blocking")
        def grade(ctx):
            return Verdict.ok()

        p = Prompt("sys", name="SAMPLE_CHK", post=[grade])
        client = wrap(FakeClient(response=make_response(content="hi")))
        response = client.chat.completions.create(
            model="m", messages=[{"role": "developer", "content": p}]
        )
        assert response.promptkeep.run_key is None
        assert response.promptkeep.verification == "ok"
        assert [c.name for c in response.promptkeep.checks] == ["grade"]
        assert history.runs("SAMPLE_CHK") == []

    def test_configure_rejects_out_of_range(self):
        for bad in (-0.1, 1.5, True, "0.5"):
            with pytest.raises(ValueError):
                promptkeep.configure(sample_rate=bad)

    def test_env_var_sets_the_rate_and_bad_values_keep_everything(self, monkeypatch):
        monkeypatch.setenv("PROMPTKEEP_SAMPLE_RATE", "0.25")
        assert config.get_settings().sample_rate == 0.25
        for bad in ("abc", "7", "-1", ""):
            monkeypatch.setenv("PROMPTKEEP_SAMPLE_RATE", bad)
            assert config.get_settings().sample_rate == 1.0
        monkeypatch.setenv("PROMPTKEEP_SAMPLE_RATE", "0.25")
        promptkeep.configure(sample_rate=0.75)  # configure() outranks the env
        assert config.get_settings().sample_rate == 0.75


def _mask(text):
    return text.replace("secret", "[X]")


class TestRedaction:
    def test_every_stored_text_field_passes_through_the_hook(self):
        """Prompt text, inputs, outputs, variables and request params (as
        JSON), error text, and bundled verdicts are all redacted; the version
        template — code, not data — is not."""
        promptkeep.configure(redact=_mask)
        p = Prompt("hello {who}, secret template", {"who": "secret agent"}, name="REDACT")
        _ok_run(
            p,
            output_text="a secret reply",
            error="secret error",
            request_params={"user": "secret-user", "temperature": 0},
            input_text="my secret question",
            original_input_text="my secret question, unrewritten",
            checks=[
                {
                    "name": "c",
                    "phase": "pre",
                    "status": "ok",
                    "message": "found a secret",
                    "rewritten": "secret rewritten",
                }
            ],
        )
        (run,) = history.runs("REDACT")
        assert run.rendered_text == "hello [X] agent, [X] template"
        assert run.variables == {"who": "[X] agent"}
        assert run.request_params == {"user": "[X]-user", "temperature": 0}
        assert run.output_text == "a [X] reply"
        assert run.error == "[X] error"
        assert run.input_text == "my [X] question"
        assert run.original_input_text == "my [X] question, unrewritten"
        (verdict,) = history.checks(run.run_key)
        assert verdict.message == "found a [X]"
        assert verdict.rewritten == "[X] rewritten"
        (version,) = history.versions("REDACT")
        assert version.template == "hello {who}, secret template"

    def test_late_verdicts_are_redacted_too(self):
        promptkeep.configure(redact=_mask)
        p = Prompt("s", name="REDACT_LATE")
        run_key = _ok_run(p)
        storage.record_check(
            run_key,
            {
                "name": "late",
                "phase": "post",
                "status": "warn",
                "message": "secret leak",
                "score": None,
                "latency_ms": 1,
                "rewritten": None,
            },
        )
        (verdict,) = history.checks(run_key)
        assert verdict.message == "[X] leak"

    def test_through_the_wrapper_and_a_conversation(self):
        promptkeep.configure(redact=_mask)
        client = wrap(FakeClient(response=make_response(content="the secret is 42")))
        with promptkeep.conversation("redact-sess"):
            client.chat.completions.create(
                model="m", messages=[{"role": "user", "content": "tell me the secret"}]
            )
        (turn,) = history.conversation("redact-sess").turns
        assert turn.input_text == "tell me the [X]"
        assert turn.output_text == "the [X] is 42"

    def test_hook_that_raises_drops_the_run_silently(self):
        """A broken redactor must neither crash the call nor store plaintext."""

        def broken(text):
            raise RuntimeError("regex exploded")

        promptkeep.configure(redact=broken)
        p = Prompt("s", name="REDACT_RAISE")
        assert _ok_run(p, output_text="secret") is None
        assert history.runs("REDACT_RAISE") == []
        assert storage.record_check("some-key", {"name": "c", "message": "secret"}) is None

    def test_hook_returning_non_string_drops_the_run(self):
        promptkeep.configure(redact=lambda text: None)
        p = Prompt("s", name="REDACT_NONE")
        assert _ok_run(p) is None
        assert history.runs("REDACT_NONE") == []

    def test_configure_rejects_non_callable(self):
        with pytest.raises(TypeError, match="callable"):
            promptkeep.configure(redact="[REDACTED]")

    def test_background_mode_redacts_before_queueing(self, monkeypatch):
        """The row that reaches the writer is already redacted — plaintext
        never sits in the queue."""
        from promptkeep import writer

        promptkeep.configure(write_mode="background", redact=_mask)
        seen = []
        monkeypatch.setattr(writer, "submit", lambda item: seen.append(item))
        _ok_run(Prompt("s", name="REDACT_BG"), output_text="secret")
        (row,) = seen
        assert row["output_text"] == "[X]"

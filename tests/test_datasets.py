"""promptkeep.dataset(): recorded runs, filtered, out to JSONL / DSPy / promptfoo."""

import json
import sys
import types

import pytest

import promptkeep
from promptkeep import Prompt, history, storage, tracking

REVIEW = Prompt("Review {what}.", name="REVIEW")
REVIEW_V2 = Prompt("Review {what} briefly.", name="REVIEW")


def _run(prompt=REVIEW, what="code", *, feedback=None, verdicts=(), **fields):
    """Record one completed run of ``prompt``; attach verdicts (statuses) and
    feedback ((label, score) pairs). Returns the run key."""
    fields.setdefault("output_text", f"reviewed {what}")
    checks = [
        {"name": f"check{n}", "phase": "post", "status": status, "score": None}
        for n, status in enumerate(verdicts)
    ]
    key = tracking.record_prompt_run(
        prompt,
        {"what": what},
        str(prompt.format(what=what).text),
        provider="openai",
        model="gpt-test",
        checks=checks or None,
        **fields,
    )
    for label, score in feedback or ():
        promptkeep.feedback(key, score=score, label=label)
    return key


def _read_jsonl(path):
    """Every line of a JSONL file, decoded."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def fake_dspy(monkeypatch):
    """A stand-in for the dspy package: just enough Example to check shapes."""

    class Example:
        def __init__(self, **fields):
            self.fields = fields
            self.inputs = ()

        def with_inputs(self, *names):
            self.inputs = names
            return self

    module = types.ModuleType("dspy")
    module.Example = Example
    monkeypatch.setitem(sys.modules, "dspy", module)
    return module


class TestSelection:
    def test_completed_runs_with_a_reply_newest_first(self):
        first = _run(what="a")
        _run(what="b", status="error", output_text=None)
        _run(what="c", output_text=None)
        second = _run(what="d")
        ds = promptkeep.dataset("REVIEW")
        assert [run.run_key for run in ds] == [second, first]
        assert len(ds) == 2

    def test_by_version(self):
        _run(REVIEW)
        newer = _run(REVIEW_V2)
        assert [run.run_key for run in promptkeep.dataset("REVIEW", version=2)] == [newer]

    def test_passed_keeps_runs_whose_checks_all_passed(self):
        good = _run(verdicts=["ok", "ok"])
        _run(verdicts=["ok", "warn"])
        _run()  # never checked: not a pass
        assert [run.run_key for run in promptkeep.dataset("REVIEW", passed=True)] == [good]

    def test_passed_false_is_the_failures(self):
        _run(verdicts=["ok"])
        bad = _run(verdicts=["ok", "error"])
        _run()
        assert [run.run_key for run in promptkeep.dataset("REVIEW", passed=False)] == [bad]

    def test_feedback_is_not_a_check(self):
        _run(feedback=[("thumbs_down", 0.0)])
        assert len(promptkeep.dataset("REVIEW", passed=True)) == 0
        assert len(promptkeep.dataset("REVIEW", passed=False)) == 0

    def test_by_feedback_label(self):
        liked = _run(feedback=[("thumbs_up", 1.0)])
        _run(feedback=[("thumbs_down", 0.0)])
        chosen = promptkeep.dataset("REVIEW", feedback="thumbs_up")
        assert [run.run_key for run in chosen] == [liked]

    def test_by_average_feedback_score(self):
        high = _run(feedback=[("a", 1.0), ("b", 0.8)])
        _run(feedback=[("a", 1.0), ("b", 0.2)])
        _run(feedback=[("comment only", None)])
        chosen = promptkeep.dataset("REVIEW", min_feedback=0.8)
        assert [run.run_key for run in chosen] == [high]

    def test_limit_counts_examples_after_filtering(self):
        keep = [_run(verdicts=["ok"]) for _ in range(3)]
        for _ in range(3):
            _run(verdicts=["error"])
        chosen = promptkeep.dataset("REVIEW", passed=True, limit=2)
        assert [run.run_key for run in chosen] == keep[::-1][:2]

    def test_labels_travel_with_the_runs(self):
        key = _run(verdicts=["ok"], feedback=[("thumbs_up", 1.0)])
        ds = promptkeep.dataset("REVIEW")
        assert [(c.phase, c.name) for c in ds.labels[key]] == [
            ("post", "check0"),
            ("feedback", "thumbs_up"),
        ]

    def test_an_unknown_prompt_or_disabled_tracking_is_empty(self):
        _run()
        assert len(promptkeep.dataset("NOPE")) == 0
        promptkeep.configure(enabled=False)
        assert len(promptkeep.dataset("REVIEW")) == 0


class TestExports:
    def test_jsonl(self, tmp_path):
        key = _run(verdicts=["ok"], input_text="is this fine?")
        out = tmp_path / "ds.jsonl"
        assert promptkeep.dataset("REVIEW").to_jsonl(out) == 1
        (record,) = _read_jsonl(out)
        assert record["run_key"] == key
        assert (record["prompt"], record["version"], record["model"]) == ("REVIEW", 1, "gpt-test")
        assert record["variables"] == {"what": "code"}
        assert (record["input"], record["output"]) == ("is this fine?", "reviewed code")
        assert record["labels"][0]["name"] == "check0"

    def test_a_prompt_that_was_the_user_turn_is_not_a_separate_input(self, tmp_path):
        _run(input_text="Review code.")  # the rendered prompt itself was the user turn
        out = tmp_path / "ds.jsonl"
        promptkeep.dataset("REVIEW").to_jsonl(out)
        assert _read_jsonl(out)[0]["input"] is None

    def test_promptfoo_test_cases(self, tmp_path):
        key = _run(input_text="is this fine?")
        out = tmp_path / "tests.jsonl"
        assert promptkeep.dataset("REVIEW").to_promptfoo(out) == 1
        (case,) = _read_jsonl(out)
        assert case["vars"] == {"what": "code", "input": "is this fine?"}
        assert case["description"] == f"REVIEW v1 · run {key[:8]}"
        assert case["metadata"]["promptkeep_run_key"] == key
        assert case["metadata"]["reference_output"] == "reviewed code"
        assert "assert" not in case

    def test_dspy_examples(self, fake_dspy):
        _run(input_text="is this fine?")
        _run(what="docs")
        with_turn, without = reversed(promptkeep.dataset("REVIEW").to_dspy())
        assert with_turn.fields == {
            "what": "code",
            "input": "is this fine?",
            "output": "reviewed code",
        }
        assert with_turn.inputs == ("what", "input")
        assert (without.fields, without.inputs) == (
            {"what": "docs", "output": "reviewed docs"},
            ("what",),
        )

    def test_dspy_field_names_can_be_chosen(self, fake_dspy):
        _run(input_text="q")
        (example,) = promptkeep.dataset("REVIEW").to_dspy(
            input_field="question", output_field="answer"
        )
        assert set(example.fields) == {"what", "question", "answer"}

    def test_a_variable_named_like_a_field_is_refused(self, fake_dspy, tmp_path):
        clash = Prompt("Answer {input} as {output}.", name="CLASH")
        tracking.record_prompt_run(
            clash,
            {"input": "x", "output": "y"},
            "Answer x as y.",
            provider="openai",
            input_text="separate turn",
            output_text="done",
        )
        ds = promptkeep.dataset("CLASH")
        with pytest.raises(ValueError, match="input_field"):
            ds.to_promptfoo(tmp_path / "t.jsonl")
        with pytest.raises(ValueError, match="output_field"):
            ds.to_dspy(input_field="turn")

    def test_dspy_missing_says_how_to_install_it(self, monkeypatch):
        _run()
        monkeypatch.setitem(sys.modules, "dspy", None)  # makes the import fail
        with pytest.raises(ImportError, match="pip install dspy"):
            promptkeep.dataset("REVIEW").to_dspy()


class TestLabelsRead:
    """history.labels(): checks() for many runs in one go."""

    def test_every_key_asked_for_is_answered(self):
        labelled = _run(verdicts=["ok"])
        bare = _run()
        found = history.labels([labelled, bare, "never-recorded"])
        assert [c.name for c in found[labelled]] == ["check0"]
        assert found[bare] == [] and found["never-recorded"] == []

    def test_many_keys_are_chunked(self, monkeypatch):
        monkeypatch.setattr(history, "_KEYS_PER_QUERY", 2)
        keys = [_run(verdicts=["ok"]) for _ in range(5)]
        assert all(len(labels) == 1 for labels in history.labels(keys).values())

    def test_matches_checks_one_run_at_a_time(self):
        key = _run(verdicts=["ok", "warn"], feedback=[("meh", 0.5)])
        storage.record_check(key, {"name": "late", "phase": "post", "status": "ok"})
        assert history.labels([key])[key] == history.checks(key)

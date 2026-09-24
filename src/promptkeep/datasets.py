"""Datasets out of history: a prompt's recorded runs, filtered by how they
went, exported to the tools that evaluate and optimize prompts.

promptkeep already holds the corpus an optimizer needs — every version, every
run, every verdict and piece of feedback. This module is the hand-off, not an
optimizer: ``dataset()`` picks the runs worth learning from and a ``Dataset``
writes them out as JSONL, DSPy examples or promptfoo test cases. The queries
are ``history``'s; what lives here is the filtering and the formats.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from . import history
from .history import CheckInfo, RunInfo


@dataclass(frozen=True)
class Dataset:
    """Recorded runs of one prompt, chosen by ``promptkeep.dataset()``, with
    the labels on each — newest first.

    Every run completed (status "ok") and has a reply. An example's inputs are
    the run's variables, plus the user turn when the call had one separate
    from the prompt (``input_text``); its output is the reply. Iterating
    yields the RunInfo rows; ``labels`` maps each run key to its check
    verdicts and feedback.
    """

    prompt: str
    version: int | None
    runs: list[RunInfo]
    labels: dict[str, list[CheckInfo]]

    def __len__(self) -> int:
        """How many examples the dataset holds."""
        return len(self.runs)

    def __iter__(self) -> Iterator[RunInfo]:
        """The runs, newest first."""
        return iter(self.runs)

    def to_jsonl(self, path: str | Path) -> int:
        """Write one JSON object per example — run key, prompt and version,
        model, variables, input, output and labels — and return how many.
        The neutral format: anything that reads JSON lines can take it."""
        with open(path, "w", encoding="utf-8") as out:
            for run in self.runs:
                record = {
                    "run_key": run.run_key,
                    "prompt": run.prompt_name,
                    "version": run.version,
                    "model": run.model,
                    "variables": run.variables or {},
                    "input": _turn_input(run),
                    "output": run.output_text,
                    "labels": [asdict(c) for c in self.labels.get(run.run_key, [])],
                    "created_at": run.created_at,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
        return len(self.runs)

    def to_dspy(self, input_field: str = "input", output_field: str = "output") -> list[Any]:
        """The examples as ``dspy.Example`` objects, inputs marked.

        Each example's fields are the run's variables, the user turn under
        ``input_field`` (when the call had one) and the reply under
        ``output_field``; every field but the output is an input. Needs DSPy
        installed (``pip install dspy``) — promptkeep doesn't depend on it.
        """
        try:
            import dspy  # type: ignore[import-not-found, unused-ignore]
        except ImportError:
            raise ImportError(
                "Dataset.to_dspy() needs DSPy, which promptkeep doesn't install: pip install dspy"
            ) from None

        examples = []
        for run in self.runs:
            fields = _inputs(run, input_field)
            if output_field in fields:
                raise ValueError(
                    f"variable {output_field!r} collides with the output field; "
                    "pass output_field= to name it something else"
                )
            example = dspy.Example(**fields, **{output_field: run.output_text})
            examples.append(example.with_inputs(*fields))
        return examples

    def to_promptfoo(self, path: str | Path, input_field: str = "input") -> int:
        """Write the examples as promptfoo test cases (JSONL, one per line)
        and return how many — load them with ``tests: file://<path>``.

        Each case's ``vars`` are the run's variables, plus the user turn under
        ``input_field`` when the call had one, so the same inputs can be run
        against a new prompt version. The recorded reply travels as
        ``metadata.reference_output`` alongside the run key and version, for
        an assertion or a grader to compare against; no assertions are
        written — what counts as a pass is yours to say.
        """
        with open(path, "w", encoding="utf-8") as out:
            for run in self.runs:
                case = {
                    "description": f"{run.prompt_name} v{run.version} · run {run.run_key[:8]}",
                    "vars": _inputs(run, input_field),
                    "metadata": {
                        "promptkeep_run_key": run.run_key,
                        "prompt": run.prompt_name,
                        "version": run.version,
                        "model": run.model,
                        "reference_output": run.output_text,
                    },
                }
                out.write(json.dumps(case, ensure_ascii=False) + "\n")
        return len(self.runs)


def dataset(
    prompt: str,
    version: int | None = None,
    *,
    passed: bool | None = None,
    feedback: str | None = None,
    min_feedback: float | None = None,
    limit: int | None = None,
) -> Dataset:
    """A prompt's recorded runs as a dataset — the runs worth learning from.

        ds = promptkeep.dataset("REVIEW_SYSTEM", version=4, passed=True, limit=500)
        ds.to_jsonl("review_v4.jsonl")
        trainset = ds.to_dspy()
        ds.to_promptfoo("review_tests.jsonl")

    Only completed runs with a reply are included — an error or a blocked call
    taught nothing about the prompt. The filters combine (all must hold):

    - ``version``: runs of that version only (default: every version).
    - ``passed``: True keeps runs whose checks all returned ok (and that had
      at least one); False keeps runs with any other verdict — the failures,
      for a negative set. Feedback is not a check and never counts here.
    - ``feedback``: runs that received feedback with this label.
    - ``min_feedback``: runs whose feedback scores average at least this;
      runs with no scored feedback are left out.
    - ``limit``: the newest this many, counted *after* filtering.

    A read: raises for a broken database, and is empty when tracking is
    disabled or nothing matches.
    """
    # Every completed run of the prompt, newest first, with its labels.
    candidates = [
        run
        for run in history.runs(prompt, version=version, limit=None)
        if run.status == "ok" and run.output_text is not None
    ]
    labels = history.labels([run.run_key for run in candidates])

    # The filters, then the limit — so ``limit`` counts examples, not runs.
    chosen = [
        run for run in candidates if _matches(labels[run.run_key], passed, feedback, min_feedback)
    ]
    if limit is not None:
        chosen = chosen[:limit]
    return Dataset(
        prompt=prompt,
        version=version,
        runs=chosen,
        labels={run.run_key: labels[run.run_key] for run in chosen},
    )


# --- helpers -----------------------------------------------------------------------


def _matches(
    run_labels: list[CheckInfo],
    passed: bool | None,
    feedback: str | None,
    min_feedback: float | None,
) -> bool:
    """Whether one run's labels satisfy every filter that was given."""
    verdicts = [c.status for c in run_labels if c.phase != "feedback"]
    given = [c for c in run_labels if c.phase == "feedback"]

    # Checks: all ok (and some ran), or anything but.
    if passed is True and not (verdicts and all(s == "ok" for s in verdicts)):
        return False
    if passed is False and all(s == "ok" for s in verdicts):
        return False

    # Feedback: a label given, and an average score reached.
    if feedback is not None and not any(c.name == feedback for c in given):
        return False
    if min_feedback is not None:
        scores = [c.score for c in given if c.score is not None]
        if not scores or sum(scores) / len(scores) < min_feedback:
            return False
    return True


def _turn_input(run: RunInfo) -> str | None:
    """The user turn a run answered, when it had one of its own. A Prompt
    that *was* the user turn is already the variables' rendering, not a
    separate input."""
    if run.input_text is None or run.input_text == run.rendered_text:
        return None
    return run.input_text


def _inputs(run: RunInfo, input_field: str) -> dict[str, Any]:
    """An example's inputs: the run's variables, plus its user turn under
    ``input_field``. A variable of that name would be overwritten, so that is
    refused rather than silently resolved."""
    fields = dict(run.variables or {})
    turn = _turn_input(run)
    if turn is not None:
        if input_field in fields:
            raise ValueError(
                f"variable {input_field!r} collides with the input field; "
                "pass input_field= to name it something else"
            )
        fields[input_field] = turn
    return fields

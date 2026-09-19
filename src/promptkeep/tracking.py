"""Recording a run for a Prompt, labelling a run after the fact
(``feedback()``), and ``flush()`` — the "everything recorded so far is
durable" call.

``record_prompt_run`` is the one thing an integration (or a script that
tracks calls by hand) needs beyond ``storage.record_run``: it resolves the
Prompt to its version row first. Everything underneath is shielded, so this
never raises into a request path.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from . import checks, storage, writer
from .prompts import Prompt

logger = logging.getLogger("promptkeep")


def record_prompt_run(
    prompt: Prompt,
    variables: dict[str, Any] | None,
    rendered_text: str,
    *,
    provider: str,
    run_key: str | None = None,
    **fields: Any,
) -> str | None:
    """Record one execution of ``prompt``: resolve its version, then write the
    run row through ``storage.record_run`` (which takes every remaining
    keyword — model, output_text, usage, conversation_id, checks, ...).
    Returns the run's key when recorded, None when tracking is off or the
    prompt could not be registered. Never raises — this sits on the caller's
    request path, and a completely broken storage layer must cost telemetry,
    not the call.
    """
    try:
        # None means tracking is off or registration failed — either way
        # there is no lineage to attach the run to.
        registration = prompt._ensure_registered()
        if registration is None:
            return None
        return storage.record_run(
            run_key=run_key,
            version_id=registration[0],
            variables=variables,
            rendered_text=rendered_text,
            provider=provider,
            **fields,
        )
    except Exception:
        logger.warning("promptkeep: failed to record run", exc_info=True)
        return None


def feedback(
    run_key: str | None,
    *,
    score: float | None = None,
    label: str | None = None,
    comment: str | None = None,
) -> None:
    """Attach a human (or downstream) judgement to a run, by its key.

        response = client.chat.completions.create(...)
        key = response.promptkeep.run_key          # keep it with your own records
        ...
        promptkeep.feedback(key, score=1.0, label="thumbs_up")
        promptkeep.feedback(key, score=0.0, label="hallucination", comment="invented a citation")

    It is stored where check verdicts are — the ``checks`` table is the label
    store — as a row with ``phase="feedback"``: ``label`` is its name,
    ``comment`` its message. So it rides the same write path as a late
    verdict (queued behind its run in background mode, redacted, skipped with
    a warning if the run was never persisted), ``history.checks(run_key)``
    returns it next to the automatic verdicts, and a run can collect any
    number of them. Feedback never changes a run's pass/fail headline.

    ``run_key=None`` — what a handle carries when its run wasn't recorded —
    is a no-op, so ``feedback(response.promptkeep.run_key, ...)`` is always
    safe. Passing no judgement at all is a usage error and raises; nothing
    else here does.
    """
    # A label with nothing in it is a bug at the call site, not telemetry.
    if score is None and label is None and comment is None:
        raise ValueError("feedback() needs at least one of score=, label= or comment=")
    if score is not None and (isinstance(score, bool) or not isinstance(score, (int, float))):
        raise TypeError(f"feedback(score=...) must be a number, got {type(score).__name__}")

    # The same row shape a check produces, under its own phase.
    storage.record_check(
        run_key,
        {
            "name": label or "feedback",
            "phase": "feedback",
            "status": "ok",
            "score": None if score is None else float(score),
            "message": comment,
            "latency_ms": None,
            "rewritten": None,
        },
    )


def flush(timeout: float | None = None) -> bool:
    """Block until everything recorded so far is on disk, or ``timeout`` passes.

    Two things can still be in flight after a call returns: async post-checks
    (their verdicts don't exist until the check finishes) and the background
    write queue. flush() waits for the checks first — each verdict joins the
    queue as it lands — then drains the queue. Returns True when both have
    fully settled, False on timeout. Cheap when nothing is pending.

        promptkeep.flush(timeout=5)   # before a script exits, or before a
                                      # test reads back what it just wrote
    """
    deadline = None if timeout is None else time.monotonic() + timeout

    def remaining() -> float | None:
        return None if deadline is None else max(0.0, deadline - time.monotonic())

    if not checks.wait_for_pending(remaining()):
        return False
    return writer.drain(remaining())

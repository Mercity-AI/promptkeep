"""Provider-agnostic run recording: the bridge between an integration wrapper
(which knows about requests/responses) and storage (which knows about rows).
Also home to ``flush()``, the "everything recorded so far is durable" call.

Never raises into the caller's request path — losing telemetry is always
preferable to breaking an LLM call.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from . import checks, writer
from .prompts import Prompt

logger = logging.getLogger("promptkeep")


def record_prompt_run(
    prompt: Prompt,
    variables: Optional[Dict[str, Any]],
    rendered_text: str,
    *,
    run_key: Optional[str] = None,
    provider: str,
    model: Optional[str] = None,
    request_params: Optional[Dict[str, Any]] = None,
    response_id: Optional[str] = None,
    output_text: Optional[str] = None,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    latency_ms: Optional[int] = None,
    status: str = "ok",
    error: Optional[str] = None,
    conversation_id: Optional[int] = None,
    turn_index: Optional[int] = None,
    input_text: Optional[str] = None,
    original_input_text: Optional[str] = None,
    checks: Optional[list] = None,
) -> Optional[str]:
    """Record one execution of a prompt: resolve its version, insert a run row.
    Returns the run's key when recorded (see storage.record_run).

    Silently skips when tracking is disabled; swallows (and logs) all errors.
    """
    try:
        # Resolve the prompt to its version row; None means tracking is off
        # or registration failed — either way there's nothing to attach to.
        registration = prompt._ensure_registered()
        if registration is None:
            return None
        from . import storage

        return storage.record_run(
            run_key=run_key,
            version_id=registration[0],
            variables=variables,
            rendered_text=rendered_text,
            provider=provider,
            model=model,
            request_params=request_params,
            response_id=response_id,
            output_text=output_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            status=status,
            error=error,
            conversation_id=conversation_id,
            turn_index=turn_index,
            input_text=input_text,
            original_input_text=original_input_text,
            checks=checks,
        )
    except Exception:
        logger.warning("promptkeep: failed to record run", exc_info=True)
        return None


def record_conversation_turn(
    *,
    run_key: Optional[str] = None,
    provider: str,
    model: Optional[str] = None,
    request_params: Optional[Dict[str, Any]] = None,
    response_id: Optional[str] = None,
    output_text: Optional[str] = None,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    latency_ms: Optional[int] = None,
    status: str = "ok",
    error: Optional[str] = None,
    conversation_id: Optional[int] = None,
    turn_index: Optional[int] = None,
    input_text: Optional[str] = None,
    original_input_text: Optional[str] = None,
    checks: Optional[list] = None,
) -> Optional[str]:
    """Record a turn with no wrapped Prompt — a plain message, or a checked
    call whose only reason to exist as a run is to hang check verdicts off.
    Returns the run's key when recorded.

    There's no version to resolve, so this forwards straight to storage
    instead of going through a Prompt's lineage (that resolution is the only
    reason record_prompt_run does more than this). storage.record_run is
    already fully shielded, so no guard of its own is needed.
    """
    from . import storage

    return storage.record_run(
        run_key=run_key,
        version_id=None,
        variables=None,
        rendered_text=None,
        provider=provider,
        model=model,
        request_params=request_params,
        response_id=response_id,
        output_text=output_text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        latency_ms=latency_ms,
        status=status,
        error=error,
        conversation_id=conversation_id,
        checks=checks,
        turn_index=turn_index,
        input_text=input_text,
        original_input_text=original_input_text,
    )


def flush(timeout: Optional[float] = None) -> bool:
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

    def remaining() -> Optional[float]:
        return None if deadline is None else max(0.0, deadline - time.monotonic())

    if not checks.wait_for_pending(remaining()):
        return False
    return writer.drain(remaining())

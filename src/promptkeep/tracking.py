"""Provider-agnostic run recording: the bridge between an integration wrapper
(which knows about requests/responses) and storage (which knows about rows).

Never raises into the caller's request path — losing telemetry is always
preferable to breaking an LLM call.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .prompts import Prompt

logger = logging.getLogger("promptkeep")


def record_prompt_run(
    prompt: Prompt,
    variables: Optional[Dict[str, Any]],
    rendered_text: str,
    *,
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
) -> None:
    """Record one execution of a prompt: resolve its version, insert a run row.

    Silently skips when tracking is disabled; swallows (and logs) all errors.
    """
    try:
        # Resolve the prompt to its version row; None means tracking is off
        # or registration failed — either way there's nothing to attach to.
        registration = prompt._ensure_registered()
        if registration is None:
            return
        from . import storage

        storage.record_run(
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
        )
    except Exception:
        logger.warning("promptkeep: failed to record run", exc_info=True)


def record_conversation_turn(
    *,
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
    conversation_id: int,
    turn_index: Optional[int] = None,
    input_text: Optional[str] = None,
) -> None:
    """Record a conversation turn that involved no wrapped Prompt at all —
    e.g. a plain follow-up message. There's no version to resolve, so this
    forwards straight to storage instead of going through a Prompt's lineage
    (that resolution is the only reason record_prompt_run does more than this).

    Kept as a named sibling of record_prompt_run so the wrapper's two record
    paths stay symmetric through the tracking layer. No error handling of its
    own: storage.record_run is already fully shielded, including the disabled
    /off-mode early return — a guard here would only shield a call that
    cannot raise.
    """
    from . import storage

    storage.record_run(
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
        turn_index=turn_index,
        input_text=input_text,
    )

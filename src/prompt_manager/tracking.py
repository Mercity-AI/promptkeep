"""Provider-agnostic run recording. Never raises into the caller's request path."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .prompts import Prompt

logger = logging.getLogger("prompt_manager")


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
) -> None:
    try:
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
        )
    except Exception:
        logger.warning("prompt_manager: failed to record run", exc_info=True)

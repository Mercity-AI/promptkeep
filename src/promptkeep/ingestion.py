"""Normalize live framework events before they enter storage or an export queue."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime

from . import controls, storage, writer
from .config import get_settings
from .events import Event, key

logger = logging.getLogger("promptkeep")


def _plain(value):
    """Read visible strings naturally while retaining structured JSON payloads."""
    if value is None:
        return None
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, str) else value
    except ValueError:
        return value


def prepare(event: Event) -> dict | None:
    """Apply production controls before a normalized live event is queued."""
    settings = get_settings()
    if not settings.enabled or settings.write_mode == "off":
        return None
    cohort = (
        int.from_bytes(hashlib.sha256(event.trace_key.encode()).digest()[:8], "big") / 2**64
        < settings.sample_rate
    )
    if not controls.keep_run(settings, {"status": event.status}, None, cohort=cohort):
        return None
    duration = None
    if event.started_at and event.ended_at:
        value = round(
            (
                datetime.fromisoformat(event.ended_at) - datetime.fromisoformat(event.started_at)
            ).total_seconds()
            * 1000
        )
        duration = value if value >= 0 else None
    span = dict(
        span_key=event.span_key,
        trace_key=event.trace_key,
        parent_span_key=key(event.source, event.trace_id, event.parent_id)
        if event.parent_id
        else None,
        kind=event.kind,
        name=event.name,
        started_at=event.started_at,
        ended_at=event.ended_at,
        observed_at=event.observed_at or event.ended_at or event.started_at or storage._utcnow(),
        duration_ms=duration,
        status=event.status,
        sampled=cohort,
        metadata_json=json.dumps({"source": event.source, "timing_quality": event.timing_quality}),
    )
    for name, value in (
        ("input_json", event.input),
        ("output_json", event.output),
        ("error", event.error),
    ):
        if value is not None:
            text = value if name == "error" else json.dumps(value, ensure_ascii=False, default=str)
            if settings.redact:
                text = settings.redact(text)
                if not isinstance(text, str):
                    raise TypeError("redact must return a string")
            encoded = text.encode("utf-8")
            if len(encoded) > 65536:
                text = encoded[:65536].decode("utf-8", errors="ignore") + "\n[promptkeep truncated]"
            span[name] = text
    # The enrichment row contains only already-redacted visible text and numeric usage.
    usage = {k: v for k, v in event.usage.items() if type(v) in (int, float) and v >= 0}
    incoming = usage.get("input_tokens", usage.get("prompt_tokens"))
    outgoing = usage.get("output_tokens", usage.get("completion_tokens"))
    run = dict(
        run_key=event.span_key,
        provider=event.provider or "unknown",
        model=event.model,
        response_id=event.response_id,
        span_key=event.span_key,
        created_at=event.observed_at or event.ended_at or event.started_at or storage._utcnow(),
        status=event.status,
        error=span.get("error"),
        latency_ms=duration,
        input_text=_plain(span.get("input_json")),
        output_text=_plain(span.get("output_json")),
        prompt_tokens=incoming,
        completion_tokens=outgoing,
        total_tokens=usage.get(
            "total_tokens",
            incoming + outgoing if incoming is not None and outgoing is not None else None,
        ),
        usage_details=json.dumps(usage) if usage else None,
        cost_usd=event.cost_usd,
    )
    return dict(
        _kind="event",
        span=span,
        run=run if event.kind == "llm" and event.status != "running" else None,
        identity=(event.source, event.trace_id, event.span_id),
        links=[key(event.source, event.trace_id, value) for value in event.links]
        + list(event.link_keys),
    )


def record(event: Event) -> bool:
    """Live event ingestion is fail-open and follows configured persistence mode."""
    try:
        with storage.connection_context(event.destination):
            item = prepare(event)
            if item is None:
                return True
            if get_settings().write_mode == "background":
                writer.submit(item)
            elif storage.get_db() is not None:
                with storage.notification_transaction(storage.db_proxy):
                    storage._insert_event(item)
            return True
    except Exception:
        logger.warning("promptkeep: failed to ingest external event", exc_info=True)
        return False

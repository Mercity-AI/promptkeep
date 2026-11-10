"""Pure event preparation and payload reduction shared by every ingestion path."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

from . import controls
from .config import get_settings
from .events import Event, key


class PreparedPayload(str):
    """Immutable serialized payload that has already passed redaction and truncation."""


def payload(value, settings, *, serialized=False):
    """Prepare once before queueing or staging; preserve explicit JSON null."""
    if isinstance(value, PreparedPayload):
        return value
    text = value if serialized else json.dumps(value, ensure_ascii=False, default=str)
    if settings.redact:
        text = settings.redact(text)
        if not isinstance(text, str):
            raise TypeError("redact must return a string")
    encoded = text.encode("utf-8")
    if len(encoded) > 65536:
        text = encoded[:65536].decode("utf-8", errors="ignore") + "\n[promptkeep truncated]"
    return PreparedPayload(text)


def merge(existing, patch):
    """Reduce a partial update: omitted payloads survive; explicit values replace."""
    return {**existing, **patch}


def _plain(value):
    """Expose visible strings naturally and retain serialized structured payloads."""
    if value is None:
        return None
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, str) else value
    except ValueError:
        return value


def cohort_for(event, settings):
    """Use the same deterministic trace cohort for framework spans and provider runs."""
    return (
        int.from_bytes(hashlib.sha256(event.trace_key.encode()).digest()[:8], "big") / 2**64
        < settings.sample_rate
    )


def prepare(event: Event, *, settings=None, archive=False) -> dict | None:
    """Normalize one immutable observation before delivery policy is applied."""
    settings = settings or get_settings()
    if not settings.enabled or settings.write_mode == "off":
        return None
    cohort = archive or cohort_for(event, settings)
    if not archive and not controls.keep_run(
        settings,
        {"status": "ok" if event.status == "running" else event.status},
        None,
        cohort=cohort,
    ):
        return None
    duration = event.duration_ms
    if duration is None and event.started_at and event.ended_at:
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
        observed_at=event.observed_at
        or event.ended_at
        or event.started_at
        or datetime.now(UTC).isoformat(),
        duration_ms=duration,
        status=event.status,
        sampled=cohort,
        metadata_json=json.dumps(
            {"source": event.source, "timing_quality": event.timing_quality} | dict(event.metadata)
        ),
    )
    for name, value, present in (
        ("input_json", event.input, event.input_present),
        ("output_json", event.output, event.output_present),
        ("error", event.error, False),
    ):
        if value is not None or present:
            span[name] = payload(value, settings, serialized=name == "error")
    usage = {k: v for k, v in event.usage.items() if type(v) in (int, float) and v >= 0}
    incoming = usage.get("input_tokens", usage.get("prompt_tokens"))
    outgoing = usage.get("output_tokens", usage.get("completion_tokens"))
    run = dict(
        run_key=event.span_key,
        provider=event.provider or "unknown",
        model=event.model,
        response_id=event.response_id,
        span_key=event.span_key,
        created_at=span["observed_at"],
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

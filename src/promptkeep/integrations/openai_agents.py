"""OpenAI Agents SDK tracing processor; importing this module requires its extra."""

from __future__ import annotations

from agents.tracing import TracingProcessor

from ..events import Event
from ..storage import _utcnow
from ._bridge import Bridge


def _visible(value):
    """Read only model text the SDK explicitly included in its exported data."""
    if isinstance(value, str):
        return value
    if not isinstance(value, (tuple, list)):
        return None
    parts = []
    for item in value:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(
                block["text"]
                for block in content
                if isinstance(block, dict)
                and block.get("type") in ("text", "input_text", "output_text")
                and isinstance(block.get("text"), str)
            )
    return "\n".join(parts) or None


class PromptKeepTracingProcessor(Bridge, TracingProcessor):
    """Consume native Agents SDK events without replacing its other processors."""

    def on_trace_start(self, trace):
        """A native trace becomes an agent root with an observed callback start."""
        self.start(
            Event(
                "openai-agents",
                trace.trace_id,
                trace.trace_id,
                getattr(trace, "name", "agent"),
                kind="agent",
                started_at=_utcnow(),
                status="running",
            )
        )

    def on_trace_end(self, trace):
        """Finish the root independently of caught child failures."""
        self.end(
            Event(
                "openai-agents",
                trace.trace_id,
                trace.trace_id,
                getattr(trace, "name", "agent"),
                kind="agent",
                ended_at=_utcnow(),
            )
        )

    def _event(self, span, status):
        """Read SDK-exported fields; absent sensitive data remains absent."""
        data = span.span_data.export()
        kind = data.get("type")
        mapped = (
            "llm"
            if kind in ("generation", "response")
            else "tool"
            if kind == "function"
            else "agent"
            if kind == "agent"
            else "operation"
        )
        response = getattr(span.span_data, "response", None)
        usage = data.get("usage") or {}
        if response is not None:
            raw_usage = getattr(response, "usage", None)
            if raw_usage is not None:
                usage = raw_usage.model_dump()
        return Event(
            "openai-agents",
            span.trace_id,
            span.span_id,
            data.get("name") or data.get("model") or kind or "operation",
            kind=mapped,
            parent_id=span.parent_id or span.trace_id,
            started_at=span.started_at,
            ended_at=span.ended_at,
            status="error"
            if span.error
            else "blocked"
            if kind == "guardrail" and data.get("triggered")
            else status,
            error=str(span.error.get("message", "framework error")) if span.error else None,
            input=data.get("input") if kind == "function" else _visible(data.get("input")),
            output=data.get("output") if kind == "function" else _visible(data.get("output")),
            model=data.get("model") or getattr(response, "model", None),
            response_id=data.get("response_id"),
            usage=usage,
        )

    def on_span_start(self, span):
        """Keep native parentage and native start timing."""
        self.start(self._event(span, "running"))

    def on_span_end(self, span):
        """Generation, tool and custom spans share the same ingestion path."""
        self.end(self._event(span, "ok"))

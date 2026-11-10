"""OpenCode saved exports: modern session-transfer and legacy message/parts JSON."""

from __future__ import annotations

from dataclasses import replace

from ..events import TraceEvent, key
from .base import AgentTrace, _timestamp


def _time(value):
    """OpenCode encoded DateTime values are epoch milliseconds (or aware ISO)."""
    return _timestamp(value, unit="milliseconds")


def _text(parts):
    """Read visible text only; reasoning, media and injected content are excluded."""
    return (
        "\n".join(
            p["text"]
            for p in parts
            if isinstance(p, dict)
            and p.get("type") == "text"
            and isinstance(p.get("text"), str)
            and not p.get("synthetic")
            and not p.get("ignored")
        )
        or None
    )


def _usage(tokens):
    """Preserve cache counters separately while counting all input tokens once."""
    if not isinstance(tokens, dict):
        return {}
    cache = tokens.get("cache") or {}
    result = {
        target: tokens[name]
        for name, target in (
            ("input", "input_tokens"),
            ("output", "output_tokens"),
            ("reasoning", "reasoning_output_tokens"),
            ("total", "total_tokens"),
        )
        if type(tokens.get(name)) in (int, float) and tokens[name] >= 0
    }
    for name, target in (("read", "cached_input_tokens"), ("write", "cache_write_input_tokens")):
        if type(cache.get(name)) in (int, float) and cache[name] >= 0:
            result[target] = cache[name]
    if "input_tokens" in result:
        result["input_tokens"] += result.get("cached_input_tokens", 0) + result.get(
            "cache_write_input_tokens", 0
        )
    if "total_tokens" not in result and "input_tokens" in result and "output_tokens" in result:
        result["total_tokens"] = (
            result["input_tokens"]
            + result["output_tokens"]
            + result.get("reasoning_output_tokens", 0)
        )
    return result


class OpenCodeTrace(AgentTrace):
    """Interpret complete export snapshots; never read the native database."""

    source = "opencode"
    revision = 1
    snapshot = True
    per_call_complete = True

    @classmethod
    def detect(cls, record):
        """Recognize the verified session-transfer export envelope."""
        return (
            isinstance(record, dict)
            and isinstance(record.get("info"), dict)
            and isinstance(record["info"].get("id"), str)
            and isinstance(record.get("messages"), list)
        )

    def model_records(self):
        """Model observations are prepared with their usage during consumption."""

    def consume(self, record, line):
        """Read modern messages or legacy info/parts in recorded array order."""
        if not self.detect(record):
            raise ValueError("OpenCode invalid_export")
        info = record["info"]
        self.session = info["id"]
        self.trace = key(self.source, self.session, self.session)
        times = info.get("time") or {}
        self.first = _time(times.get("created"))
        self.last = _time(times.get("updated")) or self.first
        root = TraceEvent(
            self.source,
            self.session,
            self.session,
            "opencode",
            kind="agent",
            started_at=self.first,
            observed_at=self.first,
            status="incomplete",
            links=(info["parentID"],) if info.get("parentID") else (),
        )
        if info.get("parentID"):
            root = replace(
                root, links=(), link_keys=(key(self.source, info["parentID"], info["parentID"]),)
            )
        self.stage_event(root)
        if info.get("parentID"):
            self.stage.put("spans", self.trace, {"_link_keys": list(root.link_keys)})
        message_turns = {}
        for index, entry in enumerate(record["messages"], 1):
            if not isinstance(entry, dict):
                raise ValueError("OpenCode invalid_message")
            legacy = "info" in entry
            message = entry.get("info") if legacy else entry
            if not isinstance(message, dict) or not isinstance(message.get("id"), str):
                raise ValueError("OpenCode invalid_message_identity")
            identity = message["id"]
            parts = entry.get("parts", []) if legacy else message.get("content", [])
            kind = message.get("role") if legacy else message.get("type")
            time = message.get("time") or {}
            started, ended = _time(time.get("created")), _time(time.get("completed"))
            observed = ended or started
            if observed:
                self.last = max(self.last or observed, observed)
            if kind == "user":
                self.current = identity
                message_turns[identity] = identity
                self.stage_event(
                    TraceEvent(
                        self.source,
                        self.session,
                        identity,
                        "opencode.turn",
                        kind="agent",
                        parent_id=self.session,
                        started_at=started,
                        observed_at=started,
                        status="incomplete",
                        input=_text(parts) if legacy else message.get("text"),
                    )
                )
                continue
            parent = message_turns.get(message.get("parentID")) or self.current or self.session
            if kind == "assistant":
                message_turns[identity] = parent
                ref = message.get("model") or {}
                model = message.get("modelID") if legacy else ref.get("id")
                provider = message.get("providerID") if legacy else ref.get("providerID")
                completed = ended
                ended = _time(time.get("streamed")) or ended
                error = message.get("error")
                status = (
                    "cancelled"
                    if isinstance(error, dict)
                    and error.get("name", error.get("type"))
                    in ("MessageAbortedError", "AbortError")
                    else "error"
                    if error
                    else "ok"
                    if ended
                    else "incomplete"
                )
                usage = _usage(message.get("tokens"))
                event = TraceEvent(
                    self.source,
                    self.session,
                    identity,
                    model or "model",
                    kind="llm",
                    parent_id=parent,
                    started_at=started,
                    ended_at=ended,
                    observed_at=observed,
                    status=status,
                    output=_text(parts),
                    model=model,
                    provider=provider,
                    usage=usage,
                    error=self._error(error),
                    metadata={
                        "cost_provenance": "harness_reported",
                        "source_cost_usd": message.get("cost"),
                        "source_completed_at": completed,
                    },
                )
                self.stage_event(event)
                if usage:
                    self.stage.put(
                        "usage",
                        "message:" + identity,
                        dict(
                            scope="response",
                            operation_id=identity,
                            span_key=event.span_key,
                            turn_id=parent,
                            observed_at=observed,
                            usage=usage,
                        ),
                    )
                for part_index, part in enumerate(parts):
                    if not isinstance(part, dict) or part.get("type") != "tool":
                        continue
                    state = part.get("state") or {}
                    tool_time = state.get("time", {}) if legacy else part.get("time", {})
                    begin = _time(tool_time.get("start" if legacy else "ran")) or _time(
                        tool_time.get("created")
                    )
                    end = _time(tool_time.get("end" if legacy else "completed"))
                    tool_id = (
                        part.get("callID") or part.get("id") or f"{identity}:tool:{part_index}"
                    )
                    result = state.get("output") if legacy else _text(state.get("content") or [])
                    self.stage_event(
                        TraceEvent(
                            self.source,
                            self.session,
                            tool_id,
                            part.get("tool") or part.get("name") or "opencode.tool",
                            kind="tool",
                            parent_id=identity,
                            started_at=begin,
                            ended_at=end,
                            observed_at=end or begin or observed,
                            status="error"
                            if state.get("status") == "error"
                            else "ok"
                            if state.get("status") == "completed"
                            else "incomplete",
                            input=state.get("input"),
                            output=result,
                            error=self._error(state.get("error")),
                        )
                    )
                if ended and parent != self.session:
                    self.finish(key(self.source, self.session, parent), ended, status, index)
            elif kind == "idle":
                if self.current:
                    status = {"succeeded": "ok", "failed": "error", "interrupted": "cancelled"}.get(
                        message.get("outcome"), "incomplete"
                    )
                    self.finish(
                        key(self.source, self.session, self.current), observed, status, index
                    )
            elif kind in (
                "compaction",
                "model-switched",
                "agent-switched",
                "system",
                "synthetic",
                "shell",
            ):
                self.stage_event(
                    TraceEvent(
                        self.source,
                        self.session,
                        identity,
                        kind,
                        parent_id=parent,
                        started_at=started,
                        ended_at=ended,
                        observed_at=observed,
                        status="ok",
                        output=message.get("text") if kind in ("system", "synthetic") else None,
                    )
                )
            else:
                self.counts["ignored"] += 1

    @staticmethod
    def _error(value):
        """Keep the error message without unrelated provider environment metadata."""
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            data = value.get("data") or {}
            return (
                value.get("message")
                or data.get("message")
                or value.get("name")
                or value.get("type")
            )
        return None

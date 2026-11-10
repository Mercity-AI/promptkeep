"""Native Claude Code transcript import using the shared atomic archive machinery."""

from __future__ import annotations

import json
import re

from .base import AgentTrace, ImportReport, _key, _timestamp, _visible
from .engine import import_trace as _import_trace

_REVISION = 3


def _classify(event, message):
    """Use native flags and complete recognized envelopes, never substring matches."""
    content = message.get("content", [])
    blocks = content if isinstance(content, list) else []
    if any(isinstance(b, dict) and b.get("type") == "tool_result" for b in blocks):
        return "tool_result"
    if event.get("isCompactSummary"):
        return "compaction"
    if event.get("isMeta") or event.get("isVisibleInTranscriptOnly"):
        return "injected_note"
    text = _visible(content)
    if text in ("[Request interrupted by user]", "[Request interrupted by user for tool use]"):
        return "interruption"
    if isinstance(text, str) and re.fullmatch(
        r"\s*<(local-command-stdout|local-command-stderr|command-message|command-name|system-reminder)>[\s\S]*</\1>\s*",
        text,
    ):
        return "command_output" if text.lstrip().startswith("<local-command-") else "injected_note"
    return "prompt"


class ClaudeCodeTrace(AgentTrace):
    """Preserve explicit message ancestry and response/tool identities."""

    source = "claude-code"
    provider = "anthropic"
    revision = _REVISION
    per_call_complete = True

    def __init__(self, stage, strict=False, session=None, native_session=None, agent_id=None):
        """Session identity is discovered before payloads are staged."""
        super().__init__(stage, strict)
        self.session = session
        self.native_session = native_session or session
        self.agent_id = agent_id
        self.trace = _key(session, "agent", "session", self.source)
        self.message_turns = {}

    def parse(self, event: dict, line: int) -> None:
        """Read native messages; ignore unrelated snapshots and environment records."""
        if not isinstance(event, dict):
            self.diagnose("invalid_record", line, True)
            return
        if event.get("sessionId") not in (None, self.native_session):
            raise ValueError(f"Claude conflicting_session_identity at line {line}")
        if event.get("type") in ("user", "assistant") and event.get("agentId") not in (
            None,
            self.agent_id,
        ):
            raise ValueError(f"Claude conflicting_agent_identity at line {line}")
        timestamp = _timestamp(event.get("timestamp"))
        if timestamp:
            self.first = min(self.first or timestamp, timestamp)
            self.last = max(self.last or timestamp, timestamp)
        if self.stage.get("spans", self.trace) is None:
            self.stage.put(
                "spans",
                self.trace,
                dict(
                    span_key=self.trace,
                    trace_key=self.trace,
                    parent_span_key=None,
                    kind="agent",
                    name=self.source,
                    started_at=timestamp,
                    observed_at=timestamp,
                    status="incomplete",
                    sampled=True,
                    metadata_json=json.dumps({"source": self.source, "timing_quality": "observed"}),
                ),
            )
        if self.agent_id is not None:
            parent_tool = (
                event.get("parentToolUseID")
                or event.get("parentToolUseId")
                or event.get("parent_tool_use_id")
            )
            parent = (
                _key(self.native_session, "tool", parent_tool, self.source)
                if parent_tool
                else _key(self.native_session, "agent", "session", self.source)
            )
            self.stage.put("spans", self.trace, {"_link_keys": [parent]})
        root = self.stage.get("spans", self.trace)
        if timestamp and root.get("started_at") is None:
            self.stage.put("spans", self.trace, {"started_at": timestamp, "observed_at": timestamp})
        kind = event.get("type")
        message = event.get("message")
        if kind not in ("user", "assistant") or not isinstance(message, dict):
            self.counts["ignored"] += 1
            if kind == "system" and event.get("subtype") == "compact_boundary":
                identity = event.get("uuid") or f"compaction:{line}"
                self.span(identity, "operation", "compaction", timestamp, self.trace, status="ok")
            return
        identity = event.get("uuid") or f"line:{line}"
        if not event.get("uuid"):
            self.diagnose("fallback_message_identity", line)
        content = message.get("content", [])
        blocks = content if isinstance(content, list) else []
        parent_turn = self.message_turns.get(event.get("parentUuid"))
        if parent_turn:
            self.current = parent_turn
        classification = _classify(event, message) if kind == "user" else "assistant"
        if kind == "user" and classification not in ("prompt", "tool_result"):
            self.message_turns[identity] = self.current
            parent = (
                _key(self.session, "agent", self.current, self.source)
                if self.current
                else self.trace
            )
            self.span(
                identity,
                "operation",
                classification,
                timestamp,
                parent,
                status="ok",
                output_json=_visible(content),
            )
            if classification == "interruption" and self.current:
                self.finish(parent, timestamp, "cancelled", line)
            return
        if kind == "user" and classification == "prompt":
            self.current = identity
            turn = self.turn(identity, timestamp)
            if turn:
                self.stage.put("spans", turn, {"input_json": _visible(content)})
                if parent_turn and parent_turn != identity:
                    self.stage.put(
                        "spans",
                        turn,
                        {"parent_span_key": _key(self.session, "agent", parent_turn, self.source)},
                    )
        else:
            turn = self.turn(None, timestamp)
        self.message_turns[identity] = self.current
        if kind == "assistant":
            response = message.get("id")
            if not isinstance(response, str) or not response:
                self.diagnose("missing_response_identity", line, True)
                return
            model = message.get("model")
            model_key = _key(self.session, "llm", response, self.source)
            # Only observation time is known in ordinary assistant transcript records.
            existing = self.stage.get("spans", model_key)
            if existing is None:
                self.stage.put(
                    "spans",
                    model_key,
                    dict(
                        span_key=model_key,
                        trace_key=self.trace,
                        parent_span_key=turn,
                        kind="llm",
                        name=model or "model",
                        started_at=None,
                        observed_at=timestamp,
                        status="ok",
                        sampled=True,
                        metadata_json=json.dumps(
                            {"source": self.source, "timing_quality": "unknown"}
                        ),
                    ),
                )
            text = (
                _visible(content)
                or "\n".join(
                    b["refusal"]
                    for b in blocks
                    if isinstance(b, dict)
                    and b.get("type") == "refusal"
                    and isinstance(b.get("refusal"), str)
                )
                or None
            )
            if text:
                self.stage.append_text(model_key, identity, text)
                if turn:
                    self.stage.put("spans", turn, {"output_json": text})
            raw_usage = message.get("usage")
            if isinstance(raw_usage, dict):
                usage = {k: v for k, v in raw_usage.items() if type(v) is int and v >= 0}
                incoming = usage.get("input_tokens")
                outgoing = usage.get("output_tokens")
                if incoming is not None:
                    incoming += (usage.get("cache_read_input_tokens") or 0) + (
                        usage.get("cache_creation_input_tokens") or 0
                    )
                    usage["input_tokens"] = incoming
                if incoming is not None and outgoing is not None:
                    usage["total_tokens"] = incoming + outgoing
                old = self.stage.get("usage", f"response:{response}")
                if old:
                    usage = {k: max(v, old["usage"].get(k, 0)) for k, v in usage.items()}
                if "input_tokens" in usage and "output_tokens" in usage:
                    usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
                self.stage.put(
                    "usage",
                    f"response:{response}",
                    dict(
                        scope="response",
                        turn_id=self.current,
                        response_id=response,
                        model=model,
                        observed_at=timestamp,
                        usage=usage,
                    ),
                )
            for block in blocks:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                self.tool_record(
                    dict(
                        type="function_call",
                        call_id=block.get("id"),
                        name=block.get("name"),
                        arguments=block.get("input"),
                    ),
                    timestamp,
                    line,
                )
                tool_key = _key(self.session, "tool", block.get("id"), self.source)
                if self.stage.get("spans", tool_key):
                    self.stage.put("spans", tool_key, {"parent_span_key": model_key})
            refusal = message.get("stop_reason") == "refusal" or any(
                isinstance(b, dict) and b.get("type") == "refusal" for b in blocks
            )
            if refusal:
                metadata = json.loads(self.stage.get("spans", model_key)["metadata_json"])
                self.stage.put(
                    "spans", model_key, {"metadata_json": json.dumps(metadata | {"refusal": True})}
                )
            if turn and (
                refusal or message.get("stop_reason") in ("end_turn", "stop_sequence", "max_tokens")
            ):
                self.finish(turn, timestamp, "ok", line)
        else:
            for block in blocks:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    self.tool_record(
                        dict(
                            type="function_call_output",
                            call_id=block.get("tool_use_id"),
                            output=block.get("content"),
                            is_error=block.get("is_error", False),
                        ),
                        timestamp,
                        line,
                    )

    def tool_record(self, record, timestamp, line):
        """Correlate explicit tool-use IDs using the common span lifecycle."""
        identity = record.get("call_id")
        if not isinstance(identity, str) or not identity:
            self.diagnose("missing_tool_identity", line, True)
            return
        turn_id = self.call_turns.get(identity) or self.current
        parent = _key(self.session, "agent", turn_id, self.source) if turn_id else self.trace
        span = _key(self.session, "tool", identity, self.source)
        if record["type"] == "function_call":
            self.call_turns[identity] = self.current
            self.span(
                identity,
                "tool",
                record.get("name") or "tool",
                timestamp,
                parent,
                input_json=record.get("arguments"),
            )
        else:
            existing = self.stage.get("spans", span)
            if existing is None:
                self.diagnose("unmatched_tool_output", line)
                span = self.span(identity, "tool", "tool", timestamp, parent)
            if span:
                self.finish(
                    span,
                    timestamp,
                    "error" if record.get("is_error") else "ok" if existing else "incomplete",
                    line,
                    output_json=record.get("output"),
                )

    @classmethod
    def detect(cls, record):
        """Identify Claude transcript identity envelopes."""
        return (
            isinstance(record, dict)
            and isinstance(record.get("sessionId"), str)
            and record.get("type") in ("user", "assistant", "system")
        )

    def discover(self, records):
        """Namespace independent subagent transcripts before consumption."""
        native, agent = None, None
        for _, event in records:
            if not isinstance(event, dict):
                continue
            identity = event.get("sessionId")
            if isinstance(identity, str):
                if native is not None and native != identity:
                    raise ValueError("Claude conflicting_session_identity")
                native = identity
            candidate = event.get("agentId") if event.get("type") in ("user", "assistant") else None
            if candidate:
                if not isinstance(candidate, str) or (agent is not None and agent != candidate):
                    raise ValueError("Claude conflicting_agent_identity")
                agent = candidate
        if not native:
            raise ValueError("Claude missing_session_identity")
        self.native_session, self.agent_id = native, agent
        self.session = native + ":agent:" + agent if agent else native
        self.trace = _key(self.session, "agent", "session", self.source)

    def consume(self, event, line):
        """Public ordered record consumption contract."""
        return self.parse(event, line)


_ClaudeParser = ClaudeCodeTrace


def import_claude_code(path, *, db_path=None, strict=False) -> ImportReport:
    """Compatibility entry point for explicit Claude Code archives."""
    return _import_trace(path, source="claude-code", db_path=db_path, strict=strict)

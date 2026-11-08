"""Native Claude Code transcript import using the shared atomic archive machinery."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from .. import storage
from ..config import get_settings
from .codex import ImportReport, _key, _Parser, _source, _timestamp, _visible

_REVISION = 2


class _ClaudeParser(_Parser):
    """Preserve explicit message ancestry and response/tool identities."""

    source = "claude-code"

    def __init__(self, stage, strict, session, native_session=None, agent_id=None):
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
        is_result = any(isinstance(b, dict) and b.get("type") == "tool_result" for b in blocks)
        if kind == "user" and not is_result:
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
            text = _visible(content)
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
                self.item(
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
            if turn and message.get("stop_reason") in ("end_turn", "stop_sequence", "max_tokens"):
                self.finish(turn, timestamp, "ok", line)
        else:
            for block in blocks:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    self.item(
                        dict(
                            type="function_call_output",
                            call_id=block.get("tool_use_id"),
                            output=block.get("content"),
                            is_error=block.get("is_error", False),
                        ),
                        timestamp,
                        line,
                    )


def import_claude_code(
    path: str | Path, *, db_path: str | Path | None = None, strict: bool = False
) -> ImportReport:
    """Import one explicit native transcript; never scan unrelated subagent files."""
    settings = get_settings()
    if not settings.enabled or settings.write_mode == "off":
        raise ValueError("Claude import requires enabled tracking and write_mode other than off")
    destination = str(Path(db_path or settings.db_path).absolute())
    native_session = None
    agent_id = None
    # A subagent can share its parent's sessionId; agentId must namespace its archive.
    with _source(Path(path)) as source:
        for raw in source:
            try:
                event = json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                continue
            if not isinstance(event, dict):
                continue
            identity = event.get("sessionId")
            if isinstance(identity, str):
                if native_session is not None and native_session != identity:
                    raise ValueError("Claude conflicting_session_identity")
                native_session = identity
            if event.get("type") in ("user", "assistant") and event.get("agentId"):
                candidate = event["agentId"]
                if not isinstance(candidate, str) or (
                    agent_id is not None and agent_id != candidate
                ):
                    raise ValueError("Claude conflicting_agent_identity")
                agent_id = candidate
    if not native_session:
        raise ValueError("Claude missing_session_identity")
    session = native_session + ":agent:" + agent_id if agent_id else native_session
    digest = hashlib.sha256()
    with (
        _source(Path(path)) as source,
        tempfile.TemporaryDirectory(prefix="promptkeep-import-") as temp,
    ):
        stage = storage._ImportStage(str(Path(temp) / "stage.db"), settings)
        try:
            parser = _ClaudeParser(stage, strict, session, native_session, agent_id)
            for line, raw in enumerate(source, 1):
                digest.update(raw)
                parser.counts["records"] += 1
                try:
                    event = json.loads(raw)
                except (ValueError, UnicodeDecodeError):
                    parser.diagnose("invalid_json", line, True)
                    continue
                try:
                    parser.parse(event, line)
                except (TypeError, AttributeError, KeyError):
                    parser.diagnose("invalid_record_shape", line, True)
            summary = parser.summary()
            outcome = storage.persist_import(
                stage,
                db_path=destination,
                session_id=session,
                trace_key=parser.trace,
                digest=digest.hexdigest(),
                revision=_REVISION,
                summary=summary,
                source="claude-code",
            )
            values = {
                k: summary[k]
                for k in ImportReport.__dataclass_fields__
                if k not in ("outcome", "diagnostics")
            }
            return ImportReport(**values, outcome=outcome, diagnostics=tuple(parser.diagnostics))
        finally:
            stage.close()

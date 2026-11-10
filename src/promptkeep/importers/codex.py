"""Native Codex rollout interpretation, independent of persistence and other agents."""

from __future__ import annotations

import hashlib
import json
import re

from .base import AgentTrace, ImportReport, _counts, _key, _timestamp, _visible
from .engine import import_trace as _import_trace

_REVISION = 3


class CodexTrace(AgentTrace):
    """Consume Codex rollout envelopes in their original source order."""

    source = "codex"
    provider = "openai"
    revision = _REVISION

    @classmethod
    def detect(cls, record):
        """Identify native rollout metadata and event envelopes."""
        return (
            isinstance(record, dict)
            and record.get("type") == "session_meta"
            and isinstance(record.get("payload"), dict)
        )

    def consume(self, event, line):
        """Public ordered record consumption contract."""
        return self.parse(event, line)

    def parse(self, event: dict, line: int) -> None:
        """Dispatch known native rollout families, ignoring unrelated records."""
        if not isinstance(event, dict) or not isinstance(event.get("payload"), dict):
            self.diagnose("invalid_record", line, True)
            return
        family, payload = event.get("type"), event["payload"]
        timestamp = _timestamp(event.get("timestamp"))
        if timestamp:
            self.first = min(self.first or timestamp, timestamp)
            self.last = max(self.last or timestamp, timestamp)
        if family == "session_meta":
            session = payload.get("id") or payload.get("session_id")
            if not isinstance(session, str) or not session.strip():
                raise ValueError(f"Codex invalid_session_identity at line {line}")
            if self.session and self.session != session:
                raise ValueError(f"Codex conflicting_session_identity at line {line}")
            self.session = session
            self.trace = _key(session, "agent", "session", self.source)
            observed = _timestamp(payload.get("timestamp")) or timestamp
            if observed is None:
                raise ValueError(f"Codex missing_session_timestamp at line {line}")
            self.span("session", "agent", self.source, observed, None)
            return
        if self.session is None:
            raise ValueError(f"Codex missing_session_metadata at line {line}")
        if family == "turn_context":
            self.turn(payload.get("turn_id"), timestamp)
            self.task_pending = True
            if payload.get("model"):
                self.models[self.current] = payload["model"]
                self.effective_model = payload["model"]
        elif family == "response_item":
            self.item(payload, timestamp, line)
        elif family == "token_usage_record":
            self.usage(payload, timestamp, line)
        elif family == "event_msg":
            kind = payload.get("type")
            if kind in ("task_started", "turn_started"):
                if not payload.get("turn_id") and not self.task_pending:
                    self.current = None
                self.turn(
                    payload.get("turn_id"), _timestamp(payload.get("started_at")) or timestamp
                )
                self.task_pending = True
            elif kind in ("task_complete", "turn_complete", "turn_aborted"):
                key = self.turn(payload.get("turn_id"), timestamp)
                if key:
                    values = {}
                    if type(payload.get("duration_ms")) in (int, float):
                        values["duration_ms"] = payload["duration_ms"]
                    if isinstance(payload.get("last_agent_message"), str):
                        values["output_json"] = payload["last_agent_message"]
                    self.finish(
                        key,
                        _timestamp(payload.get("completed_at")) or timestamp,
                        "cancelled" if kind == "turn_aborted" else "ok",
                        line,
                        _timestamp(payload.get("started_at")),
                        **values,
                    )
                self.task_pending = False
            elif kind == "item_completed":
                if isinstance(payload.get("item"), dict):
                    self.turn(payload.get("turn_id"), timestamp)
                    self.item(
                        payload["item"],
                        timestamp,
                        line,
                        _timestamp(payload.get("started_at_ms"), unit="milliseconds"),
                        _timestamp(payload.get("completed_at_ms"), unit="milliseconds"),
                    )
                else:
                    self.diagnose("invalid_completed_item", line, True)
            elif kind == "token_count":
                self.native_usage(payload, timestamp, line)
            elif kind == "user_message":
                # response_item messages are authoritative; legacy notifications are mirrors.
                self.counts["ignored"] += 1
            else:
                self.counts["ignored"] += 1
        else:
            self.counts["ignored"] += 1

    def item(
        self,
        item: dict,
        timestamp: str | None,
        line: int,
        start: str | None = None,
        end: str | None = None,
    ) -> None:
        """Correlate messages and tool operations from old and modern records."""
        response = item.get("response_id")
        if response:
            self.observe_response(response, timestamp, item.get("model"))
        elif item.get("type") in ("function_call", "custom_tool_call", "toolCall"):
            self.diagnose("ambiguous_tool_model_association", line)
        kind = item.get("type")
        if kind in ("message", "userMessage", "agentMessage", "UserMessage", "AgentMessage"):
            role = item.get("role") or (
                "user" if kind in ("userMessage", "UserMessage") else "assistant"
            )
            if role not in ("user", "assistant"):
                self.counts["ignored"] += 1
                return
            text = _visible(item.get("content"))
            if role == "user" and not self.task_pending:
                # A new user message begins a legacy turn unless an explicit turn is already active.
                if self.current is None or self.current.startswith("legacy:"):
                    self.current = None
            key = self.turn(None, start or timestamp)
            if key and text:
                field = "input_json" if role == "user" else "output_json"
                self.stage.put("spans", key, {field: text})
                if role == "assistant" and response:
                    self.stage.append_text(
                        _key(self.session, "llm", response, self.source),
                        item.get("id") or f"line:{line}",
                        text,
                    )
            self.task_pending = True
            return
        if kind in ("CommandExecution", "McpToolCall"):
            call_id = item.get("call_id") or item.get("id")
            if call_id not in self.call_turns:
                self.counts["ignored"] += 1
                self.diagnose("unlinked_completed_tool", line)
                return
            item = {
                "type": "function_call_output",
                "call_id": call_id,
                "output": item.get("aggregated_output", item.get("result")),
                "exit_code": item.get("exit_code"),
                "status": item.get("status"),
            }
            kind = item["type"]
        call_kinds = ("function_call", "custom_tool_call", "toolCall")
        output_kinds = ("function_call_output", "custom_tool_call_output", "toolCallOutput")
        if kind not in call_kinds + output_kinds:
            self.counts["ignored"] += 1
            return
        call_id = item.get("call_id") or item.get("id")
        if not isinstance(call_id, str) or not call_id:
            call_id = f"line:{line}"
            self.diagnose("fallback_operation_identity", line)
        ambient_turn = self.current
        parent = self.turn(self.call_turns.get(call_id), start or timestamp)
        self.call_turns.setdefault(call_id, self.current)
        if ambient_turn is not None:
            self.current = ambient_turn
        key = _key(self.session, "tool", call_id, self.source)
        existing = self.stage.get("spans", key)
        if kind in call_kinds:
            name = item.get("name") or f"{self.source}.tool"
            value = item.get("arguments", item.get("input"))
            definition = hashlib.sha256(
                json.dumps([name, value], sort_keys=True).encode()
            ).hexdigest()
            if existing and existing.get("_definition") not in (None, definition):
                raise ValueError(f"Codex conflicting_operation_identity at line {line}")
            key = self.span(
                call_id,
                "tool",
                name,
                start or timestamp,
                parent,
                input_json=value,
                _definition=definition,
            )
            if key:
                patch = {"name": name}
                if response:
                    patch["parent_span_key"] = _key(self.session, "llm", response, self.source)
                if existing and timestamp and timestamp < existing["started_at"]:
                    patch["started_at"] = start or timestamp
                self.stage.put("spans", key, patch)
                if existing and existing.get("ended_at"):
                    self.finish(
                        key, existing["ended_at"], existing["status"], line, patch.get("started_at")
                    )
        elif existing is None:
            self.diagnose("unmatched_tool_output", line)
            key = self.span(call_id, "tool", f"{self.source}.tool", timestamp, parent)
        if key is None:
            self.diagnose("missing_operation_timestamp", line, True)
            return
        if kind in output_kinds or end:
            output = item.get("output")
            failed = item.get("is_error") is True or item.get("status") in ("failed", "error")
            code = item.get("exit_code")
            if isinstance(output, dict):
                failed = failed or output.get("is_error") is True
                code = output.get("exit_code", code)
                metadata = output.get("metadata")
                if isinstance(metadata, dict):
                    code = metadata.get("exit_code", code)
            elif isinstance(output, str):
                # Older rollouts encode the structured terminal result as a JSON string.
                try:
                    envelope = json.loads(output)
                except ValueError:
                    envelope = None
                if isinstance(envelope, dict):
                    failed = failed or envelope.get("is_error") is True
                    code = envelope.get("exit_code", code)
                    metadata = envelope.get("metadata")
                    if isinstance(metadata, dict):
                        code = metadata.get("exit_code", code)
                # Recognize the native terminal-tool envelope, not arbitrary prose.
                match = re.search(
                    r"(?:^|\n)(?:Process exited with code |Exit code: )(-?\d+)(?:\n|$)", output
                )
                if match:
                    code = int(match[1])
            failed = failed or (type(code) is int and code != 0)
            status = "error" if failed else "ok"
            if existing is None and kind in output_kinds:
                status = "incomplete"
            self.finish(key, end or timestamp, status, line, start, output_json=output)

    def usage(self, payload: dict, timestamp: str | None, line: int) -> None:
        """Prefer explicit response usage; keep cumulative snapshots separate."""
        response = payload.get("response_id")
        usage = _counts(payload.get("usage"))
        if all(k in usage for k in ("input_tokens", "output_tokens", "total_tokens")):
            if usage["input_tokens"] + usage["output_tokens"] != usage["total_tokens"]:
                self.diagnose("inconsistent_usage_total", line)
        if response and usage:
            key = f"response:{response}"
            previous = self.stage.get("usage", key)
            if previous and previous["usage"] != usage:
                raise ValueError(f"Codex conflicting_usage_identity at line {line}")
            self.stage.put(
                "usage",
                key,
                dict(
                    scope="response",
                    response_id=response,
                    turn_id=payload.get("turn_id") or self.current,
                    model=payload.get("model") or self.effective_model,
                    observed_at=timestamp,
                    usage=usage,
                ),
            )
        for scope in ("turn", "thread"):
            value = _counts(payload.get(f"{scope}_token_usage"))
            if value:
                identity = payload.get("turn_id") if scope == "turn" else "session"
                self.stage.put(
                    "usage",
                    f"{scope}:{identity}",
                    dict(
                        scope=scope,
                        turn_id=payload.get("turn_id") if scope == "turn" else None,
                        observed_at=timestamp,
                        usage=value,
                    ),
                )

    def observe_response(self, response, timestamp, model=None):
        """An explicit provider response identity proves a call, even without usage."""
        span = _key(self.session, "llm", response, self.source)
        if self.stage.get("spans", span) is None:
            self.stage.put(
                "spans",
                span,
                dict(
                    span_key=span,
                    trace_key=self.trace,
                    parent_span_key=_key(self.session, "agent", self.current, self.source)
                    if self.current
                    else self.trace,
                    kind="llm",
                    name=model or self.effective_model or "model",
                    started_at=None,
                    observed_at=timestamp,
                    status="ok",
                    sampled=True,
                    metadata_json=json.dumps({"source": self.source, "timing_quality": "unknown"}),
                ),
            )

    def native_usage(self, payload, timestamp, line):
        """Separate last-call observations, repeated snapshots and accounting resets."""
        info = payload.get("info")
        if not isinstance(info, dict):
            return
        total = _counts(info.get("total_token_usage"))
        last = _counts(info.get("last_token_usage"))
        previous = self.usage_snapshot
        reset = bool(
            total and previous and any(total[k] < v for k, v in previous.items() if k in total)
        )
        if reset:
            self.stage.put(
                "usage",
                f"epoch:{self.usage_epoch}",
                dict(scope="accounting_epoch", observed_at=timestamp, usage=previous),
            )
            self.usage_epoch += 1
            self.diagnose("usage_accounting_reset", line)
        fingerprint = json.dumps([self.current, self.effective_model, total, last], sort_keys=True)
        repeated = fingerprint == self.usage_notification or (bool(total) and total == previous)
        if total:
            self.stage.put(
                "usage", "legacy:thread", dict(scope="thread", observed_at=timestamp, usage=total)
            )
            self.usage_snapshot = total
        if repeated:
            return
        self.usage_notification = fingerprint
        if last:
            operation = f"usage:{self.usage_epoch}:{line}"
            self.stage.put(
                "usage",
                f"observation:{operation}",
                dict(
                    scope="response",
                    operation_id=operation,
                    response_id=None,
                    turn_id=self.current,
                    model=self.effective_model,
                    observed_at=timestamp,
                    usage=last,
                ),
            )
            self.diagnose("usage_observation_without_response_id", line)
        elif total:
            self.diagnose("unattributed_usage_snapshot", line)


_Parser = CodexTrace


def import_codex(path, *, db_path=None, strict=False) -> ImportReport:
    """Compatibility entry point for explicit Codex archives."""
    return _import_trace(path, source="codex", db_path=db_path, strict=strict)

"""Native Codex rollout JSONL import, preserving gaps rather than inventing calls."""

from __future__ import annotations

import hashlib
import importlib
import json
import re
import tempfile
import uuid
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .. import storage
from ..config import get_settings

_REVISION = 2
_NAMESPACE = uuid.UUID("e75b9474-b29a-4e2d-b39b-185cf8c9b424")


@dataclass(frozen=True)
class ImportDiagnostic:
    """A payload-free parse diagnostic, safe to show in CLI output."""

    severity: str
    code: str
    line: int


@dataclass(frozen=True)
class ImportReport:
    """The committed import's identity, counts and source-quality diagnostics."""

    session_id: str
    trace_key: str
    outcome: str
    input_records: int
    agent_spans: int
    tool_spans: int
    model_spans: int
    usage_records: int
    ignored_records: int
    malformed_records: int
    diagnostics: tuple[ImportDiagnostic, ...]


@contextmanager
def _source(path: Path):
    """Open decompressed bytes; missing compression support has no DB effects."""
    with path.open("rb") as raw:
        if path.name.endswith(".jsonl.zst"):
            try:
                zstd = importlib.import_module("zstandard")
            except ImportError:
                raise ImportError(
                    'Compressed Codex imports require pip install "promptkeep[codex]"'
                ) from None
            with zstd.ZstdDecompressor().stream_reader(raw) as reader:
                # BufferedReader supports bounded line-by-line iteration.
                import io

                with io.BufferedReader(reader) as buffered:
                    yield buffered
        elif path.suffix == ".jsonl":
            yield raw
        else:
            raise ValueError("Codex imports require .jsonl or .jsonl.zst")


def _timestamp(value: Any) -> str | None:
    """Accept aware ISO timestamps or documented epoch-millisecond values."""
    try:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            parsed = datetime.fromtimestamp(value / 1000, UTC)
        elif isinstance(value, str):
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                return None
        else:
            return None
        return parsed.astimezone(UTC).isoformat()
    except (ValueError, OverflowError, OSError):
        return None


def _key(session: str, category: str, identity: str, source: str = "codex") -> str:
    """Deterministic IDs survive source-file renames and compression changes."""
    return str(uuid.uuid5(_NAMESPACE, json.dumps([source, session, category, identity])))


def _visible(content: Any) -> str | None:
    """Only visible message text; never encrypted reasoning or binary content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            b["text"]
            for b in content
            if isinstance(b, dict)
            and b.get("type") in ("text", "input_text", "output_text")
            and isinstance(b.get("text"), str)
        ]
        return "\n".join(parts) or None
    return None


def _counts(value: Any) -> dict[str, int]:
    """Accept only nonnegative numeric usage counters, never rate limits."""
    names = {
        "input_tokens",
        "cached_input_tokens",
        "cache_write_input_tokens",
        "output_tokens",
        "reasoning_output_tokens",
        "total_tokens",
    }
    return (
        {k: v for k, v in value.items() if k in names and type(v) is int and v >= 0}
        if isinstance(value, dict)
        else {}
    )


class _Parser:
    """One session's small correlation state; payloads live in redacted staging."""

    source = "codex"

    def __init__(self, stage: storage._ImportStage, strict: bool) -> None:
        """Track source identities without retaining full session payloads."""
        self.stage, self.strict = stage, strict
        self.session: str | None = None
        self.trace: str | None = None
        self.current: str | None = None
        self.first: str | None = None
        self.last: str | None = None
        self.diagnostics: list[ImportDiagnostic] = []
        self.counts = Counter()
        self.turns: set[str] = set()
        self.call_turns: dict[str, str] = {}
        self.legacy_turn = 0
        self.task_pending = False
        self.models = {}

    def diagnose(self, code: str, line: int, malformed: bool = False) -> None:
        """Never include user content in errors or stored diagnostics."""
        if malformed:
            self.counts["malformed"] += 1
            if self.strict:
                raise ValueError(f"Codex {code} at line {line}")
        self.diagnostics.append(ImportDiagnostic("warning", code, line))

    def span(
        self,
        identity: str,
        kind: str,
        name: str,
        timestamp: str | None,
        parent: str | None,
        **values: Any,
    ) -> str | None:
        """Create a span only when a historical observation exists."""
        key = _key(self.session, kind, identity, self.source)
        existing = self.stage.get("spans", key)
        if existing is None:
            if timestamp is None:
                self.diagnose("missing_span_timestamp", 0)
                return None
            self.stage.put(
                "spans",
                key,
                dict(
                    span_key=key,
                    trace_key=self.trace,
                    parent_span_key=parent,
                    kind=kind,
                    name=name,
                    started_at=timestamp,
                    status="incomplete",
                    sampled=True,
                    metadata_json=json.dumps({"source": self.source, "timing_quality": "observed"}),
                ),
            )
        self.stage.put("spans", key, values)
        return key

    def turn(self, turn_id: str | None, timestamp: str | None) -> str | None:
        """Ensure a turn container without duplicating mirrored user events."""
        if turn_id:
            self.current = str(turn_id)
        if self.current is None:
            self.legacy_turn += 1
            self.current = f"legacy:{self.legacy_turn}"
            self.diagnose("fallback_turn_identity", 0)
        self.turns.add(self.current)
        return self.span(self.current, "agent", f"{self.source}.turn", timestamp, self.trace)

    def finish(
        self,
        key: str,
        end: str | None,
        status: str,
        line: int,
        start: str | None = None,
        **values: Any,
    ) -> None:
        """Prefer explicit intervals; absence or invalid order stays visible."""
        existing = self.stage.get("spans", key)
        if existing is None:
            return
        if start:
            values["started_at"] = start
        begin = start or existing["started_at"]
        if end:
            duration = (
                round(
                    (datetime.fromisoformat(end) - datetime.fromisoformat(begin)).total_seconds()
                    * 1000
                )
                if begin
                else -1
            )
            values["ended_at"] = end
            if duration >= 0:
                values["duration_ms"] = duration
            else:
                values["duration_ms"] = None
                if begin:
                    self.diagnose("negative_interval", line)
        values["status"] = status
        if start and end:
            values["metadata_json"] = json.dumps(
                {"source": self.source, "timing_quality": "explicit"}
            )
        self.stage.put("spans", key, values)

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
                        _timestamp(payload.get("started_at_ms")),
                        _timestamp(payload.get("completed_at_ms")),
                    )
                else:
                    self.diagnose("invalid_completed_item", line, True)
            elif kind == "token_count":
                info = payload.get("info") or {}
                usage = _counts(info.get("total_token_usage")) if isinstance(info, dict) else {}
                if usage:
                    self.stage.put(
                        "usage",
                        "legacy:thread",
                        dict(scope="thread", observed_at=timestamp, usage=usage),
                    )
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
            name = item.get("name") or "codex.tool"
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
                if existing and timestamp and timestamp < existing["started_at"]:
                    patch["started_at"] = start or timestamp
                self.stage.put("spans", key, patch)
                if existing and existing.get("ended_at"):
                    self.finish(
                        key, existing["ended_at"], existing["status"], line, patch.get("started_at")
                    )
        elif existing is None:
            self.diagnose("unmatched_tool_output", line)
            key = self.span(call_id, "tool", "codex.tool", timestamp, parent)
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
                    turn_id=payload.get("turn_id"),
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

    def model_records(self) -> None:
        """A response ID proves a call occurred, but does not invent its timing."""
        for _, usage in self.stage.rows("usage"):
            if usage["scope"] != "response" or not usage.get("response_id"):
                continue
            response = usage["response_id"]
            span_key = _key(self.session, "llm", response, self.source)
            observed = usage.get("observed_at")
            values = usage["usage"]
            model = usage.get("model") or self.models.get(usage.get("turn_id"))
            parent = (
                _key(self.session, "agent", usage["turn_id"], self.source)
                if usage.get("turn_id")
                else self.trace
            )
            if self.stage.get("spans", span_key) is None:
                self.stage.put(
                    "spans",
                    span_key,
                    dict(
                        span_key=span_key,
                        trace_key=self.trace,
                        parent_span_key=parent,
                        kind="llm",
                        name=model or "model",
                        started_at=None,
                        observed_at=observed,
                        status="ok",
                        sampled=True,
                        metadata_json=json.dumps(
                            {"source": self.source, "timing_quality": "unknown"}
                        ),
                    ),
                )
            self.stage.put("usage", "response:" + response, {"span_key": span_key})
            span = self.stage.get("spans", span_key)
            output = span.get("output_json")
            if output:
                try:
                    decoded = json.loads(output)
                    output = decoded if isinstance(decoded, str) else output
                except ValueError:
                    pass
            incoming, outgoing = values.get("input_tokens"), values.get("output_tokens")
            self.stage.put(
                "runs",
                span_key,
                dict(
                    run_key=span_key,
                    span_key=span_key,
                    provider="anthropic" if self.source == "claude-code" else "openai",
                    model=model,
                    response_id=response,
                    created_at=observed or span.get("observed_at") or self.first,
                    status=span["status"],
                    output_text=output,
                    prompt_tokens=incoming,
                    completion_tokens=outgoing,
                    total_tokens=values.get("total_tokens"),
                    usage_details=json.dumps(values),
                ),
            )

    def summary(self) -> dict:
        """Compute counts and usage without mixing deltas with snapshots."""
        if self.session is None:
            raise ValueError("Codex missing_session_metadata")
        self.finish(self.trace, self.last, "incomplete", 0)
        self.model_records()
        statuses = []
        counts = Counter()
        for key, row in self.stage.rows("spans"):
            counts[row["kind"]] += 1
            if row["kind"] == "agent" and key != self.trace:
                statuses.append(row["status"])
        if statuses and all(s != "incomplete" for s in statuses):
            self.stage.put(
                "spans", self.trace, {"status": "cancelled" if "cancelled" in statuses else "ok"}
            )
        usage_rows = list(self.stage.rows("usage"))
        modern = [(k, v) for k, v in usage_rows if not k.startswith("legacy:")]
        if modern:
            self.stage.db.execute_sql("DELETE FROM usage WHERE key LIKE 'legacy:%'")
            usage_rows = modern
        total = Counter()
        for _, row in usage_rows:
            if row["scope"] == "response":
                total.update(row["usage"])
        response_total = dict(total)
        snapshot = next((v["usage"] for _, v in usage_rows if v["scope"] == "thread"), None)
        if snapshot:
            total = Counter(snapshot)
        required = ("input_tokens", "output_tokens", "total_tokens")
        complete = bool(snapshot) and all(
            k in snapshot and k in response_total and snapshot[k] == response_total[k]
            for k in required
        )
        if self.source == "claude-code":
            responses = [v for _, v in usage_rows if v["scope"] == "response"]
            complete = (
                bool(responses)
                and len(responses) == counts["llm"]
                and all(all(k in v["usage"] for k in required) for v in responses)
            )
        completeness = (
            "complete"
            if complete and not self.counts["malformed"]
            else "partial"
            if any(v["usage"] for _, v in usage_rows)
            else "unknown"
        )
        unattributed = {
            k: v - response_total.get(k, 0)
            for k, v in dict(total).items()
            if v > response_total.get(k, 0)
        }
        return dict(
            session_id=self.session,
            trace_key=self.trace,
            input_records=self.counts["records"],
            agent_spans=counts["agent"],
            tool_spans=counts["tool"],
            model_spans=counts["llm"],
            usage_records=len(usage_rows),
            ignored_records=self.counts["ignored"],
            malformed_records=self.counts["malformed"],
            diagnostics=[asdict(d) for d in self.diagnostics],
            total_usage=dict(total),
            usage_completeness=completeness,
            response_usage=response_total,
            unattributed_usage=unattributed,
            timing_quality="observed",
            parser_revision=_REVISION,
        )


def import_codex(
    path: str | Path, *, db_path: str | Path | None = None, strict: bool = False
) -> ImportReport:
    """Import one saved session atomically; errors preserve the previous snapshot.

    Imports bypass live sampling and the telemetry queue. Redaction still applies,
    and original activity timestamps determine eligibility for normal retention.
    """
    settings = get_settings()
    if not settings.enabled or settings.write_mode == "off":
        raise ValueError("Codex import requires enabled tracking and write_mode other than off")
    destination = str(Path(db_path or settings.db_path).absolute())
    digest = hashlib.sha256()
    with (
        _source(Path(path)) as source,
        tempfile.TemporaryDirectory(prefix="promptkeep-import-") as temp,
    ):
        stage = storage._ImportStage(str(Path(temp) / "stage.db"), settings)
        try:
            parser = _Parser(stage, strict)
            for number, raw in enumerate(source, 1):
                digest.update(raw)
                parser.counts["records"] += 1
                try:
                    event = json.loads(raw)
                except (ValueError, UnicodeDecodeError):
                    parser.diagnose("invalid_json", number, True)
                    continue
                try:
                    parser.parse(event, number)
                except (TypeError, AttributeError, KeyError):
                    parser.diagnose("invalid_record_shape", number, True)
            summary = parser.summary()
            outcome = storage.persist_import(
                stage,
                db_path=destination,
                session_id=parser.session,
                trace_key=parser.trace,
                digest=digest.hexdigest(),
                revision=_REVISION,
                summary=summary,
            )
            report = {
                k: summary[k]
                for k in ImportReport.__dataclass_fields__
                if k not in ("outcome", "diagnostics")
            }
            return ImportReport(**report, outcome=outcome, diagnostics=tuple(parser.diagnostics))
        finally:
            stage.close()

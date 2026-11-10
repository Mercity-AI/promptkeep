"""Public source adapters and shared source-neutral trace lifecycle."""

from __future__ import annotations

import json
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from .. import storage
from ..events import TraceEvent
from ..normalization import prepare

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


def _timestamp(value: Any, *, unit: str = "seconds") -> str | None:
    """Parse aware ISO values or numeric epochs with an explicitly selected unit."""
    try:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            parsed = datetime.fromtimestamp(value / (1000 if unit == "milliseconds" else 1), UTC)
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


class AgentTrace:
    """Public adapter lifecycle; subclasses interpret ordered source records.

    Override detect(), discover(), and consume(). Use span(), finish(), and
    stage_event() to submit observations without SQL or another agent parser.
    Checkpoints contain identifiers and counters, never source payloads.
    """

    source = "unknown"
    provider = "unknown"
    revision = 3
    per_call_complete = False
    snapshot = False

    def __init__(self, stage: storage._ImportStage, strict: bool = False) -> None:
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
        self.effective_model = None
        self.usage_snapshot = {}
        self.usage_epoch = 0
        self.usage_notification = None

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
        metadata = json.loads(existing.get("metadata_json") or "{}")
        source_duration = values.pop("duration_ms", metadata.get("source_duration_ms"))
        begin = start or existing.get("started_at")
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
            if source_duration is not None:
                values["duration_ms"] = source_duration
                if duration >= 0 and abs(duration - source_duration) > 1:
                    self.diagnose("duration_timestamp_disagreement", line)
            elif duration >= 0:
                values["duration_ms"] = duration
            else:
                values["duration_ms"] = source_duration
            if duration < 0 and begin:
                self.diagnose("negative_interval", line)
        if source_duration is not None:
            values["duration_ms"] = source_duration
            metadata["source_duration_ms"] = source_duration
            if source_duration < 0:
                self.diagnose("negative_source_duration", line)
        values["status"] = status
        if start and end:
            metadata.update(source=self.source, timing_quality="explicit")
        values["metadata_json"] = json.dumps(metadata)
        self.stage.put("spans", key, values)

    def model_records(self) -> None:
        """A response ID proves a call occurred, but does not invent its timing."""
        for usage_key, usage in self.stage.rows("usage"):
            if usage["scope"] != "response":
                continue
            response = usage.get("response_id")
            operation = response or usage.get("operation_id")
            if not operation:
                continue
            span_key = _key(self.session, "llm", operation, self.source)
            observed = usage.get("observed_at")
            values = usage["usage"]
            model = usage.get("model")
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
            self.stage.put("usage", usage_key, {"span_key": span_key})
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
                    provider=self.provider,
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
        root = self.stage.get("spans", self.trace) or {}
        status = (
            root.get("status")
            if root.get("status") in ("ok", "error", "cancelled")
            else "incomplete"
        )
        self.finish(self.trace, self.last, status, 0)
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
        if any(v["scope"] == "thread" for _, v in modern):
            self.stage.delete_prefix("usage", "legacy:")
            usage_rows = [(k, v) for k, v in usage_rows if not k.startswith("legacy:")]
        total = Counter()
        for _, row in usage_rows:
            if row["scope"] == "response":
                total.update(row["usage"])
        response_total = dict(total)
        snapshot = next((v["usage"] for _, v in usage_rows if v["scope"] == "thread"), None)
        if snapshot:
            total = Counter(snapshot)
            for _, row in usage_rows:
                if row["scope"] == "accounting_epoch":
                    total.update(row["usage"])
        required = ("input_tokens", "output_tokens", "total_tokens")
        complete = bool(snapshot) and all(
            k in snapshot and k in response_total and snapshot[k] == response_total[k]
            for k in required
        )
        if self.per_call_complete:
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
            parser_revision=self.revision,
        )

    @classmethod
    def detect(cls, record: dict) -> bool:
        """Return true only for a recognized native record envelope."""
        return False

    def discover(self, records) -> None:
        """Inspect identity metadata before ordered consumption, when necessary."""

    def consume(self, record: dict, line: int) -> None:
        """Interpret one source record; third parties implement this method."""
        raise NotImplementedError

    def finalize(self) -> dict:
        """Produce the final normalized snapshot and payload-free summary."""
        return self.summary()

    def checkpoint_state(self) -> dict:
        """Serialize correlation state; payloads remain in redacted staging."""
        return {
            name: value
            for name, value in self.__dict__.items()
            if name not in ("stage", "strict", "diagnostics", "turns", "counts")
        } | {
            "counts": dict(self.counts),
            "turns": sorted(self.turns),
            "diagnostics": [asdict(d) for d in self.diagnostics],
        }

    def restore_state(self, state: dict) -> None:
        """Restore a checkpoint made by this adapter revision."""
        for name, value in state.items():
            if name not in ("stage", "strict", "counts", "turns", "diagnostics"):
                setattr(self, name, value)
        self.counts = Counter(state.get("counts", {}))
        self.turns = set(state.get("turns", []))
        self.diagnostics = [ImportDiagnostic(**d) for d in state.get("diagnostics", [])]

    def stage_event(self, event: TraceEvent) -> None:
        """Submit an agent-neutral observation through common preparation/reduction."""
        item = prepare(event, settings=self.stage.settings, archive=True)
        if item is not None:
            self.stage.event(item)

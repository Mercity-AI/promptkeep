"""Optional OTel SDK ingestion and explicitly configured outbound OTLP export."""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import queue
import threading
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import Link, SpanContext, SpanKind, Status, StatusCode, TraceFlags

from .. import ingestion, storage
from ..config import get_settings
from ..events import Event, key
from ..tracking import flush

logger = logging.getLogger("promptkeep")


def _time(nanoseconds):
    """Convert SDK epoch nanoseconds without treating absent clocks as zero."""
    return (
        datetime.fromtimestamp(nanoseconds / 1e9, UTC).isoformat()
        if nanoseconds is not None
        else None
    )


def _integer(key, bits):
    """Preserve valid hexadecimal IDs and deterministically map other identities."""
    try:
        value = int(key.replace("-", ""), 16)
    except (ValueError, AttributeError):
        value = int.from_bytes(hashlib.sha256(str(key).encode()).digest(), "big")
    return (value & ((1 << bits) - 1)) or 1


class PromptKeepSpanExporter(SpanExporter):
    """Attach to an existing Python TracerProvider; never replace its processors."""

    def __init__(self):
        """Initialize an explicitly owned exporter without global registration."""
        self.closed = False
        self.destination = str(get_settings().db_path)

    def export(self, spans):
        """Map known GenAI attributes and ignore PromptKeep-origin export loops."""
        if self.closed:
            return SpanExportResult.FAILURE
        try:
            failed = False
            for span in spans:
                attributes = span.attributes or {}
                if attributes.get("promptkeep.origin") == "promptkeep":
                    continue
                trace_id = f"{span.context.trace_id:032x}"
                span_id = f"{span.context.span_id:016x}"
                operation = attributes.get("gen_ai.operation.name")
                kind = (
                    "llm"
                    if operation in ("chat", "generate_content", "text_completion")
                    else "tool"
                    if operation == "execute_tool"
                    else "operation"
                )
                status = "error" if span.status.status_code == StatusCode.ERROR else "ok"
                start, end = _time(span.start_time), _time(span.end_time)
                if span.parent is None:
                    failed = (
                        not ingestion.record(
                            Event(
                                "opentelemetry",
                                trace_id,
                                trace_id,
                                span.name,
                                kind="agent",
                                started_at=start,
                                ended_at=end,
                                status=status,
                                destination=self.destination,
                            )
                        )
                        or failed
                    )
                parent = f"{span.parent.span_id:016x}" if span.parent else trace_id
                usage = {
                    normalized: attributes[key]
                    for key, normalized in (
                        ("gen_ai.usage.input_tokens", "input_tokens"),
                        ("gen_ai.usage.output_tokens", "output_tokens"),
                    )
                    if key in attributes
                }
                failed = (
                    not ingestion.record(
                        Event(
                            "opentelemetry",
                            trace_id,
                            span_id,
                            span.name,
                            kind=kind,
                            parent_id=parent,
                            started_at=start,
                            ended_at=end,
                            status=status,
                            error=span.status.description if status == "error" else None,
                            model=attributes.get("gen_ai.response.model")
                            or attributes.get("gen_ai.request.model"),
                            response_id=attributes.get("gen_ai.response.id"),
                            usage=usage,
                            link_keys=tuple(
                                key(
                                    "opentelemetry",
                                    f"{link.context.trace_id:032x}",
                                    f"{link.context.span_id:016x}",
                                )
                                for link in span.links
                            ),
                            destination=self.destination,
                            provider=attributes.get("gen_ai.provider.name")
                            or attributes.get("gen_ai.system"),
                        )
                    )
                    or failed
                )
            return SpanExportResult.FAILURE if failed else SpanExportResult.SUCCESS
        except Exception:
            return SpanExportResult.FAILURE

    def force_flush(self, timeout_millis=30000):
        """Respect the SDK timeout without touching the application's provider."""
        return flush(timeout_millis / 1000)

    def shutdown(self):
        """Disable only this exporter."""
        self.closed = True


class OTLPExport:
    """Bounded redacted-span queue; official SDK exporters own transport and retries."""

    def __init__(self, endpoint, *, protocol="http/protobuf", exporter=None, include_content=False):
        """Network export is opt-in and starts only with an explicit endpoint."""
        if not endpoint:
            raise ValueError("an explicit OTLP endpoint is required")
        if protocol not in ("http/protobuf", "grpc"):
            raise ValueError("OTLP protocol must be http/protobuf or grpc")
        if exporter is None:
            module = importlib.import_module(
                "opentelemetry.exporter.otlp.proto."
                + ("http" if protocol == "http/protobuf" else "grpc")
                + ".trace_exporter"
            )
            exporter = module.OTLPSpanExporter(endpoint=endpoint, timeout=5)
        self.destination = str(get_settings().db_path)
        self.exporter = exporter
        self.include_content = include_content
        self.queue = queue.Queue(maxsize=2048)
        self.dropped_count = 0
        self.failed_count = 0
        self.skipped_unknown_timing = 0
        self._stop = threading.Event()
        self._unsubscribe = storage.observe_spans(self._accept)
        self._thread = threading.Thread(target=self._work, daemon=True, name="promptkeep-otlp")
        self._thread.start()

    def _readable_span(self, row, database):
        """Convert one already stored span and its primary usage to an OTel value."""
        attributes = {
            "promptkeep.origin": "promptkeep",
            "promptkeep.kind": row["kind"],
            "promptkeep.status": row["status"],
        }
        # Numeric model metrics come from the primary run only.
        run = database.execute_sql(
            "SELECT provider,model,prompt_tokens,completion_tokens,total_tokens,cost_usd FROM runs "
            "WHERE span_key=? ORDER BY id LIMIT 1",
            (row["span_key"],),
        ).fetchone()
        if run:
            for name, value in zip(
                (
                    "gen_ai.provider.name",
                    "gen_ai.response.model",
                    "gen_ai.usage.input_tokens",
                    "gen_ai.usage.output_tokens",
                    "promptkeep.total_tokens",
                    "promptkeep.cost_usd",
                ),
                run,
            ):
                if value is not None:
                    attributes[name] = value
        if self.include_content:
            for field in ("input_json", "output_json", "error"):
                if row.get(field) is not None:
                    attributes["promptkeep." + field] = row[field]

        trace_id = _integer(row["trace_key"], 128)
        context = SpanContext(trace_id, _integer(row["span_key"], 64), False, TraceFlags(1))
        parent = (
            SpanContext(trace_id, _integer(row["parent_span_key"], 64), False, TraceFlags(1))
            if row.get("parent_span_key")
            else None
        )
        links = []
        for (linked_key,) in database.execute_sql(
            "SELECT linked_span_key FROM span_links WHERE span_key=?", (row["span_key"],)
        ):
            linked = database.execute_sql(
                "SELECT trace_key FROM spans WHERE span_key=?", (linked_key,)
            ).fetchone()
            if linked:
                links.append(
                    Link(
                        SpanContext(
                            _integer(linked[0], 128),
                            _integer(linked_key, 64),
                            False,
                            TraceFlags(1),
                        )
                    )
                )
        return ReadableSpan(
            row["name"],
            context=context,
            parent=parent,
            resource=Resource({"service.name": "promptkeep"}),
            attributes=attributes,
            links=links,
            kind=SpanKind.INTERNAL,
            status=Status(
                StatusCode.ERROR if row["status"] in ("error", "blocked") else StatusCode.UNSET
            ),
            start_time=int(datetime.fromisoformat(row["started_at"]).timestamp() * 1e9),
            end_time=int(datetime.fromisoformat(row["ended_at"]).timestamp() * 1e9),
        )

    def _queue_row(self, row, database, *, timeout=0):
        """Queue a terminal span with known timing, applying the source loop guard."""
        metadata = json.loads(row.get("metadata_json") or "{}")
        if metadata.get("source") == "opentelemetry":
            return False
        if not row.get("started_at") or not row.get("ended_at"):
            self.skipped_unknown_timing += 1
            return False
        span = self._readable_span(row, database)
        try:
            self.queue.put(span, timeout=timeout)
        except queue.Full:
            self.dropped_count += 1
            return False
        return True

    def _accept(self, row):
        """Only already-redacted terminal live spans are eligible for export."""
        if (
            self._stop.is_set()
            or storage.db_proxy.obj is None
            or storage.db_proxy.obj.database != self.destination
        ):
            return
        self._queue_row(row, storage.db_proxy.obj)

    def replay(self, trace_keys: Iterable[str], *, db_path=None, timeout=None):
        """Explicitly export stored traces with known timing; never re-export OTel input.

        ``trace_keys`` must name the exact traces to replay. The source database is
        read without modifying it, stored redacted content is included only when
        this exporter was created with ``include_content=True``, and the method
        waits until the queued spans finish or ``timeout`` expires.
        """
        if self._stop.is_set():
            raise RuntimeError("cannot replay traces after OTLP export shutdown")
        keys = (trace_keys,) if isinstance(trace_keys, str) else tuple(dict.fromkeys(trace_keys))
        if not keys:
            return 0
        source = str(Path(db_path or get_settings().db_path))
        deadline = None if timeout is None else time.monotonic() + timeout
        queued = 0
        with storage.connection_context(source):
            database = storage.get_db()
            for trace_key in keys:
                exists = database.execute_sql(
                    "SELECT 1 FROM spans WHERE trace_key=? LIMIT 1", (trace_key,)
                ).fetchone()
                if not exists:
                    raise ValueError(f"no trace found for trace_key {trace_key!r}")
            for trace_key in keys:
                cursor = database.execute_sql(
                    "SELECT * FROM spans WHERE trace_key=? ORDER BY started_at,span_key",
                    (trace_key,),
                )
                columns = [column[0] for column in cursor.description]
                for values in cursor.fetchall():
                    row = dict(zip(columns, values))
                    remaining = None if deadline is None else max(0, deadline - time.monotonic())
                    if self._queue_row(row, database, timeout=remaining):
                        queued += 1
                    elif deadline is not None and time.monotonic() >= deadline:
                        raise TimeoutError("OTLP replay timed out while queueing spans")
        remaining = None if deadline is None else max(0, deadline - time.monotonic())
        if not self.flush(remaining):
            raise TimeoutError("OTLP replay timed out while exporting spans")
        return queued

    def _work(self):
        """Transport failures cost exported telemetry, never application work."""
        try:
            while not self._stop.is_set() or not self.queue.empty():
                try:
                    first = self.queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                batch = [first]
                while len(batch) < 512:
                    try:
                        batch.append(self.queue.get_nowait())
                    except queue.Empty:
                        break
                try:
                    if self.exporter.export(batch) != SpanExportResult.SUCCESS:
                        self.failed_count += len(batch)
                except Exception:
                    self.failed_count += len(batch)
                finally:
                    for _ in batch:
                        self.queue.task_done()
        finally:
            try:
                self.exporter.shutdown()
            except Exception:
                logger.warning("promptkeep: OTLP exporter shutdown failed", exc_info=True)

    def flush(self, timeout=None):
        """Wait for queued exports while honoring the caller's deadline."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self.queue.all_tasks_done:
            while self.queue.unfinished_tasks:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self.queue.all_tasks_done.wait(remaining)
        return True

    def shutdown(self, timeout=None):
        """Unsubscribe first, then retire only this installation's worker."""
        self._unsubscribe()
        self._stop.set()
        self._thread.join(timeout)
        return not self._thread.is_alive()

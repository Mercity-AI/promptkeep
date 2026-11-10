"""Registered archive ingestion: reading, detection, discovery and atomic delivery."""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import tempfile
from contextlib import contextmanager
from pathlib import Path

from .. import storage
from ..config import get_settings
from .base import AgentTrace, ImportReport

_ADAPTERS: dict[str, type[AgentTrace]] = {}


def register_agent_trace(adapter_class):
    """Register an independent adapter under its unique source name."""
    if not isinstance(adapter_class, type) or not issubclass(adapter_class, AgentTrace):
        raise TypeError("adapter must subclass AgentTrace")
    name = adapter_class.source
    if not name or name in ("auto", "unknown"):
        raise ValueError("adapter requires a source name")
    if name in _ADAPTERS and _ADAPTERS[name] is not adapter_class:
        raise ValueError("source already registered")
    _ADAPTERS[name] = adapter_class
    return adapter_class


@contextmanager
def _source(path):
    """Read plain JSON/JSONL or streaming Zstandard without database effects."""
    path = Path(path)
    with path.open("rb") as raw:
        if path.name.endswith(".zst"):
            try:
                zstd = importlib.import_module("zstandard")
            except ImportError:
                raise ImportError(
                    'Compressed imports require pip install "promptkeep[codex]"'
                ) from None
            with zstd.ZstdDecompressor().stream_reader(raw) as reader:
                with io.BufferedReader(reader) as buffered:
                    yield buffered
        else:
            yield raw


def records(path, *, complete_lines=False):
    """Yield ordered records; an unfinished watched JSONL tail stays unconsumed."""
    with _source(path) as stream:
        if str(path).endswith((".json", ".json.zst")):
            yield 1, json.load(stream)
        else:
            for line, raw in enumerate(stream, 1):
                if complete_lines and not raw.endswith(b"\n"):
                    break
                try:
                    yield line, json.loads(raw)
                except (ValueError, UnicodeDecodeError):
                    yield line, None


def select_adapter(path, source="auto"):
    """Reject ambiguous/unknown content without echoing any source payload."""
    if source != "auto":
        try:
            return _ADAPTERS[source]
        except KeyError:
            raise ValueError("unknown trace source") from None
    matches = set()
    for line, record in records(path):
        if record is not None:
            matches.update(cls for cls in _ADAPTERS.values() if cls.detect(record))
        if line >= 100:
            break
    if len(matches) != 1:
        raise ValueError("ambiguous trace format" if matches else "unknown trace format")
    return matches.pop()


def import_trace(
    path, *, source="auto", db_path=None, strict=False, checkpoint=None, complete_lines=False
):
    """Import an explicit saved session through its registered source adapter.

    Archives bypass sampling and queues, apply payload controls before staging,
    and replace their owned snapshot atomically. Failed imports preserve history.
    """
    settings = get_settings()
    if not settings.enabled or settings.write_mode == "off":
        raise ValueError("Trace import requires enabled tracking and write_mode other than off")
    destination = str(Path(db_path or settings.db_path).absolute())
    adapter = select_adapter(path, source)
    if adapter.snapshot:
        with _source(path) as stream:
            raw = stream.read()
        document = json.loads(raw)
        return import_session(
            lambda: iter(((1, document),)),
            adapter=adapter,
            digest=hashlib.sha256(raw).hexdigest(),
            db_path=destination,
            strict=strict,
            checkpoint=checkpoint,
        )
    digest = hashlib.sha256()
    with _source(path) as stream:
        for raw in iter(lambda: stream.read(65536), b""):
            digest.update(raw)
    return import_session(
        lambda: records(path, complete_lines=complete_lines),
        adapter=adapter,
        digest=digest.hexdigest(),
        db_path=destination,
        strict=strict,
        checkpoint=checkpoint,
    )


def import_session(
    record_factory,
    *,
    adapter,
    digest,
    db_path=None,
    strict=False,
    checkpoint=None,
    metadata=None,
    expected_session=None,
    diagnostics=(),
):
    """Ingest a replayable ordered session input from a file or dataset adapter."""
    settings = get_settings()
    if not settings.enabled or settings.write_mode == "off":
        raise ValueError("Trace import requires enabled tracking")
    destination = str(Path(db_path or settings.db_path).absolute())
    with tempfile.TemporaryDirectory(prefix="promptkeep-import-") as temporary:
        stage = storage._ImportStage(str(Path(temporary) / "stage.db"), settings)
        try:
            parser = adapter(stage, strict)
            parser.discover(record_factory())
            for code, line in diagnostics:
                parser.diagnose(code, line)
            for line, event in record_factory():
                parser.counts["records"] += 1
                if event is None:
                    parser.diagnose("invalid_json", line, True)
                    continue
                try:
                    parser.consume(event, line)
                except (TypeError, AttributeError, KeyError):
                    parser.diagnose("invalid_record_shape", line, True)
            if expected_session is not None and parser.session != expected_session:
                raise ValueError("dataset session identity mismatch")
            summary = parser.finalize()
            summary.update(metadata or {})
            if checkpoint is not None:
                checkpoint["state"] = parser.checkpoint_state()
                checkpoint["parser_revision"] = adapter.revision
            outcome = storage.persist_import(
                stage,
                db_path=destination,
                session_id=parser.session,
                trace_key=parser.trace,
                digest=digest,
                revision=adapter.revision,
                summary=summary,
                source=adapter.source,
                checkpoint=checkpoint,
            )
            values = {
                k: summary[k]
                for k in ImportReport.__dataclass_fields__
                if k not in ("outcome", "diagnostics")
            }
            return ImportReport(**values, outcome=outcome, diagnostics=tuple(parser.diagnostics))
        finally:
            stage.close()

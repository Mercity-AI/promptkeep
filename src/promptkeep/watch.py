"""Incremental, restartable watching of explicitly selected native session logs."""

from __future__ import annotations

import hashlib
import json
import logging
import tempfile
import threading
import uuid
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from . import storage
from .config import get_settings
from .importers.claude_code import _ClaudeParser
from .importers.codex import ImportDiagnostic, _Parser

logger = logging.getLogger("promptkeep")


class Watcher:
    """Watch append-only files without launching or modifying agent processes."""

    def __init__(self, source, path, *, db_path=None, interval=1.0, strict=False):
        """Capture configuration and validate the explicit watch target."""
        if source not in ("codex", "claude-code"):
            raise ValueError("watch source must be codex or claude-code")
        if interval <= 0:
            raise ValueError("watch interval must be positive")
        self.source, self.path = source, Path(path).absolute()
        if not self.path.exists():
            raise ValueError("watch path does not exist")
        if self.path.is_file() and self.path.suffix != ".jsonl":
            raise ValueError("incremental watching requires uncompressed .jsonl files")
        self.settings = get_settings()
        if not self.settings.enabled or self.settings.write_mode == "off":
            raise ValueError("watching requires enabled tracking")
        self.db_path = str(Path(db_path or self.settings.db_path).absolute())
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.interval, self.strict = interval, strict
        self._stop = threading.Event()
        self._thread = None
        self._sessions = {}
        self._temporary = None
        self.failed_count = 0
        self.revision = 2

    def start(self):
        """Start one explicitly owned worker, idempotently."""
        if self._thread and self._thread.is_alive():
            return self
        self._stop.clear()
        self._temporary = tempfile.TemporaryDirectory(prefix="promptkeep-watch-")
        self._thread = threading.Thread(target=self._work, daemon=True, name="promptkeep-watch")
        self._thread.start()
        return self

    def stop(self):
        """Request shutdown without terminating or modifying the watched agent."""
        self._stop.set()

    def join(self, timeout=None):
        """Wait only up to the caller's timeout; return whether the worker retired."""
        if self._thread:
            self._thread.join(timeout)
        return not self._thread or not self._thread.is_alive()

    def __enter__(self):
        """Start watching within an explicit application lifetime."""
        return self.start()

    def __exit__(self, *args):
        """Stop and release private redacted staging on normal or exceptional exit."""
        self.stop()
        self.join()
        return False

    def _work(self):
        """Poll complete lines and close staging connections on their owner thread."""
        try:
            while not self._stop.is_set():
                files = [self.path] if self.path.is_file() else self.path.rglob("*.jsonl")
                for path in files:
                    if self._stop.is_set():
                        break
                    try:
                        self._poll(path)
                    except Exception:
                        failed = self._sessions.pop(path.absolute(), None)
                        if failed:
                            failed[0].close()
                        self.failed_count += 1
                        logger.warning("promptkeep: watch update failed", exc_info=True)
                self._stop.wait(self.interval)
        finally:
            for path, (stage, parser, checkpoint) in self._sessions.items():
                try:
                    if parser is None:
                        continue
                    for key, row in stage.rows("spans"):
                        if row["status"] == "running":
                            stage.put("spans", key, {"status": "incomplete"})
                    self._commit(stage, parser, checkpoint, finalize=True)
                except Exception:
                    logger.warning("promptkeep: watch finalization failed", exc_info=True)
                finally:
                    stage.close()
            self._sessions.clear()
            if self._temporary:
                self._temporary.cleanup()

    def _restore(self, stage, state):
        """Restore only identifiers/counters; raw payloads are never checkpointed."""
        parser = (
            _ClaudeParser(
                stage,
                self.strict,
                state["session"],
                state.get("native_session"),
                state.get("agent_id"),
            )
            if self.source == "claude-code"
            else _Parser(stage, self.strict)
        )
        for name in (
            "session",
            "trace",
            "current",
            "first",
            "last",
            "legacy_turn",
            "task_pending",
            "call_turns",
            "models",
            "message_turns",
            "native_session",
            "agent_id",
        ):
            if name in state:
                setattr(parser, name, state[name])
        parser.counts = Counter(state.get("counts", {}))
        parser.diagnostics = [ImportDiagnostic(**d) for d in state.get("diagnostics", [])]
        return parser

    def _commit(self, stage, parser, checkpoint, finalize=False):
        """State and source offset advance only when all changed rows commit."""
        if parser.session is None:
            return
        summary = parser.summary()
        # An actively watched operation is running, not a completed historical snapshot.
        if not self._stop.is_set() and not finalize:
            for key, row in stage.rows("spans"):
                if row["status"] == "incomplete" or key == parser.trace:
                    stage.put("spans", key, {"status": "running"})
        names = (
            "session",
            "trace",
            "current",
            "first",
            "last",
            "legacy_turn",
            "task_pending",
            "call_turns",
            "models",
            "message_turns",
            "native_session",
            "agent_id",
        )
        checkpoint["state"] = {
            name: getattr(parser, name) for name in names if hasattr(parser, name)
        }
        checkpoint["state"].update(
            counts=dict(parser.counts), diagnostics=[asdict(d) for d in parser.diagnostics]
        )
        storage.persist_watch(
            stage,
            db_path=self.db_path,
            source=self.source,
            session_id=parser.session,
            trace_key=parser.trace,
            summary=summary,
            checkpoint=checkpoint,
        )

    def _poll(self, path):
        """Read appended complete records; generation changes never delete stored history."""
        path = path.absolute()
        source_key = hashlib.sha256((self.source + ":" + str(path)).encode()).hexdigest()
        stat = path.stat()
        generation = f"{stat.st_dev}:{stat.st_ino}"
        session = self._sessions.get(path)
        if session is None:
            checkpoint, data = storage.watch_snapshot(self.db_path, source_key)
            stage = storage._ImportStage(
                str(Path(self._temporary.name) / (source_key + uuid.uuid4().hex + ".db")),
                self.settings,
            )
            if checkpoint:
                for kind, rows in data.items():
                    for value in rows:
                        key, row = (
                            value
                            if kind == "usage"
                            else (value["span_key"] if kind == "spans" else value["run_key"], value)
                        )
                        stage.db.execute_sql(
                            f"INSERT INTO {kind}(key,data) VALUES (?,?)", (key, json.dumps(row))
                        )
                parser = self._restore(stage, checkpoint["state"])
            else:
                parser = _Parser(stage, self.strict) if self.source == "codex" else None
                checkpoint = dict(
                    source_key=source_key,
                    generation=generation,
                    byte_offset=0,
                    parser_revision=self.revision,
                    state={},
                )
            checkpoint["source_key"] = source_key
            session = (stage, parser, checkpoint)
        stage, parser, checkpoint = session
        self._sessions[path] = session
        if (
            checkpoint["generation"] != generation
            or stat.st_size < checkpoint["byte_offset"]
            or checkpoint["parser_revision"] != self.revision
        ):
            # Reparse a replacement prefix using stable IDs; old observations are retained.
            checkpoint.update(generation=generation, byte_offset=0, parser_revision=self.revision)
            for kind in ("spans", "runs", "usage"):
                stage.db.execute_sql(f"DELETE FROM {kind}")
                stage.dirty[kind].clear()
            parser = _Parser(stage, self.strict) if self.source == "codex" else None
        offset = checkpoint["byte_offset"]
        changed = False
        with path.open("rb") as source:
            source.seek(offset)
            while True:
                raw = source.readline()
                if not raw or not raw.endswith(b"\n"):
                    break
                offset = source.tell()
                try:
                    event = json.loads(raw)
                except ValueError:
                    if self.strict:
                        raise ValueError("invalid watched JSON record") from None
                    continue
                if parser is None:
                    identity = event.get("sessionId") if isinstance(event, dict) else None
                    if not identity or event.get("type") not in ("user", "assistant"):
                        continue
                    agent_id = event.get("agentId")
                    logical = identity + ":agent:" + agent_id if agent_id else identity
                    parser = _ClaudeParser(stage, self.strict, logical, identity, agent_id)
                parser.counts["records"] += 1
                parser.parse(event, parser.counts["records"])
                changed = True
        if parser is not None and changed:
            checkpoint["byte_offset"] = offset
            self._commit(stage, parser, checkpoint)
        self._sessions[path] = (stage, parser, checkpoint)

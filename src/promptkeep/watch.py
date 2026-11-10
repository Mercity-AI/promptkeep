"""Incremental, restartable watching of explicitly selected native session logs."""

from __future__ import annotations

import hashlib
import json
import logging
import tempfile
import threading
import uuid
from pathlib import Path

from . import storage
from .config import get_settings
from .importers.engine import import_session, records, select_adapter

logger = logging.getLogger("promptkeep")


class Watcher:
    """Watch append-only files without launching or modifying agent processes."""

    def __init__(self, *args, source="auto", db_path=None, interval=1.0, strict=False):
        """Capture configuration and validate the explicit watch target."""
        if len(args) == 2:
            source, path = args  # Historical Watcher(source, path).
        elif len(args) == 1:
            path = args[0]
        else:
            raise TypeError("Watcher requires a path")
        if interval <= 0:
            raise ValueError("watch interval must be positive")
        self.source, self.path = source, Path(path).absolute()
        if not self.path.exists():
            raise ValueError("watch path does not exist")
        if self.path.is_file() and self.path.suffix not in (".jsonl", ".json"):
            raise ValueError("watching requires uncompressed .jsonl or .json files")
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
        self._snapshots = {}

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
                files = (
                    [self.path]
                    if self.path.is_file()
                    else (p for p in self.path.rglob("*") if p.suffix in (".jsonl", ".json"))
                )
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
        checkpoint["state"] = parser.checkpoint_state()
        storage.persist_watch(
            stage,
            db_path=self.db_path,
            source=parser.source,
            session_id=parser.session,
            trace_key=parser.trace,
            summary=summary,
            checkpoint=checkpoint,
        )

    def _poll(self, path):
        """Read appended complete records; generation changes never delete stored history."""
        path = path.absolute()
        if path.suffix == ".json":
            try:
                json.loads(path.read_bytes())
            except (ValueError, UnicodeDecodeError):
                return
        adapter = select_adapter(path, self.source)
        source_key = hashlib.sha256((adapter.source + ":" + str(path)).encode()).hexdigest()
        if adapter.snapshot:
            return self._poll_snapshot(path, source_key, adapter)
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
                parser = adapter(stage, self.strict)
                parser.restore_state(checkpoint["state"])
            else:
                parser = adapter(stage, self.strict)
                parser.discover(records(path, complete_lines=True))
                checkpoint = dict(
                    source_key=source_key,
                    generation=generation,
                    byte_offset=0,
                    parser_revision=adapter.revision,
                    state={},
                )
            checkpoint["source_key"] = source_key
            session = (stage, parser, checkpoint)
        stage, parser, checkpoint = session
        self._sessions[path] = session
        if (
            checkpoint["generation"] != generation
            or stat.st_size < checkpoint["byte_offset"]
            or checkpoint["parser_revision"] != adapter.revision
        ):
            # A revision rebuild reconciles obsolete owned rows; rotation retains history.
            checkpoint["reconcile"] = checkpoint["parser_revision"] != adapter.revision
            checkpoint.update(
                generation=generation, byte_offset=0, parser_revision=adapter.revision
            )
            for kind in ("spans", "runs", "usage"):
                stage.db.execute_sql(f"DELETE FROM {kind}")
                stage.dirty[kind].clear()
            parser = adapter(stage, self.strict)
            parser.discover(records(path, complete_lines=True))
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
                parser.counts["records"] += 1
                parser.consume(event, parser.counts["records"])
                changed = True
        if parser is not None and changed:
            checkpoint["byte_offset"] = offset
            self._commit(stage, parser, checkpoint)
        self._sessions[path] = (stage, parser, checkpoint)

    def _poll_snapshot(self, path, source_key, adapter):
        """Commit only complete valid JSON replacements with their checkpoints."""
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if self._snapshots.get(path) == digest:
            return
        try:
            document = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            return  # An editor may still be writing the replacement.
        stat = path.stat()
        checkpoint = dict(
            source_key=source_key,
            generation=f"{stat.st_dev}:{stat.st_ino}",
            byte_offset=len(raw),
            parser_revision=adapter.revision,
            state={},
        )
        import_session(
            lambda: iter(((1, document),)),
            adapter=adapter,
            digest=digest,
            db_path=self.db_path,
            strict=True,
            checkpoint=checkpoint,
        )
        self._snapshots[path] = digest

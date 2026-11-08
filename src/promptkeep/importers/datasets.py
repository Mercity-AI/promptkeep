"""Thin adapters for explicitly downloaded, revision-pinned dataset formats."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .. import storage
from ..config import get_settings
from .claude_code import import_claude_code
from .codex import _REVISION, ImportReport, _Parser


@dataclass(frozen=True)
class DatasetImportReport:
    """Independent session outcomes; one bad session cannot corrupt other commits."""

    dataset: str
    sessions: tuple[ImportReport, ...]
    failures: tuple[dict, ...]


def import_dataset(
    path: str | Path, *, dataset: str, db_path=None, strict: bool = False
) -> DatasetImportReport:
    """Import a downloaded local JSONL file without network access or raw staging."""
    if dataset == "ultralazr/claude-code-traces":
        return DatasetImportReport(
            dataset, (import_claude_code(path, db_path=db_path, strict=strict),), ()
        )
    if dataset != "build-small-hackathon/hackathon-advisor-codex-traces":
        raise ValueError("unsupported dataset adapter")
    settings = get_settings()
    if not settings.enabled or settings.write_mode == "off":
        raise ValueError("Dataset import requires enabled tracking")
    destination = str(Path(db_path or settings.db_path).absolute())
    groups = defaultdict(dict)
    failures, reports = [], []
    invalid_sessions = set()
    # Keep only offsets and hashes in memory; payloads are reread from the caller's file.
    with Path(path).open("rb") as source:
        line = 0
        while True:
            offset = source.tell()
            raw = source.readline()
            if not raw:
                break
            line += 1
            try:
                value = json.loads(raw)
                session, index = value["session_id"], value["record_index"]
                if not isinstance(session, str) or type(index) is not int:
                    raise ValueError()
                digest = hashlib.sha256(raw).hexdigest()
                previous = groups[session].get(index)
                if previous and previous[2] != digest:
                    invalid_sessions.add(session)
                    raise ValueError("conflicting record index")
                groups[session][index] = (offset, line, digest)
            except (ValueError, KeyError, TypeError):
                failures.append({"line": line, "code": "invalid_dataset_record"})
                if strict:
                    raise ValueError(f"invalid dataset record at line {line}") from None
        for session, offsets in groups.items():
            if session in invalid_sessions:
                continue
            with tempfile.TemporaryDirectory(prefix="promptkeep-dataset-") as temporary:
                stage = storage._ImportStage(str(Path(temporary) / "stage.db"), settings)
                try:
                    parser = _Parser(stage, strict)
                    digest = hashlib.sha256()
                    last = None
                    for index, (offset, line, _) in sorted(offsets.items()):
                        if last is not None and index != last + 1:
                            parser.diagnose("dataset_record_gap", line)
                        last = index
                        source.seek(offset)
                        raw = source.readline()
                        digest.update(raw)
                        event = json.loads(raw)
                        parser.counts["records"] += 1
                        parser.parse(event, line)
                    if parser.session != session:
                        raise ValueError("dataset session identity mismatch")
                    summary = parser.summary()
                    summary["dataset"] = dataset
                    outcome = storage.persist_import(
                        stage,
                        db_path=destination,
                        session_id=session,
                        trace_key=parser.trace,
                        digest=digest.hexdigest(),
                        revision=_REVISION,
                        summary=summary,
                    )
                    values = {
                        k: summary[k]
                        for k in ImportReport.__dataclass_fields__
                        if k not in ("outcome", "diagnostics")
                    }
                    reports.append(
                        ImportReport(
                            **values, outcome=outcome, diagnostics=tuple(parser.diagnostics)
                        )
                    )
                except Exception as exc:
                    failures.append({"session_id": session, "code": type(exc).__name__})
                finally:
                    stage.close()
    return DatasetImportReport(dataset, tuple(reports), tuple(failures))

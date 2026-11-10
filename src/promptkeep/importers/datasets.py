"""Registered dataset adapters yield ordered session inputs to the shared engine."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .base import ImportReport
from .engine import import_session, import_trace, select_adapter


@dataclass(frozen=True)
class DatasetImportReport:
    """Independent session outcomes; one bad session cannot corrupt other commits."""

    dataset: str
    sessions: tuple[ImportReport, ...]
    failures: tuple[dict, ...]


class NativeDataset:
    """A dataset that publishes native session files without a row wrapper."""

    source = "claude-code"

    def sessions(self, path, failures, strict):
        """Yield the native file directly to the ordinary import engine."""
        yield dict(path=path, source=self.source)


class IndexedDataset:
    """Index wrapped records by session and source order without raw disk staging."""

    source = "codex"

    def sessions(self, path, failures, strict):
        """Keep offsets and digests only, then replay independently validated sessions."""
        groups = defaultdict(dict)
        invalid = set()
        with Path(path).open("rb") as stream:
            line = 0
            while True:
                offset = stream.tell()
                raw = stream.readline()
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
                        invalid.add(session)
                        raise ValueError()
                    groups[session][index] = offset, line, digest
                except (ValueError, KeyError, TypeError):
                    failures.append(dict(line=line, code="invalid_dataset_record"))
                    if strict:
                        raise ValueError(f"invalid dataset record at line {line}") from None
        for session, offsets in groups.items():
            if session not in invalid:
                yield dict(path=path, source=self.source, session=session, offsets=offsets)


_DATASETS = {
    "ultralazr/claude-code-traces": NativeDataset(),
    "build-small-hackathon/hackathon-advisor-codex-traces": IndexedDataset(),
}


def _indexed_records(path, offsets):
    """Reread selected records in source order on each engine pass."""
    with Path(path).open("rb") as stream:
        for _, (offset, line, _) in sorted(offsets.items()):
            stream.seek(offset)
            yield line, json.loads(stream.readline())


def import_dataset(path, *, dataset, db_path=None, strict=False) -> DatasetImportReport:
    """Adapt a local pinned dataset without source-specific dispatch or persistence."""
    if dataset not in _DATASETS:
        raise ValueError("unsupported dataset adapter")
    reports, failures = [], []
    for session in _DATASETS[dataset].sessions(path, failures, strict):
        try:
            if "offsets" not in session:
                report = import_trace(
                    session["path"], source=session["source"], db_path=db_path, strict=strict
                )
            else:
                offsets = session["offsets"]
                digest = hashlib.sha256()
                with Path(path).open("rb") as stream:
                    for _, (offset, _, _) in sorted(offsets.items()):
                        stream.seek(offset)
                        digest.update(stream.readline())
                report = import_session(
                    lambda: _indexed_records(path, offsets),
                    adapter=select_adapter(path, session["source"]),
                    digest=digest.hexdigest(),
                    db_path=db_path,
                    strict=strict,
                    metadata={"dataset": dataset},
                    expected_session=session["session"],
                    diagnostics=[
                        ("dataset_record_gap", offsets[index][1])
                        for previous, index in zip(sorted(offsets), sorted(offsets)[1:])
                        if index != previous + 1
                    ],
                )
            reports.append(report)
        except Exception as exc:
            failures.append(dict(session_id=session.get("session"), code=type(exc).__name__))
    return DatasetImportReport(dataset, tuple(reports), tuple(failures))

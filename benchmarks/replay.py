"""Offline application replay: warm up, then five measured runs per write mode."""

import json
import statistics
import tempfile
import time
from pathlib import Path

import peewee
from tests.fakes import FakeClient

import promptkeep
from promptkeep import storage


def measure(mode):
    """Replay the same 500 tools and 50 model calls with a fresh local destination."""
    opens = 0
    original = peewee.SqliteDatabase._connect

    def connect(database):
        """Count physical connection opens independently of PromptKeep internals."""
        nonlocal opens
        opens += 1
        return original(database)

    storage.reset_caches()
    with tempfile.TemporaryDirectory() as directory:
        promptkeep.configure(
            db_path=Path(directory) / "trace.db", write_mode=mode, queue_size=10000
        )
        client = promptkeep.wrap(FakeClient())
        peewee.SqliteDatabase._connect = connect
        try:
            start = time.perf_counter()
            with promptkeep.agent("replay", input={"task": "offline"}):
                for index in range(500):
                    if index % 10 == 0:
                        client.chat.completions.create(
                            model="fixture", messages=[dict(role="user", content="hello")]
                        )
                    with promptkeep.tool("step", input={"index": index}) as step:
                        step.output("done")
            assert promptkeep.flush(timeout=20)
            elapsed = time.perf_counter() - start
        finally:
            peewee.SqliteDatabase._connect = original
            storage.reset_caches()
    return elapsed, opens


if __name__ == "__main__":
    result = {}
    for mode in ("sync", "background"):
        measure(mode)
        runs = [measure(mode) for _ in range(5)]
        result[mode] = dict(
            median_seconds=statistics.median(t for t, _ in runs),
            seconds=[t for t, _ in runs],
            connection_opens=[n for _, n in runs],
        )
    print(json.dumps(result, indent=2))

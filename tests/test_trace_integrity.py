"""Stress the SQLite read-before-write lock-upgrade regression with real writers."""

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

import promptkeep
from promptkeep import Prompt, history, storage, writer
from tests.fakes import FakeClient


@pytest.mark.parametrize("repetition", range(3))
def test_registration_and_3000_background_tool_spans(caplog, repetition):
    promptkeep.configure(write_mode="background", queue_size=20000, batch_size=100)

    def register():
        """A separate connection competes with the background writer."""
        try:
            for i in range(100):
                assert Prompt(f"version {i}", name="CONCURRENT").version is not None
        finally:
            storage.db_proxy.close()

    client = promptkeep.wrap(FakeClient())
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(register)
        with promptkeep.agent("stress") as root:
            for i in range(3000):
                if i % 10 == 0:
                    client.chat.completions.create(
                        model="test", messages=[dict(role="user", content="hi")]
                    )
                with promptkeep.tool("step"):
                    pass
        future.result()
    assert promptkeep.flush(timeout=20)
    assert writer.dropped_count() == 0
    assert len(history.trace(root.trace_key).spans) == 3301
    assert len(history.all_runs(limit=1000)) == 300
    assert "database is locked" not in caplog.text


def test_independent_processes_share_background_database(isolated_db):
    # Initialize schema before launching independent writers; this test targets batch contention.
    storage.get_db()
    script = """
import sys
import promptkeep
from promptkeep import Prompt
promptkeep.configure(db_path=sys.argv[1], write_mode="background", queue_size=10000)
with promptkeep.agent("process"):
    for i in range(1500):
        if i % 20 == 0:
            assert Prompt(f"version {i}", name="PROCESS").version is not None
        with promptkeep.tool("step"):
            pass
assert promptkeep.flush(timeout=20)
"""
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script, str(isolated_db)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(4)
    ]
    for process in processes:
        out, err = process.communicate(timeout=40)
        assert process.returncode == 0, out + err
        assert "database is locked" not in err
    traces = history.list_traces()
    assert len(traces) == 4
    assert all(len(history.trace(root.span_key).spans) == 1501 for root in traces)

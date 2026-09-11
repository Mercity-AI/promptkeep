"""The background writer: a bounded queue drained by one daemon thread.

In write_mode="background" (the default), `storage.record_run` and
`storage.record_check` enqueue their items here and return immediately; this
module's worker thread batches queued items into single transactions off the
caller's hot path. Design rules, each a known telemetry-library footgun:

- **Bounded queue, drop-oldest.** A full queue drops the oldest item and
  counts it (rate-limited warning) — a slow disk must never become an OOM.
- **The worker never dies.** Every batch is exception-shielded; a broken DB
  loses telemetry, never the thread.
- **drain() and an atexit hook** give short-lived processes (scripts,
  Lambdas, tests) a way to guarantee queued items are on disk. The public
  `promptkeep.flush()` (in `tracking`) first waits for async post-checks
  still running, then calls drain() — by exit time the check pool has
  already been joined by the interpreter, so the hook only needs drain().
- **Fork safety.** A forked child inherits a dead thread and a queue of
  items the parent will also write; the child resets to empty state instead
  of double-writing or hanging (os.register_at_fork, POSIX only).

The thread starts lazily on the first enqueue — importing promptkeep never
spawns threads.
"""

from __future__ import annotations

import atexit
import logging
import os
import queue
import threading
import time
from typing import Optional

logger = logging.getLogger("promptkeep")

_lock = threading.Lock()
_queue: Optional["queue.Queue[dict]"] = None
_thread: Optional[threading.Thread] = None
_dropped_total = 0
_last_drop_log = 0.0
_DROP_LOG_INTERVAL = 5.0
# The atexit drain is registered once per process, not once per worker start:
# reset() retires the thread, and re-registering on each restart would pile up
# duplicate hooks (one per test, in a test suite). Deliberately not cleared by
# reset() — a stale hook is harmless (flush() no-ops on an empty queue), a
# leaked one per reset is not.
_atexit_registered = False


def submit(item: dict) -> None:
    """Enqueue one item (a run row or a late verdict) for the background
    thread; never blocks, never raises.

    On overflow the oldest queued item is discarded (and counted) to make
    room — recent telemetry is worth more than old telemetry.
    """
    global _dropped_total, _last_drop_log
    try:
        q = _ensure_started()
        try:
            q.put_nowait(item)
            return
        except queue.Full:
            pass
        # Full: evict the oldest, then retry once. A second Full (the worker
        # drained the queue between our two calls) just means there's room.
        with _lock:
            try:
                q.get_nowait()
                q.task_done()
                _dropped_total += 1
            except queue.Empty:
                pass
            try:
                q.put_nowait(item)
            except queue.Full:
                _dropped_total += 1
        now = time.monotonic()
        if now - _last_drop_log >= _DROP_LOG_INTERVAL:
            _last_drop_log = now
            logger.warning(
                "promptkeep: write queue full — %d run(s) dropped so far; "
                "raise queue_size or call promptkeep.flush()",
                _dropped_total,
            )
    except Exception:
        logger.warning("promptkeep: failed to enqueue run", exc_info=True)


def drain(timeout: Optional[float] = None) -> bool:
    """Block until every queued item is written (or timeout seconds pass).

    Returns True when the queue fully drained, False on timeout. A process
    that never wrote in background mode returns True immediately. This is the
    queue half of `promptkeep.flush()`; use that from application code.
    """
    q = _queue
    if q is None:
        return True
    deadline = None if timeout is None else time.monotonic() + timeout
    with q.all_tasks_done:
        while q.unfinished_tasks:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                return False
            q.all_tasks_done.wait(remaining)
    return True


def dropped_count() -> int:
    """Total items discarded because the queue was full (for tests/diagnostics)."""
    return _dropped_total


def reset() -> None:
    """Discard everything queued, the queue itself, and the drop counter.
    Mainly for tests.

    Queued items are *discarded*, not flushed: between tests they point at a
    database that no longer exists, and flushing them into the next test's
    fresh DB would be cross-contamination. The queue object is dropped too so
    the next use re-reads queue_size — the worker notices its queue was
    retired (module _queue no longer points at it) and exits within one
    flush_interval.
    """
    global _queue, _thread, _dropped_total
    with _lock:
        q = _queue
        if q is None:
            return
        while True:
            try:
                q.get_nowait()
                q.task_done()
            except queue.Empty:
                break
        _queue = None
        _thread = None
        _dropped_total = 0


def _ensure_started() -> "queue.Queue[dict]":
    """Create the queue and start the daemon thread on first use."""
    global _queue, _thread
    q = _queue
    if q is not None and _thread is not None and _thread.is_alive():
        return q
    with _lock:
        if _queue is None:
            from .config import get_settings

            # Queue bound is read once at creation; changing queue_size later
            # needs a process restart (or reset_caches in tests).
            _queue = queue.Queue(maxsize=get_settings().queue_size)
        if _thread is None or not _thread.is_alive():
            _thread = threading.Thread(target=_worker, name="promptkeep-writer", daemon=True)
            _thread.start()
            # Drain on interpreter exit: atexit runs in the main thread while
            # daemon threads are still alive, so queued items can still land.
            # Registered once per process (see _atexit_registered) — restarts
            # after reset() must not stack duplicate hooks.
            global _atexit_registered
            if not _atexit_registered:
                atexit.register(drain, 2.0)
                _atexit_registered = True
        return _queue


def _worker() -> None:
    """Drain the queue: batch rows, write each batch in one transaction.

    Serves the queue it was started for; retires when reset() swaps it out.
    """
    from .config import get_settings

    q = _queue
    assert q is not None
    while True:
        try:
            if _queue is not q:
                return  # this queue was retired by reset()
            settings = get_settings()
            try:
                first = q.get(timeout=settings.flush_interval)
            except queue.Empty:
                continue
            batch = [first]
            while len(batch) < settings.batch_size:
                try:
                    batch.append(q.get_nowait())
                except queue.Empty:
                    break
            try:
                from . import storage

                storage.write_batch(batch)
            except Exception:
                logger.warning(
                    "promptkeep: background writer failed to persist %d item(s)",
                    len(batch),
                    exc_info=True,
                )
            finally:
                for _ in batch:
                    q.task_done()
        except Exception:
            # The worker must survive anything — even failures in the
            # bookkeeping above. Losing telemetry is fine; dying is not.
            logger.warning("promptkeep: background writer error", exc_info=True)


def _after_fork_in_child() -> None:
    """Forget inherited state: the thread didn't survive the fork, and the
    parent still owns (and will write) everything that was queued."""
    global _queue, _thread, _dropped_total
    _queue = None
    _thread = None
    _dropped_total = 0


if hasattr(os, "register_at_fork"):  # POSIX only; Windows cannot fork.
    os.register_at_fork(after_in_child=_after_fork_in_child)

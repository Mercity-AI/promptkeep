"""Checks: user functions that run before a call (gates) and after it (audits).

A *check* is just a function returning a ``Verdict``. Two kinds:

- **pre** — runs before the provider call, in registration order, and can
  stop it. Blocking by nature (the whole point is to gate the request).
- **post** — runs after the response comes back. Defaults to ``mode="async"``
  so the caller gets their response immediately and the verdict is written
  a moment later; ``mode="blocking"`` waits inline when the verdict must
  gate what the user sees.

Whatever a check does inside — a regex, a JSON-schema check, or another LLM
call — is opaque to the framework; it only cares about the returned Verdict.

Two safety rules, same spirit as the rest of the library:

- **A crashing check never crashes the call.** It's recorded ``status="error"``
  and treated as non-blocking.
- **A check that calls an LLM must not record itself.** Check execution runs
  inside ``suppress()``, a contextvar the wrapper honors to skip tracking for
  nested calls — otherwise a groundedness judge would log its own runs and,
  worse, recurse into its own checks.

Verdicts persist through the same write path as run rows (``storage``), so
``write_mode`` governs them too, and ``promptkeep.flush()`` waits for async
post-checks still running (``wait_for_pending``) before draining the queue —
"everything is on disk" includes the verdicts that were mid-flight.
"""

from __future__ import annotations

import contextvars
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field, replace
from typing import Any

from . import storage

logger = logging.getLogger("promptkeep")

# Set while a check function runs. The wrapper treats a call made under
# suppression as an untracked passthrough — no run row, no nested checks.
_suppressed: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "promptkeep_suppress", default=False
)

# Shared pool for *async* post-checks only (fire-and-forget audits that run
# after the response is already back). Its size bounds how many run at once —
# backpressure on background telemetry, which never gates a request. Blocking
# checks (pre and blocking-post) do NOT use this pool: each runs in its own
# thread (see _run_with_timeout), so a saturated pool can't make a fast check
# look slow or cap how many requests can be in flight.
_executor: ThreadPoolExecutor | None = None


def _get_executor() -> ThreadPoolExecutor:
    """Lazily create the shared async-post-check pool (never at import time)."""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="promptkeep-post")
    return _executor


# Async post-checks still running. flush() waits on these before draining the
# writer queue, so a verdict that was mid-flight is on disk when flush returns.
_pending: set[Future] = set()
_pending_lock = threading.Lock()


def schedule_async(chk: Check, ctx: CheckContext, run_key: str | None) -> Future:
    """Run an async post-check on the shared pool; persist its verdict when it lands.

    Returns the Future (the RunHandle waits on it for ``wait()``). The check
    is tracked as pending until it finishes so ``wait_for_pending`` — and
    therefore ``promptkeep.flush()`` — can wait for it. Persistence goes
    through ``storage.record_check``, which follows the configured write_mode:
    in background mode the verdict queues behind its own run row.
    """

    def _work() -> CheckResult:
        result = chk.run_inline(ctx)
        storage.record_check(run_key, result.to_row())
        return result

    future = _get_executor().submit(_work)
    # Register before attaching the done-callback: a check that finishes
    # between the two would otherwise leave a stale entry behind.
    with _pending_lock:
        _pending.add(future)
    future.add_done_callback(_forget_pending)
    return future


def _forget_pending(future: Future) -> None:
    with _pending_lock:
        _pending.discard(future)


def wait_for_pending(timeout: float | None = None) -> bool:
    """Block until every async post-check in flight has finished, or ``timeout``
    seconds pass. Returns True when none remain, False on timeout."""
    with _pending_lock:
        snapshot = list(_pending)
    if not snapshot:
        return True
    _done, not_done = wait(snapshot, timeout=timeout)
    return not not_done


def _run_with_timeout(fn: Callable[[], Any], timeout: float) -> tuple[bool, Any]:
    """Run ``fn()`` in a dedicated daemon thread, waiting up to ``timeout`` seconds.

    A fresh thread per call — not a shared pool — is deliberate. It means the
    timeout measures the check's *own* execution: there's no queue to wait in
    first, so a saturated pool can never make a fast check look like it timed
    out (which, with on_timeout='closed', would block real traffic). The
    number of check threads is then bounded by request concurrency, not a
    fixed worker count.

    Returns ``(finished, result)``: ``(True, value)`` when ``fn`` completed,
    ``(False, None)`` when it ran past the timeout. A timed-out check keeps
    running — a synchronous function can't be interrupted — but the thread is
    a daemon, so a check that hangs forever neither holds up interpreter exit
    nor permanently retires a pooled worker (both of which the old shared pool
    was prone to).
    """
    box: list[Any] = []
    thread = threading.Thread(target=lambda: box.append(fn()), name="promptkeep-check", daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        return False, None
    return True, box[0]


def suppressed() -> bool:
    """True while inside a check — the wrapper skips tracking for such calls."""
    return _suppressed.get()


class suppress:
    """Context manager that marks nested calls as check-internal (untracked)."""

    def __enter__(self):
        """Mark the current context as inside a check."""
        self._token = _suppressed.set(True)
        return self

    def __exit__(self, *exc):
        """Lift the mark, restoring whatever it was before."""
        _suppressed.reset(self._token)
        return False


# --- verdict -------------------------------------------------------------------


@dataclass(frozen=True)
class Verdict:
    """What a check decided. Build with the classmethods, not the constructor.

    status is one of 'ok' | 'warn' | 'block' | 'error'. score is an optional
    number (for threshold checks). rewritten is set only by rewrite().
    """

    status: str
    message: str | None = None
    score: float | None = None
    rewritten: str | None = None

    @classmethod
    def ok(cls, message: str | None = None) -> Verdict:
        """Pass — continue normally."""
        return cls("ok", message)

    @classmethod
    def warn(cls, message: str) -> Verdict:
        """Continue, but record a concern on the run."""
        return cls("warn", message)

    @classmethod
    def block(cls, message: str) -> Verdict:
        """Stop — do not call the provider (pre-checks only)."""
        return cls("block", message)

    @classmethod
    def rewrite(cls, text: str, message: str | None = None) -> Verdict:
        """Continue, but replace the newest outgoing message with ``text``
        (pre-checks only). Build ``text`` from ``ctx.last_text`` (the current
        turn), not ``ctx.rendered`` (every message joined) — the latter would
        fold the system prompt into the user turn.
        """
        return cls("ok", message, rewritten=text)

    @classmethod
    def from_score(
        cls, score: float, threshold: float = 0.5, message: str | None = None
    ) -> Verdict:
        """A numeric check: 'ok' at/above the threshold, 'warn' below it."""
        status = "ok" if score >= threshold else "warn"
        return cls(status, message, score=score)


# --- context passed to a check -------------------------------------------------


@dataclass(frozen=True)
class CheckContext:
    """What a check function receives.

    Pre-checks see the outgoing request (rendered, messages, prompt);
    post-checks additionally see the response (output_text, response). For
    a streamed call ``response`` is the stream's accumulated ResponseFields
    summary — there is no single provider object to hand over.
    """

    rendered: str
    messages: Any = None
    prompt: Any = None
    variables: dict[str, Any] | None = None
    model: str | None = None
    provider: str = "openai"
    output_text: str | None = None
    response: Any = None
    # The newest outgoing message's text (the current turn). This — not
    # `rendered`, which is every message joined for scanning — is what a
    # rewrite should be built from, since rewrite replaces exactly this.
    last_text: str | None = None
    # The current turn as the caller passed it, before any pre-check rewrote
    # it. None until a rewrite happens; from then on every later check (the
    # remaining pre-checks and the post-checks) can compare the two.
    original_text: str | None = None


# --- a registered check --------------------------------------------------------


@dataclass
class Check:
    """A check function plus how to run it. Build via check.pre / check.post."""

    fn: Callable[[CheckContext], Verdict]
    name: str
    phase: str  # 'pre' | 'post'
    mode: str = "blocking"  # post only: 'async' | 'blocking'
    timeout: float | None = 5.0
    on_timeout: str = "open"  # 'open' (continue) | 'closed' (block)

    def _execute(self, ctx: CheckContext) -> CheckResult:
        """Call the function under suppression, shielded. Never raises — a
        crash becomes an 'error' result rather than touching the caller."""
        start = time.perf_counter()
        try:
            with suppress():
                verdict = self.fn(ctx)
        except Exception as exc:
            latency = int((time.perf_counter() - start) * 1000)
            logger.warning(
                "promptkeep: check %r raised — recording error", self.name, exc_info=True
            )
            return CheckResult(self.name, self.phase, "error", None, repr(exc), latency)
        latency = int((time.perf_counter() - start) * 1000)
        if not isinstance(verdict, Verdict):
            return CheckResult(
                self.name, self.phase, "error", None, "check did not return a Verdict", latency
            )
        return CheckResult(
            self.name,
            self.phase,
            verdict.status,
            verdict.score,
            verdict.message,
            latency,
            rewritten=verdict.rewritten,
        )

    def run(self, ctx: CheckContext) -> CheckResult:
        """Hot-path execution with a timeout. On timeout, fail open (record a
        warning and continue) unless on_timeout='closed' on a pre-check, which
        blocks. A slow check must never become an outage.

        The timeout is enforced in a dedicated thread (not a shared pool), so
        it measures this check's execution rather than time spent queued behind
        other checks — under load a fast check no longer spuriously times out.
        """
        if self.timeout is None:
            return self._execute(ctx)
        start = time.perf_counter()
        finished, result = _run_with_timeout(lambda: self._execute(ctx), self.timeout)
        if finished:
            return result
        latency = int((time.perf_counter() - start) * 1000)
        msg = f"check timed out after {self.timeout}s"
        if self.on_timeout == "closed" and self.phase == "pre":
            return CheckResult(self.name, self.phase, "block", None, msg, latency)
        logger.warning("promptkeep: check %r timed out — failing open", self.name)
        return CheckResult(self.name, self.phase, "warn", None, msg, latency)

    def run_inline(self, ctx: CheckContext) -> CheckResult:
        """Execution without the timeout wrapper — for async post-checks, which
        already run off the caller's thread so no nested pool submit is needed."""
        return self._execute(ctx)


@dataclass
class CheckResult:
    """The outcome of one check run — mirrors a row in the checks table."""

    name: str
    phase: str
    status: str
    score: float | None = None
    message: str | None = None
    latency_ms: int | None = None
    rewritten: str | None = None

    def to_row(self) -> dict:
        """As a storage row. ``rewritten`` rides along so the audit trail names
        which check changed the turn and to what."""
        return {
            "name": self.name,
            "phase": self.phase,
            "status": self.status,
            "score": self.score,
            "message": self.message,
            "latency_ms": self.latency_ms,
            "rewritten": self.rewritten,
        }


# --- the check.pre / check.post decorators -------------------------------------


class _CheckFactory:
    """The `check` object: `@check.pre(...)` / `@check.post(...)` decorators."""

    def pre(
        self, name: str | None = None, timeout: float | None = 5.0, on_timeout: str = "open"
    ) -> Callable[[Callable], Check]:
        """Register a pre-check (a gate). Blocking by nature."""

        def decorate(fn: Callable[[CheckContext], Verdict]) -> Check:
            return Check(fn, name or fn.__name__, "pre", "blocking", timeout, on_timeout)

        return decorate

    def post(
        self,
        name: str | None = None,
        mode: str = "async",
        timeout: float | None = 5.0,
        on_timeout: str = "open",
    ) -> Callable[[Callable], Check]:
        """Register a post-check (an audit). Async by default (non-blocking)."""

        def decorate(fn: Callable[[CheckContext], Verdict]) -> Check:
            return Check(fn, name or fn.__name__, "post", mode, timeout, on_timeout)

        return decorate


check = _CheckFactory()


# --- running lists of checks ---------------------------------------------------


@dataclass
class PreOutcome:
    """Result of running all pre-checks: what to record, and whether to stop."""

    results: list[CheckResult] = field(default_factory=list)
    blocked: CheckResult | None = None
    rewritten: str | None = None


def run_pre_checks(checks: list[Check], ctx: CheckContext) -> PreOutcome:
    """Run pre-checks in order. Stop at the first block; apply rewrites."""
    outcome = PreOutcome()
    rendered = ctx.rendered
    for chk in checks:
        result = chk.run(ctx)
        outcome.results.append(result)
        if result.status == "block":
            outcome.blocked = result
            return outcome
        if result.rewritten is not None:
            outcome.rewritten = result.rewritten
            rendered = result.rewritten
            # Later checks see the rewritten text (as both the scan target and
            # the current turn — a rewrite replaces the newest message), and
            # keep sight of the original: the first rewrite pins it.
            ctx = replace(
                ctx,
                rendered=rendered,
                last_text=result.rewritten,
                original_text=ctx.original_text if ctx.original_text is not None else ctx.last_text,
            )
    return outcome


class PromptBlocked(Exception):
    """Raised when a pre-check blocks a call and on_block='raise' (the default)."""

    def __init__(self, check_name: str, message: str | None):
        """Carry the blocking check's name and its message."""
        self.check_name = check_name
        self.message = message
        super().__init__(f"blocked by check {check_name!r}: {message}")


# --- the handle attached to a response -----------------------------------------


class RunHandle:
    """Attached to a response as ``response.promptkeep``.

    Carries the run's key (its identity — pass it to ``history.checks()``),
    the prompt version that ran, and the check results. ``wait()`` blocks for
    async post-checks if you need their verdicts. run_key is None when the
    run was not recorded (tracking disabled, write_mode "off", or sampled out
    by ``sample_rate``); the checks still ran and the verdicts are still here.
    """

    def __init__(self, run_key, prompt_version, results, futures=None):
        """Bind the run's key, the prompt version that ran, the verdicts known
        so far, and the futures of any async post-checks still running."""
        self.run_key = run_key
        self.prompt_version = prompt_version
        self.checks = list(results)
        self._futures = futures or []

    @property
    def verification(self) -> str:
        """Aggregate verdict, in precedence order: 'failed' if anything
        blocked/errored, else 'pending' if async checks are still running
        (a later verdict could still fail), else 'warn' if any warned, else
        'ok'."""
        statuses = [c.status for c in self.checks]
        if any(s in ("block", "error") for s in statuses):
            return "failed"
        if self._pending():
            return "pending"
        return "warn" if any(s == "warn" for s in statuses) else "ok"

    def _pending(self) -> bool:
        return any(not f.done() for f in self._futures)

    def wait(self, timeout: float | None = None) -> RunHandle:
        """Block until async post-checks finish (or timeout), then return self."""
        for future in self._futures:
            try:
                result = future.result(timeout=timeout)
                if result is not None:
                    self.checks.append(result)
            except Exception:
                logger.warning("promptkeep: waiting on async check failed", exc_info=True)
        self._futures = []
        return self

    async def awaited(self, timeout: float | None = None) -> RunHandle:
        """Async twin of wait(): await the async post-checks off the event loop,
        then return self. Use from async code so waiting never blocks the loop."""
        import asyncio

        await asyncio.to_thread(self.wait, timeout)
        return self

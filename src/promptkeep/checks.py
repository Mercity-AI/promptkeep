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
"""

from __future__ import annotations

import contextvars
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

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
_executor: Optional[ThreadPoolExecutor] = None


def _get_executor() -> ThreadPoolExecutor:
    """Lazily create the shared async-post-check pool (never at import time)."""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="promptkeep-post")
    return _executor


def _run_with_timeout(fn: Callable[[], Any], timeout: float) -> Tuple[bool, Any]:
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
    box: List[Any] = []
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
        self._token = _suppressed.set(True)
        return self

    def __exit__(self, *exc):
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
    message: Optional[str] = None
    score: Optional[float] = None
    rewritten: Optional[str] = None

    @classmethod
    def ok(cls, message: Optional[str] = None) -> "Verdict":
        """Pass — continue normally."""
        return cls("ok", message)

    @classmethod
    def warn(cls, message: str) -> "Verdict":
        """Continue, but record a concern on the run."""
        return cls("warn", message)

    @classmethod
    def block(cls, message: str) -> "Verdict":
        """Stop — do not call the provider (pre-checks only)."""
        return cls("block", message)

    @classmethod
    def rewrite(cls, text: str, message: Optional[str] = None) -> "Verdict":
        """Continue, but replace the newest outgoing message with ``text``
        (pre-checks only). Build ``text`` from ``ctx.last_text`` (the current
        turn), not ``ctx.rendered`` (every message joined) — the latter would
        fold the system prompt into the user turn.
        """
        return cls("ok", message, rewritten=text)

    @classmethod
    def from_score(
        cls, score: float, threshold: float = 0.5, message: Optional[str] = None
    ) -> "Verdict":
        """A numeric check: 'ok' at/above the threshold, 'warn' below it."""
        status = "ok" if score >= threshold else "warn"
        return cls(status, message, score=score)


# --- context passed to a check -------------------------------------------------


@dataclass(frozen=True)
class CheckContext:
    """What a check function receives.

    Pre-checks see the outgoing request (rendered, messages, prompt);
    post-checks additionally see the response (output_text, response).
    """

    rendered: str
    messages: Any = None
    prompt: Any = None
    variables: Optional[Dict[str, Any]] = None
    model: Optional[str] = None
    provider: str = "openai"
    output_text: Optional[str] = None
    response: Any = None
    # The newest outgoing message's text (the current turn). This — not
    # `rendered`, which is every message joined for scanning — is what a
    # rewrite should be built from, since rewrite replaces exactly this.
    last_text: Optional[str] = None


# --- a registered check --------------------------------------------------------


@dataclass
class Check:
    """A check function plus how to run it. Build via check.pre / check.post."""

    fn: Callable[[CheckContext], Verdict]
    name: str
    phase: str  # 'pre' | 'post'
    mode: str = "blocking"  # post only: 'async' | 'blocking'
    timeout: Optional[float] = 5.0
    on_timeout: str = "open"  # 'open' (continue) | 'closed' (block)

    def _execute(self, ctx: CheckContext) -> "CheckResult":
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

    def run(self, ctx: CheckContext) -> "CheckResult":
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

    def run_inline(self, ctx: CheckContext) -> "CheckResult":
        """Execution without the timeout wrapper — for async post-checks, which
        already run off the caller's thread so no nested pool submit is needed."""
        return self._execute(ctx)


@dataclass
class CheckResult:
    """The outcome of one check run — mirrors a row in the checks table."""

    name: str
    phase: str
    status: str
    score: Optional[float] = None
    message: Optional[str] = None
    latency_ms: Optional[int] = None
    rewritten: Optional[str] = None

    def to_row(self) -> dict:
        """As a storage row (drops rewritten — it lives on the run, not here)."""
        return {
            "name": self.name,
            "phase": self.phase,
            "status": self.status,
            "score": self.score,
            "message": self.message,
            "latency_ms": self.latency_ms,
        }


# --- the check.pre / check.post decorators -------------------------------------


class _CheckFactory:
    """The `check` object: `@check.pre(...)` / `@check.post(...)` decorators."""

    def pre(
        self, name: Optional[str] = None, timeout: Optional[float] = 5.0, on_timeout: str = "open"
    ) -> Callable[[Callable], Check]:
        """Register a pre-check (a gate). Blocking by nature."""

        def decorate(fn: Callable[[CheckContext], Verdict]) -> Check:
            return Check(fn, name or fn.__name__, "pre", "blocking", timeout, on_timeout)

        return decorate

    def post(
        self,
        name: Optional[str] = None,
        mode: str = "async",
        timeout: Optional[float] = 5.0,
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

    results: List[CheckResult] = field(default_factory=list)
    blocked: Optional[CheckResult] = None
    rewritten: Optional[str] = None


def run_pre_checks(checks: List[Check], ctx: CheckContext) -> PreOutcome:
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
            # the current turn — a rewrite replaces the newest message).
            ctx = CheckContext(
                rendered=rendered,
                messages=ctx.messages,
                prompt=ctx.prompt,
                variables=ctx.variables,
                model=ctx.model,
                provider=ctx.provider,
                last_text=result.rewritten,
            )
    return outcome


class PromptBlocked(Exception):
    """Raised when a pre-check blocks a call and on_block='raise' (the default)."""

    def __init__(self, check_name: str, message: Optional[str]):
        self.check_name = check_name
        self.message = message
        super().__init__(f"blocked by check {check_name!r}: {message}")


# --- the handle attached to a response -----------------------------------------


class RunHandle:
    """Attached to a response as ``response.promptkeep``.

    Carries the run id, the prompt version that ran, and the check results.
    ``wait()`` blocks for async post-checks if you need their verdicts.
    """

    def __init__(self, run_id, prompt_version, results, futures=None):
        self.run_id = run_id
        self.prompt_version = prompt_version
        self.checks = list(results)
        self._futures = futures or []

    @property
    def verification(self) -> str:
        """Aggregate: 'failed' if anything blocked/errored, else 'warn' if any
        warning, 'pending' if async checks are still running, else 'ok'."""
        if self._pending():
            statuses = [c.status for c in self.checks]
            if any(s in ("block", "error") for s in statuses):
                return "failed"
            return "pending"
        statuses = [c.status for c in self.checks]
        if any(s in ("block", "error") for s in statuses):
            return "failed"
        if any(s == "warn" for s in statuses):
            return "warn"
        return "ok"

    def _pending(self) -> bool:
        return any(not f.done() for f in self._futures)

    def wait(self, timeout: Optional[float] = None) -> "RunHandle":
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

    async def awaited(self, timeout: Optional[float] = None) -> "RunHandle":
        """Async twin of wait(): await the async post-checks off the event loop,
        then return self. Use from async code so waiting never blocks the loop."""
        import asyncio

        await asyncio.to_thread(self.wait, timeout)
        return self


# --- the explicit call() shape -------------------------------------------------


@dataclass(frozen=True)
class CallResult:
    """What ``call()`` returns — the explicit shape, no attribute-poking.

    text is the model's reply; verification is the aggregate verdict; run_id
    and checks come from the same RunHandle the attach path exposes. response
    is the untouched provider object, still available if you need it.
    """

    text: Optional[str]
    verification: str
    run_id: Optional[int]
    checks: list
    response: Any


def _result_from_response(response) -> CallResult:
    """Build a CallResult from a (possibly promptkeep-annotated) response."""
    text = None
    choices = getattr(response, "choices", None)
    if choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            text = content
    handle = getattr(response, "promptkeep", None)
    if handle is not None:
        return CallResult(text, handle.verification, handle.run_id, handle.checks, response)
    return CallResult(text, "ok", None, [], response)


def call(client, **kwargs) -> CallResult:
    """Make a tracked, checked call and get a result object directly.

        result = promptkeep.call(client, model="gpt-5.5", messages=[...])
        result.text          # the reply
        result.verification  # "ok" | "warn" | "failed" | "pending"

    Same rows as the attach path — just a nicer shape for new code. Blocked
    calls raise PromptBlocked (or, under on_block="return", come back with
    verification="failed" and text=None). Non-streaming only.
    """
    response = client.chat.completions.create(**kwargs)
    return _result_from_response(response)


async def acall(client, **kwargs) -> CallResult:
    """Async twin of call(), for AsyncOpenAI clients."""
    response = await client.chat.completions.create(**kwargs)
    return _result_from_response(response)

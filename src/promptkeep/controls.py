"""The production controls: run sampling and the redaction hook.

Both are pure functions of the configured settings and a row about to be
stored. ``storage.record_run`` / ``record_check`` apply them at the one point
every run row and verdict passes through, before anything reaches a queue or
a disk — so the wrapper, the tracking helpers and direct storage calls are
all covered, and plaintext never sits in the background queue.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable
from typing import Any

from .config import Settings

# Run-row fields that can carry user data, and the check-row ones. Everything
# else on a row is an id, a number, a status or a timestamp.
RUN_TEXT_FIELDS = (
    "rendered_text",
    "variables",
    "request_params",
    "output_text",
    "error",
    "input_text",
    "original_input_text",
)
CHECK_TEXT_FIELDS = ("message", "rewritten")


def keep_run(settings: Settings, row: dict[str, Any], conversation_id: int | None) -> bool:
    """The sampling decision for one run.

    Anything worth investigating is always kept: a call that failed or was
    blocked, or one carrying a verdict that isn't plain ok (a warning
    included — ``Verdict.from_score`` below its threshold is a warn). Only an
    uneventful run is subject to ``sample_rate``.

    Inside a conversation the decision is derived from the conversation's id
    rather than drawn per turn, so one session is kept whole or dropped whole
    (no holes in a replay) and every process writing to the same file makes
    the same call. Async post-check verdicts land after this decision and
    can't rescue a dropped run — use ``mode="blocking"`` for a check whose
    failures must always be kept.
    """
    rate = settings.sample_rate
    if rate >= 1.0:
        return True

    # The always-keep rules: a problem is never sampled away.
    if row["status"] != "ok":
        return True
    if any(check.get("status") != "ok" for check in row.get("_checks", ())):
        return True

    # Uneventful: a coin flip per run, or per conversation for a turn.
    if conversation_id is None:
        return random.random() < rate
    digest = hashlib.sha256(str(conversation_id).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < rate


def redact_run(redact: Callable[[str], str], row: dict[str, Any]) -> dict[str, Any]:
    """A run row (and any verdicts bundled with it) with every text field
    passed through the hook. Raises if the hook does or returns a non-string
    — the caller turns that into "don't store"."""
    redacted = _redact_fields(redact, row, RUN_TEXT_FIELDS)
    if redacted.get("_checks"):
        redacted["_checks"] = [redact_check(redact, check) for check in redacted["_checks"]]
    return redacted


def redact_check(redact: Callable[[str], str], row: dict[str, Any]) -> dict[str, Any]:
    """A check verdict row with its message and rewrite passed through the hook."""
    return _redact_fields(redact, row, CHECK_TEXT_FIELDS)


def _redact_fields(
    redact: Callable[[str], str], row: dict[str, Any], fields: tuple[str, ...]
) -> dict[str, Any]:
    """A copy of ``row`` with each string-valued field in ``fields`` redacted."""
    redacted = dict(row)
    for field in fields:
        value = redacted.get(field)
        if isinstance(value, str):
            result = redact(value)
            if not isinstance(result, str):
                raise TypeError(f"redact hook must return a str, got {type(result).__name__}")
            redacted[field] = result
    return redacted

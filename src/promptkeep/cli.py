"""The `promptkeep` command-line entry point: the history, from a terminal.

    promptkeep list                              every prompt, with version and run counts
    promptkeep versions REVIEW_SYSTEM            a prompt's lineage
    promptkeep diff REVIEW_SYSTEM 4 5            what changed between two versions
    promptkeep runs REVIEW_SYSTEM --version 5    recorded calls, newest first
    promptkeep convo user-42-session-9           a conversation, turn by turn
    promptkeep stats REVIEW_SYSTEM               how each version has performed
    promptkeep export -o runs.jsonl              runs (and their labels) as JSON lines
    promptkeep serve                             the local dashboard

Every command but `serve` is a thin presentation of a ``history`` read —
nothing here queries the database itself, and nothing writes to it. Plain
text, no dependencies. `serve` lazily imports fastapi/uvicorn/jinja2: those
are an optional extra (`pip install promptkeep[serve]`), so importing
`promptkeep` itself never requires a server stack.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import history
from .config import configure, get_settings

# ANSI colours for `diff`, used only when writing to a terminal.
_RED, _GREEN, _CYAN, _RESET = "\033[31m", "\033[32m", "\033[36m", "\033[0m"


def main(argv: Sequence[str] | None = None) -> None:
    """Parse args and dispatch to the requested subcommand."""
    args = _build_parser().parse_args(argv)
    if args.db:
        configure(db_path=args.db)
    if args.command == "serve":
        _serve(args)
        return

    # A read command must never create the file it was asked to read: a
    # mistyped --db would otherwise leave an empty database behind.
    db_path = Path(get_settings().db_path)
    if not db_path.exists():
        raise SystemExit(f"promptkeep: no database at {db_path} (pass --db PATH)")
    try:
        _COMMANDS[args.command](args)
    except ValueError as exc:
        # history raises ValueError for "no such version / conversation".
        raise SystemExit(f"promptkeep: {exc}") from None


def _build_parser() -> argparse.ArgumentParser:
    """The argument parser: one subcommand per view, `--db` on every one."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--db", default=None, help="path to the promptkeep SQLite file (default: ./.promptkeep.db)"
    )
    parser = argparse.ArgumentParser(prog="promptkeep", description="Inspect a promptkeep history.")
    commands = parser.add_subparsers(dest="command", required=True)

    # The listings.
    commands.add_parser("list", parents=[common], help="every prompt, with version and run counts")
    versions = commands.add_parser("versions", parents=[common], help="a prompt's versions")
    versions.add_argument("name")
    versions.add_argument("--full", action="store_true", help="print each template in full")

    diff = commands.add_parser("diff", parents=[common], help="diff two versions of a prompt")
    diff.add_argument("name")
    diff.add_argument("old", type=int)
    diff.add_argument("new", type=int)
    diff.add_argument("--no-color", action="store_true", help="never colour the output")

    runs = commands.add_parser("runs", parents=[common], help="recorded runs, newest first")
    runs.add_argument("name", nargs="?", help="a prompt name (default: every run)")
    runs.add_argument("--version", type=int, default=None, help="only this version (needs a name)")
    runs.add_argument("--limit", type=int, default=20)

    # The detail views.
    convo = commands.add_parser("convo", parents=[common], help="a conversation, turn by turn")
    convo.add_argument("external_id")
    stats = commands.add_parser("stats", parents=[common], help="how each version has performed")
    stats.add_argument("name")

    export = commands.add_parser("export", parents=[common], help="runs and their labels, as JSONL")
    export.add_argument("--format", choices=["jsonl"], default="jsonl")
    export.add_argument("--prompt", default=None, help="only this prompt's runs")
    export.add_argument(
        "--version", type=int, default=None, help="only this version (needs --prompt)"
    )
    export.add_argument("--limit", type=int, default=None, help="newest N runs (default: all)")
    export.add_argument("-o", "--output", default=None, help="write to a file instead of stdout")

    serve = commands.add_parser("serve", parents=[common], help="run the local dashboard")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8420)
    return parser


# --- the commands --------------------------------------------------------------------


def _list(args: argparse.Namespace) -> None:
    """Every prompt with its version and run counts."""
    rows = [
        (p.name, p.version_count, p.run_count, _when(p.created_at)) for p in history.list_prompts()
    ]
    _table(("PROMPT", "VERSIONS", "RUNS", "CREATED"), rows, empty="no prompts recorded yet")


def _versions(args: argparse.Namespace) -> None:
    """A prompt's lineage: one line per version, or each template in full."""
    found = _require(history.versions(args.name), f"no prompt named {args.name!r}")
    if not args.full:
        rows = [
            (f"v{v.version}", v.source, _when(v.created_at), _one_line(v.template)) for v in found
        ]
        _table(("VERSION", "SOURCE", "CREATED", "TEMPLATE"), rows)
        return
    for v in found:
        print(f"--- {args.name} v{v.version} · {v.source} · {_when(v.created_at)}")
        print(v.template)
        print()


def _diff(args: argparse.Namespace) -> None:
    """Unified diff between two versions, coloured on a terminal."""
    text = history.diff(args.name, args.old, args.new)
    if not text:
        print(f"{args.name} v{args.old} and v{args.new} are identical")
        return
    colour = sys.stdout.isatty() and not args.no_color and "NO_COLOR" not in os.environ
    for line in text.splitlines():
        print(_coloured(line) if colour else line)


def _runs(args: argparse.Namespace) -> None:
    """Recorded runs, newest first — for one prompt or for everything."""
    if args.version is not None and args.name is None:
        raise SystemExit("promptkeep: --version needs a prompt name")
    if args.name is None:
        found = history.all_runs(limit=args.limit)
    else:
        found = history.runs(args.name, version=args.version, limit=args.limit)
    rows = [
        (
            r.run_key[:8],
            r.prompt_name or "—",
            f"v{r.version}" if r.version is not None else "—",
            r.model or "—",
            _number(r.total_tokens),
            history.format_cost(r.cost_usd),
            _number(r.latency_ms, "ms"),
            r.status,
            _when(r.created_at),
        )
        for r in found
    ]
    headers = ("RUN", "PROMPT", "VER", "MODEL", "TOKENS", "COST", "LATENCY", "STATUS", "WHEN")
    _table(headers, rows, right={4, 5, 6}, empty="no runs recorded")


def _convo(args: argparse.Namespace) -> None:
    """A conversation's transcript: a header, then each turn with its labels."""
    convo = history.conversation(args.external_id)
    used = ", ".join(
        f"{name} {'/'.join(f'v{v}' for v in versions)}"
        for name, versions in convo.versions_used.items()
    )
    turn_count = len({turn.turn_index for turn in convo.turns})
    print(convo.title or convo.external_id)
    print(
        f"{turn_count} turn{'s' if turn_count != 1 else ''} · {convo.total_tokens} tokens · "
        f"{history.format_cost(convo.total_cost)} · {convo.duration:.1f}s"
        + (f" · {used}" if used else "")
    )
    forks = convo.forks
    for turn in convo.turns:
        # One block per row: who drove it, what went in, what came out.
        lineage = f" · {turn.prompt_name} v{turn.version}" if turn.prompt_name else ""
        print(f"\n#{turn.turn_index} · {turn.status}{lineage} · {turn.model or '—'}")
        if turn.turn_index in forks:
            origin = forks[turn.turn_index]
            print(f"  ↳ branches from {'another conversation' if origin is None else f'#{origin}'}")
        if turn.original_input_text is not None:
            print(f"  user (as received): {turn.original_input_text}")
        if turn.input_text is not None:
            print(f"  user: {turn.input_text}")
        if turn.output_text is not None:
            print(f"  assistant: {turn.output_text}")
        if turn.error:
            print(f"  error: {turn.error}")
        for label in history.checks(turn.run_key):
            score = f" ({label.score:g})" if label.score is not None else ""
            status = "" if label.phase == "feedback" else f": {label.status}"
            print(f"  [{label.phase}] {label.name}{status}{score}")


def _stats(args: argparse.Namespace) -> None:
    """How each version of a prompt has performed, side by side."""
    found = _require(history.stats(args.name), f"no prompt named {args.name!r}")
    rows = [
        (
            f"v{s.version}",
            s.runs,
            s.errors,
            s.blocked,
            _percent(s.check_pass_rate),
            _number(s.avg_score, digits=2),
            _number(s.avg_feedback, digits=2),
            _number(s.avg_latency_ms, "ms"),
            _number(s.avg_tokens),
            history.format_cost(s.total_cost),
        )
        for s in found
    ]
    headers = ("VER", "RUNS", "ERRORS", "BLOCKED", "CHECKS OK", "SCORE", "FEEDBACK")
    _table(headers + ("LATENCY", "TOKENS", "COST"), rows, right=set(range(1, 10)))


def _export(args: argparse.Namespace) -> None:
    """Runs as JSON lines, newest first, each with the labels recorded on it
    (check verdicts and feedback) — the raw material for an eval set."""
    if args.version is not None and args.prompt is None:
        raise SystemExit("promptkeep: --version needs --prompt")
    if args.prompt is None:
        found = history.all_runs(limit=args.limit)
    else:
        found = history.runs(args.prompt, version=args.version, limit=args.limit)

    # One object per line; a file when asked for one, else stdout.
    by_run = history.labels([run.run_key for run in found])
    out = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    try:
        for run in found:
            record = {**asdict(run), "checks": [asdict(c) for c in by_run[run.run_key]]}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        if args.output:
            out.close()
    if args.output:
        print(f"wrote {len(found)} runs to {args.output}", file=sys.stderr)


def _serve(args: argparse.Namespace) -> None:  # pragma: no cover - starts a server
    """Launch the dashboard, pointed at the given (or default-configured) DB."""
    try:
        import uvicorn
    except ImportError:
        raise SystemExit(
            "The dashboard needs extra dependencies. Install with:\n"
            "    pip install 'promptkeep[serve]'"
        ) from None

    from .dashboard.app import create_app

    app = create_app()
    print(f"Serving promptkeep dashboard at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


_COMMANDS = {
    "list": _list,
    "versions": _versions,
    "diff": _diff,
    "runs": _runs,
    "convo": _convo,
    "stats": _stats,
    "export": _export,
}


# --- presentation helpers --------------------------------------------------------------


def _require(found: list[Any], message: str) -> list[Any]:
    """``found``, or exit with ``message`` when the read came back empty."""
    if not found:
        raise SystemExit(f"promptkeep: {message}")
    return found


def _table(
    headers: tuple[str, ...],
    rows: list[tuple[Any, ...]],
    right: set[int] | None = None,
    empty: str = "nothing to show",
) -> None:
    """Print rows as an aligned text table; ``right`` names the columns to
    right-align (numbers). Widths come from the content — no truncation, so
    the output stays greppable and pipeable."""
    if not rows:
        print(empty)
        return
    right = right or set()
    cells = [headers] + [tuple(str(value) for value in row) for row in rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(headers))]
    for row in cells:
        line = "  ".join(
            value.rjust(widths[i]) if i in right else value.ljust(widths[i])
            for i, value in enumerate(row)
        )
        print(line.rstrip())


def _when(timestamp: str) -> str:
    """A stored ISO timestamp, trimmed to the minute for a table column."""
    return timestamp[:16].replace("T", " ")


def _one_line(template: str, width: int = 60) -> str:
    """A template's first line, shortened to fit a table column."""
    first = template.strip().splitlines()[0] if template.strip() else ""
    return first if len(first) <= width else first[: width - 1] + "…"


def _number(value: float | None, unit: str = "", digits: int = 0) -> str:
    """A number for a table, or a dash when nothing was reported."""
    return "—" if value is None else f"{value:.{digits}f}{unit}"


def _percent(share: float | None) -> str:
    """A 0..1 share as a percentage, or a dash when there is nothing to share."""
    return "—" if share is None else f"{share:.0%}"


def _coloured(line: str) -> str:
    """One unified-diff line in its conventional colour."""
    if line.startswith(("+++", "---")):
        return line
    if line.startswith("@@"):
        return f"{_CYAN}{line}{_RESET}"
    if line.startswith("+"):
        return f"{_GREEN}{line}{_RESET}"
    if line.startswith("-"):
        return f"{_RED}{line}{_RESET}"
    return line


if __name__ == "__main__":
    main()

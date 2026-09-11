"""FastAPI app for `promptkeep serve`: a read-only, offline view onto the
same SQLite file `history`/`storage` already know how to read.

Nothing here writes to the database — this module only renders what
tracking already recorded. Imported lazily by `promptkeep.cli`, never from
`promptkeep/__init__.py`, so the core package has no server dependency.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from .. import history

TEMPLATES_DIR = Path(__file__).parent / "templates"


def create_app() -> FastAPI:
    """Build the dashboard app. Call after promptkeep.configure(), if needed —
    routes read through history/storage at request time, not at import time."""
    app = FastAPI(title="promptkeep")
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    @app.get("/")
    def root():
        """Land on the prompts overview."""
        return RedirectResponse(url="/prompts")

    @app.get("/prompts")
    def prompts_page(request: Request):
        """Every prompt with its version/run counts."""
        return templates.TemplateResponse(
            request, "prompts.html", {"active": "prompts", "prompts": history.list_prompts()}
        )

    @app.get("/prompts/{name}")
    def prompt_detail(request: Request, name: str):
        """One prompt's version lineage."""
        versions = history.versions(name)
        if not versions:
            raise HTTPException(404, f"no prompt named {name!r}")
        return templates.TemplateResponse(
            request,
            "prompt_detail.html",
            {"active": "prompts", "name": name, "versions": versions},
        )

    @app.get("/prompts/{name}/diff")
    def prompt_diff(request: Request, name: str, old: int, new: int):
        """Unified diff between two versions of one prompt."""
        try:
            diff_text = history.diff(name, old, new)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc
        return templates.TemplateResponse(
            request,
            "diff.html",
            {"active": "prompts", "name": name, "old": old, "new": new, "diff_text": diff_text},
        )

    @app.get("/runs")
    def runs_page(
        request: Request,
        prompt: str | None = None,
        version: str | None = None,
        limit: int = 100,
    ):
        """All runs, or one prompt's (optionally one version's), newest first.

        version is declared str, not int: an HTML form with an empty number
        box submits `version=`, and FastAPI would reject "" as a 422 before
        the handler ever ran — which read as "the filter is broken". Blank
        or non-numeric input just means "no version filter".
        """
        try:
            version_number = int(version) if version else None
        except ValueError:
            version_number = None
        if prompt:
            runs = history.runs(prompt, version=version_number, limit=limit)
        else:
            runs = history.all_runs(limit=limit)
        # A run's headline verdict for the badge column: run_key -> "ok" |
        # "warn" | "blocked" | "failed" | None (no checks). Cheap per-run
        # lookups — fine for a local single-user dashboard.
        verdicts = {r.run_key: history.verdict(r.status, history.checks(r.run_key)) for r in runs}
        return templates.TemplateResponse(
            request,
            "runs.html",
            {
                "active": "runs",
                "prompts": history.list_prompts(),
                "selected_prompt": prompt,
                "selected_version": version,
                "runs": runs,
                "verdicts": verdicts,
            },
        )

    @app.get("/conversations")
    def conversations_page(request: Request, prompt: str | None = None, version: str | None = None):
        """Every recorded conversation, most recently active first — or only
        those a given prompt (version) drove. version is a str for the same
        reason as on /runs: an empty form box must mean "no filter"."""
        try:
            version_number = int(version) if version else None
        except ValueError:
            version_number = None
        conversations = history.list_conversations(
            prompt=prompt or None, version=version_number if prompt else None
        )
        return templates.TemplateResponse(
            request,
            "conversations.html",
            {
                "active": "conversations",
                "prompts": history.list_prompts(),
                "selected_prompt": prompt,
                "selected_version": version,
                "conversations": conversations,
            },
        )

    @app.get("/conversations/{external_id}")
    def conversation_detail(request: Request, external_id: str):
        """One conversation's full transcript, turn by turn, with each turn's
        check verdicts shown inline."""
        try:
            convo = history.conversation(external_id)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc
        # run_key -> that turn's check verdicts, for the transcript.
        turn_checks = {t.run_key: history.checks(t.run_key) for t in convo.turns}
        return templates.TemplateResponse(
            request,
            "conversation_detail.html",
            {"active": "conversations", "convo": convo, "turn_checks": turn_checks},
        )

    return app

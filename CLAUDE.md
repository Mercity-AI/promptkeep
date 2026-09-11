# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                                          # install (editable) + dev deps
uv run pytest -q                                 # full test suite
uv run pytest tests/test_storage.py -q           # one file
uv run pytest tests/test_prompt.py::TestVersioning::test_changed_text_bumps_version  # one test
uv run pytest -q --cov --cov-fail-under=85       # what the CI coverage gate runs
uv run ruff format src tests examples && uv run ruff check src tests examples   # line-length 100
uv run python examples/playground.py             # narrated sandbox (throwaway DB, no network)
uv build                                         # build sdist+wheel into dist/
```

**There is no CI yet.** The workflows are written (`docs/workflows/ci.yml`, `release.yml`) but
not enabled — see `TODO.md` for why and how. Run the lint + coverage gate above locally before
pushing. Releases, for now: bump `version` in `pyproject.toml` (the only place it lives —
`__version__` reads it back from package metadata), add the `CHANGELOG.md` entry, tag
`vX.Y.Z`, then `uv build && uv publish --token pypi-...`.

Published on PyPI as `promptkeep`; GitHub remote is `Mercity-AI/promptkeep`. The directory is
still named `prompt-manager` — everything inside uses `promptkeep`. `plan.md` is the original
design doc, kept as history; don't update it to match code changes.

## Architecture

Three-entity data model, strictly layered: **Prompt** (`name` = permanent identity) →
**Version** (one template text under a name, deduplicated by sha256 of the *normalized*
template, numbered sequentially) → **Run** (one LLM execution: variables + rendered text +
response metadata). Variables are run data, never version identity — changing variables must
never create a version. Normalization (`rendering.normalize_template`) canonicalizes
placeholder names to `{v0}`, `{v1}`, ... so renaming `{var1}` → `{x}` dedups to the same
version; static text, repetition patterns, and format specs still distinguish versions.

Flow between modules: `prompts.Prompt.render()` → lazily registers its template via
`storage.register_version()` (memoized per object *and* per process) → integrations wrapper
intercepts `chat.completions.create` → `tracking.record_prompt_run()` → `storage.record_run()`.
`history.py` is the read side, turning storage's dict rows into frozen dataclasses.

Load-bearing design decisions (breaking these breaks the library's contract):

- **`RenderedText` is a `str` subclass carrying `_pm_prompt`/`_pm_variables`.** This is how
  `prompt.text` stays a plain string for unwrapped SDKs while the wrapped client can still
  trace it back to its Prompt for run tracking. `Prompt` itself is deliberately NOT a str
  subclass.
- **Prompt is frozen** (`__slots__` + blocked `__setattr__`); `.format()` derives a new one.
  Immutability is what keeps an object consistent with the version hash it registered under.
- **Version registration is lazy** — first `.text`/`.render()`/`.version` access, never at
  construction. Prompts are defined at module import time; import must not do I/O.
- **All implicit write paths are exception-shielded** (`storage.register_version`,
  `storage.record_run`, `tracking`, `writer`): a broken DB logs a warning and loses telemetry,
  it must never raise into the user's request. History reads raise normally.
- **Run writes are asynchronous by default** (`write_mode="background"`): `record_run` builds
  the complete row (timestamps/turn numbers reflect call time) and hands it to `writer.py` —
  a bounded queue drained by one lazy-started daemon thread that batches rows per transaction.
  Overflow drops oldest and counts; `promptkeep.flush()` + an atexit hook drain it. Version
  registration stays synchronous (`.version` is a read-back value). Turn numbers come from
  in-process counters (`storage.reserve_turn_index`, reserve semantics — calling it claims the
  turn), because the DB's `MAX(turn_index)` is stale while rows sit in the queue. Tests run
  `write_mode="sync"` via the conftest fixture.
- **Runs are identified by a client-minted `run_key`** (`storage.new_run_key`, a UUID), never
  by the integer row id. That is what lets checked calls use the background writer too: the
  `RunHandle` and any late verdict name the run before its row exists. Check verdicts travel
  the same queue (`_kind: "check"` items, resolved to the run by key at write time; a verdict
  whose run was evicted is skipped, not raised). `promptkeep.flush()` lives in `tracking.py`:
  it waits for in-flight async post-checks (`checks.wait_for_pending`), then `writer.drain()`.
  A pre-check rewrite stores both sides: `runs.original_input_text` and `checks.rewritten`.
- **The wrapper never monkey-patches the `openai` module** — only the object passed to
  `wrap()` gets its `create` replaced (idempotent via `_pm_instrumented`). Message dicts are
  copied, never mutated. Streaming defers run recording until the stream ends
  (`_StreamRecorder.finish()` is write-once).
- **Rendering is lenient by default** (`rendering.py`): unknown `{placeholders}` and JSON
  braces pass through literally; unparseable templates return unrendered. Strict mode is
  opt-in per Prompt or via `configure(strict=True)`.
- **Checks** (`checks.py`) are user functions returning a `Verdict`. `pre` checks gate the
  request (block/rewrite/warn/ok) before the provider call; `post` checks audit the response
  (async by default via a shared `ThreadPoolExecutor`, blocking opt-in). Attach at three
  scopes merged most-specific-wins: global (`configure(pre=/post=)`) < prompt (`Prompt(pre=/
  post=)`) < per-call (`promptkeep_pre=/promptkeep_post=` kwargs, stripped before the request).
  A checked call records its run through the normal write path (background queue included):
  the `RunHandle` (`response.promptkeep`) carries the client-minted `run_key`, pre +
  blocking-post verdicts bundle into the run row, and async-post verdicts go through
  `storage.record_check` when they land — queued behind their run in background mode. Two invariants: a crashing/timing-out check never
  breaks the call (recorded `status="error"`/fail-open `warn`; `on_timeout="closed"` opts a
  pre-check into fail-closed), and check execution runs under `checks.suppress()` — a
  contextvar the wrapper honors to make an LLM-judge check's own calls untracked, so it can't
  recurse. Checks run on all four paths — sync/async × non-streaming/streaming; the async and
  streaming orchestrators share the same pure-sync phase helpers (`_checked_pre/_block/_post`),
  the async ones running each phase via `asyncio.to_thread` so a slow check never blocks the
  loop, and streaming running post-checks in `_CheckedStreamRecorder.finish()` (output only
  exists once the stream drains) into a `RunHandle` already attached to the proxy.
  `promptkeep.call()`/`acall()` return the explicit `CallResult` shape; `RunHandle.awaited()`
  is the async twin of `wait()`. Schema v4 added the `checks` table (the label store for later
  optimization); v5 added `run_key`, `original_input_text`, and `checks.rewritten`.

## SQLite/peewee specifics

- Models bind to a `DatabaseProxy`; `storage._get_db()` initializes it from config under a
  lock (fresh-file WAL switch races otherwise — this was a real test failure, twice).
- Registration uses `atomic("IMMEDIATE")` + retry on IntegrityError/OperationalError:
  peewee's default deferred transaction reads before writing, and SQLite refuses to wait on
  read→write lock upgrades. Don't "simplify" this back to plain `atomic()`.
- Schema changes: bump `_SCHEMA_VERSION`, add a forward-only step in `_migrate()` using
  `playhouse.migrate` operations. The DB tracks its schema in `PRAGMA user_version`.
- `config.py` resolves settings fresh on every call: `configure()` overrides >
  `PROMPTKEEP_DB`/`PROMPTKEEP_DISABLED`/`PROMPTKEEP_WRITE_MODE`/`PROMPTKEEP_SAMPLE_RATE` env
  vars > defaults (`./.promptkeep.db`, enabled, background, keep everything).
- **Sampling and redaction live in `storage.record_run`/`record_check`**, the one point every
  run row and verdict passes through, so they cover the wrapper, `tracking`, and direct
  storage calls alike. Sampling always keeps non-ok runs/verdicts and decides per
  conversation from its row id; a failing `redact` hook drops the row rather than storing it
  unredacted. Templates are never redacted (they're code, and the version hash depends on
  them).

## Tests

`tests/conftest.py` has an autouse fixture giving every test a fresh tmp DB and reset config —
tests never touch a real `.promptkeep.db`. OpenAI wrapper tests run against hand-rolled fakes
in `tests/fakes.py` (no network, no `openai` dependency; core must never import `openai`).
`test_storage.py::test_concurrent_registration_from_threads` is the canary for the SQLite
locking subtleties above — if a storage change makes it flaky, the change is wrong, not the
test. The module for the Prompt class is `prompts.py` (plural) because the public `prompt`
decorator would collide with a `prompt.py` submodule name in the package namespace.

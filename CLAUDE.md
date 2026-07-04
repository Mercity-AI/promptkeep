# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                                          # install (editable) + dev deps
uv run pytest -q                                 # full test suite
uv run pytest tests/test_storage.py -q           # one file
uv run pytest tests/test_prompt.py::TestVersioning::test_changed_text_bumps_version  # one test
uv run ruff format src tests && uv run ruff check src tests   # format + lint (line-length 100)
uv build                                         # build sdist+wheel into dist/
uv publish --token pypi-...                      # release (bump version in pyproject.toml first)
```

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
  `storage.record_run`, `tracking`): a broken DB logs a warning and loses telemetry, it must
  never raise into the user's request. History reads raise normally.
- **The wrapper never monkey-patches the `openai` module** — only the object passed to
  `wrap()` gets its `create` replaced (idempotent via `_pm_instrumented`). Message dicts are
  copied, never mutated. Streaming defers run recording until the stream ends
  (`_StreamRecorder.finish()` is write-once).
- **Rendering is lenient by default** (`rendering.py`): unknown `{placeholders}` and JSON
  braces pass through literally; unparseable templates return unrendered. Strict mode is
  opt-in per Prompt or via `configure(strict=True)`.

## SQLite/peewee specifics

- Models bind to a `DatabaseProxy`; `storage._get_db()` initializes it from config under a
  lock (fresh-file WAL switch races otherwise — this was a real test failure, twice).
- Registration uses `atomic("IMMEDIATE")` + retry on IntegrityError/OperationalError:
  peewee's default deferred transaction reads before writing, and SQLite refuses to wait on
  read→write lock upgrades. Don't "simplify" this back to plain `atomic()`.
- Schema changes: bump `_SCHEMA_VERSION`, add a forward-only step in `_migrate()` using
  `playhouse.migrate` operations. The DB tracks its schema in `PRAGMA user_version`.
- `config.py` resolves settings fresh on every call: `configure()` overrides >
  `PROMPTKEEP_DB`/`PROMPTKEEP_DISABLED` env vars > defaults (`./.promptkeep.db`, enabled).

## Tests

`tests/conftest.py` has an autouse fixture giving every test a fresh tmp DB and reset config —
tests never touch a real `.promptkeep.db`. OpenAI wrapper tests run against hand-rolled fakes
in `tests/fakes.py` (no network, no `openai` dependency; core must never import `openai`).
`test_storage.py::test_concurrent_registration_from_threads` is the canary for the SQLite
locking subtleties above — if a storage change makes it flaky, the change is wrong, not the
test. The module for the Prompt class is `prompts.py` (plural) because the public `prompt`
decorator would collide with a `prompt.py` submodule name in the package namespace.

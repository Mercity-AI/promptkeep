# AGENTS.md

Guidance for AI coding agents (Claude Code, Codex, Cursor, and the like) working in this
repository. `CLAUDE.md` imports this file; edit here, not there.

## Commands

```bash
uv sync                                          # install (editable) + dev deps
uv run pytest -q                                 # full test suite
uv run pytest tests/test_storage.py -q           # one file
uv run pytest tests/test_prompt.py::TestVersioning::test_changed_text_bumps_version  # one test
uv run pytest -q --cov --cov-fail-under=85       # what the CI coverage gate runs
uv run ruff format src tests examples && uv run ruff check src tests examples   # line-length 100
uv run promptkeep stats NAME --db demo.promptkeep.db   # the CLI (seed one: examples/seed_demo.py)
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

## Code style

Beyond ruff (line length 100, isort, pyupgrade to 3.11), the codebase follows two conventions
that ruff does not enforce — keep them when adding or editing code:

- **Logical blocks.** A function body is written as a sequence of steps. Each step — a loop, a
  query, a branch that does one thing — is its own block, separated from the next by a blank
  line and introduced by a one-line comment saying what that block does (or why). A reader
  should be able to skim the comments alone and get the algorithm. Don't comment single
  obvious lines; do comment the block.
- **Docstrings on everything.** Every module, class, function and method has one, including
  private helpers and dunder methods. Say what it does and, where it isn't obvious, why it is
  that way — the docstrings are where the design reasoning lives (see `storage._register`,
  `writer`, `controls.keep_run`). A one-liner is fine when that is all there is to say.

The public API is written down in `docs/API.md` and pinned by
`tests/test_package.py::test_the_public_api_changes_only_on_purpose`. Adding a public name
means updating that test and the CHANGELOG; removing or renaming one goes through a
`DeprecationWarning` for at least one minor release first.

Module graph is a DAG: no function-level intra-package imports (`tests/test_package.py`
enforces it; `cli.py`'s lazy dashboard import is the one exception). If an import cycle appears,
one dependency is pointing the wrong way — fix that rather than deferring the import.

## Architecture

Three-entity data model, strictly layered: **Prompt** (`name` = permanent identity) →
**Version** (one template text under a name, deduplicated by sha256 of the *normalized*
template, numbered sequentially) → **Run** (one LLM execution: variables + rendered text +
response metadata). Variables are run data, never version identity — changing variables must
never create a version. Normalization (`rendering.normalize_template`) canonicalizes
placeholder names to `{v0}`, `{v1}`, ... so renaming `{var1}` → `{x}` dedups to the same
version; static text, repetition patterns, and format specs still distinguish versions.

Modules, bottom up: `rendering`, `conversation`, `config` (no package imports) → `writer`
(the queue; its sink is injected by storage) → `models` (peewee tables) → `migrations` →
`controls` (sampling, redaction) → `storage` (the connection and every write) → `prompts` →
`history` (the read side: queries → frozen dataclasses) → `datasets` (history filtered
and exported to JSONL / DSPy / promptfoo) → `checks` → `tracking` →
`integrations/` (`base` = the adapter interface, `core` = the shared interceptor built around
one `_Call` object per call, `registry` = `wrap()` and the adapter list, `call` = the explicit
`call()` shape, `openai_wrapper` = the one adapter).

Flow: `prompts.Prompt.render()` → lazily registers its template via
`storage.register_version()` (memoized per object *and* per process) → the interceptor
`wrap()` installed on the client → `tracking.record_prompt_run()` → `storage.record_run()`
(sampling, redaction, then the writer queue or a direct insert).

Load-bearing design decisions (breaking these breaks the library's contract):

- **`RenderedText` is a `str` subclass carrying `_pm_prompt`/`_pm_variables`.** This is how
  `prompt.text` stays a plain string for unwrapped SDKs while the wrapped client can still
  trace it back to its Prompt for run tracking. `Prompt` itself is deliberately NOT a str
  subclass.
- **Prompt is frozen** (`__slots__` + blocked `__setattr__`); `.format()` derives a new one.
  Immutability is what keeps an object consistent with the version hash it registered under.
- **`Prompt.variants(name)` rebuilds stored versions as Prompts** with `_registration`
  pre-seeded from the row, so a variant is bound to the version it was loaded from. It must
  reproduce the identity the version was hashed under — `exact_match` is inferred by
  comparing the stored hash with the normalized one — or loading would mint a new version.
  The lineage query lives in `storage.version_rows` (shared with `history.versions`) because
  `prompts` sits below `history` in the module graph.
- **Sync or async is decided on the unwrapped method** (`core._instrument_target` uses
  `inspect.unwrap`), because SDKs decorate: `iscoroutinefunction` is False for the real
  `AsyncOpenAI.chat.completions.create`. The sync interceptor also hands any awaitable result
  to `_settle`, the shared async tail, so an undetectable async method still records after the
  await instead of storing an empty run.
- **A Prompt's registration memo is `(db_path, result)`.** Prompts are module-level objects and
  outlive `configure(db_path=...)`; a bare version id would be meaningless (or, worse, valid)
  in another file. `.format()` and `variants()` carry/seed the memo in the same shape.
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
- **Provider knowledge lives only in adapters** (`integrations/base.ProviderAdapter`): where the
  call method lives (`locate`), where Prompts hide in a request and which text is the current
  turn (`parse_request`, `apply_rewrite`), how to read a response and fold a stream
  (`read_response`, `stream_absorber`), the `on_block="return"` stub. Everything else —
  conversations, checks, recording, the stream proxies — is written once in
  `integrations/core.py` and must stay provider-agnostic; if core needs to know a provider's
  shape, that's a new adapter method, not an `if provider ==`. Adapters are registered in
  `integrations/__init__.py` (`register_adapter()` for third parties); `wrap()` instruments
  every surface any adapter locates. Core shields adapter calls made inside the request path
  (`read_response`, the absorber), so an adapter bug loses telemetry, never a call.
  `tests/test_adapters.py` is the contract every adapter must pass — a new provider adds one
  `Scenario` there. Two adapters today, both on the OpenAI client: `chat.completions`
  (`openai_wrapper.py`) and the Responses API (`openai_responses.py`, which reuses the chat
  adapter's message helpers — input items are message-shaped). `accepts(kwargs)` tells the
  two surfaces apart for `call()`.
- **Conversations are never inferred — but a request may name its predecessor.**
  `ProviderAdapter.conversation_hint()` returns the response id a call continues (Responses'
  `previous_response_id`); `storage.chain_conversation()` files the call in the conversation
  of the run that produced it, or — for a chain's second call — starts `response:<id>` and
  adopts the first run as turn 0. The predecessor may still be in the writer queue, so the
  lookup goes through an in-process `response_id → conversation` index before the DB, the
  adoption is a queue item (`_kind: "adopt"`, behind its run, like late verdicts), and the
  turn counter is seeded past the adopted turn rather than read from `MAX(turn_index)`.
  Explicit conversations win; a predecessor promptkeep never recorded means no chaining.
- **A conversation is a tree; NULL `parent_run_key` means "continues the turn before".**
  A run names its parent by key (`promptkeep_parent=`, or automatically from the chain
  lookup above — the index maps a response id to its *primary* run's key), never by row id,
  because the parent may still be queued. `history.ConversationInfo` derives everything from
  that rule (`_predecessors`): `replay()` follows the branch ending at the latest turn, so a
  linear conversation reads exactly as it always did. A parent that isn't in the
  conversation (or was pruned) starts a branch; readers never assume it exists.
- **Feedback is a row in `checks`, not a table of its own.** `tracking.feedback(run_key, ...)`
  writes `phase="feedback"` (`name` = the label, `message` = the comment, `status="ok"`)
  through `storage.record_check`, so it inherits the late-verdict machinery: queue ordering
  behind its run, redaction, skip-with-a-warning when the run never landed. Anything that
  reads verdicts as pass/fail (`history.verdict`, stats) must filter that phase out. Its
  prerequisite: every call that records — checked or not — mints its `run_key` up front and
  gets a `RunHandle` on the response.
- **Cost is reported, never estimated.** `runs.cost_usd` (schema v6) holds what the provider
  itself said the call cost — adapters read it off the usage block (`ResponseFields.cost_usd`;
  OpenRouter sends `usage.cost` on every response and on a stream's last chunk). A provider
  that reports nothing leaves it NULL. Don't add a bundled price table: it goes stale, and
  "unknown" is a better answer than a wrong number.
- **The wrapper never monkey-patches a provider module** — only the object passed to
  `wrap()` gets its method replaced (idempotent via `_pm_instrumented` on the method's owner).
  Message dicts are copied, never mutated. Streaming defers run recording until the stream
  ends (`_StreamRecorder.finish()` is write-once).
- **The CLI presents, `history` reads.** Every `cli.py` command except `serve` formats the
  result of a `history` function; a command that needs new data gets a new read in
  `history` (as `stats` did), never a query in `cli.py`. Output is plain aligned text with no
  truncation, so it pipes and greps. Read commands refuse a missing DB file rather than
  letting `get_db()` create an empty one.
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
- **Retention rides the write path too.** `record_run` schedules the sweep
  (`storage._schedule_prune`: first run, then hourly per process), which in background mode
  is a `_kind: "prune"` queue item run *after* its batch's transaction, never inside it.
  `prune()` deletes conversations whole by `updated_at` and other runs by `created_at`, in
  chunked IMMEDIATE transactions, and evicts deleted conversation ids from the in-process
  caches. Because a queued or another-process-cached turn can outlive its conversation,
  `_insert_run` catches the FK failure and records the turn unattached. Never delete
  prompts or versions.

## Tests

`tests/conftest.py` has an autouse fixture giving every test a fresh tmp DB and reset config —
tests never touch a real `.promptkeep.db`. OpenAI wrapper tests run against hand-rolled fakes
in `tests/fakes.py` (no network, no `openai` dependency; core must never import `openai`).
**The fakes are our belief about the SDK; `tests/test_real_sdk.py` is the SDK.** It drives the
real `openai` clients (in the dev group — the package still doesn't depend on it) over an httpx
`MockTransport`, and it exists because a fake can't catch a wrong assumption about what it
imitates: the real `AsyncOpenAI.chat.completions.create` is an `async def` behind a sync
decorator, a fake `async def create` never modelled that, and async chat runs were recorded
empty until that file ran. When an adapter learns a new shape, add the case there too, not only
to the fakes. `examples/live_smoke.py` is the by-hand check against a live endpoint.
`test_storage.py::test_concurrent_registration_from_threads` is the canary for the SQLite
locking subtleties above — if a storage change makes it flaky, the change is wrong, not the
test. The module for the Prompt class is `prompts.py` (plural) because the public `prompt`
decorator would collide with a `prompt.py` submodule name in the package namespace.

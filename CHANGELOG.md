# Changelog

All notable changes to promptkeep are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/). Until 1.0, minor versions may change
public APIs; each such change is called out below.

## [Unreleased]

### Added

- Provider adapter interface (`promptkeep.integrations.ProviderAdapter`): all
  provider-specific knowledge now lives in an adapter, and the tracking
  orchestration (conversations, checks, recording, streaming) is written once
  in `integrations/core.py`. `register_adapter()` lets a third party teach
  `wrap()` a new client shape; `tests/test_adapters.py` is the contract every
  adapter must pass. OpenAI `chat.completions` is the first (and so far only)
  adapter; its behavior is unchanged.

- Conversation read model: `ConversationInfo.replay()` rebuilds a session as a
  chat `messages` list (pass `system=` to re-run it against another prompt),
  plus `versions_used`, `total_tokens` and `duration`.
- `history.list_conversations(prompt=, version=)` filters sessions by the
  prompt version that drove them; the dashboard's conversations page gets the
  same filter and a per-conversation stats line.
- `@prompt` on an `async def` template function: awaiting the call yields the
  Prompt.
- `configure(sample_rate=...)` (or `PROMPTKEEP_SAMPLE_RATE`) stores a fraction
  of uneventful runs while always keeping errors, blocked calls and runs with a
  non-ok verdict; conversations are kept or dropped whole.
- `configure(redact=fn)` passes every stored text field of a run or verdict
  through `fn` before it is written, in every write mode; a failing hook drops
  the row rather than storing it unredacted.

- `Prompt.variants(name)` returns every stored version of a prompt as a
  usable `Prompt`, oldest first — to re-run against an old template, or split
  traffic between two. A variant knows its version with no database lookup,
  records its runs under it, and never creates a version (an `exact_match`
  version is rebuilt as one). `.format()` now carries a prompt's registration
  over to the prompt it derives, saving that lookup everywhere.
- The rest of the CLI: `promptkeep list`, `versions`, `diff` (coloured on a
  terminal), `runs`, `convo`, `stats` and `export --format jsonl` (each run
  with its check verdicts and feedback). Plain text, no new dependencies,
  `--db PATH` on every command; read commands never create the file they were
  pointed at.
- `history.stats(name)` — per-version runs, errors, blocked, check pass rate,
  average check score, average feedback score, latency, tokens and cost.
  `history.runs()` / `all_runs()` accept `limit=None` for "everything".
- OpenAI Responses API adapter: `wrap()` now also tracks
  `client.responses.create` (provider `"openai-responses"`) — Prompts as
  `instructions`, as the `input` string or inside `input` items; typed stream
  events; `input_tokens`/`output_tokens` usage; checks, cost and async as on
  chat completions. `promptkeep.call()` picks the surface from how the kwargs
  are spelled (`messages` vs `input`/`instructions`).
- Automatic conversation chaining: a call carrying `previous_response_id`
  joins the conversation of the run that produced that response; the chain's
  first call is adopted into a new conversation (`response:<id>`) as turn 0.
  No user code. An explicit conversation wins; an untracked chain is not
  followed. Schema v7 indexes `runs.response_id` for the lookup. Adapters opt
  in through a new optional `ProviderAdapter.conversation_hint()`; a second
  optional method, `accepts()`, tells surfaces on one client apart.
- `tests/test_real_sdk.py` drives the real `openai` SDK — sync and async
  clients, both surfaces, SSE streams — through promptkeep over an httpx
  `MockTransport`. `openai` joins the dev dependency group for it (the package
  itself still depends only on peewee); it is what caught the async bug below.
- `examples/live_smoke.py`: three tiny real calls (key from the environment)
  that print what was recorded — the check that a live endpoint's responses
  are shaped the way the adapters read them.
- `promptkeep.feedback(run_key, score=, label=, comment=)` attaches a human or
  downstream judgement to a run after the fact. It is stored in the `checks`
  table with `phase="feedback"` (no schema change), rides the same write path
  as a late verdict, shows up in `history.checks()` and the dashboard, and
  never moves a run's pass/fail headline.
- Every recorded call now carries `response.promptkeep` (a `RunHandle` with
  the run's `run_key`), not only checked ones — that key is what `feedback()`
  takes. A call that records nothing is still returned untouched.
- Cost tracking: a run records what the provider said the call cost, as
  `RunInfo.cost_usd` (schema v6 adds `runs.cost_usd`). It is read off the
  response's usage block — OpenRouter reports `usage.cost` on every response,
  streamed or not — and is never estimated: a provider that reports nothing
  (OpenAI's own API) leaves it `None`. `ConversationInfo.total_cost` sums it,
  `storage.record_run(cost_usd=...)` takes it for hand-recorded runs, and the
  dashboard shows it per run and per conversation.

- Conversations are trees (schema v8: `runs.parent_run_key`). A call can
  name the run it continues from — `promptkeep_parent=` takes a response, a
  `RunHandle` or a `run_key` — for regenerations, retries and sub-agent calls;
  a Responses API call's `previous_response_id` sets it automatically, so
  continuing an older response is a branch. `RunInfo.parent_run_key`;
  `ConversationInfo.forks`, `.leaves` and `.path(run_key)`;
  `replay(upto=run_key)`. `replay()` now follows the latest branch rather than
  every turn — identical for a conversation that never branched. `promptkeep
  convo` and the dashboard mark where a branch forks.

- `Prompt.load(name, version=None, *, strict=, pre=, post=)` — one stored
  version as a Prompt, pinned or the latest (highest-numbered): the prompt
  registry, so the template in play can change without a deploy. Bound to
  its version like a variant (no lookup, never a new version); checks and
  strictness, which versions don't store, are given at load. Raises
  `ValueError` for an unknown prompt or version.

### Changed

- Internal restructuring, no public API change: `storage` is split into
  `models`, `migrations`, `controls` and the write path; `history` builds its
  dataclasses straight from its own queries; the wrapper's orchestration is
  one `_Call` object; `call()`/`acall()` live in `promptkeep.integrations`
  and work through the adapter; the module graph has no import cycles (a
  test enforces it). `template_hash` moved to `rendering`, `new_run_key` to
  `models`; `tracking.record_conversation_turn` is gone (use
  `storage.record_run`).
- **Python 3.11 is now the minimum** (was 3.9; 3.9 and 3.10 are end-of-life).
- `wrap()` on an unrecognized object now raises
  `TypeError: wrap() found no supported provider surface ...` naming the
  registered adapters.
- For a *streamed* checked call, a post-check's `ctx.response` is the
  stream's `ResponseFields` summary rather than a synthetic response object.

### Fixed

- **Async chat completions on the real `AsyncOpenAI` were recorded empty.**
  The SDK hides `chat.completions.create` behind a plain-`def` decorator, so
  it did not look like a coroutine function; promptkeep took it for a sync
  method and recorded the run the moment the coroutine was created — no
  output, no usage, zero latency — and a post-check audited the coroutine
  instead of the response. The method is now unwrapped before it is asked, and
  a call that returns an awaitable anyway is finished the async way. (Async
  *Responses* calls, and everything on the sync client, were unaffected.)
- A Prompt's remembered version id is now tied to the database it came from.
  A module-level Prompt that had already rendered kept its old id across a
  `configure(db_path=...)`, so its runs failed the foreign key and were lost —
  or were filed under whichever version owned that id in the new file.
- The dashboard's check chips inherited `white-space: pre-wrap` from the
  transcript fields and rendered the template's own indentation inside them.
- `ConversationInfo.total_tokens` counted a call once per tracked Prompt: a
  call carrying two Prompts writes two rows repeating the same usage. Totals
  now count each API call once.

## [0.3.0] - 2026-09-11

The release that makes promptkeep safe to deploy: run writes leave the request
path, multi-turn sessions become first-class, and calls can be gated and audited.

### Added

- **Conversations.** `promptkeep.conversation(external_id, **metadata)` groups
  every tracked call inside the block (sync or `async with`) into one session;
  the `promptkeep_conversation=` kwarg attaches a single call without a block.
  Every turn is recorded, with or without a `Prompt` in the messages, so a
  session can be read back in full. Read side: `history.conversation()`,
  `history.list_conversations()`, and `conversation_id` / `turn_index` /
  `input_text` on `RunInfo`.
- **Background writes.** `write_mode="background"` is the new default: run rows
  are built at call time and persisted by a bounded, drop-oldest queue drained
  on a daemon thread. `promptkeep.flush(timeout)` and an `atexit` hook drain it;
  `"sync"` and `"off"` remain available (`PROMPTKEEP_WRITE_MODE`). Fork-safe on
  POSIX. Version registration stays synchronous.
- **Checks.** `@check.pre` gates (can `block` or `rewrite` the outgoing turn)
  and `@check.post` audits (async by default) attach per prompt, per call
  (`promptkeep_pre=` / `promptkeep_post=`) or globally via `configure()`.
  Verdicts land in a `checks` table and ride back on the response as
  `response.promptkeep` (a `RunHandle`); `promptkeep.call()` / `acall()` return
  a `CallResult` directly. Timeouts fail open by default (`on_timeout="closed"`
  per check), a crashing check records an error instead of raising, and
  `suppress()` keeps LLM-judge checks from recording themselves.
- **Local dashboard.** `pip install "promptkeep[serve]"` then `promptkeep serve`
  opens a read-only, localhost-only, fully offline view: prompts and version
  lineage with diffs, a filterable runs explorer with check verdicts, and
  turn-by-turn conversation transcripts.
- `history.all_runs()`, `history.list_prompts()`, `history.checks(run_key)` and
  `history.verdict()`.
- `examples/seed_demo.py` (rich demo data for the dashboard),
  `examples/pii_conversation_demo.py`, and `examples/playground.py` (moved from
  the repository root).
- GitHub Actions workflows for CI (ruff, the test suite on Python 3.11–3.14
  across Linux, macOS and Windows, an 85% coverage gate) and tag-driven PyPI
  publishing, staged in `docs/workflows/` — not yet enabled (see `TODO.md`).

### Changed

- Runs are identified by a client-minted `run_key` (UUID) rather than their row
  id; `RunInfo.run_key` is the value to pass to `history.checks()`.
- `__version__` is read from the installed package's metadata; `pyproject.toml`
  is the only place the version is declared.
- The sdist no longer bundles repository-only files.

### Schema

- Database schema is now version 5 (`PRAGMA user_version`); existing files are
  migrated forward automatically on first open. New: `conversations` and
  `checks` tables; `runs` gains `run_key`, `conversation_id`, `turn_index`,
  `input_text`, `original_input_text`, and nullable `version_id` /
  `rendered_text`.

## [0.2.0] - 2026-07-04

### Changed

- Versions dedupe on the **normalized** template: placeholder names are
  canonicalized before hashing, so renaming `{var1}` to `{x}` resolves to the
  same version. Static text, repetition patterns, attribute paths and format
  specs still distinguish versions. Existing rows are re-hashed on migration.

### Added

- `exact_match=True` (on `Prompt` and `@prompt`) opts a prompt back into
  raw-text identity.

## [0.1.0] - 2026-07-04

Initial release: `Prompt` (named, immutable, lazily versioned templates), the
`@prompt` decorator for computed prompts, lenient/strict rendering, SQLite
lineage via peewee, `wrap()` for the OpenAI SDK (sync, async, streaming), and
`history.versions()` / `diff()` / `runs()`.

[Unreleased]: https://github.com/Mercity-AI/promptkeep/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/Mercity-AI/promptkeep/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Mercity-AI/promptkeep/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Mercity-AI/promptkeep/releases/tag/v0.1.0

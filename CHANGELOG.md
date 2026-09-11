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

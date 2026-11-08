# Public API and stability policy

What you can depend on in promptkeep, how it changes, and how you'll hear about it before
something you use goes away.

## Versioning

promptkeep follows [Semantic Versioning](https://semver.org/).

- **Before 1.0** (now): a minor release (`0.x` → `0.x+1`) may change or remove public API.
  Every such change is listed under **Changed** or **Removed** in `CHANGELOG.md`, with what
  to do instead. Patch releases (`0.x.y` → `0.x.y+1`) never break anything.
- **From 1.0**: public API only breaks in a major release. Minor releases add; patches fix.

## What is public

The public API is exactly this — everything else is internal, however it is reachable:

| Where | What |
|---|---|
| `promptkeep` | every name in `promptkeep.__all__` |
| `promptkeep.history` | every function and dataclass whose name has no leading underscore |
| `promptkeep.importers` | every name in `promptkeep.importers.__all__`: `import_codex`, `import_claude_code`, `import_dataset`, `DatasetImportReport`, `ImportReport`, `ImportDiagnostic` |
| `promptkeep.integrations` | every name in `promptkeep.integrations.__all__` — including the `ProviderAdapter` interface a third-party adapter implements |
| `promptkeep.watch` | `Watcher` and its documented lifecycle methods |
| Optional integration modules | `PromptKeepTracingProcessor`, `PromptKeepCallbackHandler`, `PromptKeepSpanExporter`, and `OTLPExport`, documented below |
| Public classes | their methods, properties and attributes without a leading underscore (`Prompt`, `RenderedText`, `RunHandle` as `response.promptkeep`, `CallResult`, `Verdict`, `CheckContext`, `Dataset`, `Span` — the `agent()`/`tool()` handle — `SpanInfo`, `TraceInfo`, and the other `history` dataclasses) |
| Call-site kwargs | `promptkeep_conversation=`, `promptkeep_parent=`, `promptkeep_pre=`, `promptkeep_post=` on a wrapped client |
| Configuration | every `configure()` argument and every `PROMPTKEEP_*` environment variable |
| The CLI | command names and flags. Output is for people: its layout may change in any release. For machines, use `promptkeep export` (JSONL), `promptkeep import codex --json`, or `promptkeep trace --json` |
| The database file | see below |

Not public, even though Python lets you import them: anything with a leading underscore, and
the modules `storage`, `tracking`, `models`, `migrations`, `writer`, `controls`, `rendering`
(beyond what `promptkeep` re-exports), `integrations.core`, and the dashboard's routes and
HTML. **Recording runs by hand** through `storage.record_run` / `tracking.record_prompt_run`
works and is tested, but is provisional: it may change before 1.0.

### Contracts, not just names

Some behaviour is API too. Changing any of these is a breaking change, treated as one:

- **Tracking never breaks your call.** A broken database, a crashing check, a failing redact
  hook or an adapter bug costs telemetry, never the completion. History *reads* raise.
- **Version identity.** A template's version is decided by its normalized text (placeholder
  names don't count, unless `exact_match=True`). A change to normalization that would give
  existing templates new versions is breaking.
- **Variables are run data.** Changing a prompt's variables never creates a version.
- **Rendering is lenient by default.** Unknown placeholders pass through; strict is opt-in.
- **Wrapping is transparent.** A wrapped client returns the provider's own response objects;
  promptkeep only adds `response.promptkeep`.
- **Trace keys identify attempts, not rows.** `agent()` and `tool()` handles expose their
  `span_key` before entry, and a root exposes `trace_key` after entry. Sampling, redaction,
  a full queue, `write_mode="off"`, or a failed write can still drop the row; a key never
  proves that persistence succeeded.
- **Traces are allowed to be partial.** A missing root, parent, linked run, or payload is
  reported as missing rather than reconstructed. `history.trace()` raises `ValueError` only
  for a key with no surviving spans. A caught tool/provider failure is a child failure; it
  does not rewrite the root's exit status. `history.list_traces()` returns root metadata
  without payloads; `history.trace()` loads the payloads and children.
- **Span payloads are bounded and redacted.** Agent/tool input, output, and error text
  go through the configured `redact` hook before queueing; names do not (they are the
  trace's labels, like prompt names). One serialized payload is capped at 64 KiB before a
  visible truncation marker is appended; the cap bounds the serialized value, not the cost
  of serializing an arbitrary object. Omitted output is distinct from an explicitly
  recorded JSON `null`; `SpanInfo.has_input`/`has_output` answer which one it is.
- **Sampling keeps traces coherent where it can.** A root's cohort is chosen once and shared
  by its normal model steps; errors, blocks, and non-ok known verdicts still force a
  problem-only trace. A call explicitly moved to another conversation follows that
  conversation's cohort, and its model link may be missing. Earlier successful steps already
  dropped are not recovered, and a late async verdict cannot restore them.
- **Retention expires traces whole.** The scheduled sweep uses the latest span or linked-run
  activity, so a recent child keeps the trace alive and an abandoned running root eventually
  expires. Prompt/version rows survive; conversation pruning can remove a linked run before
  its trace expires, and the remaining span then reads as run-unavailable.

### Result types grow

The `history` dataclasses (`RunInfo`, `ConversationInfo`, `SpanInfo`, `TraceInfo`, ...) and
`Dataset` are results you read. Any release may add fields to them (at the end, with a
default) and that is not a breaking change. Building them yourself is not supported.

## The database file

- **Forward-compatible.** Every release opens a file written by any earlier release and
  upgrades it in place (schema migrations run on open, forward only).
- **Not backward-compatible.** Once a newer release has opened a file, an older release may
  not read it. Keep a copy before downgrading.
- **The schema is not API.** Read your history through `promptkeep.history`, the CLI or
  `export`; tables and columns may change in any release. Ad-hoc SQL against the file works
  and is fine for exploring — just don't build on it.

## Deprecation

When a public name or behaviour is going away:

1. The release that deprecates it keeps it working, emits a `DeprecationWarning` pointing at
   your line of code (`stacklevel` set to the caller), and lists it under **Deprecated** in
   `CHANGELOG.md` with its replacement.
2. It stays for **at least one minor release** before 1.0, and until the **next major**
   after 1.0.
3. The release that removes it lists it under **Removed**.

Run your test suite with `-W error::DeprecationWarning` to hear about every one early.

A change that can't go through a deprecation — a contract above, or a fix for a security
issue — is called out at the top of its release's changelog entry.


## Provider coverage

`wrap()` supports OpenAI Chat Completions/Responses and Anthropic Messages.
The Anthropic compatibility matrix covers SDK 1.17.0 and 2.6.0, with dependency
range `>=1.17,<3`. Supported methods are sync/async `create`, event streaming,
`.stream()` managers, `.parse()`, `.with_raw_response.create()`, and
`.with_streaming_response.create()`.

Helpers retain native accumulation and typed results. Raw wrappers retain native
headers, body parsing, and lazy reads; parse/close paths finalize exactly once.
An unread close records `incomplete`, not an empty successful output. Blocked calls
make no HTTP request and return compatible empty stubs when `on_block="return"`.
Beta resources, batches, tool runners, and cloud-specific clients are deferred.

`AnthropicMessagesAdapter` is exported by `integrations`. `locate_targets(client)`
allows multiple method surfaces; its default delegates to `locate(client)`.
`Target.invocation_style` defaults to `call`; specialized adapters may implement
manager/raw hooks. Existing two-argument Target construction remains valid.
`Request.asynchronous` defaults to False and describes the invocation protocol.
`ResponseFields.usage_details` and `RunInfo.usage_details` retain reported numeric
usage. Anthropic input includes uncached, cache-read and cache-creation tokens once.
Missing usage remains unknown; reported cost is never estimated.

## External spans and timing

`SpanInfo.started_at` may be None for externally observed calls without a recorded
request start. `observed_at` is separate, `duration_ms` stays None without a valid
interval, `metadata` carries timing quality, and `links` holds explicit related span
keys. This widens the started_at type in 0.4; live manually scoped spans retain starts.
Timing quality is `explicit`, `observed`, or `unknown`. Missing parents are tolerated;
late starts enrich terminal spans without regressing their status. Generic framework
steps use kind `operation`; only `llm` spans count as model calls.

External identities are source/trace/operation scoped. Wrapper run rows can enrich
the framework-owned model span. Response IDs or explicit callback context establish
correlation; matching text or nearby clocks do not. Model metrics count one primary
run per model span even when several prompt versions share the physical call.

## Archive imports

```python
from promptkeep.importers import import_codex, import_claude_code, import_dataset
report = import_codex(path, db_path=None, strict=False)
```

Native importers accept one `.jsonl` or `.jsonl.zst` file. Zstandard requires
`promptkeep[codex]`. Claude files may contain unrelated leading snapshot records;
identity is discovered from native session metadata. Separate subagent files must
be supplied explicitly. Branches and compaction markers retain source relationships.

`ImportReport` contains session/trace identity, outcome (`created`, `replaced`,
`unchanged`), input/span/usage counts, ignored/malformed counts, and diagnostics.
`ImportDiagnostic(severity, code, line)` is payload-free; zero denotes a session-level
diagnostic. `DatasetImportReport` contains dataset identity, session reports, and
per-session failures. Each session commits independently.

Supported local dataset adapters are `ultralazr/claude-code-traces` and
`build-small-hackathon/hackathon-advisor-codex-traces`. Codex records are grouped and
ordered by explicit session/index fields. Conflicting indexes reject that session.
Downloads are never implicit. Test provenance pins the inspected revisions
`afe3c108c148427625f7b2275791517f99f8115d` and
`c7a79f391597b1e0a0b3ee8d89f2c436adb23210`, respectively.

Imports capture their destination without changing global configuration. They bypass
live sampling and the lossy writer queue, honor redaction, and reject disabled/off
tracking. Redacted staging is removed on exit. Identity ignores filenames and
compression. Changed snapshots atomically replace owned rows; failed replacements
preserve the previous import. Retention follows historical activity and removes
import-owned usage with its trace.

`history.imported_session(trace_key)` returns `ImportedSessionInfo`: source/session
identity, trace key, import time, parser revision, summary and usage records. Summary
includes total source usage, response-level usage, unattributed remainder, diagnostics,
and `complete`/`partial`/`unknown` usage completeness. Source totals may include usage
already linked to model runs; the source and trace views are not additive. Cumulative
turn/thread snapshots are never summed. Unknown provider timing remains unknown.

Default parsing diagnoses malformed known records; `strict=True` rejects them.
Unknown types remain ignorable. Invalid identities, conflicts, corrupt compression,
redaction failures, and write failures always abort the affected session. No fake
prompt versions are created. Hidden reasoning and environment/account metadata are
excluded. Diagnostics expose missing and unlinkable source details explicitly.

## Framework bridges and setup

```python
installation = promptkeep.setup(
    integrations=("openai_agents", "langchain"),
    clients=(openai_client, anthropic_client),
    otlp_endpoint=None,
    otlp_protocol="http/protobuf",
)
```

All dependencies and requested client surfaces are validated before installation.
Equivalent setup calls share one Installation and lifetime. Conflicting installations
cannot own the same client. Failed setup restores installed methods. Shutdown only
restores methods still owned by that installation; unrelated later wrappers survive.

`Installation` exposes `langchain_callbacks`, `otel_exporter`, `otlp_export`,
`flush(timeout=None)`, `shutdown(timeout=None)`, and context-manager lifetime.
Flush/shutdown share a timeout across owned work. Repeated shutdown is safe.

- `integrations.openai_agents.PromptKeepTracingProcessor` consumes native Agents SDK
  trace/span callbacks. Setup adds a package-owned dispatcher without replacing other
  processors. Agent/model/tool steps retain native identities; handoffs/guardrails and
  custom steps remain operations. Only explicitly exported sensitive data is read.
- `integrations.langchain.PromptKeepCallbackHandler` records sync/async chains, models,
  tools, errors and interrupted runs. Pass it through runnable callbacks; global callback
  managers are unchanged. Native run/parent IDs determine ancestry across threads/tasks.
  LangGraph parallel execution and interruption/resume are tested. Retry attempts remain
  separate only when the framework supplies distinct identities.

Compatibility currently covers openai-agents 0.24.0, langchain-core 1.7.6, and
LangGraph 1.2.20. These are Python SDK integrations, not process-wide CLI hooks.

## OpenTelemetry

`integrations.opentelemetry.PromptKeepSpanExporter` attaches to an application's own
Python SDK processor. It maps known GenAI usage/model attributes, generic operations,
parentage and explicit links. The existing tracer provider is never replaced.
An OTLP network receiver is not included.

`OTLPExport(endpoint, protocol="http/protobuf", exporter=None, include_content=False)`
exports committed redacted live spans using official OTLP HTTP/protobuf or gRPC
transports. The default queue is bounded to 2,048 spans, batching up to 512. SDK
transports own retries. `dropped_count`, `failed_count`, and
`skipped_unknown_timing` expose delivery limitations. It supports `flush(timeout)`
and `shutdown(timeout)`. Endpoint configuration is explicit; payload content requires
an additional opt-in. Imported snapshots do not automatically enter export queues.
Call `replay(trace_keys, db_path=None, timeout=None)` to explicitly export selected
stored traces. Replay reads but does not modify the source database, includes content
only when `include_content=True`, skips inbound OTel spans, and raises for unknown
trace keys or an expired timeout. Inbound PromptKeep-origin spans are ignored to
prevent loops.

## Watching saved files

`Watcher(source, path, db_path=None, interval=1.0, strict=False)` accepts source
`codex` or `claude-code` and an explicit file/directory. It provides `start`, `stop`,
`join(timeout)`, context-manager support and `failed_count`. Only uncompressed JSONL
is watched. Start is idempotent while running; stop never terminates an agent.

Only complete lines advance the persisted cursor. Normalized updates and checkpoint
state commit together; restart resumes from the committed offset. Rotation/truncation
retain old observations rather than running archive replacement. Active observations
are running; shutdown without completion preserves incomplete state. Cursor paths are
hashed and payload staging is redacted. Reimporting a final saved snapshot reconciles
through the same source identities.

## Storage and migrations

Schema v11 adds nullable external starts, observation time, external identities and
links; v12 adds durable watch checkpoints. Existing v9/v10 files migrate forward.
Connection ownership is thread-local. Operation scopes close only connections they
opened, queued items retain their original destination, retired writers are joined,
and late checks retain their originating database. Export observers run after commit;
rollback cannot publish a span. Existing IMMEDIATE locking remains load-bearing.

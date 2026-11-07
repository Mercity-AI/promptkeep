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
| `promptkeep.integrations` | every name in `promptkeep.integrations.__all__` — including the `ProviderAdapter` interface a third-party adapter implements |
| Public classes | their methods, properties and attributes without a leading underscore (`Prompt`, `RenderedText`, `RunHandle` as `response.promptkeep`, `CallResult`, `Verdict`, `CheckContext`, `Dataset`, `Span` — the `agent()`/`tool()` handle — `SpanInfo`, `TraceInfo`, and the other `history` dataclasses) |
| Call-site kwargs | `promptkeep_conversation=`, `promptkeep_parent=`, `promptkeep_pre=`, `promptkeep_post=` on a wrapped client |
| Configuration | every `configure()` argument and every `PROMPTKEEP_*` environment variable |
| The CLI | command names and flags. Output is for people: its layout may change in any release. For machines, use `promptkeep export` (JSONL) |
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

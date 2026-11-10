## Behavior

Saved Codex, Claude Code and OpenCode sessions now import and watch through registered independent source adapters. `promptkeep import PATH` and `promptkeep watch PATH` detect content, including individual files in mixed-source directories; explicit sources and the previous Python/CLI forms remain supported. Every source is readable through `history.trace()`.

- Codex interprets native turn epochs as seconds and explicit millisecond fields as milliseconds, preserves source durations, and diagnoses invalid/disagreeing intervals. Native last-call usage produces stable model operations with source-order model switches and unknown provider response IDs/timing. Repeated snapshots do not inflate calls or tokens; cumulative totals, accounting resets and unattributed usage stay separate.
- Claude classifies prompts, tool results, injected notes, command output, compaction and interruptions before counting turns. Refusals complete with metadata. Response-part deduplication, assembled turn output, cache accounting, branches, subagents and tool links are preserved.
- OpenCode supports verified modern session-transfer and legacy message/parts saved JSON exports. Models, session parents, tool failures and recorded timestamps survive; source costs retain harness provenance. Partial/invalid export replacements preserve the last committed snapshot. Native database access is outside this change.
- The public `AgentTrace` contract supports third-party registration, ordered consumption, finalization and checkpoint serialization. Common event/payload preparation retains immutable redacted input without repeating redaction. Parser upgrades rebuild old checkpoints and reconcile obsolete importer-owned rows atomically.
- Live tracing preserves input when a start row is evicted, keeps known sampled-out ancestors for retained tool relationships, shares physical-call sampling decisions, and rate-limits outside-agent warnings across names. Nested thread-owned connections reuse trace/worker lifetimes and close on exit, failures, destination changes and shutdown. Atomic span upserts preserve short IMMEDIATE transactions and avoid unnecessary read-back.

## Validation

- **719 tests passed** on Python **3.11, 3.12, 3.13 and 3.14**; repository coverage gate passed at **91.61%**.
- Ruff lint/format checks, `git diff --check`, source distribution and wheel builds passed.
- Native fixtures cover seconds-based timing, a 68-second interval, repeated notifications, switches, resets, Claude classification/refusals and both OpenCode formats. Contract tests cover third-party import/watch/resume, archive/watch parity, checkpoint upgrades, rotation and failed commits.
- Three repetitions of the 3,000-tool contention case and a four-process 6,000-tool workload completed with zero unexpected loss. Cleanup instrumentation matched 500 opens with 500 closes across short-lived threads, with no descriptor growth. Forced eviction and non-idempotent redaction preserve input; 500 outside-agent calls produce one warning in the window.
- Same-machine replay against `487d193`: one warm-up and five measured runs per mode. Median sync **0.27435 → 0.08806 s**; background **0.20274 → 0.04469 s**. Both pass the 20% regression target, with one physical connection open per measured replay. Raw samples and reproduction commands are in `benchmarks/`.

See `docs/PR5_VALIDATION.md` and fixture provenance for details. Repairing previously imported dates/counts/model calls requires the original saved files. Reviewer-private inputs were unavailable, so the reviewer's exact workload comparison remains **unverified**.

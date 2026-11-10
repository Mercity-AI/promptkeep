# PR #5 verification

Verification performed locally on 2026-11-10. The fixture provenance and manually
enumerated expected source facts are in
[tests/fixtures/agent_traces/README.md](../tests/fixtures/agent_traces/README.md).

- Full pytest suite: **719 passed** on CPython **3.11.17, 3.12.15, 3.13.17 and 3.14.8**.
- Repository coverage gate: **91.61%**, above the required 85%.
- `ruff check src tests examples benchmarks`, formatting checks and `git diff --check`:
  clean.
- `uv build`: source archive and wheel built successfully.
- `examples/import_trace.py`: imported the modern OpenCode fixture and read the same
  trace through `history.trace()`, with two model calls and 41 recorded tokens.
- Repeated the 3,000-tool-span contention case three times, each with 300 model calls
  and concurrent prompt registration. No unexpected loss or locking warnings.
- Four independent processes wrote 1,500 tools each to one database: 6,000 tools,
  four roots, zero unexpected loss or locking warnings.
- 500 short-lived threads: 500 physical connection opens, 500 closes, no descriptor
  growth within the supported `/dev/fd` inventory check.
- A forced queue eviction preserves terminal input with a non-idempotent redactor
  called once. Omitted input and explicit JSON null remain distinct.
- 500 outside-agent calls across different tool names produce one warning in the
  rate-limit window; the next warning reports 499 suppressed calls.
- Manual and framework sampled-out model ancestors survive structurally when retained
  tools reference them; discarded content and usage remain unavailable.
- Archive/watch parity covers the complete Codex seconds fixture and both OpenCode
  export formats. Existing tests cover Claude checkpoints, parts, subagents and rotation.
- Native format checks cover repeated usage, model changes, accounting resets,
  source durations, injected Claude records, refusals, branches and tool failures.
- A small independently defined adapter registers, imports, watches and resumes without
  changing built-in source dispatch. Invalid/ambiguous detection stays payload-free.
- Failed writes, redaction failures, malformed input, incomplete replacements and parser
  upgrade reconciliation preserve committed history/checkpoints.

The [offline replay](../benchmarks/README.md) ran one warm-up and five measured runs
per write mode against baseline `487d193`, using the same machine and Python. Medians
were 0.27435 → 0.08806 seconds sync and 0.20274 → 0.04469 seconds background. Both
pass the requirement to stay within 20% of baseline; raw samples are committed.

The reviewer's private logs, exact workload comparison and fixes were unavailable.
That comparison remains **unverified**. The checked-in fixtures are synthetic and
source-derived; independent external review of their expected facts is still welcome.
OpenCode native database access is outside this change. Repairing previous imported
source facts requires the original saved files.

# Fixture provenance

All payloads are synthetic, with invented IDs, prompts and paths. No reviewer-private
logs, credentials, account details or reasoning contents were copied. The explicit
`PRIVATE REASONING` placeholder verifies exclusion rather than representing reasoning.

- `codex-seconds.jsonl` models the reported native epoch-seconds turn envelope and
  existing Codex JSONL fixtures. Manually enumerated expectations: nine records,
  one user turn plus root, three usage observations, two model names, one repeated
  notification, one accounting reset, 42 total tokens across both epochs, and a
  68,000 ms source interval. Observation time does not establish call start/end.
- `opencode-legacy.json` follows OpenCode v1.2.27's
  [export command](https://github.com/anomalyco/opencode/blob/v1.2.27/packages/opencode/src/cli/cmd/export.ts)
  and [message/parts schema](https://github.com/anomalyco/opencode/blob/v1.2.27/packages/opencode/src/session/message-v2.ts).
- `opencode-modern.json` follows upstream commit
  `d4aeee551ebf554f37803b57618926c828a70b39`, inspected 2026-11-10:
  [session transfer](https://github.com/anomalyco/opencode/blob/d4aeee551ebf554f37803b57618926c828a70b39/packages/schema/src/session-transfer.ts),
  [messages](https://github.com/anomalyco/opencode/blob/d4aeee551ebf554f37803b57618926c828a70b39/packages/schema/src/session-message.ts),
  [token usage](https://github.com/anomalyco/opencode/blob/d4aeee551ebf554f37803b57618926c828a70b39/packages/schema/src/token-usage.ts),
  and [export command](https://github.com/anomalyco/opencode/blob/d4aeee551ebf554f37803b57618926c828a70b39/packages/cli/src/commands/handlers/session/export.ts).
  Both OpenCode fixtures contain one user turn, two model calls, one failing tool,
  one external parent session and 41 tokens (17 + 24, including recorded cache and
  reasoning counts). Source cost 0.01 is harness-reported and excluded from billed
  cost. The modern response records its streamed timestamp separately from tool
  completion.

Claude classification fixtures extend `tests/test_remaining.py::claude_records`
with native `isMeta`/`isCompactSummary` flags, complete command envelopes, a quoted
interruption marker and an explicit refusal. Expected turns are enumerated in
`test_claude_classifies_envelopes_and_completes_refusals`.

These are source-derived synthetic fixtures, not a claim that private workloads have
been reproduced or independently reviewed. The reviewer's exact workload comparison
remains unverified until its original inputs are available.

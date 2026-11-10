# Offline replay comparison

`replay.py` executes the identical saved application workload against a selected
checkout: one task, 500 tool scopes and 50 fake model calls. There are no provider or
network requests. Each write mode gets one warm-up and five measured runs with a
fresh database. The timer includes trace exit and `flush()`. Reset/worker shutdown
happen after timing. Physical SQLite opens are instrumented independently through
Peewee. Raw samples and environment details are in [results.json](results.json).

Measured 2026-11-10 on macOS arm64, CPython 3.14.8:

| Write mode | `487d193` median | PR #5 corrected median | Ratio |
| --- | ---: | ---: | ---: |
| sync | 0.27435 s | 0.08806 s | 0.321 |
| background | 0.20274 s | 0.04469 s | 0.220 |

Every measured replay opened one physical connection in both checkouts. The corrected
implementation reuses owned connections across a sync trace or background worker;
separate cleanup tests verify 500 opens and 500 closes across 500 short-lived threads
and check `/dev/fd` for descriptor growth. Async finalization gets its own bounded
worker-thread scope. No SQLite connection crosses threads or provider calls inside a
write transaction.

The initial corrected replay exceeded the 20% regression limit. Profiling identified
per-span ORM read-back, including when there were no export observers. A single SQL
upsert now preserves terminal status and partial payloads atomically; read-back occurs
only when an observer needs the reduced row. Both final medians pass the target.
These small local timings characterize this replay, not every production workload.

To reproduce from the repository root, use the same installed dependencies and Python:

```bash
git worktree add --detach /tmp/promptkeep-baseline 487d193
PYTHONPATH=/tmp/promptkeep-baseline/src:/tmp/promptkeep-baseline uv run python benchmarks/replay.py
PYTHONPATH=src:. uv run python benchmarks/replay.py
```

The reviewer's exact workload and private inputs were unavailable. Their original
reported comparison remains **unverified**; this replay does not claim to reproduce it.

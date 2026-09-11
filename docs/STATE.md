# promptkeep — Current State

*Snapshot: 10 September 2026 · `main` at `d76a5ad` · v0.2.0 on PyPI (main is ahead of the
release) · `feat/checks` open and unmerged*

This is the handoff document: what the library is, what is actually built, how to use it, and
where the edges are. Anyone picking this project up should be able to read this and be
productive without reading the whole source tree first. Section 7 is the scorecard against
`ROADMAP-v1.md`.

---

## 1. What this is

A Python library that treats prompts as **first-class objects instead of strings**.

A prompt gets a permanent name, its text gets versioned automatically, and every LLM call made
with it gets recorded — which version ran, with which variables, against which model, what came
back, how many tokens, how long it took. Multi-turn sessions are grouped into conversations.
All of it lands in a local SQLite file, written off the request path by a background thread,
and `promptkeep serve` opens a read-only local dashboard over it. No server, no account, no
network calls.

The design constraint that shapes everything: **it must never get in the way.** If you don't
wrap your OpenAI client, `prompt.text` is a plain string and the library is just nice template
ergonomics. If the database is broken or missing, you lose telemetry — never a completion.

### The problem it solves

Prompts live as string literals scattered across a codebase. They get edited constantly, and
nobody can answer basic questions afterwards:

- Which version of this prompt was live when that bad output happened?
- What did this prompt look like three weeks ago, before someone "improved" it?
- Did the change actually help, or did it just feel better?
- This conversation went wrong at turn 6 — which prompt version was in play?

Git technically has the answer, but git tracks *files*, not prompts, and it can't link a prompt
revision to the outputs it produced. promptkeep tracks the prompt as the unit and hangs the
runs off it.

---

## 2. Status

| | |
|---|---|
| **Package** | `promptkeep` on PyPI — v0.2.0, published 4 July 2026. **Everything below from conversations onward is unreleased.** `pyproject.toml` still says 0.2.0; bump to 0.3.0 before the next `uv publish`. |
| **Repo** | `github.com/Mercity-AI/promptkeep` (local dir still named `prompt-manager`) |
| **Branches** | `main` at `d76a5ad` (16 commits; PR #1 merged `feat/conversations-dashboard`). `origin/feat/checks` — 14 commits ahead of main, branched from `a72590c`, not merged. |
| **Tests** | main: 147 passing (~1.8s, no network). `feat/checks`: 182 passing. |
| **Size** | main: ~2,800 LOC Python + ~390 LOC HTML templates · ~1,700 LOC tests |
| **Python** | ≥ 3.9, classifiers through 3.14 (no CI actually runs the matrix) |
| **Deps** | `peewee>=3.17` only. Extras: `[openai]` → `openai>=1.0`; `[serve]` → fastapi, uvicorn, jinja2 |
| **License** | MIT |
| **Maturity** | Beta. Core is solid and covered. Deployable now that writes are off the hot path. Still OpenAI-only. |

```bash
pip install promptkeep            # core
pip install "promptkeep[openai]"  # with the OpenAI integration
pip install "promptkeep[serve]"   # with the local dashboard
```

`docs/` (this file and the roadmap) is currently **untracked** — commit it.

---

## 3. The data model

Four entities now. Conversations sit beside the prompt lineage and link to it through runs.

```
Prompt  (name — permanent identity)
  └── Version  (one template text, content-hashed, numbered 1..n)
        └── Run  (one LLM execution: variables + rendered text + response)
              ▲
Conversation  (external_id — caller's session/thread id)
  └── Run, Run, Run ...   (ordered by turn_index; a turn need not involve a Prompt)
```

| Entity | Identity | What creates a new one |
|---|---|---|
| **Prompt** | `name` — unique, never changes | first use of a new name |
| **Version** | sha256 of the *normalized* template | any meaningful edit to the template text |
| **Run** | auto id | every call through a wrapped client (tracked prompt *or* active conversation) |
| **Conversation** | `external_id` — caller-supplied, unique | first tracked call under a new id |

**The rule that matters: variables are run data, never version identity.** Changing
`{"focus": "security"}` to `{"focus": "speed"}` does not create a version. Editing the sentence
around it does. Breaking this rule would make the version history useless within a day.

### Tables

`prompts` → `prompt_versions` → `runs` ← `conversations`. Schema version is tracked in
`PRAGMA user_version` (**3** on main; **5** on `feat/checks`, which adds `checks`, `run_key`, and the rewrite audit columns) with a
forward-only migration runner in `storage._migrate()`. Each migration step runs in its own
transaction on the branch.

`runs` columns added in v3: `conversation_id`, `turn_index`, `input_text` (the user turn's text,
so a conversation can be replayed even when no `Prompt` was in the message). The roadmap's
`parent_run_id` (tree-shaped conversations) was **not** built — conversations are linear.

---

## 4. Features, as built

### 4.1 The `Prompt` object

```python
from promptkeep import Prompt

review = Prompt(
    text="You are a code reviewer. Focus on {focus}.",
    variables={"focus": "correctness"},
    name="REVIEW_SYSTEM",          # required — this is the identity
)

review.text          # 'You are a code reviewer. Focus on correctness.'  (rendered)
review.raw           # 'You are a code reviewer. Focus on {focus}.'      (template)
review.version       # 1 — resolved lazily from the DB
review.placeholders  # {'focus'}
review.name          # 'REVIEW_SYSTEM'

harsher = review.format(focus="security")   # NEW Prompt, same name/template/version
```

`Prompt` is **immutable** — `__slots__` plus a blocked `__setattr__`. `.format()` derives a new
object rather than mutating. This is what guarantees an object can never drift away from the
version hash it registered under.

Registration is **lazy**: constructing a `Prompt` touches nothing. The version row is written on
first `.text` / `.render()` / `.version` access, memoized per object *and* per process. Version
registration is the one write that stays **synchronous** — `.version` is a value the caller
reads back — but it's one write per unique template per process, not per call.

### 4.2 Version matching

Versions dedup on a **normalized** template: placeholder names are canonicalized to `{v0}`,
`{v1}`, … in order of first appearance.

```python
Prompt("Grade this essay on {var1}.", name="GRADER").version   # 1
Prompt("Grade this essay on {topic}.", name="GRADER").version  # 1 — rename, not a new version
Prompt("Grade this essay harshly on {topic}.", name="GRADER").version  # 2 — wording changed
```

Structure is still respected — repetition patterns, attribute paths and format specs all
distinguish versions. Reverting to old text resolves **back to the original version number**
rather than minting a new one. If variable names carry meaning in your workflow, opt out per
prompt with `exact_match=True`.

### 4.3 Rendering

`{placeholder}` syntax, full `string.Formatter` semantics. **Lenient by default**: unknown
placeholders and JSON braces pass through literally; unparseable templates return unrendered.
Strict mode (`strict=True` per prompt, or `configure(strict=True)`) raises
`MissingVariableError` listing every unresolved placeholder.

### 4.4 The `@prompt` decorator — computed prompts

```python
from promptkeep import prompt

@prompt(name="SUMMARIZE")
def summarize_prompt(style="bullet points", max_words=50):
    return f"Summarize the text as {{style}}. Use at most {max_words} words.\n\nText: {{text}}"

p = summarize_prompt(max_words=100)       # -> a Prompt object
```

The function returns the raw template; the decorator packages it into a `Prompt` with the call's
arguments as the variables dict, and stores `fn_source_hash` on the version row. **Still breaks
on `async def`** — the wrapper gets a coroutine and trips the "must return a template string"
check. Roadmap item, not done on either branch.

### 4.5 The OpenAI wrapper

```python
from openai import OpenAI
from promptkeep import wrap

OpenAI = wrap(OpenAI)
client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-5.5",
    messages=[
        {"role": "developer", "content": review},   # a Prompt object, passed directly
        {"role": "user", "content": "How do I check isinstance?"},
    ],
)
```

The API receives a **plain string** — byte-for-byte what an unwrapped client would send. A run
row records version, variables, rendered text, model, request params, output text, token usage,
response id and latency.

Supported: `wrap(cls)` and `wrap(instance)` · sync and `AsyncOpenAI` · streaming (proxy writes
the run when the stream closes, including the context-manager form) · multi-part content ·
multiple tracked prompts per call · errors recorded with `status="error"` and re-raised
unchanged · an explicit `promptkeep_conversation=` kwarg, stripped before the request goes out.

Not supported: **`chat.completions` only** — no `responses.create`, no Anthropic, no LiteLLM,
no provider adapter interface. Never monkey-patches the `openai` module; message dicts are
copied, never mutated; idempotent via `_pm_instrumented`.

### 4.6 Provenance-carrying strings

`prompt.text` returns `RenderedText`, a `str` subclass carrying hidden `_pm_prompt` /
`_pm_variables` attributes. It behaves identically to `str` everywhere, which is how the wrapper
tracks a prompt without any extra API. `Prompt` itself is deliberately **not** a `str` subclass.

### 4.7 Conversations *(new since v0.2.0)*

```python
import promptkeep

with promptkeep.conversation("user-42-session-9", user_id=42, tenant="acme") as convo:
    client.chat.completions.create(...)     # turn 0
    client.chat.completions.create(...)     # turn 1
    convo.external_id

async with promptkeep.conversation("user-42-session-9"):   # async twin
    await client.chat.completions.create(...)

# escape hatch for call sites that can't wrap a block
client.chat.completions.create(..., promptkeep_conversation="user-42-session-9")
```

`contextvars`-based, so it survives threads and async tasks. The context manager itself never
touches the DB — the conversation row is created (or reused) at record time by the wrapper.
Keyword arguments beyond `title` become metadata, stored once on first sight of the id.

Every call inside a conversation gets a run row **whether or not a `Prompt` was involved**, with
`input_text` capturing the user turn, so a session can be reassembled in full. Turn numbers come
from in-process counters (`storage.reserve_turn_index`) because the DB's `MAX(turn_index)` is
stale while rows sit in the write queue.

Not built from the roadmap's conversation spec: `convo.replay()`, `versions_used`,
`total_tokens`, `duration`, `history.conversations(prompt=, version=)` filtering,
`parent_run_id`, and automatic chaining via `previous_response_id` (needs the Responses API
first). Conversation inference from message-prefix matching remains deliberately unbuilt.

### 4.8 Background writes *(new since v0.2.0)*

`write_mode="background"` is the default. `storage.record_run` builds the complete row (so
timestamps and turn numbers reflect call time) and hands it to `writer.py`: a bounded
`queue.Queue` drained by one lazily-started daemon thread that batches rows per transaction.

- **Bounded, drop-oldest.** Overflow discards the oldest queued run and counts it; a warning
  logs at most every 5s.
- **The worker never dies.** Every batch is exception-shielded.
- **`promptkeep.flush(timeout)`** blocks until the queue drains; an `atexit` hook (registered
  once, on first start) flushes with a 2s budget on interpreter exit.
- **Fork-safe** via `os.register_at_fork` — a child resets to empty state instead of
  double-writing what the parent will also write.
- **`write_mode="sync"`** writes before the call returns (tests use this via conftest);
  **`"off"`** drops run telemetry entirely while version registration keeps working.

### 4.9 History

```python
from promptkeep import history

history.versions("REVIEW_SYSTEM")            # [VersionInfo(...)] oldest first
print(history.diff("REVIEW_SYSTEM", 1, 3))   # unified diff between two versions
history.runs("REVIEW_SYSTEM", version=3, limit=20)   # [RunInfo(...)] newest first
history.all_runs(limit=100)                  # across every prompt
history.list_prompts()                       # [PromptSummary] name + version/run counts
history.list_conversations(limit=100)        # [ConversationSummary] id + turn count
history.conversation("user-42-session-9")    # ConversationInfo: metadata + ordered turns
```

`RunInfo` now also carries `conversation_id`, `turn_index`, `input_text`. Read paths raise
normally — a broken query is a bug you want to see.

### 4.10 CLI and local dashboard *(new since v0.2.0)*

```bash
pip install "promptkeep[serve]"
promptkeep serve                                   # http://127.0.0.1:8420, reads ./.promptkeep.db
promptkeep serve --db path/to/prompts.db --port 8420
```

`serve` is the **only** CLI subcommand. It launches a FastAPI + Jinja2 app that is read-only,
localhost-bound, and fully offline (no CDN assets, no account, automatic light/dark). Pages:
prompts overview → version lineage → diff between two versions; runs (filterable by prompt and
version); conversations → turn-by-turn transcript with the driving version shown inline. The
server stack is imported lazily by `cli.py`, never by `promptkeep/__init__.py`.

`examples/seed_demo.py` populates a throwaway DB with rich demo data for the dashboard.

### 4.11 Configuration

```python
promptkeep.configure(
    db_path="path/to/prompts.db",   # default ./.promptkeep.db  ($PROMPTKEEP_DB)
    enabled=True,                   # False = zero persistence   ($PROMPTKEEP_DISABLED)
    strict=False,                   # raise on missing variables
    write_mode="background",        # "background" | "sync" | "off"  ($PROMPTKEEP_WRITE_MODE)
    queue_size=10_000,
    flush_interval=0.5,
    batch_size=100,
)
```

Precedence, resolved fresh on every access: `configure()` overrides → env vars → defaults.
Invalid values raise from `configure()`; a bad env value falls back to the default.

### 4.12 Checks — on `feat/checks`, not yet on main

The roadmap's v0.4 milestone is essentially complete on the branch. Public surface:
`check`, `Verdict`, `CheckContext`, `PromptBlocked`, `RunHandle`, `call`, `acall`, `suppress`.

```python
from promptkeep import Prompt, check, Verdict

@check.pre(name="no_pii")                          # gate: blocking, before the request
def block_pii(ctx) -> Verdict:
    return Verdict.block("email in prompt") if EMAIL_RE.search(ctx.rendered) else Verdict.ok()

@check.post(name="grounded", mode="async")         # audit: default async, off the caller's path
def is_grounded(ctx) -> Verdict:
    return Verdict.from_score(score, threshold=0.7)

review = Prompt("...", name="REVIEW_SYSTEM", pre=[block_pii], post=[is_grounded])

response = client.chat.completions.create(...)     # native response object, untouched
response.promptkeep.run_id / .checks / .verification / .wait() / await .awaited()

result = promptkeep.call(client, model=..., messages=[...])   # or acall(); non-streaming only
result.text, result.verification, result.run_id
```

Built: `Verdict.ok/warn/block/rewrite/from_score` · three scopes (prompt, per-call kwargs,
`configure(pre=, post=)`), deduped · `on_block="raise" | "return"` (stub response) · per-check
`timeout` in a dedicated thread with `on_timeout="open"` (default) or `"closed"` for pre-checks ·
post `mode="async" | "blocking"` · `suppress()` contextvar so an LLM-judge check doesn't record
itself · a crashing check records `status="error"` and never crashes the call · works on the
streaming path · schema v5: `checks` table, `runs.run_key`, `runs.original_input_text`,
`checks.rewritten` · `history.checks(run_key)` / `history.verdict()` · a checks view in the
dashboard · `examples/pii_conversation_demo.py` · README section · 43 tests. The branch also
fixes `__version__` to `"0.2.0"` (hardcoded, not `importlib.metadata`).

Durability, as of 10 September: runs are identified by a client-minted `run_key` (UUID), so
checked calls go through the background writer like every other call — nothing on the request
path waits for SQLite. Async verdicts queue behind their run row; `promptkeep.flush()` waits for
in-flight post-checks, then drains the queue. A pre-check rewrite stores both the original turn
and the rewritten one, and the rewriting check's verdict carries the replacement text.

Not built: `promptkeep.feedback()`.

---

## 5. Architecture

```
src/promptkeep/
├── __init__.py        public API: Prompt, prompt, wrap, configure, history, conversation, flush
├── prompts.py         Prompt + RenderedText          (plural — `prompt` is the decorator)
├── rendering.py       render / normalize / placeholder extraction
├── decorator.py       @prompt
├── config.py          configure() / get_settings() / reset()
├── conversation.py    contextvar-based conversation() context manager
├── storage.py         peewee models, migrations, register_version, record_run, write_batch, fetch_*
├── writer.py          background queue + daemon thread, flush(), fork hook
├── tracking.py        provider-agnostic run recording (prompt runs and bare conversation turns)
├── history.py         read side — dict rows -> frozen dataclasses
├── cli.py             `promptkeep serve` (lazy-imports the server stack)
├── dashboard/
│   ├── app.py         FastAPI routes, read-only
│   └── templates/     base, prompts, prompt_detail, diff, runs, conversations, conversation_detail
└── integrations/
    ├── __init__.py    wrap() dispatcher
    └── openai_wrapper.py
```

Flow: `Prompt.render()` → `storage.register_version()` (lazy, memoized, sync) → wrapper
intercepts `chat.completions.create` → resolves the active conversation → `tracking` →
`storage.record_run()` → `writer.submit()` → daemon thread → `storage.write_batch()`.

### Load-bearing decisions

Breaking any of these breaks the library's contract:

1. **All implicit write paths are exception-shielded** — `register_version`, `record_run`,
   `tracking`, and the whole of `writer`. Losing telemetry is always better than breaking
   someone's LLM call.
2. **Registration is lazy, never at construction.** Import must not do I/O or spawn threads.
3. **Prompt is frozen.** Immutability is what keeps an object consistent with its version hash.
4. **Run writes are asynchronous by default; version registration is not.** Row contents are
   fixed at call time; only the insert is deferred.
5. **The wrapper touches only the object passed to `wrap()`.** No module-level patching.
6. **Rendering is lenient by default.** Strict is opt-in.
7. **Core never imports `openai`** or the server stack. Wrapper tests run against hand-rolled
   fakes; the dashboard is an optional extra.
8. **Conversations are explicit only.** Never inferred from message-history prefixes.

### SQLite specifics (the sharp edges)

- Models bind to a `DatabaseProxy`; `storage._get_db()` initializes it under a lock. Concurrent
  first-connections to a fresh file otherwise race on the WAL switch — a real intermittent test
  failure, twice.
- Registration uses `atomic("IMMEDIATE")` plus a retry loop on `IntegrityError` /
  `OperationalError`. **Do not "simplify" this back to plain `atomic()`.**
- Turn indices are reserved from in-process counters, not `MAX(turn_index)` — the DB is stale
  while rows sit in the background queue.
- Pragmas: WAL, `foreign_keys=1`, `busy_timeout=5000`.
- Schema changes: bump `_SCHEMA_VERSION`, add a forward-only step in `_migrate()` using
  `playhouse.migrate`.

---

## 6. Tests

147 tests on main, no network, no `openai` dependency, ~1.8s.

| File | Tests | Covers |
|---|---:|---|
| `test_rendering.py` | 30 | lenient/strict matrix, JSON braces, normalization |
| `test_prompt.py` | 25 | immutability, versioning, provenance, equality |
| `test_openai_wrapper.py` | 19 | substitution, run rows, streaming, async, error paths |
| `test_storage.py` | 15 | dedup, version counters, concurrency, migrations, conversations |
| `test_dashboard.py` | 15 | every route via `TestClient`, 404s, filters |
| `test_conversation.py` | 14 | context manager, kwarg path, turn ordering, async, bare turns |
| `test_decorator.py` | 14 | defaults, kwargs capture, fn source hash |
| `test_writer.py` | 8 | batching, overflow/drop counting, flush, reset |
| `test_history.py` | 7 | versions/diff/runs shaping |

`feat/checks` adds `test_checks.py` (32) and extends storage/writer tests → 182.

`tests/conftest.py` gives every test a fresh tmp DB, reset config, `write_mode="sync"`, and a
reset writer — tests never touch a real `.promptkeep.db`.

**`test_storage.py::test_concurrent_registration_from_threads` is the canary.** If a storage
change makes it flaky, the change is wrong, not the test.

```bash
uv sync
uv run pytest -q
uv run ruff format src tests && uv run ruff check src tests   # line-length 100
uv run python examples/seed_demo.py && uv run promptkeep serve --db demo.promptkeep.db
```

---

## 7. Roadmap scorecard

Status against `ROADMAP-v1.md`, milestone by milestone. **Done** = on main. **Branch** = on
`origin/feat/checks`, tests green, awaiting merge. **Open** = not started.

### v0.3 — Foundations

| Item | Status | Notes |
|---|---|---|
| Background writer, `flush()`, atexit, bounded drop-oldest queue | **Done** | `writer.py`, `configure(write_mode/queue_size/flush_interval/batch_size)` |
| Fork safety | **Done** | `os.register_at_fork`, POSIX only |
| `write_mode="sync"` in tests | **Done** | conftest |
| Conversations: schema v3, context manager, async twin, kwarg escape hatch | **Done** | linear only |
| `history.conversation()` with ordered turns | **Done** | |
| `convo.replay()`, `versions_used`, `total_tokens`, `duration` | Open | `ConversationInfo` has metadata + `turns` only |
| `history.conversations(prompt=, version=)` filter | Open | only `list_conversations(limit)` |
| `parent_run_id` (tree-shaped conversations) | Open | |
| `@prompt` on `async def` | Open | still raises |
| `sample_rate` | Open | |
| `redact` hook | Open | `rendered_text`, `input_text`, `output_text` stored in full |
| `__version__` via `importlib.metadata` | Open | main says `"0.1.0"`; branch hardcodes `"0.2.0"` |
| `testing.py` → `examples/`, drop `testing.db` | Partial | `testing.db` is now gitignored and untracked; `testing.py` is still committed in the repo root |
| CI (matrix, ruff, coverage gate) | Open | no `.github/` at all |

### v0.4 — Checks

| Item | Status | Notes |
|---|---|---|
| Pre/post checks, `Verdict`, `RunHandle`, `call()` / `acall()` | **Branch** | |
| Schema v4 `checks` table, `history.checks()` | **Branch** | |
| Three scopes, `on_block`, per-check timeout + `on_timeout`, `suppress()` | **Branch** | answers open question 3 |
| Dashboard checks view | **Branch** | |
| `promptkeep.feedback()` | Open | |
| Docs page with real examples | Partial | README section + PII demo on branch; no docs site |

### v0.5 — Reach and visibility (public beta)

| Item | Status | Notes |
|---|---|---|
| `promptkeep serve` — read-only, localhost | **Done** | shipped early, ahead of the milestone |
| Rest of the CLI: `list`, `versions`, `diff`, `runs`, `convo`, `stats`, `export` | Open | `serve` is the only subcommand |
| Provider adapter interface (`integrations/base.py`) | Open | |
| OpenAI Responses API + automatic conversation chaining | Open | |
| Anthropic adapter | Open | |
| LiteLLM adapter | Open | |
| Cost tracking (`cost_usd`, price table) | Open | |
| `compare()` and `Prompt.variants()` A/B routing | Open | |
| Docs site | Open | |

### v1.0 — Stability

| Item | Status |
|---|---|
| API freeze + deprecation policy | Open |
| `Prompt.load()` version pinning | Open |
| Dataset export (`to_dspy` / `to_jsonl` / `to_promptfoo`) | Open |
| Storage backend interface | Open |
| Published benchmarks | Open |
| Retention | Open |

### Engineering standards (roadmap §8)

None started: no CI, no release automation, no `CHANGELOG.md`, no `CONTRIBUTING.md` /
`SECURITY.md` / issue templates, no `mypy --strict`, no benchmarks, no docs site. The README was
updated for conversations, the dashboard and background writes, but not rewritten to the
"sell in fifteen seconds" shape (no GIF).

### Open questions (roadmap §10) — where they landed

1. **Default `write_mode`** — decided: background, with `flush()` and a loud README note.
2. **Auto-naming prompts** — not decided; `name` is still required.
3. **Pre-check timeout** — decided on branch: fail open by default, per-check `on_timeout="closed"`.
4. **Does `serve` exist** — decided: yes, read-only and localhost-only.
5. **Chase enterprise** — not decided.
6. **Naming** — decided on branch: `checks`, with `check.pre` / `check.post`.

### Suggested next moves, in order

1. Merge `feat/checks` (rebase onto main first — it's one commit behind).
2. Hygiene batch: `__version__` from `importlib.metadata`, move `testing.py` into
   `examples/`, commit `docs/`, add CI. Cheap, and everything after this gets safer.
3. Bump to 0.3.0 and publish — PyPI is two milestones behind main.
4. Finish the conversation read model (`replay()`, `versions_used`, filtered listing); these
   are the pieces the eval and compare work will consume.
5. `sample_rate` + `redact` — the last v0.3 items and the blockers for regulated users.

---

## 8. Reading order for someone new

1. `README.md` — 5 minutes, the whole user-facing surface
2. `uv run python examples/seed_demo.py` then `promptkeep serve --db demo.promptkeep.db` — see
   versioning, conversations and run tracking in the dashboard
3. `CLAUDE.md` — the design decisions and the SQLite traps, already written down
4. `src/promptkeep/prompts.py`, `storage.py`, `writer.py` — the three files that carry the model
5. `docs/ROADMAP-v1.md` — where this goes next; section 7 above is the scorecard against it
6. `plan.md` — the original design doc, kept as history. **Do not update it to match code**

---

*Companion document: `ROADMAP-v1.md` — where this goes next.*

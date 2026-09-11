# promptkeep — Road to v1

*Written 27 July 2026, against v0.2.0. Companion to `STATE.md`.*

v0.2.0 is a good library that does one narrow thing well. It is not yet a tool anyone would
build a production system on, and it is not yet a thing anyone would find. This document is the
plan for both: the engineering that makes it real-world-applicable, and the work that makes a
launch land.

---

## 0. The thesis, stated plainly

Everything else in this document should serve one sentence:

> **promptkeep is the local, zero-config memory of your LLM application — every prompt version,
> every run, every verdict — in a file you own, with no server to run and no account to create.**

The market is crowded with platforms: Langfuse, LangSmith, Braintrust, PromptLayer, Weave,
Helicone, Opik. Every one of them is a *destination* — you send your data somewhere, you sign up,
you configure a project, you look at a dashboard. That's the right shape for a team of thirty
with a compliance function. It's absurd overhead for the person who just wants to know which
version of a prompt produced last Tuesday's bad answer.

The gap is the same one SQLite occupies against Postgres, and the one `ruff` occupied against a
plugin ecosystem: **the thing that is embarrassingly easy to start using.** `pip install`, one
line of wrapping, and it works — offline, in CI, in a notebook, in a Lambda.

Three commitments that follow from that, and should be treated as non-negotiable:

1. **No server required, ever.** A remote backend can be an option. It can never be the
   happy path.
2. **Never break the caller.** Already the rule. Extend it to every new subsystem.
3. **Overhead you can measure and publish.** Developer tools with an invisible cost get ripped
   out at the first latency investigation. We should be able to say "under 100µs per call" and
   have a benchmark that proves it.

What we are explicitly **not** building: a dashboard SaaS, an eval platform, an agent framework,
a model router. We integrate with those; we don't compete with them.

---

## 1. Where v0.2.0 falls over in real usage

Honest inventory of what breaks once someone actually deploys this.

| # | Gap | Why it's blocking |
|---|---|---|
| 1 | Writes happen inline in the request path | A SQLite write on a busy async server blocks the event loop. Under process-level concurrency, WAL's single-writer constraint turns it into a queue nobody asked for. |
| 2 | No conversation concept | Almost every real app is multi-turn. Runs are isolated rows with no way to reassemble a thread. |
| 3 | No verification hooks | People need to block a call before it goes out and check the output after it comes back. Today there's no seam for it. |
| 4 | OpenAI `chat.completions` only | Responses API, Anthropic, Gemini, and LiteLLM users are all locked out. |
| 5 | Nothing to *do* with the data | We collect a rich corpus and expose three read functions. No CLI, no stats, no export, no optimization. |
| 6 | No production controls | No sampling, no redaction, no cost, no retention. |

Items 1–3 are the ones raised as priorities. They're also, correctly, the ones that have to
happen in that order — verification and conversations both depend on the write path being
non-blocking, and both produce data that optimization later consumes.

---

## 2. Asynchronicity — the honest answer

**Is async included today?** Partly, and the part that's missing is the part that matters.

What works: `AsyncOpenAI` is detected and wrapped with an `async def` interceptor. Async
streaming works via `_AsyncStreamProxy`.

What doesn't:

- **Every DB write is synchronous and happens on the caller's thread.** In an async handler,
  `record_run()` performs a blocking SQLite insert inside the coroutine — it stalls the event
  loop. Version registration does the same on first render. The wrapper is async-aware; the
  storage layer is not.
- **`@prompt` on an `async def` breaks.** The wrapper calls `fn(...)`, gets a coroutine, and
  raises "must return a template string".
- **No flush/shutdown story.** Short-lived processes (Lambda, CLI, a script that exits fast) have
  no way to guarantee writes landed.

### The fix: a background writer

A single daemon thread draining a bounded `queue.Queue`, batching inserts inside one
transaction.

```python
promptkeep.configure(
    write_mode="background",   # "background" (default) | "sync" | "off"
    queue_size=10_000,         # bounded — drop oldest and count, never grow unbounded
    flush_interval=0.5,        # seconds
    batch_size=100,
)

promptkeep.flush(timeout=5)    # explicit drain, for scripts and tests
```

Design requirements, each of which is a known footgun:

- **Bounded queue with a drop counter.** An unbounded queue in a telemetry library is how you
  turn a slow disk into an OOM. Drop, count, and log once per interval.
- **`atexit` flush**, plus a `flush()` for explicit control. Registered late enough not to fight
  interpreter teardown.
- **Fork safety** via `os.register_at_fork` — gunicorn/uvicorn pre-fork workers inherit a dead
  thread otherwise, which is a classic silent-data-loss bug.
- **`write_mode="sync"` in tests.** Deterministic assertions matter more than throughput there;
  the conftest fixture should set it.
- **Version registration stays synchronous.** It has to — `.version` is a value the caller reads
  back. But it's memoized per process, so it's one write per unique template per process, not
  per call. Keep it, document it.

Async correctness elsewhere:

- `@prompt` gains coroutine detection: if `fn` is a coroutine function, return an async wrapper
  that awaits the template and returns a `Prompt`.
- Never call `asyncio.run()` or block on a loop from library code.
- The whole thing must work with no event loop at all — this is a sync-first library that
  happens to support async, not the reverse.

---

## 3. Conversations — the feature only promptkeep can offer

This is the sharpest idea on the list, and worth stating why.

Observability platforms already have traces and sessions. What none of them have is the link
between a conversation and the **prompt version lineage** that drove it. Because promptkeep
already owns versions, it can answer questions nobody else can:

- "Show me every conversation where REVIEW_SYSTEM v4 was live, and how they ended."
- "We shipped v5 on Tuesday. Did conversations get longer or shorter?"
- "This conversation went wrong at turn 6 — which prompt version was in play, and what did it
  look like at that moment?"

That's the differentiated feature. Session tracking on its own is a commodity; **version-aware**
session tracking is not.

### Data model — schema v3

New `conversations` table; `runs` gains three columns:

```sql
CREATE TABLE conversations (
    id          INTEGER PRIMARY KEY,
    external_id TEXT UNIQUE,        -- caller's id: user session, thread id, ticket number
    title       TEXT,
    metadata    TEXT,               -- JSON: user_id, tenant, env, anything
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

ALTER TABLE runs ADD COLUMN conversation_id INTEGER REFERENCES conversations(id);
ALTER TABLE runs ADD COLUMN turn_index      INTEGER;   -- 0, 1, 2... within the conversation
ALTER TABLE runs ADD COLUMN parent_run_id   INTEGER REFERENCES runs(id);
CREATE INDEX idx_runs_conversation ON runs(conversation_id, turn_index);
```

`parent_run_id` is what lets a conversation be a tree rather than a list — retries, branches,
regenerations, and sub-agent calls all hang off a parent instead of pretending to be linear.

### API — three ways in, ordered by how much we want people to use them

```python
import promptkeep

# 1. Context manager (the recommended path). contextvars-based, so it survives
#    both threads and async tasks correctly.
with promptkeep.conversation("user-42-session-9", user_id=42, tenant="acme") as convo:
    client.chat.completions.create(...)     # turn 0, attached automatically
    client.chat.completions.create(...)     # turn 1
    convo.id

# async twin
async with promptkeep.conversation("user-42-session-9"):
    await client.chat.completions.create(...)

# 2. Explicit, for code that can't nest (queue workers, callbacks)
client.chat.completions.create(..., promptkeep_conversation="user-42-session-9")
#                                    ^ stripped from kwargs before the request goes out

# 3. Automatic, on the Responses API
client.responses.create(..., previous_response_id=prev.id)
#   -> chained into the same conversation with no extra code at all
```

Point 3 is worth building carefully — it's the "it just worked" moment that makes people tell
other people about a library.

### Reading it back

```python
from promptkeep import history

convo = history.conversation("user-42-session-9")
convo.turns          # [RunInfo, ...] in order
convo.versions_used  # {"REVIEW_SYSTEM": [4, 5], "SUMMARIZE": [2]}
convo.total_tokens
convo.duration

convo.replay()       # -> messages list, ready to re-send to any provider
history.conversations(prompt="REVIEW_SYSTEM", version=4, limit=20)
```

`replay()` is quietly one of the most useful things here — reconstructing a conversation to
re-run it against a new prompt version is the atom of every eval workflow, and it comes almost
free once turns are ordered.

**Deliberately not doing:** inferring conversations from message-history prefix matching. It's
tempting and it's a trap — false joins across users are worse than no grouping at all. Explicit
or `previous_response_id` only.

---

## 4. Pre and post verification

The idea from the sketch:

```python
output = call_model(...)
print(output)
>>> XYZ(text="....", verification=".....")
```

Restating what's wanted: a user-supplied function runs **before** the request goes out and can
block it, and another runs **after** the response comes back and annotates it. Either can be
deterministic code or another LLM call. Pre blocks the caller by necessity; post shouldn't have
to.

### Naming

`verification` is right for the semantics but overloaded. Proposed: **checks**, with two kinds —
`pre` (gate) and `post` (audit). Reads well in code, matches what people already call this in
CI, and leaves "eval" free for the offline-scoring meaning it already has elsewhere.

### Registration

```python
from promptkeep import Prompt, check, Verdict

@check.pre(name="no_pii")
def block_pii(ctx) -> Verdict:
    if EMAIL_RE.search(ctx.rendered):
        return Verdict.block("email address in prompt")
    return Verdict.ok()

@check.pre(name="not_too_long")
def length_gate(ctx) -> Verdict:
    return Verdict.warn("very long prompt") if len(ctx.rendered) > 40_000 else Verdict.ok()

@check.post(name="grounded", mode="async")     # default: doesn't block the caller
def is_grounded(ctx) -> Verdict:
    answer = judge_client.chat.completions.create(...)   # an LLM check is just a check
    return Verdict.from_score(float(answer.output_text), threshold=0.7)

review = Prompt(
    "You are a code reviewer. Focus on {focus}.",
    name="REVIEW_SYSTEM",
    pre=[block_pii, length_gate],
    post=[is_grounded],
)
```

Checks attach at three scopes, most specific winning: per-`Prompt`, per-call, or globally via
`configure(pre=[...], post=[...])`.

### Contracts

**Pre-checks** run before the provider call, in registration order, and are **blocking by
definition** — the whole point is to stop the request. Each returns a `Verdict`:

| Verdict | Effect |
|---|---|
| `ok()` | continue |
| `warn(msg)` | continue, record the warning on the run |
| `block(msg)` | do not call the provider |
| `rewrite(text)` | continue with substituted text; original and rewrite both recorded |

On block, behaviour is configurable: `on_block="raise"` (default — raises `PromptBlocked`) or
`on_block="return"`, which returns a response-shaped stub carrying the verdict, so a service can
degrade instead of erroring.

Pre-checks are on the hot path. They need a `timeout` (default a few seconds), and a
timeout must **fail open** with a recorded warning — a slow check must never become an outage.
That default will be argued about; make it configurable and document the reasoning loudly.

**Post-checks** run after the response and default to `mode="async"` — the caller gets their
response immediately, the check runs on the background worker, and the verdict is written to the
DB when it lands. `mode="blocking"` is opt-in for the case where the verdict must gate what the
user sees (retry-on-failure, refusal filtering).

### Getting the verdict back to the caller

The wrapper currently returns the provider's native response object untouched, and that
compatibility is worth a lot. So: attach rather than replace.

```python
response = client.chat.completions.create(...)
response.text                       # unchanged — normal SDK object, existing code fine

pk = response.promptkeep            # attached RunHandle
pk.run_id
pk.prompt_version                   # 4
pk.checks                           # [CheckResult(name="grounded", status="pending"), ...]
pk.verification                     # aggregate: "ok" | "warn" | "failed" | "pending"

pk.wait(timeout=5)                  # block for async post-checks, if you want them
await pk.awaited()                  # async twin
```

For people who want the shape from the sketch, an opt-in helper returns a result object
directly, no attribute-poking:

```python
from promptkeep import call

result = call(client, model="gpt-5.5", messages=[...])
result.text            # the output string
result.verification    # verdict
result.run_id
```

Both paths write the same rows. The attach path is the default because it preserves the "your
code doesn't change" promise; the `call()` path exists because the explicit shape is genuinely
nicer when you're writing new code.

### Schema v4

```sql
CREATE TABLE checks (
    id         INTEGER PRIMARY KEY,
    run_id     INTEGER NOT NULL REFERENCES runs(id),
    name       TEXT NOT NULL,
    phase      TEXT NOT NULL,       -- 'pre' | 'post'
    status     TEXT NOT NULL,       -- 'ok' | 'warn' | 'block' | 'error' | 'pending'
    score      REAL,                -- optional numeric, for threshold checks
    message    TEXT,
    latency_ms INTEGER,
    created_at TEXT NOT NULL
);
CREATE INDEX idx_checks_run ON checks(run_id, name);
```

Note what this table really is: **the label store.** Every post-check verdict is a
machine-generated judgement of one prompt version's output. Which is exactly the input that
optimization needs, and the reason checks should ship before optimization.

Two implementation hazards to get right early:

- **A check that calls an LLM must not recursively record itself.** A `promptkeep.suppress()`
  contextvar, set around check execution.
- **A crashing check must never crash the call.** Same shielding rule as everything else —
  record `status="error"`, continue.

---

## 5. Prompt optimization — build the substrate, not the optimizer

The honest position: DSPy has spent years on optimizers and has a research team behind them.
Reimplementing MIPRO-style search would be a year of work to arrive somewhere worse.

But DSPy's real constraint isn't the algorithm — it's that it needs a labelled trainset and a
metric function, written up front, before you have any of the data. **promptkeep arrives at the
problem from the opposite end: it already has the corpus.** Every version, every run, every
output, and — once checks ship — verdicts on many of them.

So the v1 contribution is the substrate. Four pieces, all shippable, none requiring novel
research:

**1. Feedback capture.** Optimization needs a signal. Three sources, in ascending value:
post-check verdicts (free, automatic), explicit user feedback, and implicit signals.

```python
promptkeep.feedback(run_id, score=1.0, label="thumbs_up", comment="perfect")
promptkeep.feedback(response.promptkeep.run_id, score=0.0, label="hallucination")
```

**2. Datasets out of history.** Turn logged runs into an eval set with one call.

```python
ds = promptkeep.dataset("REVIEW_SYSTEM", version=4, where="score > 0.8", limit=500)
ds.to_dspy()        # dspy.Example list
ds.to_jsonl("review_v4.jsonl")
ds.to_promptfoo()
```

This is the integration play: **promptkeep is the memory, DSPy is the optimizer.** A one-line
export into the tool that already does this well beats a mediocre in-house version, and it costs
us a day instead of a year.

**3. Version comparison.** The thing people actually want before they want optimization.

```python
promptkeep.compare("REVIEW_SYSTEM", 4, 5)
# version  runs   avg_score  check_pass  avg_latency  avg_tokens  cost
#       4  1,203      0.71         88%        1,340ms      2,100  $12.40
#       5    847      0.79         94%        1,190ms      1,950  $8.10
```

Plus A/B routing, so the comparison has honest data behind it rather than sequential
before/after with everything else changed too:

```python
review = Prompt.variants("REVIEW_SYSTEM", {4: 0.5, 5: 0.5})   # weighted; run records which fired
```

**4. Then, and only then, an optimizer.** Ships as `promptkeep[optimize]`, or not at all if the
DSPy export makes it unnecessary. The naive version — sample high-scoring runs, ask a strong
model to propose N template rewrites, evaluate each against the held-out set using the existing
post-checks as the metric, register winners as new versions under the same name — is maybe two
weeks of work *given* pieces 1–3. It is worthless without them.

Be publicly honest about the failure modes: LLM-as-judge correlates with human preference in the
70–85% range depending on task, judges are biased toward verbosity and their own outputs, and
optimizing against a biased judge produces prompts that game the judge. Retrospective
optimization without ground truth is a real technique with real limits, and saying so is better
positioning than overclaiming.

---

## 6. Everything else v1 needs

Less exciting, equally load-bearing.

**Provider coverage.** Extract `integrations/base.py` — a provider adapter interface
(`extract_prompts(request)`, `normalize_response(response)`, `stream_proxy(...)`) with the
OpenAI wrapper refactored as its first implementation. Then: OpenAI Responses API (also unlocks
automatic conversation chaining), Anthropic `messages.create`, and a LiteLLM adapter — which
gets 100+ models for one adapter's effort and is the highest-leverage single integration
available.

**CLI.** Genuinely the highest adoption-per-line-of-code item on this list. A dev tool people
can *see* gets shared.

```bash
promptkeep list                          # all prompts, version counts, run counts
promptkeep versions REVIEW_SYSTEM
promptkeep diff REVIEW_SYSTEM 4 5        # coloured diff
promptkeep runs REVIEW_SYSTEM --version 5 --limit 20
promptkeep convo user-42-session-9       # full thread, with versions per turn
promptkeep stats REVIEW_SYSTEM           # the compare table
promptkeep serve                         # localhost read-only viewer over the file
promptkeep export --format jsonl
```

`serve` is the one to be disciplined about: **read-only, localhost, no auth, no accounts, no
"sign in to see more".** The moment it becomes a product surface, we've become the thing we're
positioning against.

**Production controls.**

```python
promptkeep.configure(
    sample_rate=0.1,                    # keep 10% of runs; always keep errors and failed checks
    redact=my_redaction_fn,             # runs before anything is stored
    store_outputs=False,                # metadata only, for regulated environments
    retention_days=90,
)
```

Redaction is a hard blocker for anyone in health, finance or EU-facing products. Without it,
`rendered_text` and `output_text` are a compliance problem sitting in a file.

**Cost tracking.** A bundled model→price table plus `cost_usd` on runs. Cheap to build, and it's
the number that gets a tool shown to a manager.

**Version pinning.** Load a stored version instead of the code literal — the "prompt registry"
use case, and the thing that lets a prompt change without a deploy.

```python
review = Prompt.load("REVIEW_SYSTEM", version=4)
review = Prompt.load("REVIEW_SYSTEM")            # latest
```

**Storage abstraction.** Not a Postgres backend — just the interface that makes one possible
later, so multi-process and multi-machine deployments have an answer when they ask. SQLite's
single-writer limit is fine for one service; it isn't fine for twelve.

**Fix the small things.** `__version__` from `importlib.metadata`, `testing.py` → `examples/`,
`testing.db` gitignored and deleted.

---

## 7. Milestones

Each one is releasable and independently useful. Nothing here requires the next thing to exist.

### v0.3 — Foundations *(~3 weeks)*
Background writer + `flush()` + fork safety · async `@prompt` · conversations (schema v3, context
manager, `history.conversation`, `replay()`) · `sample_rate` and `redact` · hygiene fixes ·
**CI on day one** (matrix 3.9–3.14, ruff, coverage gate).
*Ships the thing that makes it safe to deploy.*

### v0.4 — Checks *(~3 weeks)*
Pre/post checks, `Verdict`, `RunHandle`, `promptkeep.call()` · schema v4 · `promptkeep.feedback()`
· suppression contextvar · docs page with real examples (PII gate, groundedness judge, JSON
schema validation).
*Ships the differentiator and starts accumulating labels.*

### v0.5 — Reach and visibility *(~4 weeks)* — **public beta**
Provider adapter interface · Responses API (+ automatic conversation chaining) · Anthropic ·
LiteLLM · full CLI including `serve` · cost tracking · `compare()` and variants · docs site.
*This is the version worth telling people about.*

### v1.0 — Stability *(~4 weeks after 0.5 lands with users on it)*
API freeze and a documented deprecation policy · `Prompt.load()` pinning · dataset export
(`to_dspy` / `to_jsonl` / `to_promptfoo`) · storage backend interface · published benchmarks ·
retention · **launch**.

Roughly four months at a sustainable pace for one person. Compress it by cutting scope from 0.5,
not by cutting CI or docs from 0.3.

---

## 8. Engineering standards for a real developer tool

The library is good. The repository around it is not yet a project — and for an OSS dev tool,
the repository *is* the product surface. Someone deciding whether to depend on this spends
ninety seconds on the GitHub page.

**Before anything else ships:**

- **CI** — `.github/workflows/ci.yml`, matrix across 3.9–3.14 and Linux/macOS/Windows (SQLite
  path handling differs on Windows and we'd never know), ruff format + check, pytest with
  coverage floor at 85%, required on PRs.
- **Release automation** — tag → build → publish via PyPI Trusted Publishing (OIDC, no long-lived
  token in secrets), GitHub Release with generated notes.
- **A `CHANGELOG.md`** in keep-a-changelog format, written for humans, updated in the same PR as
  the change.
- **`CONTRIBUTING.md`, `SECURITY.md`, issue and PR templates.** A bug template that asks for
  Python version, promptkeep version, provider and a minimal repro saves hours per issue.
- **Semver, stated explicitly**, plus which names are public API. Everything in `__all__` is
  covered; everything with a leading underscore is not.
- **Type coverage** — `mypy --strict` on `src/`, in CI. `py.typed` already ships, which means we
  are already making a promise here.
- **Benchmarks in CI** — µs per render, µs per tracked call, memory per 10k runs. Publish them in
  the README. This is the single most effective way to preempt "won't this slow down my app?"

**Documentation.** MkDocs Material on GitHub Pages, at `promptkeep.dev` or a GH Pages URL.
Structure: 30-second quickstart → concepts (prompt/version/run/conversation/check) → guides
(FastAPI, notebooks, CI, multi-provider) → API reference → design decisions. The last one
matters more than it looks: publishing *why* normalized version matching exists is what
convinces a reader that the library was thought about rather than assembled.

**README rewrite.** Current one explains features. It should sell in the first fifteen seconds:
one sentence on the problem, one code block showing the three-line diff to adopt it, one
terminal GIF of `promptkeep diff` showing a prompt's history, then badges, then detail. The GIF
does more work than any paragraph.

---

## 9. Launch and positioning

The package is on PyPI. That is publishing, not launching — nobody knows it exists. The launch
should happen once, on 1.0, with everything ready.

**Positioning.** One line, everywhere, identical: *"Git for your prompts — versioning, run
history and checks in a local SQLite file. No server, no account."*

Against the field: Langfuse/LangSmith/Braintrust are platforms — great at scale, heavy to start.
DSPy optimizes but doesn't remember. Promptfoo evaluates offline but doesn't touch production.
We are the small local thing you add in one line and never think about again — and the honest
line is "start with promptkeep, graduate to a platform when you need a team dashboard." Being
explicitly the *first* tool someone reaches for is a better position than pretending to be the
last.

**Sequence.**

1. **Dogfood first.** Run it in Mercity's own production for a month before telling anyone. Every
   sharp edge found internally is one not found publicly on launch day.
2. **Get three external users on 0.5.** Not stars — users, with a channel to complain in. The
   entire value of a beta is the complaints.
3. **Write the post, not the announcement.** The thing that travels is a technical article with
   an opinion: *"Why we version prompts by normalized structure, not text"*, or *"Your prompt
   history is a compliance problem you haven't noticed."* Teach something; mention the library
   as the artifact.
4. **Launch day** — Show HN in the morning US time (title stating what it does, no adjectives;
   first comment is the author explaining why it exists and what it deliberately doesn't do),
   r/LocalLLaMA and r/Python, an X/LinkedIn thread built around the GIF, and PRs adding it to the
   relevant awesome-lists.
5. **Then be present.** Respond to every issue within 24 hours for the first month. Early
   responsiveness is what converts a curious visitor into a contributor, and it's more
   determinative of whether an OSS project survives than the code is.

**What "working" looks like at 3 months:** 500+ GitHub stars is vanity; **weekly PyPI downloads
that keep rising after launch week is the real signal**, along with issues filed by people we've
never met and at least one PR from a stranger. If downloads spike and flatten, the library got
looked at but not adopted — and the fix for that is in section 8, not section 9.

---

## 10. Open questions

Worth deciding before 0.3, not during.

1. **Default `write_mode`.** Background is right for services, surprising in notebooks and
   scripts where you expect the row to exist immediately after the call. Auto-detect (sync under
   pytest and in interactive REPLs, background otherwise) is convenient but magic, and magic
   defaults are how libraries lose trust. Lean toward background + a loud docs note + `flush()`.
2. **`Prompt` requires a name today.** For adoption, an auto-name derived from the call site
   (module + line) would let people try the library with a one-character diff. It also creates
   unstable identities that break the moment someone reformats a file. Probably no — but decide
   deliberately.
3. **Pre-check timeout: fail open or fail closed?** Fail open is right for a groundedness check
   and wrong for a PII gate. Likely a per-check `on_timeout` with fail-open as the default and a
   very visible warning in the docs for security use cases.
4. **Does `serve` exist at all?** It's the best demo we could have and the first step onto a road
   that ends in a SaaS product. Ship it read-only and localhost-only, or don't ship it.
5. **Do we chase enterprise at all?** Redaction, retention and a Postgres backend are the price
   of entry for regulated buyers, and pursuing them will bend the roadmap away from the
   zero-config thesis. Fine to answer "not before 1.0" — just answer it.
6. **Naming: `checks` vs `verification` vs `guards`.** Whatever is chosen becomes public API and
   is expensive to change afterwards. Decide once, in 0.4.

---

*Companion document: `STATE.md` — where the project stands today.*

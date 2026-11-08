# promptkeep

[![PyPI](https://img.shields.io/pypi/v/promptkeep.svg)](https://pypi.org/project/promptkeep/)
[![Python](https://img.shields.io/pypi/pyversions/promptkeep.svg)](https://pypi.org/project/promptkeep/)

Prompts as first-class objects: named, versioned templates with lineage tracked in SQLite,
variable rendering, a decorator for computed prompts, and a transparent OpenAI SDK wrapper
that records every run (prompt version + variables + output + usage). Multi-turn
conversations are tracked turn by turn, run writes happen off your hot path by default,
and `promptkeep serve` opens a local dashboard over the whole history — no server, no
account, one SQLite file you own.

## The basics

```python
from promptkeep import Prompt

prompt = Prompt(
    text="You are a code reviewer. Focus on {var1}.",
    variables={"var1": "correctness"},
    name="REVIEW_SYSTEM",          # the prompt's stable identity
)

prompt.text     # rendered string — safe to pass to any SDK
prompt.raw      # raw template, placeholders intact
prompt.version  # 1 — bumps automatically whenever the template text changes
```

Same `name` + edited text ⇒ a new version row in SQLite (deduplicated by content hash).
Variables are *run data*, never versions — change them freely.

### How version matching works

By default, matching ignores placeholder *names*: renaming `{var1}` to `{x}` is **not** a new
version — only the static text and the placeholder structure count. Structure still matters:
positions, repetition patterns (`{a}..{a}` is one value used twice, so it differs from
`{a}..{b}`), attribute paths, and format specs all distinguish versions.

If you *want* renames to count — say, variable names carry meaning in your workflow — opt out
per prompt with `exact_match=True` (works on the decorator too):

```python
p = Prompt("Grade the essay on {var1}.", name="GRADER", exact_match=True)
# now "Grade the essay on {x}." registers as a NEW version under GRADER
```

Rendering is lenient by default: unknown `{placeholders}` and JSON braces in the template
pass through untouched. Use `strict=True` (per prompt or via `configure`) to raise instead.

## Computed prompts

```python
from promptkeep import prompt

@prompt(name="REVIEW_SYSTEM")
def review_sys_prompt(var1="some value", n_examples=3):
    examples = "\n".join(load_examples(n_examples))
    return f"You are a reviewer.\n{examples}\nFocus on {{var1}}."

p = review_sys_prompt(var1="security")   # -> Prompt (raw + rendered + version)
```

The function returns the raw template; the call's arguments become the variables. An
`async def` builder works the same way — `await review_sys_prompt(...)` yields the Prompt.

## OpenAI integration

```python
from openai import OpenAI
from promptkeep import wrap

OpenAI = wrap(OpenAI)                 # or: client = wrap(OpenAI(...))
client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-5.5",
    messages=[
        {"role": "developer", "content": prompt},   # Prompt object, directly
        {"role": "user", "content": "How do I check isinstance?"},
    ],
)
```

The API receives a plain string; a *run* is recorded linking this prompt version to the
variables used, the rendered text, the model, the output, token usage, cost, and latency.
Streaming, async clients, and multi-part content are supported. Tracking failures never
break the API call. Unwrapped clients work too — just pass `prompt.text`.

**Cost** is whatever the endpoint itself reports: OpenRouter (the same chat shape — point the
OpenAI SDK at it) returns `usage.cost` on every response, stream or not, and that lands on the
run as `cost_usd`. OpenAI's own API reports no cost, so those runs read `None` — promptkeep
never estimates one from a price table that would be stale by next month.

### The Responses API

The same wrapped client tracks `client.responses.create` too — a Prompt works as
`instructions`, as the `input` string, or inside `input` items:

```python
first = client.responses.create(model="gpt-5.5", instructions=prompt, input="Longest river?")
second = client.responses.create(
    model="gpt-5.5", instructions=prompt, input="And the second?",
    previous_response_id=first.id,          # <- this is all it takes
)

history.conversation(f"response:{first.id}").turns    # both calls, in order
```

Because a chained call names its predecessor, promptkeep follows the chain by itself: a call
with `previous_response_id` joins the conversation of the run that produced that response,
and the first link — recorded before there was a chain — is adopted into it as turn 0. No
`with conversation(...)`, no ids to thread through. An explicit conversation still wins, a
chain begun inside one stays in it, and a chain promptkeep never tracked (no Prompt, no
checks) is not followed: conversations are never inferred, only read off what the request
says. Streaming (`stream=True`), async clients, checks and cost work as on chat
completions; `responses.stream()` / `responses.parse()` are separate SDK methods and pass
through untracked.

OpenAI `chat.completions` and the Responses API are the built-in surfaces — which also
covers OpenRouter and every other OpenAI-compatible endpoint. The wrapper is built on a
small adapter interface (`promptkeep.integrations.ProviderAdapter`), so another SDK is one
adapter away — `register_adapter()` teaches `wrap()` a new client shape without touching
the tracking machinery.

## Conversations

Group calls into a session and promptkeep records the whole thread — every turn gets a
row, whether or not a `Prompt` was involved, so multi-turn chats can be replayed in full:

```python
import promptkeep

with promptkeep.conversation("user-42-session-9", user_id=42):
    client.chat.completions.create(...)   # turn 0
    client.chat.completions.create(...)   # turn 1
# async code: `async with` works too
```

Call sites that can't wrap a block can attach explicitly (it always wins over the
enclosing block, and is stripped before the request reaches the provider):

```python
client.chat.completions.create(..., promptkeep_conversation="user-42-session-9")
```

Each turn stores only what's *new* — the latest user message as `input_text` and the
model's reply as `output_text`. Earlier turns are earlier rows, and the system prompt is
already covered by version lineage, so nothing is duplicated as history grows. Which
prompt version drove each turn is recorded inline — so "which version was live at turn 6
of this session?" is a lookup, not an investigation.

### Branches: regenerations, retries, sub-agents

A conversation is a tree, not just a list. By default each turn continues from the one
before it; a turn that continues from somewhere else — a regenerated reply, a retry, an
edit-and-resend, a sub-agent call — names its parent:

```python
first = client.chat.completions.create(...)              # turn 0
client.chat.completions.create(...)                      # turn 1: the reply the user disliked
client.chat.completions.create(..., promptkeep_parent=first)   # turn 2: regenerated from turn 0
```

`promptkeep_parent=` takes a response, its `response.promptkeep` handle, or a bare
`run_key`, and is stripped before the request goes out. On the Responses API it is
automatic: `previous_response_id` already names the parent, so pointing it at an older
response *is* a branch. Reading a tree back:

```python
convo = history.conversation("user-42-session-9")
convo.forks                      # {2: 0} — turn 2 branches from turn 0
convo.leaves                     # the tip of every branch
convo.path(run_key)              # the turns that led to one run, root first
convo.replay()                   # the latest branch — what the model actually saw
convo.replay(upto=run_key)       # any other branch
```

`replay()` follows one branch, so a reply that was regenerated away is not replayed as if
the model had seen it. A conversation nobody branched reads exactly as before.

## Agent traces

A conversation groups messages; an agent trace groups the work around one task.
Wrap the task in `agent()` and each application-executed tool in `tool()`; a wrapped
model call inside either one becomes a linked `llm` span. No provider tool protocol is
parsed, and the scope never invokes your tool or decides whether to await it:

```python
with promptkeep.agent("research", input={"question": question}) as task:
    with promptkeep.tool("web_search", input={"query": query}) as search:
        results = web_search(query)
        search.output(results)              # omitted output means "not recorded"

    response = client.chat.completions.create(...)
    task.output(response.choices[0].message.content)

promptkeep.flush()                          # default background mode: drain the queue
trace = history.trace(task.trace_key)
trace.status                                # root exit; None if its row is missing
trace.spans                                 # agent, tool and llm spans in start order
trace.error_count                           # caught child failures/blocks, separate from root exit
trace.total_tokens, trace.total_cost        # counted once per physical model call
trace.cost_complete                         # False when some call reported no cost
trace.versions_used                         # {"REVIEW_SYSTEM": [4]}
```

`async with promptkeep.agent(...)` / `async with promptkeep.tool(...)` work the same way.
Entering a tool with no active agent records nothing and logs a warning rather than
raising, so helper code (including one that runs in a forked worker) stays safe to call.
Nested agents and tools stay in the same trace, and a model call inside a tool is that
tool's child. Pass the provider's tool-call id to parent a tool under the model call that
requested it: `tool("web_search", call_id=tool_call.id)`. Call `handle.fail(message)` when
a tool returns normally but failed (a non-zero exit code, an `is_error` result) — the span
becomes an error without an exception. The handle is a `promptkeep.Span`, exposing
`span_key` immediately; `trace_key` is known after entry. Keys identify attempts, not rows:
sampling, redaction, a full queue or `write_mode="off"` can drop the row.
`history.list_traces(limit=100)` lists root agent spans only.

A caught tool/provider error is a child failure, not a root failure. A trace can be
partial by design: a missing root, parent, or linked run is reported as missing, never
reconstructed. Agent/tool input, output and errors go through `redact` before queueing;
each payload is capped at 64 KiB with a visible marker, and names stay readable, like
prompt names. Sampling chooses one cohort for the trace's normal model steps and still
keeps errors, blocks and non-ok verdicts, so a sampled-out task that hits a problem can
appear as a problem-only trace. Retention expires spans in whole traces using their latest
activity; prompt/version rows are never deleted.

Context follows normal `contextvars` inheritance (including async tasks), not arbitrary
threads or processes. Enter a new scope inside each thread. A child task or stream started
inside a scope keeps the captured trace even if it finishes after the root; await/close it
before leaving the root when you want the root's status and duration to cover it.

## Checks

Run your own function before a call (a gate that can stop it) and after it (an
audit that grades the answer). A check is just a function returning a `Verdict`
— plain code or another LLM call, promptkeep doesn't care which.

```python
from promptkeep import Prompt, check, Verdict

@check.pre(name="no_pii")                     # runs before the request is sent
def block_pii(ctx):
    return Verdict.block("email in prompt") if "@" in ctx.rendered else Verdict.ok()

@check.post(name="grounded")                  # runs after; async by default
def is_grounded(ctx):
    score = judge(ctx.output_text)            # e.g. an LLM-as-judge call
    return Verdict.from_score(score, threshold=0.7)

review = Prompt("You are a reviewer.", name="REVIEW_SYSTEM",
                pre=[block_pii], post=[is_grounded])
```

A `Verdict` is `ok()`, `warn(msg)`, `block(msg)` (pre only — stops the call), or
`rewrite(text)` (pre only — substitutes the outgoing message). The result rides
back on the response, and existing code is untouched:

```python
response = client.chat.completions.create(...)
response.choices[0].message.content     # unchanged
response.promptkeep.verification        # "ok" | "warn" | "failed" | "pending"
response.promptkeep.checks              # each check's verdict
response.promptkeep.run_key             # the run's identity — history.checks(run_key)
response.promptkeep.wait(timeout=5)     # block for async post-checks if you want them
```

Prefer an explicit shape for new code? `call()` returns the result directly:

```python
from promptkeep import call

result = call(client, model="gpt-5.5", messages=[...])
result.text            # the reply
result.verification    # "ok" | "warn" | "failed" | "pending"
result.run_key
# async: await promptkeep.acall(client, ...), and response.promptkeep.awaited()
```

A blocked call raises `PromptBlocked` by default; `configure(on_block="return")`
makes it return a response-shaped stub instead so a service can degrade. Checks
attach globally (`configure(pre=[...])`), per prompt, or per call
(`promptkeep_pre=[...]`), merged most-specific-wins.

Because a guardrail is only useful if it can't take the request down with it,
checks are isolated from your call path: a crashing or slow check never breaks
the call (it fails open, recorded), each check's timeout measures its own
execution — so a burst of concurrent calls can't make a fast check look slow —
and on async calls nothing runs on the event loop, not even a streamed
response's post-checks. A check reads the current turn as `ctx.last_text` even
when the message carries image content, and a `pre` rewrite is applied
consistently to the outgoing request, the post-check that audits it, and the
row that's stored. The record keeps both sides of a rewrite — the turn as the
caller passed it (`original_input_text`) next to what was sent (`input_text`) —
and the rewriting check's verdict carries the replacement text, so an audit can
see exactly what changed and which check changed it. An LLM-judge check doesn't
record itself.

Checks run on every call path — sync or async, streaming or not (post-checks
fire once a stream finishes). Every verdict is saved to the `checks` table, tied
to the run it graded (and, when a `Prompt` drove the call, that prompt's
version). Runs are identified by a `run_key` minted at call time, so a checked
call goes through the background writer like any other — nothing on the request
path waits for the database. Async verdicts queue behind their run row, and
`promptkeep.flush()` waits for any post-check still running before draining the
queue: once it returns, every verdict is on disk.

## Older versions, back as Prompts

Your code only ever holds the *current* template. `Prompt.variants()` brings the stored ones
back as real Prompt objects, oldest first:

```python
v4, v5 = Prompt.variants("REVIEW_SYSTEM")[-2:]
v4.version, v4.raw                                         # 4, the template as it was

chosen = random.choice([v4, v5]).format(focus="security")  # a home-made A/B split
client.chat.completions.create(model=..., messages=[{"role": "system", "content": chosen}, ...])
```

Each variant records its runs under its own version — so `promptkeep stats REVIEW_SYSTEM`
compares the two on real traffic — and loading one never creates a version. A version stores
a template, not variables (those are run data), so `.format(...)` them in.

### Loading a version: the prompt registry

`Prompt.load()` takes one version — pinned, or the latest — so the template comes from the
database instead of a literal in your code, and the prompt in play can change without a
deploy:

```python
review = Prompt.load("REVIEW_SYSTEM", version=4)      # pinned
review = Prompt.load("REVIEW_SYSTEM")                 # the newest version
review = Prompt.load("REVIEW_SYSTEM", pre=[no_pii])   # checks aren't stored, so attach them here
```

Register a new version from anywhere that writes to the same file — a script, a notebook,
another service — and the next `load()` picks it up. "Latest" is the highest version number:
going back to an older template in code doesn't renumber it. It raises `ValueError` for an
unknown prompt or version. It reads the database, so call it at startup or per request, not
at import time.

## Feedback

Checks label a run automatically; `feedback()` is the label a person (or a downstream
system) gives it later — the thumbs-down, the support ticket, the eval score:

```python
response = client.chat.completions.create(...)
key = response.promptkeep.run_key      # every recorded call carries one — store it with your own records

promptkeep.feedback(key, score=1.0, label="thumbs_up")
promptkeep.feedback(key, score=0.0, label="hallucination", comment="invented a citation")

history.checks(key)                    # verdicts and feedback together; phase == "feedback"
```

It lands in the same `checks` table as the verdicts (so later analysis reads one label
store), travels the same write path — queued behind its run in background mode, passed
through `redact` — and a run can collect any number of them. `feedback(None, ...)` is a
no-op, which is what `run_key` is when a run wasn't stored (sampled out, tracking off), so
the call is always safe to make.

## Datasets

Everything promptkeep records is raw material for evaluating a prompt — or optimizing it.
`promptkeep.dataset()` picks out the runs worth learning from and hands them to the tools that
do that work:

```python
ds = promptkeep.dataset("REVIEW_SYSTEM", version=4, passed=True, limit=500)

ds.to_jsonl("review_v4.jsonl")          # neutral: variables, input, output, labels per line
trainset = ds.to_dspy()                 # dspy.Example objects, inputs marked (needs dspy)
ds.to_promptfoo("review_tests.jsonl")   # promptfoo test cases: `tests: file://review_tests.jsonl`
```

Only completed runs with a reply are included. The filters combine: `version=`; `passed=True`
(every check returned ok) or `passed=False` (the failures, for a negative set);
`feedback="thumbs_up"` (runs given that label); `min_feedback=0.8` (average feedback score);
and `limit=` — the newest N *after* filtering. An example's inputs are the run's variables plus
the user turn when the call had one; its output is the reply. promptfoo cases carry the
recorded reply as `metadata.reference_output` and no assertions — what counts as a pass is
yours to say. promptkeep is the memory; DSPy and promptfoo are the optimizer and the eval
harness.

## History

```python
from promptkeep import history

history.versions("REVIEW_SYSTEM")            # lineage, oldest first
print(history.diff("REVIEW_SYSTEM", 1, 3))   # unified diff between versions
history.runs("REVIEW_SYSTEM", version=3)     # recorded runs, newest first

convo = history.conversation("user-42-session-9")
convo.turns                                  # ordered turns: input, output, version, usage
convo.metadata                               # whatever you attached at the start
convo.versions_used                          # {"REVIEW_SYSTEM": [4, 5]} — what drove it
convo.total_tokens, convo.duration           # usage summed; wall-clock seconds
convo.total_cost                             # dollars, as the provider reported them (or None)
convo.replay()                               # -> messages list, ready to send again
convo.replay(system=new_prompt)              # ...against a different prompt version

history.stats("REVIEW_SYSTEM")               # per version: runs, errors, pass rate, scores, cost
history.list_prompts()                       # every prompt + version/run counts
history.list_conversations()                 # every session + turn counts
history.list_conversations(prompt="REVIEW_SYSTEM", version=4)   # sessions v4 drove
history.list_traces()                        # newest root agent spans, bounded (no payloads)
history.trace(task.trace_key)                # one task's spans, payloads + linked runs
history.all_runs()                           # everything, newest first
```

`replay()` walks the completed turns in order and emits the system prompt that was in play
(again whenever it changed mid-session), then each user message as it was actually sent and
the assistant's reply — the raw material for re-running a session against a new version.

## Command line

The same history, from a terminal — plain text, pipeable, no extra dependencies:

```bash
promptkeep list                              # every prompt, with version and run counts
promptkeep versions REVIEW_SYSTEM            # the lineage (--full prints whole templates)
promptkeep diff REVIEW_SYSTEM 4 5            # what changed (coloured on a terminal)
promptkeep runs REVIEW_SYSTEM --version 5    # recorded calls, newest first (no name: all)
promptkeep convo user-42-session-9           # a conversation, turn by turn, with its labels
promptkeep stats REVIEW_SYSTEM               # how each version has performed
promptkeep export --prompt REVIEW_SYSTEM -o runs.jsonl   # runs + their labels, as JSON lines
```

```
$ promptkeep stats REVIEW_SYSTEM
VER  RUNS  ERRORS  BLOCKED  CHECKS OK  SCORE  FEEDBACK  LATENCY  TOKENS    COST
v4   1203       9        2        88%   0.71      0.64   1340ms    2100  $12.40
v5    847       3        0        94%   0.79      0.81   1190ms    1950   $8.10
```

`stats` is the "did the change actually help?" table: per version, how many runs and how
many failed, the share of checked runs whose every verdict was ok, the average check score
next to the average `feedback()` score, latency, tokens and reported cost
(`history.stats(name)` returns the same rows). Every command takes `--db PATH`; none of them
writes — a mistyped path is an error, not a new empty database.

## Local dashboard

```bash
pip install "promptkeep[serve]"
promptkeep serve                 # http://127.0.0.1:8420, reads ./.promptkeep.db
promptkeep serve --db path/to/prompts.db --port 8420
```

A read-only web UI over the same SQLite file: every prompt with its full version lineage
and colored diffs between any two versions, a filterable runs explorer, and turn-by-turn
conversation transcripts with the driving prompt version shown inline. Works fully
offline — no CDN assets, no account, light/dark theme automatic. The server dependencies
(FastAPI, uvicorn, Jinja2) are an optional extra; the core library never needs them.

## Background writes

By default run rows are persisted **off your hot path**: the wrapped call returns
immediately and a background thread batches writes into the DB. Version registration
stays synchronous (`.version` is a value you read back), and a full queue drops oldest
rather than growing unbounded — telemetry must never take your app down with it.

The one visible consequence: a row may land a moment after the call returns. Scripts and
notebooks that read their own writes immediately should either flush or switch modes:

```python
promptkeep.flush(timeout=5)                   # block until everything queued is on disk
promptkeep.configure(write_mode="sync")       # or: write before the call returns
```

`flush()` covers check verdicts too: it waits for async post-checks still running,
then drains the queue, and returns `False` if the timeout ran out first.

An `atexit` hook flushes automatically on interpreter shutdown, so short-lived scripts
don't lose rows. `write_mode="off"` drops run telemetry entirely (versioning still works).

## Production controls

Three knobs for running this in anger — a busy service that doesn't need every row, a
regulated one that must never store certain text, and one that must not keep it forever:

```python
promptkeep.configure(
    sample_rate=0.1,          # store 10% of uneventful runs ($PROMPTKEEP_SAMPLE_RATE)
    redact=scrub_pii,         # str -> str, applied to every stored text field
    retention_days=90,        # delete history older than 90 days ($PROMPTKEEP_RETENTION_DAYS)
)
```

**Sampling** never drops what you'd want to look at: errors, blocked calls, and any run
whose checks returned something other than `ok` are always stored. Inside a conversation
the decision is made once per session (derived from the session's id, so every worker
process agrees), so a kept conversation is complete rather than full of holes. `0.0` means
"store only problems". An agent trace uses one cohort for its root and normal model steps;
errors, blocks and non-ok known verdicts can still leave a problem-only trace, without
recovering earlier steps that were already dropped. A trace that runs inside a
conversation follows that session's one keep-or-drop decision, so a session's traces
survive or drop together — problems still keep their own problem-only traces. Version
lineage is never sampled. Because the decision is made when the run is recorded, a verdict
from an *async* post-check can't rescue a dropped run — use `mode="blocking"` for a check
whose failures must always be kept. A sampled-out run's `response.promptkeep.run_key` is
`None`; its verdicts are still on the handle.

**Redaction** runs before anything is written, in every write mode — plaintext never even
enters the background queue. The hook sees each stored text field of a run (rendered prompt,
input and output, the JSON-encoded variables and request params, error text), of a check
verdict (message, rewritten text), and of an agent span (input/output payload, error).
Templates, prompt names, span names and conversation metadata are not passed through it:
templates are code, and the names and metadata are what you attached on purpose. If
the hook raises or returns a non-string, the row is dropped rather than stored unredacted.
`redact=lambda text: ""` keeps metadata only — model, tokens, cost, latency, status and
verdicts, with no prompt or response text at all.

**Retention** deletes recorded history once it is older than `retention_days`. A
conversation is deleted whole, once its *last* turn is that old, so a session still in use
never loses its opening turns; a run outside any conversation goes by its own age; verdicts
and feedback go with their run. Agent spans expire in whole traces based on the latest span
or linked-run activity, so a running root does not live forever. Prompts and versions are
never deleted — they are your code's history, not your traffic's. The sweep rides the write path: at most once an hour per
process, starting with the first run recorded, on the writer thread in background mode. A
failed sweep is logged and retried an hour later, and never costs a run. SQLite reuses the
freed space; to shrink the file itself, run `VACUUM` on it.

## Configuration

```python
import promptkeep

promptkeep.configure(
    db_path="path/to/prompts.db",   # default: ./.promptkeep.db (or $PROMPTKEEP_DB)
    enabled=True,                   # $PROMPTKEEP_DISABLED=1 turns tracking off
    strict=False,                   # raise on missing variables
    write_mode="background",        # "background" | "sync" | "off" ($PROMPTKEEP_WRITE_MODE)
    queue_size=10_000,              # background queue bound (drop-oldest when full)
    flush_interval=0.5,             # seconds the writer waits before a partial batch
    batch_size=100,                 # max rows per write transaction
    pre=[...],                      # global pre-checks (gates), run on every tracked call
    post=[...],                     # global post-checks (audits), run on every tracked call
    on_block="raise",               # blocked pre-check: "raise" PromptBlocked | "return" a stub
    sample_rate=1.0,                # fraction of uneventful runs to store ($PROMPTKEEP_SAMPLE_RATE)
    redact=None,                    # str -> str hook applied to every stored text field
    retention_days=None,            # delete history older than this ($PROMPTKEEP_RETENTION_DAYS)
)
```

## Stability

promptkeep follows semantic versioning. The public API is `promptkeep.__all__`,
`promptkeep.history`, `promptkeep.integrations.__all__`, the `promptkeep_*` call kwargs,
`configure()` and its environment variables, and the CLI's commands. Before 1.0 a minor release
may change it, always with a changelog entry saying what to do instead; anything being removed
first warns with a `DeprecationWarning` for at least one release. Database files upgrade
forward automatically. The full policy is in [docs/API.md](docs/API.md).

## Development

```bash
uv sync                                  # install with dev dependencies
uv run pytest                            # run the test suite
uv run ruff format src tests examples && uv run ruff check src tests examples
```

Supported on Python 3.11–3.14. Changes are recorded in [CHANGELOG.md](CHANGELOG.md).


## Anthropic agents

Install `pip install 'promptkeep[anthropic]'`. Stable SDK releases 1.17.0 and
2.6.0 are tested with sync and async clients:

```python
from anthropic import Anthropic
import promptkeep

client = promptkeep.wrap(Anthropic())
with promptkeep.agent("assistant"):
    response = client.messages.create(
        model="YOUR_MODEL", max_tokens=500,
        messages=[{"role": "user", "content": "Hello"}],
    )
promptkeep.flush()
```

`create(stream=True)`, `.stream()` text/final-message helpers, `.parse()`, and
raw-response helpers are supported. Raw bodies are recorded when parsed; closing
an unread body records incomplete output. Checks, redaction, cache accounting,
and `tool(..., call_id=block.id)` work across these paths. Beta APIs, batch APIs,
automatic tool runners, and cloud-specific clients remain outside the supported
surface. Wrapping Anthropic does not instrument the separate Claude Code process.
See [`examples/anthropic_trace.py`](examples/anthropic_trace.py).

## Saved Codex and Claude Code sessions

```bash
pip install 'promptkeep[codex]'  # needed only for .jsonl.zst decompression
promptkeep import codex rollout.jsonl --db history.db
promptkeep import claude-code transcript.jsonl --db history.db
promptkeep traces --db history.db
promptkeep trace TRACE_KEY --db history.db --json
```

Python entry points are `promptkeep.importers.import_codex()` and
`import_claude_code()`, accepting `path`, optional `db_path`, and `strict=False`.
Imports are synchronous, redacted before staging, and atomic per session. Reimporting
or renaming a file does not duplicate a session. A failed replacement preserves the
previous snapshot. Imports bypass live sampling and reject disabled/off tracking.
Retention uses original activity, so old imported sessions can expire normally.

An explicitly identified model response becomes a model span even when its request
start is absent. Unknown start/end/duration remain `None`; observation time is stored
separately. No provider calls are invented from ordinary assistant text. Claude tool
errors and message ancestry are preserved, including branches and compaction markers.
Hidden reasoning, environment snapshots, and unrelated private metadata are excluded.

`history.imported_session(trace_key)` exposes provenance, diagnostics, usage records,
response totals, and unattributed usage. Cumulative snapshots are never added to
response totals. The source summary may overlap linked model-span metrics; do not add
those two views together. See [`examples/import_codex.py`](examples/import_codex.py).

The two reviewed Hugging Face formats have explicit local-file adapters:

```bash
promptkeep import dataset downloaded.jsonl \
  --dataset build-small-hackathon/hackathon-advisor-codex-traces --db history.db
```

`ultralazr/claude-code-traces` is supported too. `import_dataset(path, dataset=...)`
never downloads data. It reports per-session outcomes and failures, orders Codex
records by their source index, and reuses native session identities.

## Framework integrations

```bash
pip install 'promptkeep[openai-agents,langchain]'
```

```python
import promptkeep

with promptkeep.setup(
    integrations=("openai_agents", "langchain"),
    clients=(openai_client, anthropic_client),
) as installation:
    result = runnable.invoke(
        inputs, config={"callbacks": installation.langchain_callbacks}
    )
    installation.flush(timeout=5)
```

OpenAI Agents SDK uses a tracing processor; LangChain uses native callbacks, also
verified with LangGraph parallel branches and interruption/resume. External IDs
establish parentage. Provider wrappers enrich framework model spans with prompt
versions and checks rather than creating duplicate model spans. Existing application
processors and exporters remain installed. Missing framework sensitive content is
never recovered through another API.

Setup enables only named integrations. Equivalent calls share one installation and
lifetime. `shutdown(timeout=...)` restores only methods the installation still owns.
LangChain requires the explicit callback configuration shown above; this does not
instrument unrelated applications or agent processes.

## OpenTelemetry

Install `promptkeep[otel]`. To ingest spans from an existing Python OTel SDK, attach
`PromptKeepSpanExporter` from `promptkeep.integrations.opentelemetry` to the
application's own span processor. PromptKeep does not replace its tracer provider.

For outbound export, pass an explicit `otlp_endpoint` to `setup()`, with
`otlp_protocol="http/protobuf"` (default) or `"grpc"`. Export uses a bounded worker
queue and the official OTel SDK transports. Payload content is excluded by default;
`OTLPExport(..., include_content=True)` enables already-redacted content explicitly.
Historical imports are not automatically exported. Unknown execution intervals are
skipped, and PromptKeep-origin spans cannot loop back through the inbound bridge.
Call `installation.otlp_export.replay(trace_keys, timeout=...)` to explicitly export
selected stored traces; replay reads the selected database without changing it.
Inbound OpenTelemetry spans stay excluded during replay. Export counters and
skipped-timing counts are available on `installation.otlp_export`.

## Watch explicitly selected session logs

```bash
promptkeep watch codex /path/to/sessions --db history.db
promptkeep watch claude-code /path/to/transcripts --db history.db
```

`promptkeep.watch.Watcher(source, path, interval=1.0)` offers `start()`, `stop()`,
`join(timeout)`, and context-manager usage. It reads appended complete JSONL records,
checkpoints committed offsets, and resumes without duplicating operations. Partial
last lines wait for completion. Rotation and truncation retain previously observed
history. Failed commits do not advance the cursor. Only the selected paths are
watched; compressed files remain saved-import inputs. Watchers never launch or
modify the agents. Stop CLI watching with Ctrl-C or SIGTERM.

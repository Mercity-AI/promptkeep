# promptkeep

[![CI](https://github.com/Mercity-AI/promptkeep/actions/workflows/ci.yml/badge.svg)](https://github.com/Mercity-AI/promptkeep/actions/workflows/ci.yml)
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
variables used, the rendered text, the model, the output, token usage, and latency.
Streaming, async clients, and multi-part content are supported. Tracking failures never
break the API call. Unwrapped clients work too — just pass `prompt.text`.

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
convo.replay()                               # -> messages list, ready to send again
convo.replay(system=new_prompt)              # ...against a different prompt version

history.list_prompts()                       # every prompt + version/run counts
history.list_conversations()                 # every session + turn counts
history.list_conversations(prompt="REVIEW_SYSTEM", version=4)   # sessions v4 drove
history.all_runs()                           # everything, newest first
```

`replay()` walks the completed turns in order and emits the system prompt that was in play
(again whenever it changed mid-session), then each user message as it was actually sent and
the assistant's reply — the raw material for re-running a session against a new version.

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

Two knobs for running this in anger — a busy service that doesn't need every row, and a
regulated one that must never store certain text:

```python
promptkeep.configure(
    sample_rate=0.1,          # store 10% of uneventful runs ($PROMPTKEEP_SAMPLE_RATE)
    redact=scrub_pii,         # str -> str, applied to every stored text field
)
```

**Sampling** never drops what you'd want to look at: errors, blocked calls, and any run
whose checks returned something other than `ok` are always stored. Inside a conversation
the decision is made once per session (derived from the session's id, so every worker
process agrees), so a kept conversation is complete rather than full of holes. `0.0` means
"store only problems". Version lineage is never sampled. Because the decision is made when
the run is recorded, a verdict from an *async* post-check can't rescue a dropped run — use
`mode="blocking"` for a check whose failures must always be kept. A sampled-out run's
`response.promptkeep.run_key` is `None`; its verdicts are still on the handle.

**Redaction** runs before anything is written, in every write mode — plaintext never even
enters the background queue. The hook sees each stored text field of a run (rendered prompt,
input and output, the JSON-encoded variables and request params, error text) and of a check
verdict (message, rewritten text). Templates, prompt names and conversation metadata are not
passed through it: templates are code, and the metadata is what you attached on purpose. If
the hook raises or returns a non-string, the row is dropped rather than stored unredacted.

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
)
```

## Development

```bash
uv sync                                  # install with dev dependencies
uv run pytest                            # run the test suite
uv run ruff format src tests examples && uv run ruff check src tests examples
```

CI runs the same on Python 3.9–3.14 across Linux, macOS and Windows. Changes are
recorded in [CHANGELOG.md](CHANGELOG.md).

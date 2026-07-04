# promptkeep

Prompts as first-class objects: named, versioned templates with lineage tracked in SQLite,
variable rendering, a decorator for computed prompts, and a transparent OpenAI SDK wrapper
that records every run (prompt version + variables + output + usage).

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

The function returns the raw template; the call's arguments become the variables.

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

## History

```python
from promptkeep import history

history.versions("REVIEW_SYSTEM")            # lineage, oldest first
print(history.diff("REVIEW_SYSTEM", 1, 3))   # unified diff between versions
history.runs("REVIEW_SYSTEM", version=3)     # recorded runs, newest first
```

## Configuration

```python
import promptkeep

promptkeep.configure(
    db_path="path/to/prompts.db",   # default: ./.promptkeep.db (or $PROMPTKEEP_DB)
    enabled=True,                   # $PROMPTKEEP_DISABLED=1 turns tracking off
    strict=False,                   # raise on missing variables
)
```

## Development

```bash
uv sync          # install with dev dependencies
uv run pytest    # run the test suite
```

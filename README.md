# prompt-manager

Prompts as first-class objects: named, versioned templates with lineage tracked in SQLite,
variable rendering, a decorator for computed prompts, and a transparent OpenAI SDK wrapper
that records every run (prompt version + variables + output + usage).

## The basics

```python
from prompt_manager import Prompt

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

Rendering is lenient by default: unknown `{placeholders}` and JSON braces in the template
pass through untouched. Use `strict=True` (per prompt or via `configure`) to raise instead.

## Computed prompts

```python
from prompt_manager import prompt

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
from prompt_manager import wrap

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
from prompt_manager import history

history.versions("REVIEW_SYSTEM")            # lineage, oldest first
print(history.diff("REVIEW_SYSTEM", 1, 3))   # unified diff between versions
history.runs("REVIEW_SYSTEM", version=3)     # recorded runs, newest first
```

## Configuration

```python
import prompt_manager

prompt_manager.configure(
    db_path="path/to/prompts.db",   # default: ./.prompts.db (or $PROMPT_MANAGER_DB)
    enabled=True,                   # $PROMPT_MANAGER_DISABLED=1 turns tracking off
    strict=False,                   # raise on missing variables
)
```

## Development

```bash
uv sync          # install with dev dependencies
uv run pytest    # run the test suite
```

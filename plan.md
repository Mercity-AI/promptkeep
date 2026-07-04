# Prompt Manager — Implementation Plan

A Python library for managing LLM prompts as first-class objects: versioned templates with
lineage tracking (SQLite), variable rendering, a decorator for computed prompts, and a
transparent wrapper around the OpenAI SDK that tracks every run (prompt version + variables +
output) without changing how people write their OpenAI code.

---

## 1. Goals

1. **Prompt as an object, not a string.** `Prompt(text=..., variables=..., name=...)` — the
   `name` is the stable identity; the template text is the versioned content.
2. **Lineage tracking.** Same `name`, different template text ⇒ new version recorded in SQLite.
   Full history of how a prompt evolved is queryable.
3. **Variable rendering.** Templates contain `{var}` placeholders; values come in via a
   `variables` dict (and/or kwargs). Rendered output is a plain string usable anywhere.
4. **Computed prompts via decorator.** `@prompt(name=...)` on a function that builds a prompt
   with real logic, not just substitution. Both the raw template and the rendered result are
   captured so lineage still works.
5. **Zero-friction OpenAI integration.** `OpenAI = wrap(OpenAI)` — after that, users pass a
   `Prompt` object directly as message `content` and the library (a) substitutes the rendered
   string before the request goes out, and (b) records a *run*: which prompt version ran, with
   which variables, against which model, and what came back (output, token usage, latency).
6. **Degrades gracefully.** Without wrapping, `prompt.text` is a plain string — works with any
   SDK, any framework, no lock-in.

## 2. Non-goals (for now)

- No server, no UI, no cloud sync — local SQLite only. (A CLI viewer is a stretch goal.)
- No prompt *optimization* / eval framework — we record runs; we don't judge them.
- No providers beyond OpenAI in v1 (but the wrapper layer is built so Anthropic etc. can be
  added later without touching core).
- Not tracking changes in *variables* as lineage — variable values are run-scoped data, not
  prompt identity. They're recorded per-run, never as versions.

---

## 3. Core concepts & data model

Three entities, strictly layered:

| Entity | Identity | What changes it | Where stored |
|---|---|---|---|
| **Prompt** | `name` (unique) | never — it's the ID | `prompts` table |
| **Version** | hash of raw template text | any edit to the template | `prompt_versions` table |
| **Run** | auto id | every wrapped LLM call | `runs` table |

- A **Prompt** is the named lineage ("REVIEW_SYSTEM").
- A **Version** is one concrete template text under that name. Versions are deduplicated by
  content hash: constructing the same text twice does *not* create a new version; editing the
  text does. Version numbers are monotonically increasing per prompt.
- A **Run** links a version to one execution: the variables dict used, the final rendered
  text, the model + request params, the response (output text, usage, response id), status,
  and latency.

### SQLite schema (v1)

```sql
CREATE TABLE prompts (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    created_at  TEXT NOT NULL            -- ISO-8601 UTC
);

CREATE TABLE prompt_versions (
    id             INTEGER PRIMARY KEY,
    prompt_id      INTEGER NOT NULL REFERENCES prompts(id),
    version        INTEGER NOT NULL,     -- 1, 2, 3... per prompt
    template       TEXT NOT NULL,        -- raw text, placeholders intact
    template_hash  TEXT NOT NULL,        -- sha256 of normalized template
    source         TEXT NOT NULL,        -- 'literal' | 'decorator'
    fn_source_hash TEXT,                 -- decorator only: hash of function source
    created_at     TEXT NOT NULL,
    UNIQUE (prompt_id, template_hash),
    UNIQUE (prompt_id, version)
);

CREATE TABLE runs (
    id             INTEGER PRIMARY KEY,
    version_id     INTEGER NOT NULL REFERENCES prompt_versions(id),
    variables      TEXT,                 -- JSON dict as passed by the user
    rendered_text  TEXT NOT NULL,
    provider       TEXT NOT NULL,        -- 'openai'
    model          TEXT,
    request_params TEXT,                 -- JSON (temperature, etc., minus messages)
    response_id    TEXT,
    output_text    TEXT,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    total_tokens      INTEGER,
    latency_ms     INTEGER,
    status         TEXT NOT NULL,        -- 'ok' | 'error'
    error          TEXT,
    created_at     TEXT NOT NULL
);
CREATE INDEX idx_runs_version ON runs(version_id, created_at);
```

- DB opened in WAL mode; one connection per thread (`threading.local`) so wrapped calls from
  multi-threaded apps don't fight over a cursor.
- `schema_version` pragma (`user_version`) + tiny forward-only migration runner, so the schema
  can evolve without breaking existing DBs.

### DB location & configuration

Default: `./.prompts.db` in the current working directory (a project-level artifact, like
`.env`). Overridable by:

1. `prompt_manager.configure(db_path=..., enabled=...)` — explicit wins.
2. `PROMPT_MANAGER_DB` env var.
3. `PROMPT_MANAGER_DISABLED=1` kills all persistence (Prompt still renders fine — critical for
   CI and for users who only want the rendering ergonomics).

**Registration timing:** constructing a `Prompt` does **not** hit the DB (prompts are usually
defined at module import time; import-time writes are a footgun — read-only filesystems, test
collection, etc.). The version row is written lazily on first *use* (first render access or
first tracked run), memoized per process so it's one write, not one per call.

---

## 4. Public API

Everything importable from the top-level package:

```python
from prompt_manager import Prompt, prompt, wrap, configure
```

### 4.1 `Prompt` class

```python
prompt = Prompt(
    text="You are a reviewer. Focus on {var1}.",
    variables={"var1": "correctness"},   # optional at construction
    name="REVIEW_SYSTEM",                # required — it's the identity
)

prompt.text        # -> rendered string: "You are a reviewer. Focus on correctness."
prompt.raw         # -> raw template:    "You are a reviewer. Focus on {var1}."
str(prompt)        # == prompt.text
prompt.name        # "REVIEW_SYSTEM"
prompt.version     # int, resolved lazily from the DB (None if tracking disabled)
prompt.format(var1="security")   # -> NEW Prompt with updated variables (immutable style)
prompt.render(var1="security")   # -> rendered str directly (one-shot, no new object)
```

Naming decision: **`.text` = rendered, `.raw` = template.** Rationale: `.text` is what you
reach for 95% of the time (the thing you paste into `content=`), so it gets the short name.
`rendered_text`/`raw_text` aliases can exist but the docs push `.text`/`.raw`.

**Key trick — provenance-carrying strings.** `prompt.text` doesn't return a bare `str`; it
returns `RenderedText(str)` — a `str` subclass that behaves identically everywhere (json
serialization, `+`, f-strings, the OpenAI SDK...) but carries two hidden attributes:
`_pm_prompt` (the Prompt) and `_pm_variables` (the dict used to render). Consequences:

- Unwrapped SDKs receive a real string. Nothing breaks. (User requirement: "even if I'm not
  wrapping it, it should still receive a string.")
- The wrapped OpenAI client can track runs **both** when the user passes the `Prompt` object
  directly as `content` **and** when they pass `prompt.text` — because provenance rides along
  on the string itself. No extra method calls on the prompt, ever.

`Prompt` itself is *not* a `str` subclass (str immutability forces rendering into `__new__`,
raw/rendered duality gets ugly, and `.replace()` etc. silently return plain str). Instead the
wrapper accepts `Prompt | RenderedText | str` as message content, and `Prompt.__str__`
returns `self.text` as a safety net for accidental `f"{prompt}"` usage.

**Immutability:** a `Prompt` is frozen after construction. "Changing" variables produces a new
`Prompt` sharing the same name/template (⇒ same version). This is what makes version identity
clean — the object can never drift away from the hash it registered under.

### 4.2 Rendering rules

- Syntax: Python `{var}` placeholders, rendered via `format_map`.
- **Lenient by default**: missing variables stay as literal `{var}` in the output (via a
  `SafeDict` that returns `"{key}"` for missing keys). Prompts full of JSON examples and code
  braces are the norm, and hard-crashing on `{"key": ...}` inside a prompt is the #1 pain of
  naive `.format` use. `{{` / `}}` escaping still works for users who want to be explicit.
- `strict=True` (per-Prompt or global via `configure`) raises `MissingVariableError` listing
  every unresolved placeholder — for people who want the guardrail.
- `prompt.placeholders` -> `set[str]` of declared variables (parsed with `string.Formatter`),
  so tooling/tests can validate coverage.

### 4.3 `@prompt` decorator (computed prompts)

```python
@prompt(name="REVIEW_SYSTEM")
def review_sys_prompt(var1="some value", n_examples=3):
    examples = pick_examples(n_examples)          # real computation
    return f"You are a reviewer...\n{examples}\nFocus on {{var1}}."

p = review_sys_prompt(var1="security")   # -> returns a Prompt object
p.text   # rendered
p.raw    # the template string the function returned
```

Contract: **the function returns the raw template** (placeholders intact, `{{ }}`-escaped
where needed); the decorator turns it into a `Prompt`, using the call's kwargs as the
`variables` dict. This beats the `(raw, rendered)` tuple idea from the sketch: one return
value, no way for raw and rendered to disagree, and the caller gets a full `Prompt` object
(so `.text`, `.raw`, wrap-tracking all work identically to the class path). Returning a
`Prompt` also satisfies "returns (raw_prompt, rendered_prompt)" — both are on the object.

Versioning semantics for computed prompts (the subtle part):

- The **version identity is the returned template text** (same content-hash dedup as literal
  prompts). If computation makes the template genuinely different (different examples baked
  in), that *is* a different prompt text and gets a new version — correct, if chatty.
- To keep the lineage readable we additionally store `fn_source_hash` =
  sha256(`inspect.getsource(fn)`) on each version row. History queries can then distinguish
  "the code changed" from "the same code produced different text this call".
- Guidance in docs: keep run-varying data in `{placeholders}`, keep the computed part as
  stable as possible. The library works either way; the history is just noisier otherwise.
- Kwargs used for rendering vs. kwargs used only for computation: all call kwargs are recorded
  as the run's `variables`; rendering is lenient, so kwargs without matching placeholders are
  simply ignored by the formatter. No separate declaration needed.

### 4.4 `wrap()` — OpenAI integration

```python
from openai import OpenAI
from prompt_manager import wrap

OpenAI = wrap(OpenAI)              # wrap the class…
client = OpenAI(api_key=...)       # …used exactly as before
# or: client = wrap(OpenAI(api_key=...))   # wrapping an instance also works

completion = client.chat.completions.create(
    model="gpt-5.5",
    messages=[
        {"role": "developer", "content": prompt},        # Prompt object, directly
        {"role": "user", "content": user_question},
    ],
)
```

Mechanics (no monkey-patching of the `openai` module — we only touch objects the user
explicitly passed to `wrap`):

- `wrap(cls)` returns a subclass whose `__init__` calls super then replaces
  `self.chat.completions.create` with a tracking closure around the original bound method.
  `wrap(instance)` does the same replacement on the live instance. Everything else on the
  client is untouched — same attributes, same types, `isinstance` still holds.
- The interceptor:
  1. Walks `messages`; for any `content` that is a `Prompt`, renders it; for `RenderedText`,
     uses it as-is; either way collects `(prompt, variables, rendered)` provenance. Replaces
     content with the plain rendered `str`. Multi-part content (list-of-blocks) handled by
     walking `text` blocks too.
  2. Calls the real `create()`, timing it.
  3. Records one `runs` row **per tracked prompt** in the message list (a system + a user
     prompt in one call ⇒ two runs sharing response metadata), including model, filtered
     request params, output text (`choices[0].message.content`), usage, response id.
  4. On exception: records the run with `status='error'` + the exception text, then re-raises
     unchanged.
- **Tracking must never break the user's call**: every DB write is wrapped in its own
  try/except that logs a warning and continues. A failed insert loses telemetry, never a
  completion.
- **Streaming** (`stream=True`): return a thin iterator proxy that yields chunks through
  untouched while accumulating delta content; the run row is written when the stream closes
  (with whatever usage info the final chunk carries). Same idea for the context-manager form.
- **Async** (`AsyncOpenAI`): same interceptor with `async def` + `await`; detection by
  whether the wrapped `create` is a coroutine function.
- **Responses API** (`client.responses.create`): same pattern, phase 2 of the wrapper —
  chat.completions first since that's the stated usage.
- Wrapper code lives in `integrations/openai_wrapper.py` behind a narrow interface
  (`extract_prompts(messages)`, `record_run(...)`), so an `integrations/anthropic_wrapper.py`
  later is additive.
- `openai` is an **optional dependency** (`pip install prompt-manager[openai]`); core never
  imports it.

### 4.5 History / inspection API

Minimal programmatic access so the DB isn't a black box:

```python
from prompt_manager import history

history.versions("REVIEW_SYSTEM")     # -> [VersionInfo(version=1, template=..., created_at=...), ...]
history.diff("REVIEW_SYSTEM", 1, 3)   # -> unified diff string between two versions
history.runs("REVIEW_SYSTEM", version=3, limit=20)   # -> [RunInfo(...), ...]
```

Stretch goal: `python -m prompt_manager history REVIEW_SYSTEM` CLI over the same functions.

---

## 5. Package layout

```
prompt-manager/
├── pyproject.toml               # hatchling; deps: none (core). extras: openai
├── README.md
├── plan.md                      # this file
├── src/
│   └── prompt_manager/
│       ├── __init__.py          # Prompt, prompt, wrap, configure, history
│       ├── config.py            # configure(), env vars, global settings singleton
│       ├── prompt.py            # Prompt, RenderedText
│       ├── rendering.py         # SafeDict, placeholder parsing, strict mode
│       ├── decorator.py         # @prompt
│       ├── storage.py           # connection mgmt, schema/migrations, upserts, queries
│       ├── tracking.py          # run recording (provider-agnostic)
│       ├── history.py           # versions() / diff() / runs()
│       └── integrations/
│           ├── __init__.py      # wrap() dispatcher (detects openai class/instance)
│           └── openai_wrapper.py
└── tests/
    ├── test_prompt.py
    ├── test_rendering.py
    ├── test_decorator.py
    ├── test_storage.py
    ├── test_history.py
    └── test_openai_wrapper.py   # fake client, no network
```

Tooling: `uv` for env/deps, `pytest`, `ruff` (lint + format). Python ≥ 3.9.

---

## 6. Implementation phases

### Phase 0 — Scaffolding
- [ ] `git init`, `pyproject.toml` (name TBD — see open questions), `src/` layout, `uv sync`
- [ ] pytest + ruff configured; empty package imports cleanly

### Phase 1 — Core `Prompt` + rendering (no DB yet)
- [ ] `RenderedText(str)` with `_pm_prompt` / `_pm_variables`
- [ ] `Prompt`: constructor validation (non-empty name/text), frozen attrs, `.raw`, `.text`,
      `.render()`, `.format()`, `__str__`, `__repr__`, equality by (name, template, variables)
- [ ] `rendering.py`: SafeDict lenient rendering, `{{}}` escaping, strict mode,
      `placeholders` extraction via `string.Formatter().parse`
- [ ] Tests: rendering matrix (missing vars, extra vars, JSON braces, nested braces, non-str
      values), immutability, provenance attributes survive typical string usage

### Phase 2 — Storage + lineage
- [ ] `config.py`: `configure()`, env vars, `enabled` flag, default db path
- [ ] `storage.py`: lazy connection (thread-local), WAL, schema creation, `user_version`
      migrations, `get_or_create_prompt`, `get_or_create_version` (hash dedup, version
      counter), all writes exception-shielded
- [ ] Lazy registration hook in `Prompt` (first `.text`/`.render` registers version once)
- [ ] `history.py`: `versions()`, `diff()` (difflib unified), `runs()`
- [ ] Tests: same-text ⇒ same version; edited text ⇒ v+1; dedup across processes (reopen db);
      disabled mode does zero I/O; concurrent registration from threads

### Phase 3 — `@prompt` decorator
- [ ] `decorator.py`: capture kwargs (incl. defaults via `inspect.signature.bind`), call fn,
      wrap returned str into `Prompt`, attach `fn_source_hash`
- [ ] Error if fn returns non-str; `functools.wraps` preserved
- [ ] Tests: defaults vs explicit kwargs, computed templates creating new versions,
      fn-source-hash recorded, decorated fn still introspectable

### Phase 4 — OpenAI wrapper + run tracking (sync, non-streaming)
- [ ] `wrap()` dispatcher: class vs instance detection
- [ ] Message walking: str / Prompt / RenderedText / content-block lists
- [ ] Run recording via `tracking.py` (renders, timing, usage, error path)
- [ ] Tests against a fake OpenAI-shaped client (no network): substitution happens, run rows
      correct, multiple prompts per call ⇒ multiple runs, tracking failure doesn't break the
      call, unwrapped-with-`.text` path also tracks via provenance
- [ ] One optional live smoke test behind `OPENAI_API_KEY` guard

### Phase 5 — Streaming + async
- [ ] Stream proxy (sync iterator + context manager), run written at stream end
- [ ] `AsyncOpenAI` support
- [ ] Tests with fake streaming/async clients

### Phase 6 — Polish
- [ ] `responses.create` support in the wrapper
- [ ] README with the three usage tiers (plain / decorator / wrapped)
- [ ] CLI viewer (`python -m prompt_manager history NAME`) — stretch
- [ ] Version + publish prep (classifiers, py.typed, LICENSE)

Phases 1–4 are the MVP the pseudocode describes; 5–6 can trail.

---

## 7. Open questions

1. **Package name** — `prompt-manager` is likely taken on PyPI. Alternatives: `promptline`,
   `promptvault`, `promptkeep`? (Doesn't block implementation; module name can be decided at
   Phase 0.)
2. **Default DB location** — plan says project-local `./.prompts.db`. If you'd rather have one
   global DB per machine (`~/.prompt_manager/prompts.db`), say so; it's a one-line default.
3. **Lenient rendering default** — plan says missing variables stay as `{var}` silently
   (JSON-in-prompt safety). Comfortable with that, or should the default warn/raise?
4. **Decorator contract** — plan says the function returns the raw template and the decorator
   returns a `Prompt` object (instead of a `(raw, rendered)` tuple). Flag if you specifically
   want the tuple form.
5. The original brief cut off at "run management — when it is running…". Runs-as-records
   (version + variables + output per call) is covered; if you meant something more (live
   monitoring, callbacks, cost aggregation), that's additive on top of the `runs` table.

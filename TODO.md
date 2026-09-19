# TODO

## Enable CI and release automation

Two GitHub Actions workflows are written and ready in `docs/workflows/`, but not
enabled: the GitHub token used for pushes lacks the `workflow` scope, and GitHub
rejects any push that adds a file under `.github/workflows/` from such a token.

- `docs/workflows/ci.yml` — ruff, pytest on Python 3.11–3.14 across Linux, macOS
  and Windows, and an 85% coverage gate, on every push to `main` and every PR.
- `docs/workflows/release.yml` — a pushed `v*` tag re-runs the suite, checks the
  tag matches the declared version, publishes to PyPI via Trusted Publishing
  (OIDC, no token in secrets) and opens a GitHub Release.

To enable them:

1. Re-issue the GitHub token with the `workflow` scope (or push over SSH).
2. `git mv docs/workflows .github/workflows` and commit.
3. Watch the first CI run — **Windows has never executed this test suite.**
4. For releases: on PyPI, add this repository + `release.yml` (environment
   `pypi`) as a trusted publisher for `promptkeep`, then `git push origin vX.Y.Z`.
   Until then, publish by hand: `uv build && uv publish --token ...`.

Until CI exists, run the gate locally before pushing:

```bash
uv run ruff format --check src tests examples && uv run ruff check src tests examples
uv run pytest -q --cov --cov-fail-under=85
```

## Provider adapters

Only OpenAI `chat.completions` has an adapter — which also covers OpenRouter
and every other OpenAI-compatible endpoint, so it is the shape we stick to
for now. Each of these is one `ProviderAdapter` subclass plus one `Scenario`
in `tests/test_adapters.py`:

- **OpenAI Responses API** (`client.responses.create`): Prompts in `input` /
  `instructions`; typed stream events; and `previous_response_id`, which
  needs a `conversation_hint` adapter method and a response-id → conversation
  lookup so turns chain with no user code.
- **Anthropic** (`client.messages.create`): separate `system` field,
  `content[0].text`, `input_tokens` / `output_tokens`, event-stream deltas.
- **LiteLLM**: a module function, not a client — `wrap(litellm.completion)`
  returns a wrapped function the user rebinds.

## Strict typing

`py.typed` ships, so downstream type checkers trust these annotations, but
`mypy --strict src/promptkeep` still reports errors — mostly untyped private
helpers and peewee model attributes. Get it clean and add it to CI.

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

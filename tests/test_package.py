"""Package-level checks: the version string and the public API surface."""

import ast
import re
from pathlib import Path

import promptkeep
import promptkeep.integrations

SOURCE_ROOT = Path(promptkeep.__file__).parent


def test_version_matches_pyproject():
    """__version__ is read from the installed distribution's metadata, so it
    must agree with the one place the version is declared."""
    pyproject = (Path(__file__).parent.parent / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', pyproject, re.MULTILINE)
    assert match is not None
    assert promptkeep.__version__ == match.group(1)


def test_all_names_resolve():
    """Everything advertised in __all__ actually exists on the package."""
    for name in promptkeep.__all__:
        assert hasattr(promptkeep, name), name


# The public API, spelled out (docs/API.md). A failure here means the surface
# changed: if that was deliberate, update these sets, docs/API.md if the rules
# moved, and the CHANGELOG — adding is a feature, removing or renaming is a
# breaking change that goes through a deprecation first.
PUBLIC_TOP_LEVEL = {
    "Prompt", "RenderedText", "prompt", "wrap", "configure", "get_settings", "history",
    "conversation", "dataset", "Dataset", "flush", "feedback", "check", "call", "acall",
    "Verdict", "CheckContext", "PromptBlocked", "suppress", "extract_placeholders",
    "MissingVariableError", "TemplateParseError", "__version__",
}  # fmt: skip
PUBLIC_HISTORY = {
    "CheckInfo", "ConversationInfo", "ConversationSummary", "PromptSummary", "RunInfo",
    "VersionInfo", "VersionStats", "all_runs", "checks", "conversation", "diff",
    "format_cost", "labels", "list_conversations", "list_prompts", "runs", "stats",
    "verdict", "versions",
}  # fmt: skip
PUBLIC_INTEGRATIONS = {
    "wrap", "call", "acall", "CallResult", "is_wrapped", "adapters", "register_adapter",
    "ProviderAdapter", "Request", "ResponseFields", "StreamAbsorber", "Target",
    "OpenAIChatAdapter", "OpenAIResponsesAdapter",
}  # fmt: skip


def test_the_public_api_changes_only_on_purpose():
    """promptkeep.__all__, history's public names and integrations.__all__
    are exactly the documented surface."""
    history_names = {
        name
        for name, obj in vars(promptkeep.history).items()
        if not name.startswith("_") and getattr(obj, "__module__", None) == "promptkeep.history"
    }
    assert set(promptkeep.__all__) == PUBLIC_TOP_LEVEL
    assert history_names == PUBLIC_HISTORY
    assert set(promptkeep.integrations.__all__) == PUBLIC_INTEGRATIONS


def test_no_function_level_intra_package_imports():
    """Every intra-package import is at module level.

    A `from . import x` inside a function body is how an import cycle gets
    papered over; the module graph is meant to be a DAG that never needs
    one. The single exception is cli.py, which imports the optional
    dashboard stack lazily on purpose (it is an extra, not a dependency).
    """
    offenders = []
    for path in SOURCE_ROOT.rglob("*.py"):
        if path.name == "cli.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in _imports_inside_functions(tree):
            if _is_intra_package(node):
                offenders.append(f"{path.relative_to(SOURCE_ROOT)}:{node.lineno}")
    assert offenders == []


def _imports_inside_functions(tree: ast.AST):
    """Every Import/ImportFrom node nested in a function or method body."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for inner in ast.walk(node):
                if isinstance(inner, (ast.Import, ast.ImportFrom)):
                    yield inner


def _is_intra_package(node: ast.Import | ast.ImportFrom) -> bool:
    """A relative import, or an absolute one of promptkeep itself."""
    if isinstance(node, ast.ImportFrom):
        return node.level > 0 or (node.module or "").startswith("promptkeep")
    return any(alias.name.startswith("promptkeep") for alias in node.names)

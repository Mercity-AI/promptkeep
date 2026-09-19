"""Package-level checks: the version string and the public API surface."""

import ast
import re
from pathlib import Path

import promptkeep

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

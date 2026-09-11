"""Package-level checks: the version string and the public API surface."""

import re
from pathlib import Path

import promptkeep


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

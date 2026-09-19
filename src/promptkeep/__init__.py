"""promptkeep: versioned prompt templates with lineage and run tracking.

Public API:

    from promptkeep import Prompt, prompt, wrap, configure, history
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version

from . import history
from .checks import CheckContext, PromptBlocked, Verdict, check, suppress
from .config import configure, get_settings
from .conversation import conversation
from .decorator import prompt
from .integrations import acall, call, wrap
from .prompts import Prompt, RenderedText
from .rendering import MissingVariableError, TemplateParseError, extract_placeholders
from .tracking import flush

try:
    # pyproject.toml is the single source of truth for the version; this reads
    # it back from the installed distribution so the two can never disagree.
    __version__ = _distribution_version("promptkeep")
except PackageNotFoundError:  # pragma: no cover - source tree without an install
    __version__ = "0.0.0+unknown"

__all__ = [
    "Prompt",
    "RenderedText",
    "prompt",
    "wrap",
    "configure",
    "get_settings",
    "history",
    "conversation",
    "flush",
    "check",
    "call",
    "acall",
    "Verdict",
    "CheckContext",
    "PromptBlocked",
    "suppress",
    "extract_placeholders",
    "MissingVariableError",
    "TemplateParseError",
    "__version__",
]

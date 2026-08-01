"""promptkeep: versioned prompt templates with lineage and run tracking.

Public API:

    from promptkeep import Prompt, prompt, wrap, configure, history
"""

from . import history
from .checks import CheckContext, PromptBlocked, Verdict, acall, call, check, suppress
from .config import configure, get_settings
from .conversation import conversation
from .decorator import prompt
from .integrations import wrap
from .prompts import Prompt, RenderedText
from .rendering import MissingVariableError, TemplateParseError, extract_placeholders
from .writer import flush

__version__ = "0.2.0"

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

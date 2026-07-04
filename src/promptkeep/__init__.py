"""promptkeep: versioned prompt templates with lineage and run tracking.

Public API:

    from promptkeep import Prompt, prompt, wrap, configure, history
"""

from . import history
from .config import configure, get_settings
from .decorator import prompt
from .integrations import wrap
from .prompts import Prompt, RenderedText
from .rendering import MissingVariableError, TemplateParseError, extract_placeholders

__version__ = "0.1.0"

__all__ = [
    "Prompt",
    "RenderedText",
    "prompt",
    "wrap",
    "configure",
    "get_settings",
    "history",
    "extract_placeholders",
    "MissingVariableError",
    "TemplateParseError",
    "__version__",
]

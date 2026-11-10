"""Extensible source adapters for saved agent traces and export snapshots."""

from ..events import TraceEvent
from .base import AgentTrace, ImportDiagnostic, ImportReport
from .claude_code import ClaudeCodeTrace, import_claude_code
from .codex import CodexTrace, import_codex
from .datasets import DatasetImportReport, import_dataset
from .engine import import_trace, register_agent_trace
from .opencode import OpenCodeTrace

for _adapter in (CodexTrace, ClaudeCodeTrace, OpenCodeTrace):
    register_agent_trace(_adapter)

__all__ = [
    "AgentTrace",
    "CodexTrace",
    "ClaudeCodeTrace",
    "OpenCodeTrace",
    "TraceEvent",
    "register_agent_trace",
    "import_trace",
    "ImportDiagnostic",
    "ImportReport",
    "import_codex",
    "import_claude_code",
    "import_dataset",
    "DatasetImportReport",
]

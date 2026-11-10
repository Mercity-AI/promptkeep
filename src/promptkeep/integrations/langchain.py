"""LangChain callbacks using native run IDs, also usable with LangGraph."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import replace

from langchain_core.callbacks import BaseCallbackHandler

from ..events import Event
from ..storage import _utcnow
from ._bridge import Bridge


def _visible(value):
    """Extract visible content without archiving hidden thinking or opaque blocks."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(
            b.get("text", "")
            for b in value
            if isinstance(b, dict) and b.get("type") in ("text", "input_text", "output_text")
        )
    return None


class PromptKeepCallbackHandler(Bridge, BaseCallbackHandler):
    """Run inline so provider wrappers can inherit model ownership where supported."""

    run_inline = True
    raise_error = False

    def __init__(self):
        """Keep bounded explicit ancestry independent of active callback scopes."""
        super().__init__()
        self._ancestry = OrderedDict()
        self._runs = {}
        self._interrupted = set()

    def _start(self, run_id, parent_run_id, name, kind, input=None, model=None):
        """Resolve ancestry by explicit run IDs, not callback thread identity."""
        if not self.enabled:
            return
        with self._lock:
            trace_id = self._ancestry.get(str(parent_run_id), str(parent_run_id or run_id))
            self._ancestry[str(run_id)] = trace_id
            if len(self._ancestry) > 10000:
                self._ancestry.popitem(last=False)
        event = Event(
            "langchain",
            trace_id,
            str(run_id),
            name or kind,
            kind="agent" if not parent_run_id and kind == "operation" else kind,
            parent_id=str(parent_run_id) if parent_run_id else None,
            started_at=_utcnow(),
            status="running",
            input=input,
            model=model,
        )
        event = self.start(event) or event
        with self._lock:
            self._runs[str(run_id)] = event
            if len(self._runs) > 10000:
                self._runs.pop(next(iter(self._runs)))

    def _end(self, run_id, output=None, error=None, usage=None, model=None, status_override=None):
        """Use stored identity even when completion arrives on another worker."""
        with self._lock:
            event = self._runs.pop(str(run_id), None)
        if event is None:
            return
        status = status_override or (
            "cancelled"
            if isinstance(error, asyncio.CancelledError)
            else "incomplete"
            if error and type(error).__name__ in ("GraphInterrupt", "GraphBubbleUp")
            else "error"
            if error
            else "ok"
        )
        with self._lock:
            if status == "incomplete":
                self._interrupted.add(event.trace_id)
            if event.kind == "agent" and event.trace_id in self._interrupted:
                status = "incomplete"
                self._interrupted.discard(event.trace_id)
        self.end(
            replace(
                event,
                ended_at=_utcnow(),
                status=status,
                output=output,
                error=str(error) if error else None,
                usage=usage or {},
                model=model or event.model,
            )
        )

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs):
        """Chains and graph nodes become structural operations."""
        self._start(
            run_id, parent_run_id, kwargs.get("name") or (serialized or {}).get("name"), "operation"
        )

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        """Finish structure without copying arbitrary application state."""
        interrupted = isinstance(outputs, dict) and bool(outputs.get("__interrupt__"))
        self._end(run_id, status_override="incomplete" if interrupted else None)

    def on_chain_error(self, error, *, run_id, **kwargs):
        """Preserve framework failures and interruption semantics."""
        self._end(run_id, error=error)

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, **kwargs):
        """Record only visible message content and explicitly supplied model names."""
        params = kwargs.get("invocation_params") or {}
        self._start(
            run_id,
            parent_run_id,
            kwargs.get("name") or "model",
            "llm",
            input=[[_visible(m.content) for m in batch] for batch in messages],
            model=params.get("model") or params.get("model_name"),
        )

    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, **kwargs):
        """Completion-style models use the same model lifecycle."""
        self._start(run_id, parent_run_id, kwargs.get("name") or "model", "llm", input=prompts)

    def on_llm_end(self, response, *, run_id, **kwargs):
        """Final responses own usage; token callbacks never multiply snapshots."""
        info = response.llm_output or {}
        usage = info.get("token_usage") or info.get("usage") or {}
        parts = []
        for batch in response.generations:
            for generation in batch:
                message = getattr(generation, "message", None)
                if message is not None:
                    parts.append(_visible(message.content) or "")
                    usage = getattr(message, "usage_metadata", None) or usage
                else:
                    parts.append(generation.text)
        self._end(run_id, output="".join(parts), usage=usage, model=info.get("model_name"))

    def on_llm_error(self, error, *, run_id, **kwargs):
        """Record an error without raising into the model invocation."""
        self._end(run_id, error=error)

    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, **kwargs):
        """Explicit tool execution is a child of its native framework parent."""
        self._start(
            run_id, parent_run_id, (serialized or {}).get("name") or "tool", "tool", input=input_str
        )

    def on_tool_end(self, output, *, run_id, **kwargs):
        """ToolMessage error status marks normal failed returns correctly."""
        failed = getattr(output, "status", None) == "error"
        self._end(
            run_id,
            output=_visible(getattr(output, "content", output)),
            error=RuntimeError("tool returned error") if failed else None,
        )

    def on_tool_error(self, error, *, run_id, **kwargs):
        """A failing tool cannot change its parent's recorded exit status."""
        self._end(run_id, error=error)

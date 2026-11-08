"""Thread-safe, bounded callback lifecycles shared by framework integrations."""

from __future__ import annotations

import functools
import logging
import threading
from collections import OrderedDict
from dataclasses import replace

from .. import events, ingestion
from ..config import get_settings
from ..tracking import flush

logger = logging.getLogger("promptkeep")


class Bridge:
    """Source events own span identity; SDK wrappers may enrich their model rows."""

    def __init_subclass__(cls, **kwargs):
        """Shield the entire callback, including provider-specific field parsing."""
        super().__init_subclass__(**kwargs)
        for name, method in list(cls.__dict__.items()):
            if not name.startswith("on_") or not callable(method):
                continue

            def shield(function):
                """Preserve callback signatures while failing open on telemetry errors."""

                @functools.wraps(function)
                def invoke(*args, **kw):
                    """A malformed external event must never break an agent invocation."""
                    try:
                        return function(*args, **kw)
                    except Exception:
                        logger.warning("promptkeep: framework callback failed", exc_info=True)

                return invoke

            setattr(cls, name, shield(method))

    def __init__(self):
        """No threads, network requests or database opens occur during construction."""
        self.enabled = True
        self._states = OrderedDict()
        self._lock = threading.RLock()

    def start(self, event):
        """Capture actual callback starts and bound unfinished-state memory."""
        if not self.enabled:
            return
        try:
            with self._lock:
                owner = next(
                    (
                        value[0]
                        for value in self._states.values()
                        if value[0].trace_key == event.trace_key
                    ),
                    None,
                )
                event = replace(
                    event, destination=owner.destination if owner else str(get_settings().db_path)
                )
                if event.span_key in self._states:
                    return
                previous = events.enter(event)
                self._states[event.span_key] = (event, previous)
                if len(self._states) > 10000:
                    _, (old, parent) = self._states.popitem(last=False)
                    events.leave(old, parent)
                    logger.warning("promptkeep: callback state capacity exceeded")
            ingestion.record(event)
        except Exception:
            logger.warning("promptkeep: callback start failed", exc_info=True)

    def end(self, event):
        """Finish without depending on same-thread or LIFO callback delivery."""
        if not self.enabled:
            return
        try:
            with self._lock:
                state = self._states.pop(event.span_key, None)
            if state:
                original, previous = state
                event = replace(
                    event,
                    started_at=event.started_at or original.started_at,
                    input=event.input if event.input is not None else original.input,
                    model=event.model or original.model,
                    destination=original.destination,
                )
                events.leave(original, previous)
            ingestion.record(event)
        except Exception:
            logger.warning("promptkeep: callback end failed", exc_info=True)

    def disable(self):
        """Invalidate scopes and release retained callback ancestry and payloads."""
        self.enabled = False
        with self._lock:
            for event, parent in self._states.values():
                events.leave(event, parent)
            self._states.clear()
            for name in ("_runs", "_ancestry", "_interrupted"):
                value = getattr(self, name, None)
                if value is not None:
                    value.clear()

    def shutdown(self, timeout=None):
        """Disable this bridge and flush only its own tracking pipeline."""
        self.disable()
        return self.force_flush(timeout)

    def force_flush(self, timeout=None):
        """Flush only recorded telemetry, not application tracing providers."""
        return flush(timeout)

"""Normalize live framework events before they enter storage or an export queue."""

from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict
from dataclasses import replace

from . import storage, writer
from .config import get_settings
from .events import Event, key
from .normalization import prepare

logger = logging.getLogger("promptkeep")


_omitted = OrderedDict()
_omitted_lock = threading.Lock()


def _structural(event, destination):
    """Keep bounded source facts for sampled-out ancestors, never discarded content."""
    observation = replace(
        event,
        input=None,
        output=None,
        input_present=False,
        output_present=False,
        error=None,
        usage={},
        cost_usd=None,
        model=None,
        response_id=None,
        metadata={"sampled_out": True},
        links=(),
        link_keys=(),
    )
    with _omitted_lock:
        _omitted[(destination, event.span_key)] = observation
        _omitted.move_to_end((destination, event.span_key))
        while len(_omitted) > 10000:
            _omitted.popitem(last=False)


def _ancestors(event, destination):
    """Recover observed sampling omissions only; queue losses remain unavailable."""
    result, seen = [], set()
    parent = key(event.source, event.trace_id, event.parent_id) if event.parent_id else None
    with _omitted_lock:
        while parent and parent not in seen:
            seen.add(parent)
            ancestor = _omitted.get((destination, parent))
            if ancestor is None:
                break
            item = prepare(ancestor, archive=True)
            item["span"]["sampled"], item["run"] = False, None
            result.append(item)
            parent = (
                key(ancestor.source, ancestor.trace_id, ancestor.parent_id)
                if ancestor.parent_id
                else None
            )
    return list(reversed(result))


def _after_fork():
    """A fork inherits no locks or sampling observations owned by its parent."""
    global _omitted_lock
    _omitted_lock = threading.Lock()
    _omitted.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def record(event: Event) -> bool:
    """Live event ingestion is fail-open and follows configured persistence mode."""
    try:
        with storage.connection_context(event.destination):
            settings = get_settings()
            item = prepare(event)
            destination = str(settings.db_path)
            if item is None:
                if settings.enabled and settings.write_mode != "off":
                    _structural(event, destination)
                return True
            items = _ancestors(event, destination) + [item]
            if get_settings().write_mode == "background":
                for prepared in items:
                    writer.submit(prepared)
            elif storage.get_db() is not None:
                with storage.notification_transaction(storage.db_proxy):
                    for prepared in items:
                        storage._insert_event(prepared)
            return True
    except Exception:
        logger.warning("promptkeep: failed to ingest external event", exc_info=True)
        return False

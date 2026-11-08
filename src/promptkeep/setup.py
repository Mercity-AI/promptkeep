"""Explicit, reversible installation of provider and framework instrumentation."""

from __future__ import annotations

import importlib
import threading
import time

from .config import get_settings
from .integrations.registry import adapters, wrap
from .tracking import flush

_lock = threading.RLock()
_installations = {}
_agents_processor = None
_agents_users = set()


class Installation:
    """Own only PromptKeep hooks, wrapped methods and optional export workers."""

    def __init__(self, key, modules, clients, endpoint, protocol):
        """Dependency validation is complete before any instrumentation is installed."""
        self._key = key
        self._closed = False
        self._methods = []
        self.langchain_callbacks = []
        self.otel_exporter = None
        self.otlp_export = None
        self._agents = False
        try:
            for client in clients:
                for adapter in adapters():
                    for target in adapter.locate_targets(client):
                        original = getattr(target.owner, target.attribute)
                        if getattr(original, "_pm_instrumented", False):
                            continue
                        self._methods.append([target.owner, target.attribute, original, None])
                wrap(client)
            for item in self._methods:
                item[3] = getattr(item[0], item[1])
            if "langchain" in modules:
                self.langchain_callbacks = [modules["langchain"].PromptKeepCallbackHandler()]
            if "opentelemetry" in modules:
                self.otel_exporter = modules["opentelemetry"].PromptKeepSpanExporter()
            if endpoint:
                self.otlp_export = modules["opentelemetry"].OTLPExport(endpoint, protocol=protocol)
            if "openai_agents" in modules:
                global _agents_processor
                sdk = importlib.import_module("agents.tracing")
                if _agents_processor is None:
                    _agents_processor = modules["openai_agents"].PromptKeepTracingProcessor()
                    sdk.add_trace_processor(_agents_processor)
                _agents_processor.enabled = True
                _agents_users.add(id(self))
                self._agents = True
        except BaseException:
            self.shutdown(timeout=0)
            raise

    def flush(self, timeout=None):
        """Share one deadline between model/check writes and outbound exports."""
        deadline = None if timeout is None else time.monotonic() + timeout
        if not flush(timeout):
            return False
        remaining = None if deadline is None else max(0, deadline - time.monotonic())
        return self.otlp_export.flush(remaining) if self.otlp_export else True

    def shutdown(self, timeout=None):
        """Disable only owned hooks and restore only methods still owned here."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with _lock:
            if self._closed:
                return self.otlp_export.shutdown(timeout) if self.otlp_export else True
            self._closed = True
            _installations.pop(self._key, None)
            if self._agents:
                _agents_users.discard(id(self))
                if not _agents_users:
                    _agents_processor.disable()
            for callback in self.langchain_callbacks:
                callback.disable()
            if self.otel_exporter:
                self.otel_exporter.shutdown()
            for owner, attribute, original, installed in reversed(self._methods):
                if installed is not None and getattr(owner, attribute) is installed:
                    setattr(owner, attribute, original)

        def remaining():
            """Apply one shutdown budget across all owned work."""
            return None if deadline is None else max(0, deadline - time.monotonic())

        completed = flush(remaining())
        if self.otlp_export:
            completed = self.otlp_export.shutdown(remaining()) and completed
        return completed

    def __enter__(self):
        """Use setup as a reversible application lifetime."""
        return self

    def __exit__(self, *args):
        """Retire owned resources without suppressing application exceptions."""
        self.shutdown()
        return False


def setup(
    *, integrations=(), clients=(), otlp_endpoint=None, otlp_protocol="http/protobuf"
) -> Installation:
    """Install explicitly named integrations; equivalent calls share one handle.

    LangChain requires passing ``installation.langchain_callbacks`` in runnable
    config. OTel ingestion requires attaching ``installation.otel_exporter`` to
    the application's own SDK processor. Neither global provider is replaced.
    """
    names = tuple(sorted(set(integrations)))
    clients = tuple(clients)
    supported = {
        "openai_agents": "openai_agents",
        "langchain": "langchain",
        "opentelemetry": "opentelemetry",
    }
    if any(name not in supported for name in names):
        raise ValueError("unknown PromptKeep integration")
    if otlp_protocol not in ("http/protobuf", "grpc"):
        raise ValueError("unsupported OTLP protocol")
    modules = {}
    required = set(names) | ({"opentelemetry"} if otlp_endpoint else set())
    for name in required:
        try:
            modules[name] = importlib.import_module("promptkeep.integrations." + supported[name])
        except ImportError:
            extra = (
                "openai-agents"
                if name == "openai_agents"
                else "otel"
                if name == "opentelemetry"
                else name
            )
            raise ImportError(f"Install promptkeep[{extra}] to enable {name}") from None
    for client in clients:
        if not any(adapter.locate_targets(client) for adapter in adapters()):
            raise TypeError("unsupported provider client")
    # One owner per client method prevents installations undoing one another.
    key = (
        names,
        tuple(id(client) for client in clients),
        otlp_endpoint,
        otlp_protocol,
        str(get_settings().db_path),
    )
    with _lock:
        if key in _installations:
            return _installations[key]
        for existing in _installations.values():
            if set(existing._key[1]).intersection(id(client) for client in clients):
                raise ValueError("client already belongs to a different installation")
        installation = Installation(key, modules, clients, otlp_endpoint, otlp_protocol)
        _installations[key] = installation
        for client in clients:
            try:
                client._promptkeep_installation = installation
            except Exception:
                pass
        return installation

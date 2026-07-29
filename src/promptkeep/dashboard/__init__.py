"""The local promptkeep dashboard (`promptkeep serve`).

Kept out of the core import path: promptkeep works with zero server
dependencies installed, and only `promptkeep.dashboard.app` (imported lazily
by the CLI's `serve` command) needs FastAPI/uvicorn/Jinja2.
"""

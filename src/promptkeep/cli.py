"""The `promptkeep` command-line entry point.

Only `serve` exists today. It lazily imports fastapi/uvicorn/jinja2 —
those are an optional extra (`pip install promptkeep[serve]`), not a core
dependency, so importing `promptkeep` itself never requires a server stack.
"""

from __future__ import annotations

import argparse


def main(argv=None) -> None:
    """Parse args and dispatch to the requested subcommand."""
    parser = argparse.ArgumentParser(prog="promptkeep")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser("serve", help="run the local dashboard")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8420)
    serve_parser.add_argument(
        "--db", default=None, help="path to the promptkeep SQLite file (default: ./.promptkeep.db)"
    )

    args = parser.parse_args(argv)
    if args.command == "serve":
        _serve(args)


def _serve(args) -> None:
    """Launch the dashboard, pointed at the given (or default-configured) DB."""
    try:
        import uvicorn
    except ImportError:
        raise SystemExit(
            "The dashboard needs extra dependencies. Install with:\n"
            "    pip install 'promptkeep[serve]'"
        ) from None

    if args.db:
        from . import configure

        configure(db_path=args.db)

    from .dashboard.app import create_app

    app = create_app()
    print(f"Serving promptkeep dashboard at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

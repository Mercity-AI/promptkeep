"""Import and inspect one local Codex rollout; no credentials or API calls."""

import argparse
import json

from promptkeep import configure, history
from promptkeep.importers import import_codex


def main() -> None:
    """Print observed operations and separately reported source usage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--db", default="codex.promptkeep.db")
    args = parser.parse_args()
    configure(db_path=args.db)
    report = import_codex(args.path)
    trace = history.trace(report.trace_key)
    imported = history.imported_session(report.trace_key)
    print(f"{report.outcome}: {trace.trace_key}; {len(trace.spans)} observed spans")
    print(json.dumps(imported.summary["total_usage"]))
    print(f"Source diagnostics: {len(report.diagnostics)}")


if __name__ == "__main__":
    main()
